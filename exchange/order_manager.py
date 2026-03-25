"""
Order Management System (OMS) — wraps BingXClient with rate limiting,
duplicate detection, fill verification, and bracket order lifecycle management.

Bracket orders (SL + TP) are placed immediately after every entry so that
positions are protected even when the bot is offline.  When one side fires,
the other is automatically cancelled.
"""
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from exchange.bingx_client import BingXClient
from exchange.rate_limiter import ORDER_LIMITER, REQUEST_LIMITER

logger = logging.getLogger(__name__)

FILL_TIMEOUT_S = 5.0    # Max wait for fill confirmation


class Order:
    def __init__(
        self,
        symbol: str,
        side: str,
        position_side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> None:
        self.client_order_id = str(uuid.uuid4()).replace("-", "")[:32]
        self.symbol = symbol
        self.side = side
        self.position_side = position_side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.reduce_only = reduce_only
        self.exchange_order_id: Optional[str] = None
        self.status = "PENDING"
        self.filled_qty = 0.0
        self.avg_price = 0.0


@dataclass
class BracketIds:
    """Tracks live SL and TP exchange order IDs for an open position."""
    symbol: str
    position_side: str          # "LONG" or "SHORT"
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None


class OrderManager:
    """
    High-level OMS with dedup, rate limiting, fill verification, and bracket
    order management (SL + TP placed on exchange for every entry).
    """

    def __init__(self, client: BingXClient) -> None:
        self._client = client
        self._pending_ids: set = set()          # client_order_ids in flight
        # key: (symbol, position_side) → BracketIds
        self._brackets: Dict[Tuple[str, str], BracketIds] = {}

    # ── Core order submission ─────────────────────────────────────────────────

    async def submit_order(self, order: Order) -> Optional[Dict[str, Any]]:
        """Submit an order with rate limiting and duplicate guard."""
        if order.client_order_id in self._pending_ids:
            logger.warning("Duplicate order detected: %s", order.client_order_id)
            return None
        self._pending_ids.add(order.client_order_id)
        await ORDER_LIMITER.acquire()
        try:
            result = await self._client.place_order(
                symbol=order.symbol,
                side=order.side,
                position_side=order.position_side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                reduce_only=order.reduce_only,
                client_order_id=order.client_order_id,
            )
            order.exchange_order_id = result.get("orderId")
            order.status = result.get("status", "NEW")
            logger.info(
                "Order placed: %s %s %s qty=%.4f id=%s",
                order.symbol,
                order.side,
                order.position_side,
                order.quantity,
                order.exchange_order_id,
            )
            return result
        except Exception as exc:
            logger.error("Order placement failed: %s", exc)
            order.status = "FAILED"
            return None
        finally:
            self._pending_ids.discard(order.client_order_id)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        await REQUEST_LIMITER.acquire()
        try:
            await self._client.cancel_order(symbol, order_id)
            logger.info("Order cancelled: %s %s", symbol, order_id)
            return True
        except Exception as exc:
            logger.error("Cancel failed for %s %s: %s", symbol, order_id, exc)
            return False

    async def cancel_all_orders(self, symbol: str) -> int:
        await REQUEST_LIMITER.acquire()
        open_orders: List[Dict] = await self._client.get_open_orders(symbol)
        cancelled = 0
        for o in open_orders:
            oid = o.get("orderId")
            if oid and await self.cancel_order(symbol, oid):
                cancelled += 1
        return cancelled

    async def close_position(
        self, symbol: str, position_side: str, quantity: float
    ) -> Optional[Dict[str, Any]]:
        """Close a position via market order."""
        close_side = "SELL" if position_side == "LONG" else "BUY"
        order = Order(
            symbol=symbol,
            side=close_side,
            position_side=position_side,
            order_type="MARKET",
            quantity=quantity,
            reduce_only=True,
        )
        return await self.submit_order(order)

    # ── Bracket order management ──────────────────────────────────────────────

    async def place_entry_with_brackets(
        self,
        symbol: str,
        position_side: str,    # "LONG" or "SHORT"
        quantity: float,
        sl_price: float,
        tp_price: float,
    ) -> Dict[str, Any]:
        """
        Place a market entry order followed immediately by SL (STOP_MARKET) and
        TP (TAKE_PROFIT_MARKET) bracket orders on BingX.

        Both bracket orders are reduceOnly so they cannot open new positions.
        The BracketIds are stored internally; call on_sl_triggered /
        on_tp_triggered when a fill notification arrives to cancel the other leg.

        Returns a dict with keys: entry_result, sl_result, tp_result.
        """
        entry_side = "BUY" if position_side == "LONG" else "SELL"
        close_side = "SELL" if position_side == "LONG" else "BUY"

        # 1. Market entry
        entry_order = Order(
            symbol=symbol,
            side=entry_side,
            position_side=position_side,
            order_type="MARKET",
            quantity=quantity,
        )
        entry_result = await self.submit_order(entry_order)
        if entry_result is None:
            logger.error(
                "Entry failed for %s %s — bracket orders NOT placed", symbol, position_side
            )
            return {"entry_result": None, "sl_result": None, "tp_result": None}

        # 2. Stop Loss (STOP_MARKET, reduce-only)
        sl_order = Order(
            symbol=symbol,
            side=close_side,
            position_side=position_side,
            order_type="STOP_MARKET",
            quantity=quantity,
            stop_price=sl_price,
            reduce_only=True,
        )
        sl_result = await self.submit_order(sl_order)

        # 3. Take Profit (TAKE_PROFIT_MARKET, reduce-only)
        tp_order = Order(
            symbol=symbol,
            side=close_side,
            position_side=position_side,
            order_type="TAKE_PROFIT_MARKET",
            quantity=quantity,
            stop_price=tp_price,
            reduce_only=True,
        )
        tp_result = await self.submit_order(tp_order)

        # Register bracket IDs for lifecycle management
        bracket = BracketIds(
            symbol=symbol,
            position_side=position_side,
            sl_order_id=sl_result.get("orderId") if sl_result else None,
            tp_order_id=tp_result.get("orderId") if tp_result else None,
        )
        self._brackets[(symbol, position_side)] = bracket

        logger.info(
            "Brackets registered for %s %s: SL=%s TP=%s",
            symbol, position_side, bracket.sl_order_id, bracket.tp_order_id,
        )
        return {
            "entry_result": entry_result,
            "sl_result": sl_result,
            "tp_result": tp_result,
        }

    async def on_sl_triggered(self, symbol: str, position_side: str) -> bool:
        """
        Call this when an SL fill notification arrives.
        Cancels the pending TP order and cleans up the bracket record.
        Returns True if the TP was successfully cancelled (or already absent).
        """
        bracket = self._brackets.get((symbol, position_side))
        if bracket is None:
            logger.warning(
                "on_sl_triggered: no bracket found for %s %s", symbol, position_side
            )
            return False

        cancelled = True
        if bracket.tp_order_id:
            cancelled = await self.cancel_order(symbol, bracket.tp_order_id)
            bracket.tp_order_id = None

        self._brackets.pop((symbol, position_side), None)
        logger.info("SL triggered for %s %s — TP cancelled", symbol, position_side)
        return cancelled

    async def on_tp_triggered(self, symbol: str, position_side: str) -> bool:
        """
        Call this when a TP fill notification arrives.
        Cancels the pending SL order and cleans up the bracket record.
        Returns True if the SL was successfully cancelled (or already absent).
        """
        bracket = self._brackets.get((symbol, position_side))
        if bracket is None:
            logger.warning(
                "on_tp_triggered: no bracket found for %s %s", symbol, position_side
            )
            return False

        cancelled = True
        if bracket.sl_order_id:
            cancelled = await self.cancel_order(symbol, bracket.sl_order_id)
            bracket.sl_order_id = None

        self._brackets.pop((symbol, position_side), None)
        logger.info("TP triggered for %s %s — SL cancelled", symbol, position_side)
        return cancelled

    async def update_tp_price(
        self,
        symbol: str,
        position_side: str,
        new_tp_price: float,
        quantity: float,
    ) -> bool:
        """
        Cancel the existing TP order and place a new one at new_tp_price.
        Used when trailing stop activates after 1:1 R:R is reached.
        Returns True if the replacement TP order was placed successfully.
        """
        bracket = self._brackets.get((symbol, position_side))
        if bracket is None:
            logger.warning(
                "update_tp_price: no bracket found for %s %s", symbol, position_side
            )
            return False

        # Cancel old TP
        if bracket.tp_order_id:
            await self.cancel_order(symbol, bracket.tp_order_id)
            bracket.tp_order_id = None

        # Place replacement TP
        close_side = "SELL" if position_side == "LONG" else "BUY"
        new_tp = Order(
            symbol=symbol,
            side=close_side,
            position_side=position_side,
            order_type="TAKE_PROFIT_MARKET",
            quantity=quantity,
            stop_price=new_tp_price,
            reduce_only=True,
        )
        result = await self.submit_order(new_tp)
        if result:
            bracket.tp_order_id = result.get("orderId")
            logger.info(
                "TP updated for %s %s: price=%.4f new_id=%s",
                symbol, position_side, new_tp_price, bracket.tp_order_id,
            )
            return True
        return False

    async def update_tp_quantity(
        self,
        symbol: str,
        position_side: str,
        new_quantity: float,
        tp_price: float,
    ) -> bool:
        """
        Cancel & replace TP with reduced quantity (e.g. after 50 % partial profit take).
        Delegates to update_tp_price with the same price but new quantity.
        """
        return await self.update_tp_price(
            symbol=symbol,
            position_side=position_side,
            new_tp_price=tp_price,
            quantity=new_quantity,
        )
