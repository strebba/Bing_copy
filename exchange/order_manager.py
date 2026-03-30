"""
Order Management System (OMS) — wraps BingXClient with rate limiting,
duplicate detection, fill verification, and bracket order lifecycle management.

Bracket orders (SL + TP) are placed immediately after every entry so that
positions are protected even when the bot is offline.  When one side fires,
the other is automatically cancelled.

All financial calculations use Decimal to prevent floating-point errors.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

from core.finance import (
    calculate_pnl,
    calculate_quantity_from_risk,
    calculate_risk_amount,
    calculate_slippage_bps,
    round_money,
    round_price,
    round_quantity,
    to_decimal,
)
from exchange.bingx_client import BingXClient
from exchange.rate_limiter import ORDER_LIMITER, REQUEST_LIMITER

logger = logging.getLogger(__name__)

FILL_TIMEOUT_S = 5.0  # Max wait for fill confirmation
FILL_POLL_INTERVAL_S = 0.5  # Poll interval for fill status


@dataclass
class FillResult:
    """Result of a fill price query after market order execution.

    All monetary values are stored as Decimal for precision, then converted
    to float for API compatibility.
    """

    order_id: str
    avg_price: float
    executed_qty: float
    status: str

    # Precision fields (stored as Decimal, exposed as float)
    _avg_price_decimal: Optional[Decimal] = None
    _executed_qty_decimal: Optional[Decimal] = None

    def __post_init__(self) -> None:
        self._avg_price_decimal = to_decimal(self.avg_price)
        self._executed_qty_decimal = to_decimal(self.executed_qty)

    @property
    def avg_price_decimal(self) -> Decimal:
        return self._avg_price_decimal or Decimal("0")

    @property
    def executed_qty_decimal(self) -> Decimal:
        return self._executed_qty_decimal or Decimal("0")


class Order:
    def __init__(
        self,
        symbol: str,
        side: str,
        position_side: str,
        order_type: str,
        quantity: Union[float, Decimal],
        price: Optional[Union[float, Decimal]] = None,
        stop_price: Optional[Union[float, Decimal]] = None,
        reduce_only: bool = False,
    ) -> None:
        self.client_order_id = str(uuid.uuid4()).replace("-", "")[:32]
        self.symbol = symbol
        self.side = side
        self.position_side = position_side
        self.order_type = order_type
        self._quantity = to_decimal(quantity)
        self._price = to_decimal(price) if price else None
        self._stop_price = to_decimal(stop_price) if stop_price else None
        self.reduce_only = reduce_only
        self.exchange_order_id: Optional[str] = None
        self.status = "PENDING"
        self.filled_qty: float = 0.0
        self.avg_price: float = 0.0

    @property
    def quantity(self) -> float:
        return round_quantity(self._quantity)

    @quantity.setter
    def quantity(self, value: Union[float, Decimal]) -> None:
        self._quantity = to_decimal(value)

    @property
    def quantity_decimal(self) -> Decimal:
        return self._quantity

    @property
    def price(self) -> Optional[float]:
        if self._price is None:
            return None
        return round_price(self._price)

    @price.setter
    def price(self, value: Optional[Union[float, Decimal]]) -> None:
        self._price = to_decimal(value) if value else None

    @property
    def price_decimal(self) -> Optional[Decimal]:
        return self._price

    @property
    def stop_price(self) -> Optional[float]:
        if self._stop_price is None:
            return None
        return round_price(self._stop_price)

    @stop_price.setter
    def stop_price(self, value: Optional[Union[float, Decimal]]) -> None:
        self._stop_price = to_decimal(value) if value else None

    @property
    def stop_price_decimal(self) -> Optional[Decimal]:
        return self._stop_price


@dataclass
class BracketIds:
    """Tracks live SL and TP exchange order IDs for an open position."""

    symbol: str
    position_side: str  # "LONG" or "SHORT"
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None


class OrderManager:
    """
    High-level OMS with dedup, rate limiting, fill verification, and bracket
    order management (SL + TP placed on exchange for every entry).
    """

    def __init__(self, client: BingXClient) -> None:
        self._client = client
        self._pending_ids: set = set()  # client_order_ids in flight
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
                "Order placed: %s %s %s qty=%s id=%s",
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

    async def query_fill(self, symbol: str, order_id: str) -> Optional[FillResult]:
        """
        Query the executed order to get the real fill price.
        Polls until the order is FILLED or timeout is reached.
        Returns FillResult with avgPrice and executedQty.

        Uses Decimal precision internally to avoid floating-point errors.
        """
        await REQUEST_LIMITER.acquire()
        elapsed = 0.0
        while elapsed < FILL_TIMEOUT_S:
            try:
                detail = await self._client.get_order_detail(symbol, order_id)
                status = detail.get("status", "")
                avg_price_raw = detail.get("avgPrice", "0")
                executed_qty_raw = detail.get("executedQty", "0")

                # Convert to Decimal for precision, then to float for API
                avg_price = round_price(to_decimal(avg_price_raw))
                executed_qty = round_quantity(to_decimal(executed_qty_raw))

                if status == "FILLED" and avg_price > 0:
                    logger.info(
                        "Fill confirmed: %s orderId=%s avgPrice=%s executedQty=%s",
                        symbol,
                        order_id,
                        avg_price,
                        executed_qty,
                    )
                    return FillResult(
                        order_id=order_id,
                        avg_price=avg_price,
                        executed_qty=executed_qty,
                        status=status,
                    )

                if status in ("CANCELLED", "EXPIRED", "FAILED"):
                    logger.warning(
                        "Order not filled: %s orderId=%s status=%s",
                        symbol,
                        order_id,
                        status,
                    )
                    return None

            except Exception as exc:
                logger.warning("Fill query error for %s: %s", order_id, exc)

            await asyncio.sleep(FILL_POLL_INTERVAL_S)
            elapsed += FILL_POLL_INTERVAL_S

        logger.warning(
            "Fill query timeout after %.1fs: %s orderId=%s",
            FILL_TIMEOUT_S,
            symbol,
            order_id,
        )
        return None

    async def submit_market_order_with_fill(
        self, order: Order
    ) -> tuple[Optional[Dict[str, Any]], Optional[FillResult]]:
        """
        Submit a MARKET order and immediately query for the fill price.
        Returns (placement_result, fill_result).
        """
        result = await self.submit_order(order)
        if not result:
            return None, None

        order_id = result.get("orderId")
        if not order_id:
            return result, None

        fill = await self.query_fill(order.symbol, order_id)
        if fill:
            order.avg_price = fill.avg_price
            order.filled_qty = fill.executed_qty
            order.status = fill.status
        return result, fill

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
        self, symbol: str, position_side: str, quantity: Union[float, Decimal]
    ) -> Optional[Dict[str, Any]]:
        """Close a position via market order."""
        close_side = "SELL" if position_side == "LONG" else "BUY"
        qty = to_decimal(quantity)
        order = Order(
            symbol=symbol,
            side=close_side,
            position_side=position_side,
            order_type="MARKET",
            quantity=qty,
            reduce_only=True,
        )
        return await self.submit_order(order)

    # ── Bracket order management ──────────────────────────────────────────────

    async def place_entry_with_brackets(
        self,
        symbol: str,
        position_side: str,  # "LONG" or "SHORT"
        quantity: Union[float, Decimal],
        sl_price: Union[float, Decimal],
        tp_price: Union[float, Decimal],
    ) -> Dict[str, Any]:
        """
        Place a market entry order followed immediately by SL (STOP_MARKET) and
        TP (TAKE_PROFIT_MARKET) bracket orders on BingX.

        Both bracket orders are reduceOnly so they cannot open new positions.
        The BracketIds are stored internally; call on_sl_triggered /
        on_tp_triggered when a fill notification arrives to cancel the other leg.

        All price and quantity values are converted to Decimal internally
        and rounded to exchange-preferred precision before submission.

        Returns a dict with keys: entry_result, sl_result, tp_result.
        """
        # Convert all inputs to Decimal with proper precision
        qty = to_decimal(quantity)
        sl = to_decimal(sl_price)
        tp = to_decimal(tp_price)

        entry_side = "BUY" if position_side == "LONG" else "SELL"
        close_side = "SELL" if position_side == "LONG" else "BUY"

        # 1. Market entry
        entry_order = Order(
            symbol=symbol,
            side=entry_side,
            position_side=position_side,
            order_type="MARKET",
            quantity=qty,
        )
        entry_result = await self.submit_order(entry_order)
        if entry_result is None:
            logger.error(
                "Entry failed for %s %s — bracket orders NOT placed",
                symbol,
                position_side,
            )
            return {"entry_result": None, "sl_result": None, "tp_result": None}

        # 2. Stop Loss (STOP_MARKET, reduce-only)
        sl_order = Order(
            symbol=symbol,
            side=close_side,
            position_side=position_side,
            order_type="STOP_MARKET",
            quantity=qty,
            stop_price=sl,
            reduce_only=True,
        )
        sl_result = await self.submit_order(sl_order)

        # 3. Take Profit (TAKE_PROFIT_MARKET, reduce-only)
        tp_order = Order(
            symbol=symbol,
            side=close_side,
            position_side=position_side,
            order_type="TAKE_PROFIT_MARKET",
            quantity=qty,
            stop_price=tp,
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
            symbol,
            position_side,
            bracket.sl_order_id,
            bracket.tp_order_id,
        )
        return {
            "entry_result": entry_result,
            "sl_result": sl_result,
            "tp_result": tp_result,
        }

    async def place_brackets(
        self,
        symbol: str,
        position_side: str,
        quantity: Union[float, Decimal],
        sl_price: Union[float, Decimal],
        tp_price: Union[float, Decimal],
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Place SL and TP bracket orders for an already-open position.
        Used after fill price is known and SL/TP have been recalculated.

        All inputs converted to Decimal and rounded to exchange precision.

        Returns (sl_result, tp_result).
        """
        # Convert inputs to Decimal
        qty = to_decimal(quantity)
        sl = to_decimal(sl_price)
        tp = to_decimal(tp_price)

        close_side = "SELL" if position_side == "LONG" else "BUY"

        sl_order = Order(
            symbol=symbol,
            side=close_side,
            position_side=position_side,
            order_type="STOP_MARKET",
            quantity=qty,
            stop_price=sl,
            reduce_only=True,
        )
        sl_result = await self.submit_order(sl_order)

        tp_order = Order(
            symbol=symbol,
            side=close_side,
            position_side=position_side,
            order_type="TAKE_PROFIT_MARKET",
            quantity=qty,
            stop_price=tp,
            reduce_only=True,
        )
        tp_result = await self.submit_order(tp_order)

        bracket = BracketIds(
            symbol=symbol,
            position_side=position_side,
            sl_order_id=sl_result.get("orderId") if sl_result else None,
            tp_order_id=tp_result.get("orderId") if tp_result else None,
        )
        self._brackets[(symbol, position_side)] = bracket

        logger.info(
            "Brackets placed for %s %s: SL=%s TP=%s SL_id=%s TP_id=%s",
            symbol,
            position_side,
            sl,
            tp,
            bracket.sl_order_id,
            bracket.tp_order_id,
        )
        return sl_result, tp_result

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
        new_tp_price: Union[float, Decimal],
        quantity: Union[float, Decimal],
    ) -> bool:
        """
        Cancel the existing TP order and place a new one at new_tp_price.
        Used when trailing stop activates after 1:1 R:R is reached.
        Returns True if the replacement TP order was placed successfully.

        Uses Decimal precision for all price/quantity calculations.
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
            quantity=to_decimal(quantity),
            stop_price=to_decimal(new_tp_price),
            reduce_only=True,
        )
        result = await self.submit_order(new_tp)
        if result:
            bracket.tp_order_id = result.get("orderId")
            logger.info(
                "TP updated for %s %s: price=%s new_id=%s",
                symbol,
                position_side,
                new_tp_price,
                bracket.tp_order_id,
            )
            return True
        return False

    async def update_tp_quantity(
        self,
        symbol: str,
        position_side: str,
        new_quantity: Union[float, Decimal],
        tp_price: Union[float, Decimal],
    ) -> bool:
        """
        Cancel & replace TP with reduced quantity (e.g. after 50 % partial profit take).
        Delegates to update_tp_price with the same price but new quantity.

        Uses Decimal precision.
        """
        return await self.update_tp_price(
            symbol=symbol,
            position_side=position_side,
            new_tp_price=to_decimal(tp_price),
            quantity=to_decimal(new_quantity),
        )
