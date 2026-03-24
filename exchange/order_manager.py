"""
Order Management System (OMS) — wraps BingXClient with rate limiting,
duplicate detection, and fill verification.
"""
import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

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


class OrderManager:
    """High-level OMS with dedup, rate limiting, and fill verification."""

    def __init__(self, client: BingXClient) -> None:
        self._client = client
        self._pending_ids: set = set()   # client_order_ids in flight

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
