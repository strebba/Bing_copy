"""
BingX WebSocket handler — market data and user data streams.
"""

import asyncio
import gzip
import json
import logging
import time
from typing import Any, Callable, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from config import settings
from core.event_bus import Event, EventType

logger = logging.getLogger(__name__)

MessageCallback = Callable[[Dict[str, Any]], None]


class BingXWebSocket:
    """Manages a single WebSocket connection with auto-reconnect."""

    PING_INTERVAL = 20  # seconds
    RECONNECT_DELAY = 3  # seconds
    MAX_RECONNECT_DELAY = 60  # seconds

    def __init__(self, url: str = settings.BINGX_WS_URL) -> None:
        self._url = url
        self._ws: Optional[Any] = None
        self._subscriptions: Dict[str, Dict] = {}  # stream_id → sub message
        self._callbacks: Dict[str, MessageCallback] = {}
        self._running = False
        self._reconnect_delay = self.RECONNECT_DELAY

    # ── Public API ────────────────────────────────────────────────────────────

    def subscribe(
        self, stream_id: str, sub_msg: Dict, callback: MessageCallback
    ) -> None:
        self._subscriptions[stream_id] = sub_msg
        self._callbacks[stream_id] = callback

    def unsubscribe(self, stream_id: str) -> None:
        self._subscriptions.pop(stream_id, None)
        self._callbacks.pop(stream_id, None)

    async def start(self) -> None:
        self._running = True
        while self._running:
            try:
                await self._connect()
            except Exception as exc:
                if self._running:
                    logger.warning(
                        "WebSocket disconnected (%s). Reconnecting in %ds…",
                        exc,
                        self._reconnect_delay,
                    )
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2, self.MAX_RECONNECT_DELAY
                    )

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _connect(self) -> None:
        logger.info("Connecting to BingX WebSocket %s", self._url)
        async with websockets.connect(self._url, ping_interval=None) as ws:
            self._ws = ws
            self._reconnect_delay = self.RECONNECT_DELAY
            logger.info("WebSocket connected")

            # Re-subscribe all streams
            for sub_msg in self._subscriptions.values():
                await ws.send(json.dumps(sub_msg))

            ping_task = asyncio.create_task(self._ping_loop(ws))
            try:
                async for raw in ws:
                    await self._handle_message(raw)
            except ConnectionClosed:
                pass
            finally:
                ping_task.cancel()

    async def _ping_loop(self, ws: Any) -> None:
        while True:
            await asyncio.sleep(self.PING_INTERVAL)
            try:
                await ws.send("Ping")
            except Exception:
                break

    async def _handle_message(self, raw: Any) -> None:
        try:
            if isinstance(raw, bytes):
                try:
                    text = gzip.decompress(raw).decode("utf-8")
                except Exception:
                    text = raw.decode("utf-8")
            else:
                text = raw

            if text == "Pong":
                return

            data = json.loads(text)
            data_type = data.get("dataType", "")

            for stream_id, callback in list(self._callbacks.items()):
                if stream_id in data_type or data_type == stream_id:
                    try:
                        callback(data)
                    except Exception as exc:
                        logger.error("Callback error for %s: %s", stream_id, exc)

        except Exception as exc:
            logger.error("Error handling WS message: %s", exc)


class MarketDataStream:
    """High-level market data subscription helpers."""

    def __init__(self, ws: BingXWebSocket) -> None:
        self._ws = ws

    def subscribe_kline(
        self, symbol: str, interval: str, callback: MessageCallback
    ) -> str:
        stream_id = f"{symbol}@kline_{interval}"
        sub_msg = {
            "id": stream_id,
            "reqType": "sub",
            "dataType": stream_id,
        }
        self._ws.subscribe(stream_id, sub_msg, callback)
        return stream_id

    def subscribe_depth(self, symbol: str, callback: MessageCallback) -> str:
        stream_id = f"{symbol}@depth20"
        sub_msg = {"id": stream_id, "reqType": "sub", "dataType": stream_id}
        self._ws.subscribe(stream_id, sub_msg, callback)
        return stream_id

    def subscribe_trade(self, symbol: str, callback: MessageCallback) -> str:
        stream_id = f"{symbol}@trade"
        sub_msg = {"id": stream_id, "reqType": "sub", "dataType": stream_id}
        self._ws.subscribe(stream_id, sub_msg, callback)
        return stream_id


LISTEN_KEY_REFRESH_INTERVAL_S = 30 * 60
USER_DATA_STREAM_TYPE = "ORDER_TRADE_UPDATE"


class UserDataStream:
    """WebSocket stream for user account events (ORDER_TRADE_UPDATE, ACCOUNT_UPDATE)."""

    def __init__(
        self,
        bingx_client,
        order_manager,
        state_manager,
        event_bus,
    ) -> None:
        self._client = bingx_client
        self._order_mgr = order_manager
        self._state = state_manager
        self._event_bus = event_bus
        self._ws: Optional[Any] = None
        self._listen_key: Optional[str] = None
        self._running = False
        self._refresh_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._running = True
        self._listen_key = await self._client.create_listen_key()
        logger.info("User data stream listen key created")
        asyncio.create_task(self._listen_key_refresh_loop())
        await self._connect()

    async def stop(self) -> None:
        self._running = False
        if self._refresh_task:
            self._refresh_task.cancel()
        if self._ws:
            await self._ws.close()

    async def _connect(self) -> None:
        url = f"{settings.BINGX_USER_WS_URL}?listenKey={self._listen_key}"
        logger.info("Connecting to user data stream WS: %s", url)
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=None) as ws:
                    self._ws = ws
                    logger.info("User data stream WebSocket connected")
                    async for raw in ws:
                        await self._handle_message(raw)
            except ConnectionClosed:
                logger.warning("User data stream WS closed")
            except Exception as exc:
                logger.warning(
                    "User data stream WS error: %s. Reconnecting in 5s...", exc
                )
                await asyncio.sleep(5)

            if self._running:
                try:
                    self._listen_key = await self._client.create_listen_key()
                    url = f"{settings.BINGX_USER_WS_URL}?listenKey={self._listen_key}"
                except Exception as exc:
                    logger.error("Failed to create new listen key: %s", exc)
                    await asyncio.sleep(5)

    async def _listen_key_refresh_loop(self) -> None:
        while self._running:
            await asyncio.sleep(LISTEN_KEY_REFRESH_INTERVAL_S)
            if not self._running:
                break
            try:
                await self._client.refresh_listen_key(self._listen_key)
                logger.debug("Listen key refreshed")
            except Exception as exc:
                logger.warning("Listen key refresh failed: %s — reconnecting", exc)
                try:
                    self._listen_key = await self._client.create_listen_key()
                    url = f"{settings.BINGX_USER_WS_URL}?listenKey={self._listen_key}"
                    if self._ws:
                        await self._ws.close()
                except Exception as new_key_exc:
                    logger.error("Failed to create new listen key: %s", new_key_exc)

    async def _handle_message(self, raw: Any) -> None:
        try:
            if isinstance(raw, bytes):
                try:
                    text = gzip.decompress(raw).decode("utf-8")
                except Exception:
                    text = raw.decode("utf-8")
            else:
                text = raw

            if text == "Pong":
                return

            data = json.loads(text)
            event_type = data.get("dataType", "")

            if event_type == USER_DATA_STREAM_TYPE:
                await self._on_order_trade_update(data)

        except Exception as exc:
            logger.error("Error handling user data stream message: %s", exc)

    async def _on_order_trade_update(self, data: Dict[str, Any]) -> None:
        order_data = data.get("data", {}).get("order", {})
        status = order_data.get("status", "")
        order_id = str(order_data.get("orderId", ""))
        symbol = order_data.get("symbol", "")
        side = order_data.get("side", "")
        position_side = order_data.get("positionSide", "")
        order_type = order_data.get("type", "")

        if status != "FILLED":
            return

        logger.info(
            "ORDER_TRADE_UPDATE FILLED: symbol=%s orderId=%s type=%s side=%s",
            symbol,
            order_id,
            order_type,
            side,
        )

        all_positions = self._state.all_positions()
        pos = self._state.get_position(symbol, position_side)

        if pos is None:
            return

        is_sl = order_type in ("STOP_MARKET", "STOP") and side == (
            "SELL" if position_side == "LONG" else "BUY"
        )
        is_tp = order_type in ("TAKE_PROFIT_MARKET", "TAKE_PROFIT") and side == (
            "SELL" if position_side == "LONG" else "BUY"
        )

        if not is_sl and not is_tp:
            return

        if is_sl:
            await self._handle_sl_fill(pos, symbol, position_side)
        elif is_tp:
            await self._handle_tp_fill(pos, symbol, position_side)

    async def _handle_sl_fill(self, pos, symbol: str, position_side: str) -> None:
        logger.info("SL HIT: %s %s", symbol, position_side)
        await self._order_mgr.on_sl_triggered(symbol, position_side)
        self._state.close_position(symbol, position_side)
        await self._event_bus.publish(
            Event(
                EventType.SL_HIT,
                {"symbol": symbol, "direction": position_side},
            )
        )
        await self._event_bus.publish(
            Event(
                EventType.POSITION_CLOSED,
                {"symbol": symbol, "direction": position_side, "reason": "stop_loss"},
            )
        )

    async def _handle_tp_fill(self, pos, symbol: str, position_side: str) -> None:
        logger.info("TP HIT: %s %s", symbol, position_side)
        await self._order_mgr.on_tp_triggered(symbol, position_side)
        self._state.close_position(symbol, position_side)
        await self._event_bus.publish(
            Event(
                EventType.TP_HIT,
                {"symbol": symbol, "direction": position_side},
            )
        )
        await self._event_bus.publish(
            Event(
                EventType.POSITION_CLOSED,
                {"symbol": symbol, "direction": position_side, "reason": "take_profit"},
            )
        )
