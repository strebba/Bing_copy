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

logger = logging.getLogger(__name__)

MessageCallback = Callable[[Dict[str, Any]], None]


class BingXWebSocket:
    """Manages a single WebSocket connection with auto-reconnect."""

    PING_INTERVAL = 20        # seconds
    RECONNECT_DELAY = 3       # seconds
    MAX_RECONNECT_DELAY = 60  # seconds

    def __init__(self, url: str = settings.BINGX_WS_URL) -> None:
        self._url = url
        self._ws: Optional[Any] = None
        self._subscriptions: Dict[str, Dict] = {}   # stream_id → sub message
        self._callbacks: Dict[str, MessageCallback] = {}
        self._running = False
        self._reconnect_delay = self.RECONNECT_DELAY

    # ── Public API ────────────────────────────────────────────────────────────

    def subscribe(self, stream_id: str, sub_msg: Dict, callback: MessageCallback) -> None:
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

    def subscribe_kline(self, symbol: str, interval: str, callback: MessageCallback) -> str:
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
