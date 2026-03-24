"""
Health check — monitors connectivity, heartbeat, and system vitals.
"""
import asyncio
import logging
import time
from typing import Any, Dict

from exchange.bingx_client import BingXClient

logger = logging.getLogger(__name__)

MAX_HEARTBEAT_GAP_S = 90   # 3 × 30 s heartbeat interval


class HealthChecker:
    """Checks exchange connectivity and internal heartbeat."""

    def __init__(self, client: BingXClient) -> None:
        self._client = client
        self._last_heartbeat: float = time.time()
        self._last_ws_message: float = time.time()
        self._running = False

    def record_heartbeat(self) -> None:
        self._last_heartbeat = time.time()

    def record_ws_message(self) -> None:
        self._last_ws_message = time.time()

    async def check_exchange_connectivity(self) -> bool:
        try:
            await self._client.get_ticker("BTC-USDT")
            return True
        except Exception as exc:
            logger.error("Exchange connectivity check failed: %s", exc)
            return False

    def heartbeat_ok(self) -> bool:
        gap = time.time() - self._last_heartbeat
        return gap < MAX_HEARTBEAT_GAP_S

    def ws_ok(self) -> bool:
        gap = time.time() - self._last_ws_message
        return gap < MAX_HEARTBEAT_GAP_S

    async def run_health_loop(self, interval_s: int = 60) -> None:
        self._running = True
        while self._running:
            await asyncio.sleep(interval_s)
            connected = await self.check_exchange_connectivity()
            hb_ok = self.heartbeat_ok()
            if not connected or not hb_ok:
                logger.critical(
                    "Health check FAILED — connected=%s heartbeat_ok=%s",
                    connected, hb_ok,
                )
            else:
                logger.debug("Health check OK")

    def status(self) -> Dict[str, Any]:
        return {
            "heartbeat_ok": self.heartbeat_ok(),
            "ws_ok": self.ws_ok(),
            "last_heartbeat_s": round(time.time() - self._last_heartbeat, 1),
            "last_ws_s": round(time.time() - self._last_ws_message, 1),
        }
