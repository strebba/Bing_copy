"""
Emergency stop — kill switch to close all positions immediately.
Can be triggered programmatically or via Telegram /stop command.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EmergencyStop:
    """
    Global kill switch. When activated, the trading engine should
    stop accepting new signals and close all positions immediately.
    """

    MAX_RETRIES = 5
    BASE_RETRY_DELAY = 2  # seconds, with exponential backoff

    def __init__(self, alerting: Optional[Any] = None) -> None:
        self._active = False
        self._reason: str = ""
        self._alerting = alerting

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self, reason: str = "Manual kill switch") -> None:
        self._active = True
        self._reason = reason
        logger.critical("EMERGENCY STOP ACTIVATED: %s", reason)

    def deactivate(self) -> None:
        self._active = False
        self._reason = ""
        logger.warning("Emergency stop deactivated")

    @property
    def reason(self) -> str:
        return self._reason

    async def close_all_positions(
        self,
        order_manager: Any,
        open_positions: List[Dict],
    ) -> int:
        """Close all open positions via market orders with aggressive retry."""
        closed = 0
        remaining: List[Dict] = []

        for pos in open_positions:
            symbol = pos.get("symbol")
            pos_side = pos.get("positionSide", "LONG")
            qty = abs(float(pos.get("positionAmt", 0)))
            if qty > 0 and symbol:
                remaining.append(pos)

        if not remaining:
            return 0

        for attempt in range(self.MAX_RETRIES):
            if not remaining:
                break

            failed: List[Dict] = []
            tasks = []
            for pos in remaining:
                symbol = pos.get("symbol")
                pos_side = pos.get("positionSide", "LONG")
                qty = abs(float(pos.get("positionAmt", 0)))
                tasks.append(
                    self._try_close(order_manager, symbol, pos_side, qty, attempt)
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for pos, result in zip(remaining, results):
                if isinstance(result, Exception) or not result:
                    failed.append(pos)
                else:
                    closed += 1

            remaining = failed
            if remaining:
                delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "Retrying %d positions in %ds (attempt %d/%d)",
                    len(remaining), delay, attempt + 1, self.MAX_RETRIES,
                )
                await asyncio.sleep(delay)

        logger.info(
            "Emergency close: %d/%d positions closed", closed, len(open_positions)
        )

        if remaining:
            await self._send_critical_alert(remaining)

        return closed

    async def _try_close(
        self,
        order_manager: Any,
        symbol: str,
        position_side: str,
        quantity: float,
        attempt: int,
    ) -> Any:
        """Attempt to close a single position."""
        try:
            result = await order_manager.close_position(symbol, position_side, quantity)
            logger.info(
                "Closed %s %s on attempt %d", symbol, position_side, attempt + 1
            )
            return result
        except Exception as e:
            logger.error("Failed to close %s %s: %s", symbol, position_side, e)
            raise

    async def _send_critical_alert(self, remaining: List[Dict]) -> None:
        """Send critical alert for positions that could not be closed."""
        details = []
        for p in remaining:
            symbol = p.get("symbol", "???")
            side = p.get("positionSide", "???")
            qty = abs(float(p.get("positionAmt", 0)))
            details.append(f"- {symbol} {side} {qty}")

        message = (
            "\U0001f6a8\U0001f6a8 MANUAL INTERVENTION REQUIRED \U0001f6a8\U0001f6a8\n"
            f"Failed to close {len(remaining)} positions "
            f"after {self.MAX_RETRIES} attempts:\n"
            + "\n".join(details)
        )
        logger.critical(message)

        if self._alerting:
            try:
                await self._alerting.send(message)
            except Exception as e:
                logger.error("Failed to send critical alert: %s", e)
