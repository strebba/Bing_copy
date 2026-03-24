"""
Emergency stop — kill switch to close all positions immediately.
Can be triggered programmatically or via Telegram /stop command.
"""
import asyncio
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class EmergencyStop:
    """
    Global kill switch. When activated, the trading engine should
    stop accepting new signals and close all positions immediately.
    """

    def __init__(self) -> None:
        self._active = False
        self._reason: str = ""

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
        """Close all open positions via market orders."""
        closed = 0
        tasks = []
        for pos in open_positions:
            symbol = pos.get("symbol")
            pos_side = pos.get("positionSide", "LONG")
            qty = abs(float(pos.get("positionAmt", 0)))
            if qty > 0 and symbol:
                tasks.append(order_manager.close_position(symbol, pos_side, qty))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if not isinstance(r, Exception) and r:
                closed += 1
            elif isinstance(r, Exception):
                logger.error("Emergency close error: %s", r)

        logger.info("Emergency close: %d/%d positions closed", closed, len(open_positions))
        return closed
