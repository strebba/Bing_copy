"""
Emergency stop — kill switch to close all positions immediately.
Can be triggered programmatically or via Telegram /stop command.

H-1 Fix: concurrent calls to trigger() are serialised via asyncio.Lock.
Only the first trigger within a stop-cycle executes close-all; duplicates
are discarded with a warning.
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EmergencyStop:
    """
    Global kill switch.  When activated, the trading engine stops accepting
    new signals and closes all positions immediately.

    Thread-safety: concurrent async calls to trigger() are serialised by
    _lock.  _triggered acts as a one-way latch — once set it can only be
    cleared via reset() (manual operator action, never automatic).
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._triggered: bool = False
        self._last_triggered: Optional[float] = None
        self._reason: str = ""

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._triggered

    @property
    def reason(self) -> str:
        return self._reason

    # ── Primary async API ─────────────────────────────────────────────────────

    async def trigger(
        self,
        reason: str,
        order_manager: Any = None,
        open_positions: Optional[List[Dict]] = None,
    ) -> int:
        """
        Async entry point with deduplication lock (H-1).

        Only the **first** concurrent caller executes the close-all logic.
        Later callers within the same stop-cycle receive a warning and return 0.

        Parameters
        ----------
        reason          : Human-readable trigger description.
        order_manager   : If provided, immediately close all open positions.
        open_positions  : List of position dicts from the exchange (required
                          when order_manager is supplied).

        Returns
        -------
        Number of positions closed, or 0 on a duplicate trigger.
        """
        async with self._lock:
            if self._triggered:
                logger.warning(
                    "Emergency stop already active (triggered at %s). "
                    "Ignoring duplicate trigger: %s",
                    self._last_triggered,
                    reason,
                )
                return 0

            self._triggered = True
            self._last_triggered = time.time()
            self._reason = reason
            logger.critical("EMERGENCY STOP TRIGGERED: %s", reason)

        # Close positions **outside** the lock — the lock only guards the flag
        # transition so we never block other trigger() callers while waiting
        # for potentially slow exchange API calls.
        if order_manager is not None and open_positions is not None:
            return await self.close_all_positions(order_manager, open_positions)
        return 0

    async def reset(self) -> None:
        """
        Manual reset after operator review.
        NEVER called automatically — always requires human intervention.
        """
        async with self._lock:
            self._triggered = False
            self._reason = ""
            logger.info("Emergency stop reset manually")

    # ── Backward-compatible sync API ──────────────────────────────────────────

    def activate(self, reason: str = "Manual kill switch") -> None:
        """Sync wrapper kept for backward compatibility (no concurrency guard)."""
        if not self._triggered:
            self._triggered = True
            self._last_triggered = time.time()
            self._reason = reason
            logger.critical("EMERGENCY STOP ACTIVATED: %s", reason)

    def deactivate(self) -> None:
        """Sync backward-compatible alias for reset (non-locking)."""
        self._triggered = False
        self._reason = ""
        logger.warning("Emergency stop deactivated")

    # ── Position closing ──────────────────────────────────────────────────────

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
