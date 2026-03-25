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

    MAX_RETRIES = 5
    BASE_RETRY_DELAY = 2  # seconds, with exponential backoff

    def __init__(self, alerting: Optional[Any] = None) -> None:
        self._lock = asyncio.Lock()
        self._triggered: bool = False
        self._last_triggered: Optional[float] = None
        self._reason: str = ""
        self._alerting = alerting

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
