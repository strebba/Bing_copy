"""
Drawdown monitoring and circuit-breaker logic.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


class CircuitBreakerLevel(str, Enum):
    NONE = "NONE"
    LEVEL_1 = "LEVEL_1"    # -5 %  → size *0.75
    LEVEL_2 = "LEVEL_2"    # -10 % → size *0.50, cooldown 12 h
    LEVEL_3 = "LEVEL_3"    # -15 % → halt new trades, cooldown 48 h
    LEVEL_4 = "LEVEL_4"    # -20 % → emergency close all, cooldown 168 h


CIRCUIT_BREAKER_CONFIG = {
    CircuitBreakerLevel.LEVEL_1: {
        "trigger": -0.05,
        "size_multiplier": 0.75,
        "cooldown_hours": 0,
        "halt_new_trades": False,
        "emergency_close": False,
    },
    CircuitBreakerLevel.LEVEL_2: {
        "trigger": -0.10,
        "size_multiplier": 0.50,
        "cooldown_hours": 12,
        "halt_new_trades": False,
        "emergency_close": False,
    },
    CircuitBreakerLevel.LEVEL_3: {
        "trigger": -0.15,
        "size_multiplier": 0.0,
        "cooldown_hours": 48,
        "halt_new_trades": True,
        "emergency_close": False,
    },
    CircuitBreakerLevel.LEVEL_4: {
        "trigger": -0.20,
        "size_multiplier": 0.0,
        "cooldown_hours": 168,
        "halt_new_trades": True,
        "emergency_close": True,
    },
}


@dataclass
class DrawdownState:
    peak_equity: float
    current_equity: float
    peak_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    circuit_level: CircuitBreakerLevel = CircuitBreakerLevel.NONE
    halted_until: Optional[datetime] = None
    daily_start_equity: float = 0.0
    daily_start_date: Optional[datetime] = None
    weekly_start_equity: float = 0.0
    weekly_start_date: Optional[datetime] = None


class DrawdownMonitor:
    """
    Tracks equity, peak, and current drawdown.
    Applies circuit breaker levels with cooldown periods.
    """

    def __init__(self, initial_equity: float = settings.INITIAL_CAPITAL) -> None:
        now = datetime.now(timezone.utc)
        self._state = DrawdownState(
            peak_equity=initial_equity,
            current_equity=initial_equity,
            daily_start_equity=initial_equity,
            daily_start_date=now,
            weekly_start_equity=initial_equity,
            weekly_start_date=now,
        )

    def update(self, equity: float) -> CircuitBreakerLevel:
        """Update equity and return the current circuit breaker level."""
        now = datetime.now(timezone.utc)
        s = self._state
        s.current_equity = equity

        if equity > s.peak_equity:
            s.peak_equity = equity
            s.peak_date = now

        # Reset daily / weekly baselines
        if s.daily_start_date is None or (now - s.daily_start_date).days >= 1:
            s.daily_start_equity = equity
            s.daily_start_date = now
        if s.weekly_start_date is None or (now - s.weekly_start_date).days >= 7:
            s.weekly_start_equity = equity
            s.weekly_start_date = now

        # Evaluate circuit breaker
        level = self._evaluate_level()
        if level != s.circuit_level:
            cfg = CIRCUIT_BREAKER_CONFIG.get(level, {})
            cooldown_h = cfg.get("cooldown_hours", 0)
            if cooldown_h > 0:
                s.halted_until = now + timedelta(hours=cooldown_h)
            s.circuit_level = level
            logger.warning(
                "Circuit breaker: %s | DD=%.2f%% | halted_until=%s",
                level,
                self.current_drawdown() * 100,
                s.halted_until,
            )

        return level

    def _evaluate_level(self) -> CircuitBreakerLevel:
        dd = self.current_drawdown()
        for lvl in [
            CircuitBreakerLevel.LEVEL_4,
            CircuitBreakerLevel.LEVEL_3,
            CircuitBreakerLevel.LEVEL_2,
            CircuitBreakerLevel.LEVEL_1,
        ]:
            if dd <= CIRCUIT_BREAKER_CONFIG[lvl]["trigger"]:
                return lvl
        return CircuitBreakerLevel.NONE

    def current_drawdown(self) -> float:
        """Returns current drawdown as a negative fraction."""
        s = self._state
        if s.peak_equity <= 0:
            return 0.0
        return (s.current_equity - s.peak_equity) / s.peak_equity

    def daily_pnl_pct(self) -> float:
        if self._state.daily_start_equity <= 0:
            return 0.0
        return (
            self._state.current_equity - self._state.daily_start_equity
        ) / self._state.daily_start_equity

    def weekly_pnl_pct(self) -> float:
        if self._state.weekly_start_equity <= 0:
            return 0.0
        return (
            self._state.current_equity - self._state.weekly_start_equity
        ) / self._state.weekly_start_equity

    def is_halted(self) -> bool:
        """Check if trading is halted. Auto-resets when cooldown expires (H-8)."""
        now = datetime.now(timezone.utc)
        if self._state.halted_until is not None:
            if now >= self._state.halted_until:
                # Cooldown expired — auto-reset
                logger.info(
                    "Circuit breaker cooldown expired (was until %s), resetting",
                    self._state.halted_until,
                )
                self._state.halted_until = None
                self._state.circuit_level = CircuitBreakerLevel.NONE
                return False
            return True
        return False

    def size_multiplier(self) -> float:
        """Return the current position size scalar from the circuit breaker."""
        if self.is_halted():
            return 0.0
        cfg = CIRCUIT_BREAKER_CONFIG.get(self._state.circuit_level, {})
        return cfg.get("size_multiplier", 1.0)

    def requires_emergency_close(self) -> bool:
        cfg = CIRCUIT_BREAKER_CONFIG.get(self._state.circuit_level, {})
        return cfg.get("emergency_close", False)

    @property
    def state(self) -> DrawdownState:
        return self._state
