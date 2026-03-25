"""Tests for H-8: Daily Reset Logic and Circuit Breaker Cooldown."""
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from risk.drawdown_monitor import CircuitBreakerLevel, DrawdownMonitor
from risk.position_sizer import PositionSizer
from risk.risk_engine import RiskEngine


# ── Daily Reset Tests ─────────────────────────────────────────────────────────


class TestDailyReset:
    def _make_engine(self):
        dd = DrawdownMonitor(initial_equity=10_000)
        sizer = PositionSizer()
        return RiskEngine(dd, sizer)

    def test_initial_trading_day(self):
        engine = self._make_engine()
        assert engine.current_trading_day == datetime.now(timezone.utc).date()
        assert engine.daily_pnl == 0.0
        assert engine.daily_trade_count == 0

    def test_record_daily_pnl(self):
        engine = self._make_engine()
        engine.record_daily_pnl(100.0)
        engine.record_daily_pnl(-30.0)
        assert abs(engine.daily_pnl - 70.0) < 0.01
        assert engine.daily_trade_count == 2

    def test_daily_reset_on_day_change(self):
        engine = self._make_engine()
        engine.record_daily_pnl(500.0)
        engine.record_daily_pnl(-100.0)
        assert abs(engine.daily_pnl - 400.0) < 0.01
        assert engine.daily_trade_count == 2

        # Simulate midnight UTC crossing
        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
        engine._current_trading_day = yesterday

        # Access daily_pnl triggers reset
        assert engine.daily_pnl == 0.0
        assert engine.daily_trade_count == 0
        assert engine._previous_day_pnl == 400.0
        assert engine.current_trading_day == datetime.now(timezone.utc).date()

    def test_daily_loss_limit_resets_after_midnight(self):
        engine = self._make_engine()
        engine._daily_loss_limit_hit = True
        assert engine.daily_loss_limit_hit is True

        # Simulate day change
        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
        engine._current_trading_day = yesterday

        # After reset, daily_loss_limit_hit should be False
        assert engine.daily_loss_limit_hit is False

    def test_daily_reset_preserves_previous_pnl(self):
        engine = self._make_engine()
        engine.record_daily_pnl(250.0)
        engine.record_daily_pnl(-50.0)

        # Force day change
        engine._current_trading_day = datetime.now(timezone.utc).date() - timedelta(days=1)
        engine._check_daily_reset()

        assert engine._previous_day_pnl == 200.0


# ── Circuit Breaker Cooldown Tests ────────────────────────────────────────────


class TestCircuitBreakerCooldown:
    def test_halted_until_respected(self):
        dd = DrawdownMonitor(10_000)
        future = datetime.now(timezone.utc) + timedelta(hours=12)
        dd._state.halted_until = future
        dd._state.circuit_level = CircuitBreakerLevel.LEVEL_2
        assert dd.is_halted() is True

    def test_cooldown_expires_auto_reset(self):
        dd = DrawdownMonitor(10_000)
        # Set halted_until to the past
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        dd._state.halted_until = past
        dd._state.circuit_level = CircuitBreakerLevel.LEVEL_2

        # is_halted should now return False and reset the circuit level
        assert dd.is_halted() is False
        assert dd._state.circuit_level == CircuitBreakerLevel.NONE
        assert dd._state.halted_until is None

    def test_12h_cooldown_resets_after_12h(self):
        dd = DrawdownMonitor(10_000)
        # Trigger Level 2 (12h cooldown)
        dd.update(8_900)  # -11%
        assert dd._state.circuit_level == CircuitBreakerLevel.LEVEL_2
        assert dd._state.halted_until is not None
        assert dd.is_halted() is True

        # Fast-forward past the cooldown
        dd._state.halted_until = datetime.now(timezone.utc) - timedelta(seconds=1)

        # Should auto-reset
        assert dd.is_halted() is False
        assert dd._state.circuit_level == CircuitBreakerLevel.NONE

    def test_cooldown_independent_of_midnight(self):
        """Circuit breaker cooldown should NOT depend on daily reset."""
        dd = DrawdownMonitor(10_000)
        # Set a 48h cooldown
        dd._state.halted_until = datetime.now(timezone.utc) + timedelta(hours=48)
        dd._state.circuit_level = CircuitBreakerLevel.LEVEL_3

        # Even if midnight passes, still halted because 48h not elapsed
        assert dd.is_halted() is True

    def test_size_multiplier_zero_when_halted(self):
        dd = DrawdownMonitor(10_000)
        dd._state.halted_until = datetime.now(timezone.utc) + timedelta(hours=1)
        dd._state.circuit_level = CircuitBreakerLevel.LEVEL_3
        assert dd.size_multiplier() == 0.0

    def test_size_multiplier_restored_after_cooldown(self):
        dd = DrawdownMonitor(10_000)
        dd._state.halted_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        dd._state.circuit_level = CircuitBreakerLevel.LEVEL_2

        # Calling is_halted() triggers reset
        dd.is_halted()
        # After reset, multiplier should be 1.0 (NONE level)
        assert dd.size_multiplier() == 1.0


# ── PerformanceTracker daily reset handler ────────────────────────────────────


class TestPerformanceDailyReset:
    def test_on_daily_reset_archives_pnl(self):
        from analytics.performance import PerformanceTracker
        tracker = PerformanceTracker(initial_equity=10_000)

        tracker.on_daily_reset({
            "date": "2025-01-15",
            "previous_day_pnl": 150.0,
        })

        assert len(tracker.daily_pnl_archive) == 1
        assert tracker.daily_pnl_archive[0]["date"] == "2025-01-15"
        assert tracker.daily_pnl_archive[0]["pnl"] == 150.0

    def test_multiple_daily_resets(self):
        from analytics.performance import PerformanceTracker
        tracker = PerformanceTracker(initial_equity=10_000)

        for i in range(5):
            tracker.on_daily_reset({
                "date": f"2025-01-{15+i}",
                "previous_day_pnl": 100.0 * (i + 1),
            })

        assert len(tracker.daily_pnl_archive) == 5
