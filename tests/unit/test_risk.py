"""Unit tests for the risk management layer."""
import pytest

from risk.drawdown_monitor import CircuitBreakerLevel, DrawdownMonitor
from risk.position_sizer import PositionSizer


class TestPositionSizer:
    def test_kelly_positive(self):
        sizer = PositionSizer()
        size = sizer.kelly_position_size(
            equity=10_000, win_rate=0.60, avg_win=1.5, avg_loss=1.0
        )
        assert size > 0

    def test_kelly_capped_by_max_risk(self):
        sizer = PositionSizer(max_risk_pct=0.02, kelly_fraction=1.0)
        size = sizer.kelly_position_size(
            equity=10_000, win_rate=0.99, avg_win=10.0, avg_loss=1.0
        )
        assert size <= 10_000 * 0.02 + 1e-6

    def test_kelly_zero_if_negative_edge(self):
        sizer = PositionSizer()
        size = sizer.kelly_position_size(
            equity=10_000, win_rate=0.30, avg_win=1.0, avg_loss=2.0
        )
        assert size == 0.0

    def test_fixed_fractional(self):
        sizer = PositionSizer()
        size = sizer.fixed_fractional_size(equity=5_000, risk_pct=0.01)
        assert abs(size - 50.0) < 1e-6

    def test_quantity_from_risk(self):
        sizer = PositionSizer()
        qty = sizer.quantity_from_risk(
            risk_amount_usdt=100.0,
            entry_price=50_000,
            stop_loss_price=49_500,
        )
        # risk/stop_distance = 100/500 = 0.2 BTC
        assert abs(qty - 0.2) < 1e-6

    def test_size_scalar(self):
        sizer = PositionSizer()
        base = 100.0
        assert sizer.apply_size_scalar(base, 0.5) == 50.0
        assert sizer.apply_size_scalar(base, 0.0) == 0.0
        assert sizer.apply_size_scalar(base, 1.0) == base


class TestDrawdownMonitor:
    def test_initial_state(self):
        dd = DrawdownMonitor(initial_equity=10_000)
        assert dd.current_drawdown() == 0.0
        assert dd.state.circuit_level == CircuitBreakerLevel.NONE

    def test_no_drawdown_on_equity_increase(self):
        dd = DrawdownMonitor(10_000)
        dd.update(12_000)
        assert dd.current_drawdown() == 0.0

    def test_level1_trigger(self):
        dd = DrawdownMonitor(10_000)
        dd.update(9_400)   # -6 %
        assert dd.state.circuit_level == CircuitBreakerLevel.LEVEL_1

    def test_level2_trigger(self):
        dd = DrawdownMonitor(10_000)
        dd.update(8_900)   # -11 %
        assert dd.state.circuit_level == CircuitBreakerLevel.LEVEL_2

    def test_level3_trigger(self):
        dd = DrawdownMonitor(10_000)
        dd.update(8_400)   # -16 %
        assert dd.state.circuit_level == CircuitBreakerLevel.LEVEL_3

    def test_level4_emergency(self):
        dd = DrawdownMonitor(10_000)
        dd.update(7_900)   # -21 %
        assert dd.state.circuit_level == CircuitBreakerLevel.LEVEL_4
        assert dd.requires_emergency_close() is True

    def test_size_multiplier_normal(self):
        dd = DrawdownMonitor(10_000)
        dd.update(10_500)
        assert dd.size_multiplier() == 1.0

    def test_size_multiplier_level1(self):
        dd = DrawdownMonitor(10_000)
        dd.update(9_400)
        assert dd.size_multiplier() == 0.75

    def test_size_multiplier_level3(self):
        dd = DrawdownMonitor(10_000)
        dd.update(8_400)
        assert dd.size_multiplier() == 0.0

    def test_daily_pnl(self):
        dd = DrawdownMonitor(10_000)
        dd.update(10_500)
        assert dd.daily_pnl_pct() > 0
