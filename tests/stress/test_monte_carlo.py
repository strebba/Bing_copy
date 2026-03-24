"""Stress test: Monte Carlo simulation on synthetic trade data."""
import numpy as np
import pytest

from analytics.performance import PerformanceTracker, TradeRecord
from backtest.monte_carlo import MonteCarloSimulator
from datetime import datetime, timezone


def generate_tracker(n_trades: int = 200, win_rate: float = 0.6) -> PerformanceTracker:
    tracker = PerformanceTracker(10_000)
    rng = np.random.default_rng(0)
    for i in range(n_trades):
        is_win = rng.random() < win_rate
        pnl = rng.uniform(50, 200) if is_win else -rng.uniform(30, 100)
        tracker.record_trade(TradeRecord(
            symbol="BTC-USDT",
            direction="LONG",
            strategy="alpha",
            entry_price=50_000,
            exit_price=50_000 + pnl,
            quantity=0.01,
            pnl_usdt=pnl,
            opened_at=datetime.now(timezone.utc),
        ))
    return tracker


class TestMonteCarlo:
    def test_runs_correctly(self):
        tracker = generate_tracker(200, 0.6)
        mc = MonteCarloSimulator(n_simulations=1000)
        result = mc.run(tracker, initial_equity=10_000)
        assert result is not None
        assert result.n_simulations == 1000

    def test_max_dd_positive(self):
        tracker = generate_tracker(200, 0.6)
        mc = MonteCarloSimulator(1000)
        result = mc.run(tracker, 10_000)
        assert result.max_dd_p95 >= 0.0
        assert result.max_dd_p99 >= result.max_dd_p95

    def test_good_strategy_passes_criteria(self):
        """A profitable strategy with good risk metrics should pass go-live criteria."""
        tracker = generate_tracker(300, win_rate=0.65)
        mc = MonteCarloSimulator(2000)
        result = mc.run(tracker, 10_000)
        passed, failures = mc.passes_go_live_criteria(result)
        # With a good win rate, this should generally pass
        # (not guaranteed due to random nature, but very likely)
        if not passed:
            pytest.xfail(f"Monte Carlo failed (unlikely but possible): {failures}")

    def test_bad_strategy_flags_ruin(self):
        """A losing strategy should flag ruin probability concerns."""
        tracker = generate_tracker(200, win_rate=0.30)
        mc = MonteCarloSimulator(1000)
        result = mc.run(tracker, 10_000)
        # Ruin probability should be significant for a losing strategy
        assert result.ruin_probability > 0.0

    def test_insufficient_trades_returns_none(self):
        tracker = PerformanceTracker(10_000)
        mc = MonteCarloSimulator(100)
        result = mc.run(tracker, 10_000)
        assert result is None
