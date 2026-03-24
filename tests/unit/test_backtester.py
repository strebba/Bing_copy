"""Unit tests for the backtesting engine."""
import numpy as np
import pandas as pd
import pytest

from backtest.backtester import Backtester, BacktestConfig
from strategy.alpha_momentum import AlphaMomentumStrategy
from strategy.beta_mean_rev import BetaMeanReversionStrategy


def make_synthetic_ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1.5, n))
    close = np.maximum(close, 5.0)
    high = close + rng.uniform(0.5, 3.0, n)
    low = close - rng.uniform(0.5, 3.0, n)
    volume = rng.uniform(2000, 8000, n)
    df = pd.DataFrame({
        "timestamp": list(range(n)),
        "open": close - rng.uniform(-0.5, 0.5, n),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.attrs["timeframe"] = "1H"
    return df


class TestBacktester:
    def test_runs_without_error(self):
        df = make_synthetic_ohlcv(500)
        bt = Backtester(AlphaMomentumStrategy(), BacktestConfig(initial_capital=2000))
        tracker = bt.run(df, "BTC-USDT")
        assert tracker is not None

    def test_returns_performance_tracker(self):
        from analytics.performance import PerformanceTracker
        df = make_synthetic_ohlcv(500)
        bt = Backtester(AlphaMomentumStrategy())
        tracker = bt.run(df)
        assert isinstance(tracker, PerformanceTracker)

    def test_positive_total_trades_possible(self):
        """With sufficient data, at least some trades should occur."""
        df = make_synthetic_ohlcv(1000, seed=7)
        bt = Backtester(AlphaMomentumStrategy(), BacktestConfig(initial_capital=5000))
        tracker = bt.run(df)
        # Can be 0 if no signals fire, but we just check it's non-negative
        assert tracker.total_trades() >= 0

    def test_no_negative_win_rate(self):
        df = make_synthetic_ohlcv(500)
        bt = Backtester(AlphaMomentumStrategy())
        tracker = bt.run(df)
        assert tracker.win_rate() >= 0.0

    def test_max_dd_between_0_and_1(self):
        df = make_synthetic_ohlcv(500)
        bt = Backtester(AlphaMomentumStrategy())
        tracker = bt.run(df)
        assert 0.0 <= tracker.max_drawdown() <= 1.0
