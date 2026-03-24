"""Unit tests for trading strategies."""
import numpy as np
import pandas as pd
import pytest

from strategy.alpha_momentum import AlphaMomentumStrategy
from strategy.base_strategy import Signal, SignalDirection
from strategy.beta_mean_rev import BetaMeanReversionStrategy
from strategy.gamma_breakout import GammaBreakoutStrategy


def make_trending_ohlcv(n: int = 300, trend: float = 0.5) -> pd.DataFrame:
    """Create a trending OHLCV dataset."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(trend + rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    volume = rng.uniform(1000, 5000, n)
    df = pd.DataFrame({
        "timestamp": range(n),
        "open": close - rng.uniform(-0.3, 0.3, n),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.attrs["timeframe"] = "1H"
    return df


def make_ranging_ohlcv(n: int = 300) -> pd.DataFrame:
    """Create a mean-reverting, ranging OHLCV dataset."""
    rng = np.random.default_rng(99)
    # Ornstein-Uhlenbeck-like process
    theta = 0.15
    mu = 100.0
    sigma = 0.8
    prices = [mu]
    for _ in range(n - 1):
        drift = theta * (mu - prices[-1])
        noise = sigma * rng.standard_normal()
        prices.append(max(prices[-1] + drift + noise, 1.0))
    close = np.array(prices)
    high = close + rng.uniform(0.2, 1.0, n)
    low = close - rng.uniform(0.2, 1.0, n)
    volume = rng.uniform(1000, 5000, n)
    df = pd.DataFrame({
        "timestamp": range(n),
        "open": close - rng.uniform(-0.3, 0.3, n),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.attrs["timeframe"] = "1H"
    return df


class TestBaseSignal:
    def test_valid_long_signal(self):
        sig = Signal(
            direction=SignalDirection.LONG,
            symbol="BTC-USDT",
            strategy_name="test",
            entry_price=50_000,
            stop_loss=49_000,
            take_profit=52_000,
            risk_pct=0.01,
            confidence=0.75,
        )
        assert sig.is_valid()
        assert abs(sig.risk_reward - 2.0) < 1e-6

    def test_invalid_low_rr(self):
        sig = Signal(
            direction=SignalDirection.LONG,
            symbol="BTC-USDT",
            strategy_name="test",
            entry_price=50_000,
            stop_loss=49_500,
            take_profit=50_200,    # Only 1:0.4 R:R
            risk_pct=0.01,
            confidence=0.75,
        )
        assert not sig.is_valid()

    def test_invalid_no_direction(self):
        sig = Signal(
            direction=SignalDirection.NONE,
            symbol="BTC-USDT",
            strategy_name="test",
            entry_price=50_000,
            stop_loss=49_000,
            take_profit=52_000,
            risk_pct=0.01,
            confidence=0.75,
        )
        assert not sig.is_valid()


class TestAlphaStrategy:
    def setup_method(self):
        self.strategy = AlphaMomentumStrategy()

    def test_returns_none_or_signal(self):
        df = make_trending_ohlcv(300)
        result = self.strategy.generate_signal(df, "BTC-USDT")
        assert result is None or isinstance(result, Signal)

    def test_insufficient_data_returns_none(self):
        df = make_trending_ohlcv(100)
        result = self.strategy.generate_signal(df, "BTC-USDT")
        assert result is None

    def test_signal_direction_valid(self):
        df = make_trending_ohlcv(300)
        result = self.strategy.generate_signal(df, "BTC-USDT")
        if result:
            assert result.direction in (SignalDirection.LONG, SignalDirection.SHORT)
            assert result.symbol == "BTC-USDT"
            assert result.strategy_name == "alpha"


class TestBetaStrategy:
    def setup_method(self):
        self.strategy = BetaMeanReversionStrategy()

    def test_returns_none_or_signal(self):
        df = make_ranging_ohlcv(300)
        result = self.strategy.generate_signal(df, "ETH-USDT")
        assert result is None or isinstance(result, Signal)

    def test_insufficient_data_returns_none(self):
        df = make_ranging_ohlcv(50)
        result = self.strategy.generate_signal(df, "ETH-USDT")
        assert result is None

    def test_strategy_name(self):
        df = make_ranging_ohlcv(300)
        result = self.strategy.generate_signal(df, "ETH-USDT")
        if result:
            assert result.strategy_name == "beta"


class TestGammaStrategy:
    def setup_method(self):
        self.strategy = GammaBreakoutStrategy()

    def test_returns_none_or_signal(self):
        df = make_trending_ohlcv(300, trend=0.0)
        result = self.strategy.generate_signal(df, "SOL-USDT")
        assert result is None or isinstance(result, Signal)

    def test_strategy_name(self):
        df = make_trending_ohlcv(300)
        result = self.strategy.generate_signal(df, "SOL-USDT")
        if result:
            assert result.strategy_name == "gamma"
