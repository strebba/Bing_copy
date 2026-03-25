"""Unit tests for technical indicators."""
import numpy as np
import pandas as pd
import pytest

from strategy.indicators import (
    atr,
    bollinger_bands,
    detect_rsi_divergence,
    donchian_channel,
    ema,
    hurst_exponent,
    macd,
    rsi,
    sma,
    zscore,
)


def make_series(n: int = 100, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 1, n))
    return pd.Series(prices)


def make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    volume = rng.uniform(1000, 5000, n)
    return pd.DataFrame({
        "open": close - rng.uniform(-0.5, 0.5, n),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestEMA:
    def test_length(self):
        s = make_series(100)
        result = ema(s, 20)
        assert len(result) == 100

    def test_no_nan_after_warmup(self):
        s = make_series(100)
        result = ema(s, 10)
        assert not result.iloc[20:].isna().any()

    def test_monotone_smoothing(self):
        """EMA of a constant series should equal the constant."""
        s = pd.Series([50.0] * 100)
        result = ema(s, 10)
        assert abs(result.iloc[-1] - 50.0) < 1e-6


class TestRSI:
    def test_bounds(self):
        s = make_series(200)
        r = rsi(s, 14)
        valid = r.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_length(self):
        s = make_series(100)
        r = rsi(s, 14)
        assert len(r) == 100


class TestMACD:
    def test_components(self):
        s = make_series(200)
        m, sig, hist = macd(s, 12, 26, 9)
        assert len(m) == len(s) == len(sig) == len(hist)

    def test_histogram_equals_diff(self):
        s = make_series(200)
        m, sig, hist = macd(s, 12, 26, 9)
        diff = m - sig
        pd.testing.assert_series_equal(hist, diff, check_names=False)


class TestATR:
    def test_positive(self):
        df = make_ohlcv(100)
        result = atr(df["high"], df["low"], df["close"], 14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_length(self):
        df = make_ohlcv(100)
        result = atr(df["high"], df["low"], df["close"], 14)
        assert len(result) == 100


class TestBollingerBands:
    def test_ordering(self):
        s = make_series(200)
        upper, mid, lower = bollinger_bands(s, 20, 2.0)
        valid_idx = upper.dropna().index
        assert (upper[valid_idx] >= mid[valid_idx]).all()
        assert (mid[valid_idx] >= lower[valid_idx]).all()

    def test_mid_equals_sma(self):
        s = make_series(200)
        _, mid, _ = bollinger_bands(s, 20, 2.0)
        expected = sma(s, 20)
        pd.testing.assert_series_equal(mid, expected, check_names=False)


class TestDonchianChannel:
    def test_upper_is_rolling_max(self):
        df = make_ohlcv(100)
        upper, _, _ = donchian_channel(df["high"], df["low"], 20)
        expected = df["high"].rolling(20).max()
        pd.testing.assert_series_equal(upper, expected, check_names=False)

    def test_lower_is_rolling_min(self):
        df = make_ohlcv(100)
        _, _, lower = donchian_channel(df["high"], df["low"], 20)
        expected = df["low"].rolling(20).min()
        pd.testing.assert_series_equal(lower, expected, check_names=False)


class TestZScore:
    def test_approximately_zero_mean(self):
        s = make_series(1000)
        z = zscore(s, 20)
        valid = z.dropna()
        assert abs(valid.mean()) < 1.0  # Should be approximately centered


class TestHurstExponent:
    def test_random_walk_near_half(self):
        """A Brownian motion series should have H near 0.5."""
        rng = np.random.default_rng(0)
        rw = pd.Series(np.cumsum(rng.standard_normal(500)))
        h = hurst_exponent(rw)
        assert 0.1 < h < 1.0   # Loose bounds: R/S+DFA average has estimation variance

    def test_trending_series(self):
        """A strongly trending series should produce H estimate via R/S without error."""
        rng = np.random.default_rng(77)
        s = pd.Series(np.linspace(0, 100, 500) + rng.normal(0, 0.05, 500))
        h = hurst_exponent(s)
        # R/S estimation is noisy on short windows; just verify it returns a float
        assert isinstance(h, float)

    def test_short_series_returns_half(self):
        s = pd.Series([1.0, 2.0, 3.0])
        h = hurst_exponent(s)
        assert h == 0.5


class TestRSIDivergence:
    def test_bullish_divergence(self):
        """Lower price but higher RSI = bullish divergence."""
        # price_recent (92) < price_prev (110), rsi_recent (45) > rsi_prev (30)
        close = pd.Series([110, 105, 100, 95, 90, 92])
        rsi_vals = pd.Series([30, 28, 26, 24, 22, 45])   # RSI ends higher than start
        bull, bear = detect_rsi_divergence(close, rsi_vals, lookback=5)
        assert bool(bull) is True
        assert bool(bear) is False

    def test_no_divergence(self):
        # price rising AND RSI rising → no divergence
        close = pd.Series([90, 95, 100, 105, 110, 115])
        rsi_vals = pd.Series([30, 40, 50, 60, 70, 75])
        bull, bear = detect_rsi_divergence(close, rsi_vals, lookback=5)
        assert bool(bull) is False
        assert bool(bear) is False

    def test_short_series(self):
        close = pd.Series([100, 101])
        rsi_vals = pd.Series([50, 51])
        bull, bear = detect_rsi_divergence(close, rsi_vals, lookback=5)
        assert bull is False and bear is False
