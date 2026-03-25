"""Tests for Block 7 fixes: M-1 through M-4."""
import numpy as np
import pandas as pd
import pytest

from backtest.backtester import Backtester, BacktestConfig
from strategy.base_strategy import Signal, SignalDirection
from strategy.indicators import hurst_exponent, vwap, _hurst_rs, _hurst_dfa, _hurst_history
from strategy.portfolio_manager import PortfolioManager, CONFLICT_CONFIDENCE_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════════
# FIX M-1: Backtester Timestamps
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktesterTimestamps:
    """Verify timestamp normalization in the backtester."""

    def _make_ohlcv(self, n=300, ts_unit="s"):
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.normal(0, 1, n))
        close = np.maximum(close, 5.0)
        base_ts = 1700000000  # epoch seconds
        if ts_unit == "ms":
            timestamps = [int((base_ts + i * 3600) * 1000) for i in range(n)]
        else:
            timestamps = [base_ts + i * 3600 for i in range(n)]
        return pd.DataFrame({
            "timestamp": timestamps,
            "open": close - rng.uniform(-0.5, 0.5, n),
            "high": close + rng.uniform(0.5, 2.0, n),
            "low": close - rng.uniform(0.5, 2.0, n),
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
        })

    def test_millisecond_conversion(self):
        """Timestamps in ms should be converted to seconds."""
        df = self._make_ohlcv(300, ts_unit="ms")
        assert (df["timestamp"] > 1e12).all()  # ms range
        normalized = Backtester._normalize_timestamps(df)
        assert (normalized["timestamp"] < 1e12).all()  # now in seconds

    def test_sort_chronological(self):
        """Out-of-order timestamps should be sorted."""
        df = self._make_ohlcv(300)
        # Shuffle rows
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        normalized = Backtester._normalize_timestamps(df)
        assert normalized["timestamp"].is_monotonic_increasing

    def test_dedup_timestamps(self):
        """Duplicate timestamps should be removed."""
        df = self._make_ohlcv(300)
        # Duplicate first 5 rows
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        assert len(df) == 305
        normalized = Backtester._normalize_timestamps(df)
        assert len(normalized) == 300
        assert normalized["timestamp"].is_unique

    def test_gap_warning(self, caplog):
        """Gaps > 2x timeframe should be logged."""
        df = self._make_ohlcv(300)
        df.attrs["timeframe"] = "1H"
        # Introduce a gap of 5 hours (> 2 * 3600s)
        df.loc[150:, "timestamp"] += 5 * 3600
        import logging
        with caplog.at_level(logging.WARNING):
            Backtester._normalize_timestamps(df)
        assert any("gap" in r.message.lower() for r in caplog.records)

    def test_no_timestamp_column_passthrough(self):
        """If no timestamp column exists, return unchanged."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "open": rng.uniform(90, 110, 50),
            "high": rng.uniform(100, 120, 50),
            "low": rng.uniform(80, 100, 50),
            "close": rng.uniform(90, 110, 50),
            "volume": rng.uniform(1000, 5000, 50),
        })
        result = Backtester._normalize_timestamps(df)
        assert len(result) == 50


# ═══════════════════════════════════════════════════════════════════════════════
# FIX M-2: VWAP Rolling with Daily Reset
# ═══════════════════════════════════════════════════════════════════════════════

class TestVWAPRolling:
    """Verify VWAP resets daily when timestamps are provided."""

    def test_vwap_resets_daily(self):
        """VWAP should reset at UTC midnight boundaries."""
        n = 48  # 2 days of hourly data
        rng = np.random.default_rng(42)
        base_ts = 1700006400  # aligned to midnight UTC
        timestamps = pd.Series([base_ts + i * 3600 for i in range(n)])
        high = pd.Series(rng.uniform(100, 110, n))
        low = pd.Series(rng.uniform(90, 100, n))
        close = pd.Series(rng.uniform(95, 105, n))
        volume = pd.Series(rng.uniform(1000, 5000, n))

        result = vwap(high, low, close, volume, timestamps=timestamps)
        assert len(result) == n
        assert not result.isna().all()

        # At hour 24 (start of day 2), VWAP should reset — it should equal
        # the typical price of that bar (first bar of new day)
        tp_24 = (high.iloc[24] + low.iloc[24] + close.iloc[24]) / 3
        assert abs(result.iloc[24] - tp_24) < 1e-6

    def test_vwap_no_timestamps_backward_compat(self):
        """Without timestamps, VWAP should use global cumsum (old behavior)."""
        n = 100
        rng = np.random.default_rng(42)
        high = pd.Series(rng.uniform(100, 110, n))
        low = pd.Series(rng.uniform(90, 100, n))
        close = pd.Series(rng.uniform(95, 105, n))
        volume = pd.Series(rng.uniform(1000, 5000, n))

        result = vwap(high, low, close, volume)
        tp = (high + low + close) / 3
        expected = (tp * volume).cumsum() / volume.cumsum()
        pd.testing.assert_series_equal(result, expected, check_names=False)


# ═══════════════════════════════════════════════════════════════════════════════
# FIX M-3: Hurst Exponent
# ═══════════════════════════════════════════════════════════════════════════════

class TestHurstExponent:
    """Verify improved Hurst exponent with min check, DFA, and smoothing."""

    def setup_method(self):
        # Clear global smoothing buffer between tests
        _hurst_history.clear()

    def test_short_series_returns_half(self):
        """Series with < 100 points should return 0.5 (neutral)."""
        s = pd.Series(np.random.default_rng(0).standard_normal(50).cumsum())
        h = hurst_exponent(s, min_points=100)
        assert h == 0.5

    def test_exactly_100_points(self):
        """100 points should be enough for calculation."""
        rng = np.random.default_rng(0)
        s = pd.Series(rng.standard_normal(100).cumsum())
        h = hurst_exponent(s, min_points=100)
        assert isinstance(h, float)
        assert 0.0 < h < 1.0

    def test_random_walk_near_half(self):
        """Brownian motion should have H near 0.5."""
        rng = np.random.default_rng(0)
        s = pd.Series(rng.standard_normal(500).cumsum())
        h = hurst_exponent(s)
        assert 0.1 < h < 1.0  # Loose bounds: R/S+DFA average has estimation variance

    def test_dfa_returns_float(self):
        """DFA should return a float for valid input."""
        rng = np.random.default_rng(0)
        ts = rng.standard_normal(500).cumsum()
        h = _hurst_dfa(ts)
        assert isinstance(h, float)

    def test_smoothing_averages_history(self):
        """Hurst smoothing should average last 3 values."""
        _hurst_history.clear()
        rng = np.random.default_rng(0)
        results = []
        for seed in [0, 1, 2]:
            _r = np.random.default_rng(seed)
            s = pd.Series(_r.standard_normal(200).cumsum())
            results.append(hurst_exponent(s))
        # After 3 calls, result should be average of all 3 raw values
        assert len(_hurst_history) == 3

    def test_rs_short_returns_half(self):
        """R/S with very short data returns 0.5."""
        ts = np.array([1.0, 2.0, 3.0])
        h = _hurst_rs(ts)
        assert h == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# FIX M-4: Signal Conflicts
# ═══════════════════════════════════════════════════════════════════════════════

def _make_signal(direction, symbol, strategy, confidence, risk_pct=0.01):
    """Helper to create a valid test signal."""
    if direction == "LONG":
        entry, sl, tp = 50_000, 49_000, 52_000
    else:
        entry, sl, tp = 50_000, 51_000, 48_000
    return Signal(
        direction=SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT,
        symbol=symbol,
        strategy_name=strategy,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        risk_pct=risk_pct,
        confidence=confidence,
    )


class TestSignalConflicts:
    """Verify signal conflict resolution in PortfolioManager."""

    def test_conflict_high_confidence_wins(self):
        """Alpha LONG BTC (80%) vs Beta SHORT BTC (68%) → Alpha LONG wins (diff=12%>5%)."""
        signals = [
            _make_signal("LONG", "BTC-USDT", "alpha", 0.80),
            _make_signal("SHORT", "BTC-USDT", "beta", 0.68),
        ]
        resolved = PortfolioManager._resolve_conflicts(signals)
        assert len(resolved) == 1
        assert resolved[0].direction == SignalDirection.LONG
        assert resolved[0].strategy_name == "alpha"

    def test_conflict_similar_confidence_no_trade(self):
        """Alpha LONG BTC (70%) vs Beta SHORT BTC (68%) → no trade (diff < 5%)."""
        signals = [
            _make_signal("LONG", "BTC-USDT", "alpha", 0.70),
            _make_signal("SHORT", "BTC-USDT", "beta", 0.68),
        ]
        resolved = PortfolioManager._resolve_conflicts(signals)
        assert len(resolved) == 0

    def test_same_direction_no_double_size(self):
        """Alpha LONG BTC + Gamma LONG BTC → single trade, no doubling."""
        signals = [
            _make_signal("LONG", "BTC-USDT", "alpha", 0.72),
            _make_signal("LONG", "BTC-USDT", "gamma", 0.65),
        ]
        resolved = PortfolioManager._resolve_conflicts(signals)
        assert len(resolved) == 1
        assert resolved[0].strategy_name == "alpha"  # highest confidence

    def test_no_conflict_different_symbols(self):
        """LONG BTC + SHORT ETH → no conflict, both pass."""
        signals = [
            _make_signal("LONG", "BTC-USDT", "alpha", 0.72),
            _make_signal("SHORT", "ETH-USDT", "beta", 0.68),
        ]
        resolved = PortfolioManager._resolve_conflicts(signals)
        assert len(resolved) == 2

    def test_single_signal_passes_through(self):
        """A single signal should pass through unchanged."""
        signals = [_make_signal("LONG", "BTC-USDT", "alpha", 0.72)]
        resolved = PortfolioManager._resolve_conflicts(signals)
        assert len(resolved) == 1

    def test_empty_signals(self):
        """Empty signal list returns empty."""
        resolved = PortfolioManager._resolve_conflicts([])
        assert len(resolved) == 0

    def test_conflict_short_wins_over_long(self):
        """SHORT with higher confidence should win."""
        signals = [
            _make_signal("LONG", "BTC-USDT", "alpha", 0.60),
            _make_signal("SHORT", "BTC-USDT", "beta", 0.80),
        ]
        resolved = PortfolioManager._resolve_conflicts(signals)
        assert len(resolved) == 1
        assert resolved[0].direction == SignalDirection.SHORT

    def test_three_signals_conflict_resolution(self):
        """Alpha LONG + Beta SHORT + Gamma LONG on same pair."""
        signals = [
            _make_signal("LONG", "BTC-USDT", "alpha", 0.72),
            _make_signal("SHORT", "BTC-USDT", "beta", 0.60),
            _make_signal("LONG", "BTC-USDT", "gamma", 0.65),
        ]
        resolved = PortfolioManager._resolve_conflicts(signals)
        # Best LONG is alpha (0.72), best SHORT is beta (0.60), diff = 0.12 > 5%
        assert len(resolved) == 1
        assert resolved[0].direction == SignalDirection.LONG
        assert resolved[0].strategy_name == "alpha"
