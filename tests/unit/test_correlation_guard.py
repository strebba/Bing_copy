"""
Tests for CorrelationGuard.

Scenarios:
  1. Long BTC open → Long ETH → guard blocks (corr 0.85 > 0.6)
  2. Long BTC open → Short ETH → guard allows (hedge)
  3. Long BTC open → Long SOL with corr < 0.6 → guard allows
  4. Fallback matrix used when return history is insufficient
  5. Rolling history used when sufficient data is available
  6. Unknown pair → allowed (no data, no fallback)
"""
import asyncio
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from risk.correlation_guard import CorrelationGuard, _fallback_correlation


# ── Minimal Position stub ─────────────────────────────────────────────────────

@dataclass
class _FakePosition:
    symbol: str
    position_side: str   # "LONG" | "SHORT"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _guard(max_corr: float = 0.6) -> CorrelationGuard:
    return CorrelationGuard(max_correlation=max_corr, lookback=30)


def _correlated_returns(n: int = 50, corr: float = 0.9, seed: int = 42) -> tuple:
    """Generate two return series with approx `corr` correlation."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    # y = corr*x + sqrt(1-corr^2)*noise
    other = corr * base + np.sqrt(1 - corr**2) * noise
    idx = pd.RangeIndex(n)
    return pd.Series(base, index=idx), pd.Series(other, index=idx)


# ── Fallback matrix tests ─────────────────────────────────────────────────────

class TestFallbackMatrix:
    def test_btc_eth_corr(self):
        assert _fallback_correlation("BTC-USDT", "ETH-USDT") == pytest.approx(0.85)

    def test_btc_sol_corr(self):
        assert _fallback_correlation("BTC-USDT", "SOL-USDT") == pytest.approx(0.75)

    def test_eth_sol_corr(self):
        assert _fallback_correlation("ETH-USDT", "SOL-USDT") == pytest.approx(0.80)

    def test_unknown_pair_returns_none(self):
        assert _fallback_correlation("BTC-USDT", "DOGE-USDT") is None

    def test_symmetric(self):
        assert _fallback_correlation("ETH-USDT", "BTC-USDT") == _fallback_correlation("BTC-USDT", "ETH-USDT")


# ── check() — direction-aware (uses fallback) ─────────────────────────────────

class TestCheckWithFallback:
    """No return history → guard uses hardcoded matrix."""

    @pytest.mark.asyncio
    async def test_long_btc_then_long_eth_blocked(self):
        """BTC long open → ETH long → blocked (corr=0.85 > 0.6)."""
        guard = _guard()
        open_positions = [_FakePosition("BTC-USDT", "LONG")]
        ok, reason = await guard.check("ETH-USDT", "LONG", open_positions)
        assert not ok
        assert "ETH-USDT" in reason

    @pytest.mark.asyncio
    async def test_long_btc_then_short_eth_allowed(self):
        """BTC long open → ETH short → allowed (hedge, opposite direction)."""
        guard = _guard()
        open_positions = [_FakePosition("BTC-USDT", "LONG")]
        ok, reason = await guard.check("ETH-USDT", "SHORT", open_positions)
        assert ok, f"Expected OK but got: {reason}"

    @pytest.mark.asyncio
    async def test_long_btc_then_long_sol_blocked_by_fallback(self):
        """BTC long open → SOL long → blocked (corr=0.75 > 0.6, same direction)."""
        guard = _guard()
        open_positions = [_FakePosition("BTC-USDT", "LONG")]
        ok, reason = await guard.check("SOL-USDT", "LONG", open_positions)
        assert not ok

    @pytest.mark.asyncio
    async def test_long_btc_then_long_doge_allowed(self):
        """BTC long open → DOGE long → allowed (no fallback data → unknown)."""
        guard = _guard()
        open_positions = [_FakePosition("BTC-USDT", "LONG")]
        ok, reason = await guard.check("DOGE-USDT", "LONG", open_positions)
        assert ok

    @pytest.mark.asyncio
    async def test_empty_open_positions_always_ok(self):
        guard = _guard()
        ok, reason = await guard.check("BTC-USDT", "LONG", [])
        assert ok

    @pytest.mark.asyncio
    async def test_same_symbol_ignored(self):
        """Adding to same symbol is handled by position-count check, not correlation."""
        guard = _guard()
        open_positions = [_FakePosition("BTC-USDT", "LONG")]
        ok, reason = await guard.check("BTC-USDT", "LONG", open_positions)
        assert ok  # Same symbol skipped

    @pytest.mark.asyncio
    async def test_three_longs_btc_eth_sol_blocked(self):
        """BTC long + ETH long open → SOL long → blocked (ETH-SOL corr=0.80)."""
        guard = _guard()
        open_positions = [
            _FakePosition("BTC-USDT", "LONG"),
            _FakePosition("ETH-USDT", "LONG"),
        ]
        ok, reason = await guard.check("SOL-USDT", "LONG", open_positions)
        assert not ok


# ── check() — uses rolling history when sufficient ───────────────────────────

class TestCheckWithHistory:
    @pytest.mark.asyncio
    async def test_high_corr_same_direction_blocked(self):
        guard = _guard()
        ret_a, ret_b = _correlated_returns(n=50, corr=0.9)
        guard.update_returns("PAIR-A", ret_a)
        guard.update_returns("PAIR-B", ret_b)

        open_positions = [_FakePosition("PAIR-A", "LONG")]
        ok, reason = await guard.check("PAIR-B", "LONG", open_positions)
        assert not ok

    @pytest.mark.asyncio
    async def test_high_corr_opposite_direction_allowed(self):
        guard = _guard()
        ret_a, ret_b = _correlated_returns(n=50, corr=0.9)
        guard.update_returns("PAIR-A", ret_a)
        guard.update_returns("PAIR-B", ret_b)

        open_positions = [_FakePosition("PAIR-A", "LONG")]
        ok, reason = await guard.check("PAIR-B", "SHORT", open_positions)
        assert ok, f"Expected OK but got: {reason}"

    @pytest.mark.asyncio
    async def test_low_corr_allowed(self):
        guard = _guard()
        # Low correlation: 0.2 < 0.6 threshold
        ret_a, ret_b = _correlated_returns(n=50, corr=0.2)
        guard.update_returns("PAIR-A", ret_a)
        guard.update_returns("PAIR-B", ret_b)

        open_positions = [_FakePosition("PAIR-A", "LONG")]
        ok, reason = await guard.check("PAIR-B", "LONG", open_positions)
        assert ok

    @pytest.mark.asyncio
    async def test_insufficient_history_falls_back_to_matrix(self):
        """If history < MIN_HISTORY (30), fallback matrix is used for known pairs."""
        guard = _guard()
        # Only 10 candles — below the 30-candle threshold
        short_ret = pd.Series(np.random.randn(10))
        guard.update_returns("BTC-USDT", short_ret)
        guard.update_returns("ETH-USDT", short_ret)

        # Even though we have some returns, not enough → falls back to 0.85
        open_positions = [_FakePosition("BTC-USDT", "LONG")]
        ok, reason = await guard.check("ETH-USDT", "LONG", open_positions)
        assert not ok  # fallback says 0.85 > 0.6


# ── Legacy API ────────────────────────────────────────────────────────────────

class TestLegacyAPI:
    def test_is_correlated_no_history(self):
        guard = _guard()
        # No return history → not correlated (legacy returns 0.0)
        is_corr, max_c = guard.is_correlated_with_open("BTC-USDT", ["ETH-USDT"])
        # Fallback matrix: 0.85 > 0.6 → True
        assert is_corr is True
        assert max_c == pytest.approx(0.85)

    def test_is_correlated_unknown_pair(self):
        guard = _guard()
        is_corr, max_c = guard.is_correlated_with_open("DOGE-USDT", ["BNB-USDT"])
        assert is_corr is False
        assert max_c == 0.0
