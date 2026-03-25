"""
Correlation guard — prevent entering positions that are highly correlated
with existing open positions.

Rules:
  - Reject if correlation > threshold AND same direction (both long or both short)
  - Allow  if correlation > threshold AND opposite direction (hedge)
  - Fallback hardcoded matrix when return history is insufficient (< 30 candles)

Default hardcoded correlations (major crypto):
  BTC-ETH: 0.85   BTC-SOL: 0.75   ETH-SOL: 0.80
"""
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# Minimum candles required to use rolling correlation
MIN_HISTORY = 30

# Fallback correlation matrix for pairs without enough history.
# Keyed as frozenset({base_a, base_b}) → correlation value.
_FALLBACK_CORR: Dict[frozenset, float] = {
    frozenset({"BTC", "ETH"}): 0.85,
    frozenset({"BTC", "SOL"}): 0.75,
    frozenset({"ETH", "SOL"}): 0.80,
}


def _base(pair: str) -> str:
    """Extract base asset from 'BTC-USDT' → 'BTC'."""
    return pair.split("-")[0].split("/")[0].upper()


def _fallback_correlation(pair_a: str, pair_b: str) -> Optional[float]:
    """Return hardcoded correlation or None if not in table."""
    key = frozenset({_base(pair_a), _base(pair_b)})
    return _FALLBACK_CORR.get(key)


class CorrelationGuard:
    """
    Compute pairwise correlations from recent returns and block signals
    if new position would exceed the correlation threshold.

    The main entry point for the pre-trade pipeline is ``check()``.
    The legacy ``is_correlated_with_open()`` is kept for backwards compat.
    """

    def __init__(
        self,
        max_correlation: float = settings.MAX_CORR_NEW_POSITION,
        lookback: int = MIN_HISTORY,
    ) -> None:
        self._max_corr = max_correlation
        self._lookback = lookback
        self._return_history: Dict[str, pd.Series] = {}

    # ── Feed return data ──────────────────────────────────────────────────────

    def update_returns(self, symbol: str, returns: pd.Series) -> None:
        """Feed recent log-returns for a symbol."""
        self._return_history[symbol] = returns.tail(self._lookback)

    # ── New direction-aware API ───────────────────────────────────────────────

    async def check(
        self,
        new_pair: str,
        new_direction: str,
        open_positions: list,
    ) -> Tuple[bool, str]:
        """
        Pre-trade correlation check (async for interface consistency).

        Parameters
        ----------
        new_pair      : e.g. 'BTC-USDT'
        new_direction : 'LONG' or 'SHORT'
        open_positions: list of Position objects (must have .symbol, .position_side)

        Returns
        -------
        (is_ok, reason)
          is_ok=True  → trade is allowed
          is_ok=False → trade should be rejected
        """
        for pos in open_positions:
            existing_pair = pos.symbol
            existing_direction = pos.position_side  # 'LONG' or 'SHORT'

            if existing_pair == new_pair:
                continue  # Same symbol handled by position-count check

            corr = self._compute_correlation(new_pair, existing_pair)
            if corr is None:
                continue  # No data and no fallback — allow

            same_direction = new_direction.upper() == existing_direction.upper()

            if corr >= self._max_corr:
                if same_direction:
                    reason = (
                        f"{new_pair} corr={corr:.2f} with open {existing_pair} "
                        f"({existing_direction}) — same direction blocked"
                    )
                    logger.debug("Correlation guard rejected: %s", reason)
                    return False, reason
                else:
                    # Opposite direction → hedge, allowed
                    logger.debug(
                        "Correlation guard: corr=%.2f with %s but opposite direction (hedge OK)",
                        corr,
                        existing_pair,
                    )

        return True, "OK"

    # ── Legacy API ────────────────────────────────────────────────────────────

    def is_correlated_with_open(
        self, candidate_symbol: str, open_symbols: List[str]
    ) -> Tuple[bool, float]:
        """
        Legacy check (direction-unaware).
        Return (is_too_correlated, max_correlation_found).
        """
        max_corr = 0.0
        for sym in open_symbols:
            if sym == candidate_symbol:
                continue
            corr = self._compute_correlation(candidate_symbol, sym)
            if corr is not None:
                max_corr = max(max_corr, abs(corr))

        return max_corr >= self._max_corr, max_corr

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_correlation(self, pair_a: str, pair_b: str) -> Optional[float]:
        """
        Compute correlation between pair_a and pair_b.
        Uses rolling return history if available (>= MIN_HISTORY candles).
        Falls back to the hardcoded matrix, then returns None if unknown.
        """
        # Try rolling history first
        ret_a = self._return_history.get(pair_a)
        ret_b = self._return_history.get(pair_b)

        if ret_a is not None and ret_b is not None:
            aligned = pd.concat([ret_a, ret_b], axis=1, join="inner").dropna()
            if len(aligned) >= MIN_HISTORY:
                corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                return abs(corr)

        # Fall back to hardcoded matrix
        return _fallback_correlation(pair_a, pair_b)
