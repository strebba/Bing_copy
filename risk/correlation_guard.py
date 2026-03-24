"""
Correlation guard — prevent entering positions that are highly correlated
with existing open positions.
"""
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class CorrelationGuard:
    """
    Compute pairwise correlations from recent returns and block signals
    if new position would exceed the correlation threshold.
    """

    def __init__(
        self,
        max_correlation: float = settings.MAX_CORR_NEW_POSITION,
        lookback: int = 50,
    ) -> None:
        self._max_corr = max_correlation
        self._lookback = lookback
        self._return_history: Dict[str, pd.Series] = {}

    def update_returns(self, symbol: str, returns: pd.Series) -> None:
        """Feed recent log-returns for a symbol."""
        self._return_history[symbol] = returns.tail(self._lookback)

    def is_correlated_with_open(
        self, candidate_symbol: str, open_symbols: List[str]
    ) -> tuple[bool, float]:
        """
        Return (is_too_correlated, max_correlation_found).
        """
        if candidate_symbol not in self._return_history:
            return False, 0.0

        candidate_returns = self._return_history[candidate_symbol]
        max_corr = 0.0

        for sym in open_symbols:
            if sym == candidate_symbol or sym not in self._return_history:
                continue
            existing_returns = self._return_history[sym]
            aligned = pd.concat(
                [candidate_returns, existing_returns], axis=1, join="inner"
            ).dropna()
            if len(aligned) < 10:
                continue
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            max_corr = max(max_corr, abs(corr))

        return max_corr >= self._max_corr, max_corr
