"""
Walk-forward analysis — train/test split across multiple time windows.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from analytics.performance import PerformanceTracker
from backtest.backtester import Backtester, BacktestConfig
from strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_tracker: Optional[PerformanceTracker] = None
    test_tracker: Optional[PerformanceTracker] = None

    @property
    def is_score(self) -> Optional[float]:
        if self.test_tracker:
            return self.test_tracker.sharpe_ratio()
        return None


class WalkForwardAnalyzer:
    """
    Performs walk-forward analysis with multiple train/test windows.
    Standard split: 80 % train / 20 % test per window.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        n_windows: int = 6,
        train_ratio: float = 0.8,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        self._strategy = strategy
        self._n_windows = n_windows
        self._train_ratio = train_ratio
        self._cfg = config or BacktestConfig()

    def run(self, df: pd.DataFrame, symbol: str = "BTC-USDT") -> List[WalkForwardWindow]:
        total_bars = len(df)
        window_size = total_bars // self._n_windows
        results: List[WalkForwardWindow] = []

        backtester = Backtester(self._strategy, self._cfg)

        for w in range(self._n_windows):
            w_start = w * window_size
            w_end = min(w_start + window_size, total_bars)
            split = int(w_start + (w_end - w_start) * self._train_ratio)

            window = WalkForwardWindow(
                window_id=w,
                train_start=w_start,
                train_end=split,
                test_start=split,
                test_end=w_end,
            )

            # Train run
            train_df = df.iloc[w_start:split].reset_index(drop=True)
            if len(train_df) > 250:
                window.train_tracker = backtester.run(train_df, symbol)

            # Test run (out-of-sample)
            test_df = df.iloc[split:w_end].reset_index(drop=True)
            if len(test_df) > 50:
                window.test_tracker = backtester.run(test_df, symbol)

            results.append(window)
            logger.info(
                "WF Window %d/%d: IS Sharpe=%.2f | OOS Sharpe=%.2f",
                w + 1,
                self._n_windows,
                window.train_tracker.sharpe_ratio() if window.train_tracker else 0,
                window.test_tracker.sharpe_ratio() if window.test_tracker else 0,
            )

        self._log_summary(results)
        return results

    def _log_summary(self, results: List[WalkForwardWindow]) -> None:
        is_sharpes = [r.train_tracker.sharpe_ratio() for r in results if r.train_tracker]
        oos_sharpes = [r.test_tracker.sharpe_ratio() for r in results if r.test_tracker]
        if is_sharpes and oos_sharpes:
            avg_is = sum(is_sharpes) / len(is_sharpes)
            avg_oos = sum(oos_sharpes) / len(oos_sharpes)
            degradation = (avg_is - avg_oos) / avg_is if avg_is > 0 else 0
            logger.info(
                "Walk-forward: avg IS Sharpe=%.2f | avg OOS Sharpe=%.2f | degradation=%.1f%%",
                avg_is, avg_oos, degradation * 100,
            )
