"""
Monte Carlo simulation — reshuffles trade returns to estimate risk distributions.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from analytics.performance import PerformanceTracker

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    n_simulations: int
    max_dd_p50: float
    max_dd_p95: float
    max_dd_p99: float
    ruin_probability: float          # P(equity drops 50 %)
    annual_return_p5: float          # 5th percentile annual return
    annual_return_p50: float         # Median annual return
    annual_return_p95: float         # 95th percentile annual return
    sharpe_p50: float


class MonteCarloSimulator:
    """
    10,000-simulation Monte Carlo via trade-return reshuffling.
    """

    def __init__(self, n_simulations: int = 10_000) -> None:
        self._n = n_simulations

    def run(
        self,
        tracker: PerformanceTracker,
        initial_equity: float,
        ruin_threshold: float = 0.50,
    ) -> Optional[MonteCarloResult]:
        trades = tracker._trades
        if len(trades) < 10:
            logger.warning("Insufficient trades for Monte Carlo (%d)", len(trades))
            return None

        returns = np.array([t.pnl_usdt for t in trades])
        n_trades = len(returns)

        max_dds = []
        final_equities = []

        rng = np.random.default_rng(42)

        for _ in range(self._n):
            # Resample with replacement
            sim_returns = rng.choice(returns, size=n_trades, replace=True)
            equity_curve = np.cumsum(np.concatenate([[initial_equity], sim_returns]))

            # Max drawdown
            peak = np.maximum.accumulate(equity_curve)
            dd = (peak - equity_curve) / peak
            max_dds.append(float(np.max(dd)))
            final_equities.append(float(equity_curve[-1]))

        max_dds_arr = np.array(max_dds)
        finals_arr = np.array(final_equities)

        ruin_count = np.sum(finals_arr < initial_equity * (1 - ruin_threshold))
        ruin_prob = ruin_count / self._n

        annual_returns = (finals_arr - initial_equity) / initial_equity

        result = MonteCarloResult(
            n_simulations=self._n,
            max_dd_p50=float(np.percentile(max_dds_arr, 50)),
            max_dd_p95=float(np.percentile(max_dds_arr, 95)),
            max_dd_p99=float(np.percentile(max_dds_arr, 99)),
            ruin_probability=ruin_prob,
            annual_return_p5=float(np.percentile(annual_returns, 5)),
            annual_return_p50=float(np.percentile(annual_returns, 50)),
            annual_return_p95=float(np.percentile(annual_returns, 95)),
            sharpe_p50=float(np.percentile(
                [t.r_multiple for t in trades], 50
            )),
        )

        logger.info(
            "Monte Carlo (%d sims): DD p95=%.1f%% | DD p99=%.1f%% | Ruin=%.2f%%",
            self._n,
            result.max_dd_p95 * 100,
            result.max_dd_p99 * 100,
            result.ruin_probability * 100,
        )
        return result

    def passes_go_live_criteria(self, result: MonteCarloResult) -> tuple[bool, List[str]]:
        """Check against the go-live acceptance criteria."""
        failures = []
        if result.max_dd_p95 > 0.20:
            failures.append(f"DD p95 {result.max_dd_p95:.1%} > 20%")
        if result.max_dd_p99 > 0.25:
            failures.append(f"DD p99 {result.max_dd_p99:.1%} > 25%")
        if result.ruin_probability > 0.01:
            failures.append(f"Ruin probability {result.ruin_probability:.2%} > 1%")
        return len(failures) == 0, failures
