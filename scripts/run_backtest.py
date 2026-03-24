"""
Run backtests with walk-forward analysis and Monte Carlo.
Usage: python scripts/run_backtest.py --data data/historical/BTC_USDT_1h_730d.parquet --strategy alpha
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from backtest.backtester import Backtester, BacktestConfig
from backtest.monte_carlo import MonteCarloSimulator
from backtest.walk_forward import WalkForwardAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtests")
    parser.add_argument("--data", required=True, help="Path to parquet data file")
    parser.add_argument("--strategy", default="alpha", choices=["alpha", "beta", "gamma"])
    parser.add_argument("--capital", type=float, default=2000.0)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--leverage", type=int, default=5)
    parser.add_argument("--mc-sims", type=int, default=10_000)
    parser.add_argument("--wf-windows", type=int, default=6)
    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.data)
    df.attrs["timeframe"] = "1H"
    print(f"Loaded {len(df)} bars from {args.data}")

    # Select strategy
    if args.strategy == "alpha":
        from strategy.alpha_momentum import AlphaMomentumStrategy
        strategy = AlphaMomentumStrategy()
    elif args.strategy == "beta":
        from strategy.beta_mean_rev import BetaMeanReversionStrategy
        strategy = BetaMeanReversionStrategy()
    else:
        from strategy.gamma_breakout import GammaBreakoutStrategy
        strategy = GammaBreakoutStrategy()

    cfg = BacktestConfig(
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        leverage=args.leverage,
    )

    # ── Full backtest ──────────────────────────────────────────────────────────
    print("\n=== Full Backtest ===")
    bt = Backtester(strategy, cfg)
    tracker = bt.run(df)
    summary = tracker.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # ── Walk-forward analysis ──────────────────────────────────────────────────
    print("\n=== Walk-Forward Analysis ===")
    wf = WalkForwardAnalyzer(strategy, n_windows=args.wf_windows, config=cfg)
    wf_results = wf.run(df)
    for r in wf_results:
        oos = r.test_tracker
        if oos:
            print(
                f"  Window {r.window_id}: OOS Sharpe={oos.sharpe_ratio():.2f} "
                f"WinRate={oos.win_rate():.1%} MaxDD={oos.max_drawdown():.1%}"
            )

    # ── Monte Carlo ────────────────────────────────────────────────────────────
    print("\n=== Monte Carlo Simulation ===")
    mc = MonteCarloSimulator(n_simulations=args.mc_sims)
    mc_result = mc.run(tracker, initial_equity=args.capital)
    if mc_result:
        print(f"  DD p50:  {mc_result.max_dd_p50:.1%}")
        print(f"  DD p95:  {mc_result.max_dd_p95:.1%}")
        print(f"  DD p99:  {mc_result.max_dd_p99:.1%}")
        print(f"  Ruin probability: {mc_result.ruin_probability:.2%}")
        passed, failures = mc.passes_go_live_criteria(mc_result)
        print(f"\n  Go-live criteria: {'PASS ✓' if passed else 'FAIL ✗'}")
        for f in failures:
            print(f"    - {f}")


if __name__ == "__main__":
    main()
