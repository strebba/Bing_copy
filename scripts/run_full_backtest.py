"""
Run comprehensive backtest on all 25 crypto pairs for all 3 strategies.
Usage: python scripts/run_full_backtest.py --timeframe 1h
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from backtest.backtester import Backtester, BacktestConfig
from backtest.monte_carlo import MonteCarloSimulator
from strategy.alpha_momentum import AlphaMomentumStrategy
from strategy.beta_mean_rev import BetaMeanReversionStrategy
from strategy.gamma_breakout import GammaBreakoutStrategy

TOP_25_PAIRS = [
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "XRP-USDT",
    "DOGE-USDT",
    "BNB-USDT",
    "ADA-USDT",
    "LTC-USDT",
    "AVAX-USDT",
    "LINK-USDT",
    "UNI-USDT",
    "ATOM-USDT",
    "XLM-USDT",
    "ETC-USDT",
    "FIL-USDT",
    "APT-USDT",
    "ARB-USDT",
    "OP-USDT",
    "NEAR-USDT",
    "DOT-USDT",
    "INJ-USDT",
    "SUI-USDT",
    "TRX-USDT",
    "AAVE-USDT",
    "GRT-USDT",
]

STRATEGIES = {
    "alpha": AlphaMomentumStrategy,
    "beta": BetaMeanReversionStrategy,
    "gamma": GammaBreakoutStrategy,
}


def run_backtest(
    data_path: Path, strategy_name: str, capital: float = 2000, risk: float = 0.01
) -> dict:
    """Run backtest for a single pair/strategy."""
    try:
        df = pd.read_parquet(data_path)
        df.attrs["timeframe"] = "1H"

        strategy_class = STRATEGIES[strategy_name]
        strategy = strategy_class()

        cfg = BacktestConfig(
            initial_capital=capital,
            risk_per_trade=risk,
            leverage=5,
        )

        bt = Backtester(strategy, cfg)
        tracker = bt.run(df)
        summary = tracker.summary()

        # Run Monte Carlo
        mc = MonteCarloSimulator(n_simulations=1000)
        mc_result = mc.run(tracker, initial_equity=capital)

        result = {
            "strategy": strategy_name,
            "pair": data_path.stem.replace("_1h_730d", "").replace("_4h_730d", ""),
            "timeframe": "1h" if "1h" in data_path.name else "4h",
            "status": "SUCCESS",
            "trades": summary.get("total_trades", 0),
            "win_rate": summary.get("win_rate", 0),
            "profit_factor": summary.get("profit_factor", 0),
            "sharpe_ratio": summary.get("sharpe_ratio", 0),
            "max_drawdown": summary.get("max_drawdown", 0),
            "total_return": summary.get("total_return_pct", 0),
        }

        if mc_result:
            result.update(
                {
                    "mc_dd_p95": mc_result.max_dd_p95,
                    "mc_dd_p99": mc_result.max_dd_p99,
                    "mc_ruin_prob": mc_result.ruin_probability,
                }
            )

        return result

    except Exception as e:
        return {
            "strategy": strategy_name,
            "pair": data_path.stem,
            "status": "ERROR",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run full backtest on all pairs")
    parser.add_argument("--timeframe", default="1h", choices=["1h", "4h"])
    parser.add_argument("--capital", type=float, default=2000)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--output", default="data/backtest_results")
    args = parser.parse_args()

    data_dir = Path("data/historical")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_runs = 25 * 3  # 25 pairs × 3 strategies
    run_count = 0

    print(f"Starting backtest: {25} pairs × 3 strategies = {total_runs} runs")
    print(f"Timeframe: {args.timeframe}")
    print(f"Capital: ${args.capital}, Risk: {args.risk * 100}%")
    print("=" * 60)

    for pair in TOP_25_PAIRS:
        pair_file_1h = data_dir / f"{pair.replace('-', '_')}_1h_730d.parquet"
        pair_file_4h = data_dir / f"{pair.replace('-', '_')}_4h_730d.parquet"

        if args.timeframe == "1h" and not pair_file_1h.exists():
            print(f"Skipping {pair} - no 1h data")
            continue
        if args.timeframe == "4h" and not pair_file_4h.exists():
            print(f"Skipping {pair} - no 4h data")
            continue

        data_path = pair_file_1h if args.timeframe == "1h" else pair_file_4h

        for strategy_name in ["alpha", "beta", "gamma"]:
            run_count += 1
            print(f"[{run_count}/{total_runs}] {pair} - {strategy_name}...", end=" ")

            result = run_backtest(data_path, strategy_name, args.capital, args.risk)
            results.append(result)

            if result["status"] == "SUCCESS":
                print(
                    f"Trades: {result['trades']}, WR: {result['win_rate']:.1%}, "
                    f"PF: {result['profit_factor']:.2f}, Sharpe: {result['sharpe_ratio']:.2f}, "
                    f"DD: {result['max_drawdown']:.1%}"
                )
            else:
                print(f"ERROR: {result.get('error', 'Unknown')}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"backtest_{args.timeframe}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r["status"] == "SUCCESS"]
    failed = [r for r in results if r["status"] == "ERROR"]

    print(f"Total runs: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        # Aggregate by strategy
        for strategy_name in ["alpha", "beta", "gamma"]:
            strat_results = [r for r in successful if r["strategy"] == strategy_name]
            if strat_results:
                avg_wr = sum(r["win_rate"] for r in strat_results) / len(strat_results)
                avg_pf = sum(r["profit_factor"] for r in strat_results) / len(
                    strat_results
                )
                avg_sharpe = sum(r["sharpe_ratio"] for r in strat_results) / len(
                    strat_results
                )
                avg_dd = sum(r["max_drawdown"] for r in strat_results) / len(
                    strat_results
                )
                total_trades = sum(r["trades"] for r in strat_results)

                print(f"\n{strategy_name.upper()}:")
                print(f"  Pairs tested: {len(strat_results)}")
                print(f"  Total trades: {total_trades}")
                print(f"  Avg Win Rate: {avg_wr:.1%}")
                print(f"  Avg Profit Factor: {avg_pf:.2f}")
                print(f"  Avg Sharpe: {avg_sharpe:.2f}")
                print(f"  Avg Max DD: {avg_dd:.1%}")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
