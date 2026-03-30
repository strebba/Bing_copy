"""
Download historical data for all available crypto pairs (with error handling).
Usage: python scripts/download_all_data.py --interval 1h --days 730
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from exchange.bingx_client import BingXClient

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

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
    "PEPE-USDT",
    "DOT-USDT",
    "ICP-USDT",
    "INJ-USDT",
    "AAVE-USDT",
    "GRT-USDT",
]


async def download_symbol(
    client: BingXClient, symbol: str, interval: str, days: int, output_dir: Path
) -> dict:
    """Download data for a single symbol."""
    now_ms = int(datetime.now().timestamp() * 1000)
    bar_ms = INTERVAL_MS.get(interval, 3_600_000)
    total_bars = (days * 24 * 3600 * 1000) // bar_ms
    batch_size = 1000

    all_rows = []
    end_time = now_ms
    request_count = 0
    error = None

    print(f"[{symbol}] Starting download ({total_bars} bars needed)...")

    while len(all_rows) < total_bars:
        start_time = end_time - batch_size * bar_ms
        try:
            raw = await client.get_klines(
                symbol,
                interval,
                limit=batch_size,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            error = str(e)
            print(f"\n[{symbol}] Error: {error}")
            break
        if not raw:
            break
        all_rows.extend(raw)
        end_time = start_time
        request_count += 1
        print(
            f"[{symbol}] Downloaded {len(all_rows)}/{total_bars} bars (request {request_count})...",
            end="\r",
        )

    if error:
        return {
            "symbol": symbol,
            "status": "ERROR",
            "bars": 0,
            "requests": request_count,
            "error": error,
        }

    print(f"[{symbol}] Total: {len(all_rows)} bars, {request_count} requests")

    if not all_rows:
        return {
            "symbol": symbol,
            "status": "NO_DATA",
            "bars": 0,
            "requests": request_count,
        }

    import pandas as pd

    # Handle both dict format (BingX API) and list format
    if isinstance(all_rows[0], dict):
        df = pd.DataFrame(all_rows)
        # BingX uses 'time' key for timestamp
        if "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
    else:
        df = pd.DataFrame(
            all_rows,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_base",
                "taker_quote",
                "ignore",
            ],
        )

    # Ensure timestamp is integer (milliseconds)
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(int)

    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol.replace('-', '_')}_{interval}_{days}d.parquet"
    df[["timestamp", "open", "high", "low", "close", "volume"]].to_parquet(
        out_path, index=False
    )

    return {
        "symbol": symbol,
        "status": "SUCCESS",
        "bars": len(all_rows),
        "requests": request_count,
    }


async def download_all(pairs: list, interval: str, days: int, output_dir: Path) -> list:
    """Download data for all pairs."""
    results = []
    total_requests = 0
    total_bars = 0

    async with BingXClient() as client:
        for i, symbol in enumerate(pairs, 1):
            print(f"\n{'=' * 60}")
            print(f"Downloading {symbol} ({i}/{len(pairs)}) - {interval}")
            print("=" * 60)

            result = await download_symbol(client, symbol, interval, days, output_dir)
            results.append(result)
            total_requests += result["requests"]
            total_bars += result["bars"]

            # Small delay between pairs to be nice to the API
            if i < len(pairs):
                await asyncio.sleep(0.5)

    print(f"\n{'=' * 60}")
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Total pairs: {len(pairs)}")
    print(f"Total bars: {total_bars:,}")
    print(f"Total requests: {total_requests}")
    print(f"Output directory: {output_dir}")

    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    print(f"Success rate: {success_count}/{len(pairs)}")

    print("\n--- Failed pairs ---")
    for r in results:
        if r["status"] != "SUCCESS":
            print(f"  {r['symbol']}: {r.get('error', r['status'])}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical data for all 25 crypto pairs"
    )
    parser.add_argument("--interval", default="1h", choices=list(INTERVAL_MS.keys()))
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--output", default="data/historical")
    parser.add_argument(
        "--pairs", nargs="+", default=TOP_25_PAIRS, help="List of trading pairs"
    )
    args = parser.parse_args()

    print(f"Starting download of {len(args.pairs)} pairs")
    print(f"Timeframe: {args.interval}")
    print(f"Days: {args.days}")
    print(f"Output: {args.output}")
    print()

    asyncio.run(download_all(args.pairs, args.interval, args.days, Path(args.output)))


if __name__ == "__main__":
    main()
