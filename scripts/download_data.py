"""
Historical data downloader — fetches OHLCV from BingX and saves to CSV/Parquet.
Usage: python scripts/download_data.py --symbol BTC-USDT --interval 1h --days 730
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from exchange.bingx_client import BingXClient

INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


async def download(symbol: str, interval: str, days: int, output_dir: Path) -> None:
    now_ms = int(__import__("time").time() * 1000)
    bar_ms = INTERVAL_MS.get(interval, 3_600_000)
    total_bars = (days * 24 * 3600 * 1000) // bar_ms
    batch_size = 1000

    all_rows = []
    end_time = now_ms

    print(f"Downloading {symbol} {interval} — {days} days ({total_bars} bars)")

    async with BingXClient() as client:
        while len(all_rows) < total_bars:
            start_time = end_time - batch_size * bar_ms
            raw = await client.get_klines(
                symbol, interval, limit=batch_size,
                start_time=start_time, end_time=end_time
            )
            if not raw:
                break
            all_rows.extend(raw)
            end_time = start_time
            print(f"  Downloaded {len(all_rows)}/{total_bars} bars…", end="\r")

    print(f"\nTotal bars downloaded: {len(all_rows)}")

    if not all_rows:
        print("No data downloaded")
        return

    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"
    ]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol.replace('-', '_')}_{interval}_{days}d.parquet"
    df[["timestamp", "open", "high", "low", "close", "volume"]].to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BingX historical data downloader")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--interval", default="1h", choices=list(INTERVAL_MS.keys()))
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--output", default="data/historical")
    args = parser.parse_args()

    asyncio.run(download(
        args.symbol, args.interval, args.days, Path(args.output)
    ))


if __name__ == "__main__":
    main()
