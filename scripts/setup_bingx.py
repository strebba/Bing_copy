"""
BingX account setup — sets leverage and margin mode for all configured pairs.
Usage: python scripts/setup_bingx.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from config.pairs import PAIR_CONFIGS
from exchange.bingx_client import BingXClient


async def setup() -> None:
    print("Setting up BingX account for WAGMI Copy Trading Bot…")
    print(f"Mode: {'DEMO' if settings.DEMO_MODE else 'LIVE'}")
    print(f"Pairs: {settings.TRADING_PAIRS}\n")

    async with BingXClient() as client:
        # Check balance
        try:
            balance_data = await client.get_balance()
            balance = balance_data.get("data", {}).get("balance", {})
            equity = float(balance.get("equity", 0))
            print(f"Account equity: {equity:.2f} USDT")
            if equity < settings.MIN_ACCOUNT_BALANCE:
                print(
                    f"WARNING: Equity {equity:.2f} USDT is below minimum "
                    f"{settings.MIN_ACCOUNT_BALANCE} USDT for copy trading"
                )
        except Exception as exc:
            print(f"Could not fetch balance: {exc}")

        # Setup each trading pair
        for symbol in settings.TRADING_PAIRS:
            pair_cfg = PAIR_CONFIGS.get(symbol)
            if not pair_cfg:
                print(f"No config for {symbol}, skipping")
                continue

            print(f"Configuring {symbol}…")

            # Set isolated margin
            try:
                await client.set_margin_type(symbol, "ISOLATED")
                print(f"  Margin type: ISOLATED ✓")
            except Exception as exc:
                print(f"  Margin type: {exc}")

            # Set leverage for LONG
            try:
                await client.set_leverage(symbol, pair_cfg.default_leverage, "LONG")
                print(f"  LONG leverage: {pair_cfg.default_leverage}x ✓")
            except Exception as exc:
                print(f"  LONG leverage: {exc}")

            # Set leverage for SHORT
            try:
                await client.set_leverage(symbol, pair_cfg.default_leverage, "SHORT")
                print(f"  SHORT leverage: {pair_cfg.default_leverage}x ✓")
            except Exception as exc:
                print(f"  SHORT leverage: {exc}")

    print("\nSetup complete!")


if __name__ == "__main__":
    asyncio.run(setup())
