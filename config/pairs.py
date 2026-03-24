"""
Per-pair configuration: tick size, lot size, min notional, default leverage.
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PairConfig:
    symbol: str
    tick_size: float          # Price precision
    lot_size: float           # Quantity precision
    min_notional: float       # Min order value in USDT
    max_leverage: int         # Max leverage allowed
    default_leverage: int     # Leverage to set on startup
    min_volume_24h: float     # Minimum 24 h volume in USDT
    timeframes: list = field(default_factory=lambda: ["1H", "4H"])


PAIR_CONFIGS: Dict[str, PairConfig] = {
    "BTC-USDT": PairConfig(
        symbol="BTC-USDT",
        tick_size=0.1,
        lot_size=0.001,
        min_notional=5.0,
        max_leverage=125,
        default_leverage=5,
        min_volume_24h=500_000_000,
        timeframes=["1H", "4H"],
    ),
    "ETH-USDT": PairConfig(
        symbol="ETH-USDT",
        tick_size=0.01,
        lot_size=0.001,
        min_notional=5.0,
        max_leverage=100,
        default_leverage=5,
        min_volume_24h=200_000_000,
        timeframes=["1H", "4H"],
    ),
    "SOL-USDT": PairConfig(
        symbol="SOL-USDT",
        tick_size=0.01,
        lot_size=0.01,
        min_notional=5.0,
        max_leverage=75,
        default_leverage=5,
        min_volume_24h=50_000_000,
        timeframes=["1H", "4H"],
    ),
    "BNB-USDT": PairConfig(
        symbol="BNB-USDT",
        tick_size=0.01,
        lot_size=0.001,
        min_notional=5.0,
        max_leverage=75,
        default_leverage=5,
        min_volume_24h=30_000_000,
        timeframes=["1H", "4H"],
    ),
    "XRP-USDT": PairConfig(
        symbol="XRP-USDT",
        tick_size=0.0001,
        lot_size=0.1,
        min_notional=5.0,
        max_leverage=75,
        default_leverage=5,
        min_volume_24h=30_000_000,
        timeframes=["1H", "4H"],
    ),
}


def get_pair_config(symbol: str) -> PairConfig:
    if symbol not in PAIR_CONFIGS:
        raise ValueError(f"No config found for pair: {symbol}")
    return PAIR_CONFIGS[symbol]
