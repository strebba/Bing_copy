"""
Strategy Gamma — Breakout Volatility (30 % portfolio weight).
Entry: Donchian breakout after BB squeeze + volume spike + OI increase.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.strategies import GAMMA_CONFIG, GammaConfig
from strategy.base_strategy import BaseStrategy, Signal, SignalDirection
from strategy.indicators import atr, bb_width, bollinger_bands, donchian_channel

logger = logging.getLogger(__name__)


class GammaBreakoutStrategy(BaseStrategy):
    name = "gamma"

    def __init__(self, params: GammaConfig = GAMMA_CONFIG) -> None:
        super().__init__(params)

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        open_interest_change: float = 0.0,  # Fractional change in OI
    ) -> Optional[Signal]:
        if not self._validate_df(df):
            return None

        p = self.params
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ── Indicators ────────────────────────────────────────────────────────
        bb_upper, bb_mid, bb_lower = bollinger_bands(close, p.bb_period, p.bb_std)
        bbw = bb_width(bb_upper, bb_lower, bb_mid)

        dc_upper, dc_mid, dc_lower = donchian_channel(high, low, p.donchian_period)
        atr_vals = atr(high, low, close, p.atr_period)

        avg_vol = volume.rolling(20).mean()

        # ATR percentile
        atr_lookback = min(p.atr_lookback, len(atr_vals.dropna()))
        atr_tail = atr_vals.tail(atr_lookback)
        atr_pct_rank = (atr_tail < atr_vals.iloc[-1]).mean() * 100

        # BBW percentile (squeeze detection)
        bbw_lookback = min(p.atr_lookback, len(bbw.dropna()))
        bbw_tail = bbw.tail(bbw_lookback)
        bbw_pct_rank = (bbw_tail < bbw.iloc[-1]).mean() * 100

        # Consecutive squeeze bars
        squeeze_active = bbw < bbw_tail.quantile(0.20)
        consecutive_squeeze = int(
            squeeze_active.iloc[-p.squeeze_periods_min:].all()
        )

        last_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        last_vol = volume.iloc[-1]
        last_avg_vol = avg_vol.iloc[-1]
        last_dc_upper = dc_upper.iloc[-1]
        last_dc_lower = dc_lower.iloc[-1]
        last_atr = atr_vals.iloc[-1]
        last_dc_mid = dc_mid.iloc[-1]

        if pd.isna(last_atr) or last_atr <= 0:
            return None

        # ── Breakout conditions ───────────────────────────────────────────────
        volume_spike = last_vol > p.volume_spike_mult * last_avg_vol
        oi_expanding = open_interest_change > 0
        squeeze_resolved = consecutive_squeeze or bbw_pct_rank < 25

        # LONG breakout: close above Donchian upper
        bull_breakout = (
            last_close > last_dc_upper
            and prev_close <= dc_upper.iloc[-2]          # Fresh breakout
            and volume_spike
            and squeeze_resolved
        )

        # SHORT breakout: close below Donchian lower
        bear_breakout = (
            last_close < last_dc_lower
            and prev_close >= dc_lower.iloc[-2]
            and volume_spike
            and squeeze_resolved
        )

        conf_score = sum([volume_spike, oi_expanding, squeeze_resolved]) / 3

        if bull_breakout:
            sl = last_dc_mid   # Center of the compression range
            return Signal(
                direction=SignalDirection.LONG,
                symbol=symbol,
                strategy_name=self.name,
                entry_price=last_close,
                stop_loss=sl,
                take_profit=last_close + 2 * last_atr,   # Initial TP; trailing takes over
                risk_pct=p.risk_pct,
                confidence=conf_score,
                timeframe=df.attrs.get("timeframe", "1H"),
                metadata={
                    "bbw_pct": round(bbw_pct_rank, 1),
                    "vol_spike": round(last_vol / last_avg_vol, 2),
                    "trailing": True,
                    "trailing_atr_mult": p.trailing_atr_mult,
                },
            )

        if bear_breakout:
            sl = last_dc_mid
            return Signal(
                direction=SignalDirection.SHORT,
                symbol=symbol,
                strategy_name=self.name,
                entry_price=last_close,
                stop_loss=sl,
                take_profit=last_close - 2 * last_atr,
                risk_pct=p.risk_pct,
                confidence=conf_score,
                timeframe=df.attrs.get("timeframe", "1H"),
                metadata={
                    "bbw_pct": round(bbw_pct_rank, 1),
                    "vol_spike": round(last_vol / last_avg_vol, 2),
                    "trailing": True,
                    "trailing_atr_mult": p.trailing_atr_mult,
                },
            )

        return None
