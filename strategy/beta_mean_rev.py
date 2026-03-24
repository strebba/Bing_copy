"""
Strategy Beta — Mean Reversion (30 % portfolio weight).
Entry: price beyond 2.5σ BB + Z-score extreme + RSI + funding rate contrarian.
"""
import logging
from typing import Optional

import pandas as pd

from config.strategies import BETA_CONFIG, BetaConfig
from strategy.base_strategy import BaseStrategy, Signal, SignalDirection
from strategy.indicators import bollinger_bands, hurst_exponent, rsi, vwap, zscore

logger = logging.getLogger(__name__)


class BetaMeanReversionStrategy(BaseStrategy):
    name = "beta"

    def __init__(self, params: BetaConfig = BETA_CONFIG) -> None:
        super().__init__(params)

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        funding_rate: float = 0.0,
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
        bb_upper_sl, _, bb_lower_sl = bollinger_bands(close, p.bb_period, p.bb_sl_std)

        rsi_vals = rsi(close, p.rsi_period)
        vwap_vals = vwap(high, low, close, volume)
        zs = zscore(close, p.bb_period)

        last_close = close.iloc[-1]
        last_rsi = rsi_vals.iloc[-1]
        last_bb_upper = bb_upper.iloc[-1]
        last_bb_lower = bb_lower.iloc[-1]
        last_bb_mid = bb_mid.iloc[-1]
        last_bb_upper_sl = bb_upper_sl.iloc[-1]
        last_bb_lower_sl = bb_lower_sl.iloc[-1]
        last_z = zs.iloc[-1]

        # ── Hurst regime check ────────────────────────────────────────────────
        if len(close) >= 100:
            h = hurst_exponent(close.tail(100))
        else:
            h = 0.5
        mean_reverting = h < p.hurst_max

        # ── LONG signal: price below lower BB (oversold) ──────────────────────
        conditions_long = [
            last_close < last_bb_lower,              # Below lower BB
            last_z < -p.zscore_threshold,            # Z-score extreme negative
            last_rsi < p.rsi_oversold,               # RSI oversold
            funding_rate < -0.0005,                  # Negative funding (contrarian)
            mean_reverting,                          # Regime confirms mean reversion
        ]
        # ── SHORT signal: price above upper BB (overbought) ───────────────────
        conditions_short = [
            last_close > last_bb_upper,              # Above upper BB
            last_z > p.zscore_threshold,             # Z-score extreme positive
            last_rsi > p.rsi_overbought,             # RSI overbought
            funding_rate > 0.0005,                   # Positive funding (contrarian)
            mean_reverting,
        ]

        # Require at least 3 out of 5 conditions (funding is often 0 in backtest)
        long_score = sum(conditions_long[:4])   # Exclude hurst from count
        short_score = sum(conditions_short[:4])

        if pd.isna(last_z) or pd.isna(last_bb_lower):
            return None

        if long_score >= 3 and mean_reverting:
            sl = last_bb_lower_sl - last_close * p.sl_buffer_pct
            return Signal(
                direction=SignalDirection.LONG,
                symbol=symbol,
                strategy_name=self.name,
                entry_price=last_close,
                stop_loss=sl,
                take_profit=last_bb_mid,
                risk_pct=p.risk_pct,
                confidence=long_score / 4,
                timeframe=df.attrs.get("timeframe", "1H"),
                metadata={"hurst": round(h, 3), "zscore": round(last_z, 2)},
            )

        if short_score >= 3 and mean_reverting:
            sl = last_bb_upper_sl + last_close * p.sl_buffer_pct
            return Signal(
                direction=SignalDirection.SHORT,
                symbol=symbol,
                strategy_name=self.name,
                entry_price=last_close,
                stop_loss=sl,
                take_profit=last_bb_mid,
                risk_pct=p.risk_pct,
                confidence=short_score / 4,
                timeframe=df.attrs.get("timeframe", "1H"),
                metadata={"hurst": round(h, 3), "zscore": round(last_z, 2)},
            )

        return None
