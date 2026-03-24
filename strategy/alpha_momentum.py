"""
Strategy Alpha — Momentum Reversal (40 % portfolio weight).
Entry: RSI divergence + MACD slope change + volume confirmation.
"""
import logging
from typing import Optional

import pandas as pd

from config.strategies import ALPHA_CONFIG, AlphaConfig
from strategy.base_strategy import BaseStrategy, Signal, SignalDirection
from strategy.indicators import (
    adx,
    atr,
    detect_rsi_divergence,
    ema,
    macd,
    rsi,
)

logger = logging.getLogger(__name__)


class AlphaMomentumStrategy(BaseStrategy):
    name = "alpha"

    def __init__(self, params: AlphaConfig = ALPHA_CONFIG) -> None:
        super().__init__(params)

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if not self._validate_df(df):
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ── Compute indicators ────────────────────────────────────────────────
        p = self.params
        rsi_vals = rsi(close, p.rsi_period)
        macd_line, _, histogram = macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
        ema_fast = ema(close, p.ema_fast)
        ema_slow = ema(close, p.ema_slow)
        adx_vals = adx(high, low, close, p.adx_period)
        atr_vals = atr(high, low, close, p.atr_period)

        last = df.iloc[-1]
        last_rsi = rsi_vals.iloc[-1]
        last_adx = adx_vals.iloc[-1]
        last_atr = atr_vals.iloc[-1]
        last_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]
        last_vol = volume.iloc[-1]
        avg_vol = volume.rolling(20).mean().iloc[-1]
        last_price = close.iloc[-1]
        last_ema_fast = ema_fast.iloc[-1]
        last_ema_slow = ema_slow.iloc[-1]

        bull_div, bear_div = detect_rsi_divergence(close, rsi_vals, lookback=5)

        # ── Trend context ─────────────────────────────────────────────────────
        uptrend = last_ema_fast > last_ema_slow
        downtrend = last_ema_fast < last_ema_slow

        # ── Volume divergence ─────────────────────────────────────────────────
        # Bearish vol divergence: price making higher highs but volume falling
        # Bullish vol divergence: price making lower lows but volume increasing
        vol_div_bullish = (last_price < close.iloc[-6]) and (last_vol > avg_vol)
        vol_div_bearish = (last_price > close.iloc[-6]) and (last_vol < avg_vol)

        adx_ok = last_adx > p.adx_min if pd.notna(last_adx) else False

        # ── LONG signal (bullish reversal) ────────────────────────────────────
        conditions_long = [
            bull_div,                          # RSI bullish divergence
            last_hist > prev_hist,             # MACD histogram slope turning up
            vol_div_bullish,                   # Volume confirmation
            adx_ok and downtrend,              # Confirmed downtrend (reversing)
        ]
        confluence_long = sum(conditions_long)

        # ── SHORT signal (bearish reversal) ───────────────────────────────────
        conditions_short = [
            bear_div,                          # RSI bearish divergence
            last_hist < prev_hist,             # MACD histogram slope turning down
            vol_div_bearish,                   # Volume confirmation
            adx_ok and uptrend,                # Confirmed uptrend (reversing)
        ]
        confluence_short = sum(conditions_short)

        if pd.isna(last_atr) or last_atr <= 0:
            return None

        # ── Emit signals ──────────────────────────────────────────────────────
        if confluence_long >= p.min_confluence:
            return Signal(
                direction=SignalDirection.LONG,
                symbol=symbol,
                strategy_name=self.name,
                entry_price=last_price,
                stop_loss=last_price - p.sl_atr_mult * last_atr,
                take_profit=last_price + p.tp_atr_mult * last_atr,
                risk_pct=p.max_risk_pct,
                confidence=confluence_long / 4,
                timeframe=df.attrs.get("timeframe", "1H"),
                metadata={"rsi": round(last_rsi, 2), "adx": round(last_adx, 2)},
            )

        if confluence_short >= p.min_confluence:
            return Signal(
                direction=SignalDirection.SHORT,
                symbol=symbol,
                strategy_name=self.name,
                entry_price=last_price,
                stop_loss=last_price + p.sl_atr_mult * last_atr,
                take_profit=last_price - p.tp_atr_mult * last_atr,
                risk_pct=p.max_risk_pct,
                confidence=confluence_short / 4,
                timeframe=df.attrs.get("timeframe", "1H"),
                metadata={"rsi": round(last_rsi, 2), "adx": round(last_adx, 2)},
            )

        return None
