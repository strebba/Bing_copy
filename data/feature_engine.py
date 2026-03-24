"""
Feature engineering — enriches OHLCV DataFrames with all indicators
needed by the strategy layer.
"""
import logging

import pandas as pd

from strategy.indicators import (
    adx,
    atr,
    bollinger_bands,
    bb_width,
    donchian_channel,
    ema,
    hurst_exponent,
    macd,
    rsi,
    sma,
    vwap,
    zscore,
)

logger = logging.getLogger(__name__)


def enrich(df: pd.DataFrame, timeframe: str = "1H") -> pd.DataFrame:
    """
    Add all technical indicator columns to an OHLCV DataFrame.
    Input columns required: open, high, low, close, volume.
    Returns a new DataFrame with indicator columns appended.
    """
    if len(df) < 50:
        return df

    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── Trend ──────────────────────────────────────────────────────────────────
    df["ema_9"] = ema(close, 9)
    df["ema_21"] = ema(close, 21)
    df["ema_50"] = ema(close, 50)
    df["ema_200"] = ema(close, 200)
    df["sma_20"] = sma(close, 20)

    macd_line, signal_line, histogram = macd(close, 12, 26, 9)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram

    df["adx"] = adx(high, low, close, 14)

    # ── Momentum ───────────────────────────────────────────────────────────────
    df["rsi_14"] = rsi(close, 14)
    df["rsi_7"] = rsi(close, 7)

    # ── Volatility ─────────────────────────────────────────────────────────────
    df["atr_14"] = atr(high, low, close, 14)
    df["atr_7"] = atr(high, low, close, 7)

    bb_u, bb_m, bb_l = bollinger_bands(close, 20, 2.0)
    df["bb_upper"] = bb_u
    df["bb_mid"] = bb_m
    df["bb_lower"] = bb_l
    df["bb_width"] = bb_width(bb_u, bb_l, bb_m)

    bb_u25, _, bb_l25 = bollinger_bands(close, 20, 2.5)
    df["bb_upper_25"] = bb_u25
    df["bb_lower_25"] = bb_l25

    # ── Channels ───────────────────────────────────────────────────────────────
    dc_u, dc_m, dc_l = donchian_channel(high, low, 20)
    df["dc_upper"] = dc_u
    df["dc_mid"] = dc_m
    df["dc_lower"] = dc_l

    # ── Volume ─────────────────────────────────────────────────────────────────
    df["vol_sma_20"] = sma(volume, 20)
    df["vol_ratio"] = volume / df["vol_sma_20"]
    df["vwap"] = vwap(high, low, close, volume)

    # ── Statistical ────────────────────────────────────────────────────────────
    df["zscore_20"] = zscore(close, 20)

    # Log returns
    df["log_ret"] = close.apply(lambda x: 0.0 if x <= 0 else x).pct_change()

    df.attrs["timeframe"] = timeframe
    return df
