"""
Strategy parameter configuration for all three strategies.
"""
from dataclasses import dataclass


@dataclass
class AlphaConfig:
    """Momentum Reversal — 40 % portfolio weight."""
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ema_fast: int = 50
    ema_slow: int = 200
    adx_period: int = 14
    adx_min: float = 20.0
    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 2.5
    min_confluence: int = 3         # Out of 4 conditions
    kelly_fraction: float = 0.5
    max_risk_pct: float = 0.02


@dataclass
class BetaConfig:
    """Mean Reversion — 30 % portfolio weight."""
    bb_period: int = 20
    bb_std: float = 2.5
    bb_sl_std: float = 3.0
    rsi_period: int = 14
    rsi_oversold: float = 25.0
    rsi_overbought: float = 75.0
    zscore_threshold: float = 2.0
    hurst_max: float = 0.45         # Mean-reverting regime
    sl_buffer_pct: float = 0.002
    risk_pct: float = 0.01


@dataclass
class GammaConfig:
    """Breakout Volatility — 30 % portfolio weight."""
    bb_period: int = 20
    bb_std: float = 2.0
    donchian_period: int = 20
    atr_period: int = 14
    atr_percentile_max: int = 20    # Below 20th pct = compression
    atr_lookback: int = 100
    squeeze_periods_min: int = 10
    volume_spike_mult: float = 2.0
    trailing_atr_mult: float = 2.0
    risk_pct: float = 0.0075        # Reduced for breakout strategy


ALPHA_CONFIG = AlphaConfig()
BETA_CONFIG = BetaConfig()
GAMMA_CONFIG = GammaConfig()
