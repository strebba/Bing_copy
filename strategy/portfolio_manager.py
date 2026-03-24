"""
Portfolio Manager — regime switching, strategy allocation, and signal aggregation.
"""
import logging
from typing import Dict, List, Optional

import pandas as pd

from config.settings import STRATEGY_WEIGHTS
from strategy.alpha_momentum import AlphaMomentumStrategy
from strategy.base_strategy import Signal
from strategy.beta_mean_rev import BetaMeanReversionStrategy
from strategy.gamma_breakout import GammaBreakoutStrategy
from strategy.indicators import hurst_exponent, atr

logger = logging.getLogger(__name__)


class MarketRegime:
    TRENDING = "trending"
    RANGING = "ranging"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"


def classify_regime(df: pd.DataFrame) -> str:
    """Classify market regime from price/volatility data."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    if len(close) < 100:
        return MarketRegime.RANGING

    # Hurst for trending vs ranging
    h = hurst_exponent(close.tail(100))

    # ATR percentile for volatility regime
    atr_vals = atr(high, low, close, 14)
    atr_tail = atr_vals.tail(100).dropna()
    current_atr = atr_vals.iloc[-1]
    if len(atr_tail) > 0:
        atr_pct = (atr_tail < current_atr).mean()
    else:
        atr_pct = 0.5

    if atr_pct > 0.75:
        return MarketRegime.HIGH_VOL
    elif atr_pct < 0.25:
        return MarketRegime.LOW_VOL
    elif h > 0.55:
        return MarketRegime.TRENDING
    else:
        return MarketRegime.RANGING


REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    MarketRegime.TRENDING:  {"alpha": 0.50, "beta": 0.15, "gamma": 0.35},
    MarketRegime.RANGING:   {"alpha": 0.30, "beta": 0.50, "gamma": 0.20},
    MarketRegime.HIGH_VOL:  {"alpha": 0.30, "beta": 0.20, "gamma": 0.30},  # +20% cash
    MarketRegime.LOW_VOL:   {"alpha": 0.40, "beta": 0.30, "gamma": 0.30},
}


class PortfolioManager:
    """
    Orchestrates all strategies, selects regime-appropriate weights,
    and applies drawdown-based size scaling.
    """

    def __init__(self, drawdown_monitor=None) -> None:
        self._alpha = AlphaMomentumStrategy()
        self._beta = BetaMeanReversionStrategy()
        self._gamma = GammaBreakoutStrategy()
        self._dd_monitor = drawdown_monitor
        self._base_weights = STRATEGY_WEIGHTS.copy()

    def get_weights(self, regime: str, drawdown_pct: float = 0.0) -> Dict[str, float]:
        """Return adjusted strategy weights based on regime and drawdown."""
        weights = REGIME_WEIGHTS.get(regime, self._base_weights).copy()

        # Drawdown scaling
        if drawdown_pct <= -0.10:
            scale = 0.50
        elif drawdown_pct <= -0.05:
            scale = 0.75
        else:
            scale = 1.0

        return {k: v * scale for k, v in weights.items()}

    def generate_signals(
        self,
        df_by_symbol: Dict[str, pd.DataFrame],
        funding_rates: Dict[str, float] = None,
        oi_changes: Dict[str, float] = None,
    ) -> List[Signal]:
        """
        Run all strategies on each symbol's OHLCV DataFrame.
        Returns aggregated list of valid signals, scaled by strategy weight.
        """
        if funding_rates is None:
            funding_rates = {}
        if oi_changes is None:
            oi_changes = {}

        all_signals: List[Signal] = []

        for symbol, df in df_by_symbol.items():
            # Determine regime per symbol
            regime = classify_regime(df)
            current_dd = (
                self._dd_monitor.current_drawdown() if self._dd_monitor else 0.0
            )
            weights = self.get_weights(regime, current_dd)

            # Alpha
            sig = self._alpha.generate_signal(df, symbol)
            if sig and sig.is_valid():
                sig.risk_pct *= weights.get("alpha", 0.4)
                all_signals.append(sig)
                logger.info("[Alpha] %s %s (conf=%.2f)", symbol, sig.direction, sig.confidence)

            # Beta
            fr = funding_rates.get(symbol, 0.0)
            sig = self._beta.generate_signal(df, symbol, funding_rate=fr)
            if sig and sig.is_valid():
                sig.risk_pct *= weights.get("beta", 0.3)
                all_signals.append(sig)
                logger.info("[Beta] %s %s (conf=%.2f)", symbol, sig.direction, sig.confidence)

            # Gamma
            oi = oi_changes.get(symbol, 0.0)
            sig = self._gamma.generate_signal(df, symbol, open_interest_change=oi)
            if sig and sig.is_valid():
                sig.risk_pct *= weights.get("gamma", 0.3)
                all_signals.append(sig)
                logger.info("[Gamma] %s %s (conf=%.2f)", symbol, sig.direction, sig.confidence)

        return all_signals
