"""
Portfolio Manager — regime switching, strategy allocation, and signal aggregation.

H-4 Fix: subscribes to CIRCUIT_BREAKER_LEVEL_CHANGE events and keeps an
up-to-date _cb_size_multiplier that is applied to every new signal's risk_pct.
This replaces the previous polling approach (reading dd_monitor.current_drawdown()
on every cycle) with an event-driven one so strategies react immediately.
"""
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

from config.settings import STRATEGY_WEIGHTS
from strategy.alpha_momentum import AlphaMomentumStrategy
from strategy.base_strategy import Signal, SignalDirection
from strategy.beta_mean_rev import BetaMeanReversionStrategy
from strategy.gamma_breakout import GammaBreakoutStrategy
from strategy.indicators import hurst_exponent, atr

if TYPE_CHECKING:
    from core.event_bus import Event, EventBus

logger = logging.getLogger(__name__)

CONFLICT_CONFIDENCE_THRESHOLD = 0.05  # 5% difference to resolve conflict


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

    def __init__(
        self,
        drawdown_monitor=None,
        event_bus: Optional["EventBus"] = None,
    ) -> None:
        self._alpha = AlphaMomentumStrategy()
        self._beta = BetaMeanReversionStrategy()
        self._gamma = GammaBreakoutStrategy()
        self._dd_monitor = drawdown_monitor
        self._base_weights = STRATEGY_WEIGHTS.copy()

        # H-4: size multiplier received from circuit breaker events.
        # Starts at 1.0 (no restriction) and is updated by the event handler.
        self._cb_size_multiplier: float = 1.0

        if event_bus is not None:
            from core.event_bus import EventType  # noqa: PLC0415
            event_bus.subscribe(
                EventType.CIRCUIT_BREAKER_LEVEL_CHANGE,
                self._on_circuit_breaker_level_change,
            )

    async def _on_circuit_breaker_level_change(self, event: "Event") -> None:
        """Update the size multiplier whenever the circuit breaker changes level (H-4)."""
        multiplier = event.data.get("size_multiplier", 1.0)
        new_level = event.data.get("new_level", "NONE")
        self._cb_size_multiplier = multiplier
        logger.warning(
            "PortfolioManager: circuit breaker → %s | size_multiplier=%.2f",
            new_level,
            multiplier,
        )

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

        # H-4: apply circuit breaker size multiplier from event bus (may be < 1.0)
        cb_mult = self._cb_size_multiplier

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
                sig.risk_pct *= weights.get("alpha", 0.4) * cb_mult
                all_signals.append(sig)
                logger.info("[Alpha] %s %s (conf=%.2f)", symbol, sig.direction, sig.confidence)

            # Beta
            fr = funding_rates.get(symbol, 0.0)
            sig = self._beta.generate_signal(df, symbol, funding_rate=fr)
            if sig and sig.is_valid():
                sig.risk_pct *= weights.get("beta", 0.3) * cb_mult
                all_signals.append(sig)
                logger.info("[Beta] %s %s (conf=%.2f)", symbol, sig.direction, sig.confidence)

            # Gamma
            oi = oi_changes.get(symbol, 0.0)
            sig = self._gamma.generate_signal(df, symbol, open_interest_change=oi)
            if sig and sig.is_valid():
                sig.risk_pct *= weights.get("gamma", 0.3) * cb_mult
                all_signals.append(sig)
                logger.info("[Gamma] %s %s (conf=%.2f)", symbol, sig.direction, sig.confidence)

        return self._resolve_conflicts(all_signals)

    @staticmethod
    def _resolve_conflicts(signals: List[Signal]) -> List[Signal]:
        """
        Resolve conflicting signals on the same symbol.

        Rules:
        - If signals disagree on direction for the same symbol:
          a) Take the one with highest confidence if diff >= 5%
          b) Drop both if confidence difference < 5% (uncertainty)
        - If multiple signals agree on direction for the same symbol:
          keep only the one with highest confidence (avoid double sizing)
        """
        # Group signals by symbol
        by_symbol: Dict[str, List[Signal]] = defaultdict(list)
        for sig in signals:
            by_symbol[sig.symbol].append(sig)

        resolved: List[Signal] = []
        for symbol, sym_signals in by_symbol.items():
            if len(sym_signals) == 1:
                resolved.append(sym_signals[0])
                continue

            # Separate by direction
            longs = [s for s in sym_signals if s.direction == SignalDirection.LONG]
            shorts = [s for s in sym_signals if s.direction == SignalDirection.SHORT]

            if longs and shorts:
                # Conflict: opposing directions
                best_long = max(longs, key=lambda s: s.confidence)
                best_short = max(shorts, key=lambda s: s.confidence)
                diff = abs(best_long.confidence - best_short.confidence)

                if diff < CONFLICT_CONFIDENCE_THRESHOLD:
                    logger.warning(
                        "[Conflict] %s: LONG (conf=%.2f, %s) vs SHORT (conf=%.2f, %s) "
                        "— diff %.2f%% < 5%%, NO TRADE",
                        symbol,
                        best_long.confidence, best_long.strategy_name,
                        best_short.confidence, best_short.strategy_name,
                        diff * 100,
                    )
                else:
                    winner = best_long if best_long.confidence > best_short.confidence else best_short
                    loser = best_short if winner is best_long else best_long
                    logger.warning(
                        "[Conflict] %s: %s wins (conf=%.2f, %s) over %s (conf=%.2f, %s)",
                        symbol,
                        winner.direction.value, winner.confidence, winner.strategy_name,
                        loser.direction.value, loser.confidence, loser.strategy_name,
                    )
                    resolved.append(winner)
            elif longs:
                # Multiple longs — pick best, no double sizing
                best = max(longs, key=lambda s: s.confidence)
                if len(longs) > 1:
                    logger.info(
                        "[Agree] %s: %d LONG signals, keeping best (conf=%.2f, %s)",
                        symbol, len(longs), best.confidence, best.strategy_name,
                    )
                resolved.append(best)
            elif shorts:
                # Multiple shorts — pick best, no double sizing
                best = max(shorts, key=lambda s: s.confidence)
                if len(shorts) > 1:
                    logger.info(
                        "[Agree] %s: %d SHORT signals, keeping best (conf=%.2f, %s)",
                        symbol, len(shorts), best.confidence, best.strategy_name,
                    )
                resolved.append(best)

        return resolved
