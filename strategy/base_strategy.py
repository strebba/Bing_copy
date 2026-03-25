"""
Abstract base class for all trading strategies.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class Signal:
    direction: SignalDirection
    symbol: str
    strategy_name: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_pct: float                    # Fraction of equity to risk
    confidence: float                  # 0–1 confluence score
    timeframe: str = "1H"
    trailing_activation_rr: float = 1.0  # R:R at which trailing stop activates (1.0 = 1:1)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def risk_reward(self) -> float:
        if self.direction == SignalDirection.LONG:
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
        return reward / risk if risk > 0 else 0.0

    def is_valid(self) -> bool:
        return (
            self.direction != SignalDirection.NONE
            and self.stop_loss > 0
            and self.take_profit > 0
            and self.risk_reward >= 1.5   # Minimum 1:1.5 R:R
            and 0 < self.confidence <= 1.0
        )


class BaseStrategy(ABC):
    """
    All strategies must inherit from BaseStrategy and implement `generate_signal`.
    """

    name: str = "base"

    def __init__(self, params: Any) -> None:
        self.params = params
        self._logger = logging.getLogger(f"strategy.{self.name}")

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Analyse the OHLCV DataFrame and return a Signal or None.

        Args:
            df:     DataFrame with columns [open, high, low, close, volume],
                    sorted ascending by timestamp, at least 200 rows.
            symbol: Trading pair symbol (e.g. "BTC-USDT").

        Returns:
            Signal if a valid setup is found, else None.
        """

    def _validate_df(self, df: pd.DataFrame, min_rows: int = 200) -> bool:
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            self._logger.warning("DataFrame missing required columns")
            return False
        if len(df) < min_rows:
            self._logger.warning(
                "Insufficient data: %d rows (need %d)", len(df), min_rows
            )
            return False
        return True
