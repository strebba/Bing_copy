"""
Backtesting engine — simulates strategy execution on historical OHLCV data.
Includes slippage, commission, and proper bar-by-bar simulation.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd

from analytics.performance import PerformanceTracker, TradeRecord
from data.feature_engine import enrich
from strategy.base_strategy import BaseStrategy, Signal, SignalDirection

logger = logging.getLogger(__name__)

# Timeframe to expected bar duration in seconds
_TF_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1H": 3600, "1h": 3600, "4H": 14400, "4h": 14400,
    "1D": 86400, "1d": 86400,
}

COMMISSION_RATE = 0.0005    # 0.05 % taker fee (BingX)
DEFAULT_SLIPPAGE = 0.0002   # 0.02 % slippage estimate


@dataclass
class BacktestConfig:
    initial_capital: float = 2000.0
    risk_per_trade: float = 0.01
    max_open_positions: int = 5
    leverage: int = 5
    commission_rate: float = COMMISSION_RATE
    slippage: float = DEFAULT_SLIPPAGE
    max_hold_bars: int = 48   # Time stop in bars (1 bar = 1 H candle by default)


@dataclass
class OpenTrade:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    strategy: str
    opened_bar: int
    partial_taken: bool = False


class Backtester:
    """
    Event-driven bar-by-bar backtester.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        self._strategy = strategy
        self._cfg = config or BacktestConfig()

    @staticmethod
    def _normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all timestamps are in UTC seconds, sort chronologically, deduplicate."""
        if "timestamp" not in df.columns:
            return df

        df = df.copy()

        # Convert millisecond epoch to seconds if needed
        ts = df["timestamp"]
        if ts.dtype in ("int64", "float64") and (ts > 1e12).any():
            df["timestamp"] = ts // 1000
            logger.info("Converted millisecond epoch timestamps to seconds")

        # Sort by timestamp (chronological order)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Remove duplicate timestamps
        n_before = len(df)
        df = df.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)
        n_dupes = n_before - len(df)
        if n_dupes > 0:
            logger.warning("Removed %d duplicate timestamps", n_dupes)

        # Detect gaps > 2x expected timeframe
        timeframe = df.attrs.get("timeframe", "1H")
        expected_sec = _TF_SECONDS.get(timeframe, 3600)
        diffs = df["timestamp"].diff().dropna()
        gap_mask = diffs > 2 * expected_sec
        if gap_mask.any():
            gap_count = int(gap_mask.sum())
            max_gap = int(diffs.max())
            logger.warning(
                "Detected %d timestamp gaps > 2x timeframe (%ss). Max gap: %ds",
                gap_count, expected_sec, max_gap,
            )

        return df

    def run(self, df: pd.DataFrame, symbol: str = "BTC-USDT") -> PerformanceTracker:
        """
        Run a full backtest on the provided OHLCV DataFrame.
        Returns a PerformanceTracker with all trade records.
        """
        tracker = PerformanceTracker(self._cfg.initial_capital)
        equity = self._cfg.initial_capital
        open_trades: List[OpenTrade] = []

        # Normalize timestamps: convert ms→s, sort, dedup, gap warnings
        df = self._normalize_timestamps(df)

        # Enrich with indicators
        df_enriched = enrich(df.copy())
        if df_enriched.empty:
            logger.error("Empty DataFrame after enrichment")
            return tracker

        start_bar = 200   # Warm-up period for indicators

        for i in range(start_bar, len(df_enriched)):
            current_bar = df_enriched.iloc[i]
            bar_open = float(current_bar["open"])
            bar_high = float(current_bar["high"])
            bar_low = float(current_bar["low"])
            bar_close = float(current_bar["close"])
            bar_ts = int(current_bar.get("timestamp", i))

            # ── Check existing trades ──────────────────────────────────────────
            still_open = []
            for trade in open_trades:
                closed, exit_price, reason = self._check_exit(
                    trade, bar_high, bar_low, bar_close, i
                )
                if closed:
                    pnl = self._calc_pnl(trade, exit_price)
                    equity += pnl
                    # Build timestamps
                    opened_dt = datetime.fromtimestamp(
                        bar_ts / 1000 if bar_ts > 1e10 else bar_ts, tz=timezone.utc
                    )
                    record = TradeRecord(
                        symbol=trade.symbol,
                        direction=trade.direction,
                        strategy=trade.strategy,
                        entry_price=trade.entry_price,
                        exit_price=exit_price,
                        quantity=trade.quantity,
                        pnl_usdt=pnl,
                        opened_at=opened_dt,
                        reason=reason,
                    )
                    tracker.record_trade(record)
                else:
                    still_open.append(trade)
            open_trades = still_open

            # ── Generate new signal ────────────────────────────────────────────
            if len(open_trades) >= self._cfg.max_open_positions:
                continue

            window = df_enriched.iloc[max(0, i - 300):i + 1].copy()
            window.attrs["timeframe"] = df.attrs.get("timeframe", "1H")

            try:
                signal = self._strategy.generate_signal(window, symbol)
            except Exception as exc:
                logger.debug("Strategy error at bar %d: %s", i, exc)
                continue

            if signal is None or not signal.is_valid():
                continue

            # Skip if already in same direction
            if any(t.symbol == symbol and t.direction == signal.direction.value
                   for t in open_trades):
                continue

            # Entry with slippage
            entry_price = signal.entry_price * (
                1 + self._cfg.slippage
                if signal.direction == SignalDirection.LONG
                else 1 - self._cfg.slippage
            )

            # Position sizing (fixed fractional)
            risk_usdt = equity * self._cfg.risk_per_trade
            price_risk = abs(entry_price - signal.stop_loss)
            if price_risk <= 0:
                continue
            quantity = risk_usdt / price_risk

            # Commission
            equity -= entry_price * quantity * self._cfg.commission_rate

            open_trades.append(OpenTrade(
                symbol=symbol,
                direction=signal.direction.value,
                entry_price=entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                quantity=quantity,
                strategy=self._strategy.name,
                opened_bar=i,
            ))

        # ── Close any remaining trades at last price ───────────────────────────
        if len(df_enriched) > start_bar:
            last_close = float(df_enriched["close"].iloc[-1])
            last_ts = int(df_enriched.get("timestamp", df_enriched.index).iloc[-1])
            last_dt = datetime.fromtimestamp(
                last_ts / 1000 if last_ts > 1e10 else last_ts, tz=timezone.utc
            )
            for trade in open_trades:
                pnl = self._calc_pnl(trade, last_close)
                equity += pnl
                record = TradeRecord(
                    symbol=trade.symbol,
                    direction=trade.direction,
                    strategy=trade.strategy,
                    entry_price=trade.entry_price,
                    exit_price=last_close,
                    quantity=trade.quantity,
                    pnl_usdt=pnl,
                    opened_at=last_dt,
                    reason="end_of_data",
                )
                tracker.record_trade(record)

        logger.info(
            "Backtest complete: %d trades | Win rate=%.1f%% | PF=%.2f | Sharpe=%.2f | MaxDD=%.1f%%",
            tracker.total_trades(),
            tracker.win_rate() * 100,
            tracker.profit_factor(),
            tracker.sharpe_ratio(),
            tracker.max_drawdown() * 100,
        )
        return tracker

    def _check_exit(
        self,
        trade: OpenTrade,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_idx: int,
    ) -> tuple[bool, float, str]:
        """Check if a trade should be exited. Returns (closed, exit_price, reason)."""
        if trade.direction == "LONG":
            # Stop loss hit — SL is a market order that may fill slightly below the stop
            if bar_low <= trade.stop_loss:
                sl_fill = trade.stop_loss * (1 - self._cfg.slippage)
                return True, sl_fill, "stop_loss"
            # Take profit hit — TP_MARKET order fills slightly below the TP trigger
            if bar_high >= trade.take_profit:
                tp_fill = trade.take_profit * (1 - self._cfg.slippage)
                return True, tp_fill, "take_profit"
        else:  # SHORT
            # SL: market buy fills slightly above stop
            if bar_high >= trade.stop_loss:
                sl_fill = trade.stop_loss * (1 + self._cfg.slippage)
                return True, sl_fill, "stop_loss"
            # TP: market buy fills slightly above TP trigger
            if bar_low <= trade.take_profit:
                tp_fill = trade.take_profit * (1 + self._cfg.slippage)
                return True, tp_fill, "take_profit"

        # Time stop
        bars_held = bar_idx - trade.opened_bar
        if bars_held >= self._cfg.max_hold_bars:
            return True, bar_close, "time_stop"

        return False, bar_close, "open"

    def _calc_pnl(self, trade: OpenTrade, exit_price: float) -> float:
        """Calculate net PnL after commission."""
        if trade.direction == "LONG":
            gross = (exit_price - trade.entry_price) * trade.quantity
        else:
            gross = (trade.entry_price - exit_price) * trade.quantity
        commission = exit_price * trade.quantity * self._cfg.commission_rate
        return gross - commission
