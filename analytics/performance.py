"""
Performance analytics — PnL tracking, equity curve, and KPI calculation.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    symbol: str
    direction: str
    strategy: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl_usdt: float
    opened_at: datetime
    closed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""

    @property
    def duration_hours(self) -> float:
        return (self.closed_at - self.opened_at).total_seconds() / 3600

    @property
    def is_win(self) -> bool:
        return self.pnl_usdt > 0

    @property
    def r_multiple(self) -> float:
        risk = abs(self.entry_price - self.exit_price) * self.quantity
        return self.pnl_usdt / risk if risk > 0 else 0.0


class PerformanceTracker:
    """
    Maintains trade history and computes all performance metrics.
    """

    def __init__(self, initial_equity: float) -> None:
        self._initial_equity = initial_equity
        self._current_equity = initial_equity
        self._trades: List[TradeRecord] = []
        self._equity_curve: List[tuple[datetime, float]] = [
            (datetime.now(timezone.utc), initial_equity)
        ]

    def record_trade(self, trade: TradeRecord) -> None:
        self._trades.append(trade)
        self._current_equity += trade.pnl_usdt
        self._equity_curve.append((trade.closed_at, self._current_equity))
        logger.info(
            "Trade closed: %s %s PnL=%.2f USDT (%.2fR)",
            trade.symbol, trade.direction, trade.pnl_usdt, trade.r_multiple,
        )

    # ── Metrics ───────────────────────────────────────────────────────────────

    def win_rate(self, last_n: Optional[int] = None) -> float:
        trades = self._trades[-last_n:] if last_n else self._trades
        if not trades:
            return 0.0
        return sum(1 for t in trades if t.is_win) / len(trades)

    def profit_factor(self, last_n: Optional[int] = None) -> float:
        trades = self._trades[-last_n:] if last_n else self._trades
        gross_profit = sum(t.pnl_usdt for t in trades if t.pnl_usdt > 0)
        gross_loss = abs(sum(t.pnl_usdt for t in trades if t.pnl_usdt < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    def sharpe_ratio(self, risk_free_daily: float = 0.0) -> float:
        if len(self._equity_curve) < 2:
            return 0.0
        equities = [e for _, e in self._equity_curve]
        returns = np.diff(equities) / np.array(equities[:-1])
        excess = returns - risk_free_daily
        std = np.std(excess)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(252))

    def sortino_ratio(self) -> float:
        if len(self._equity_curve) < 2:
            return 0.0
        equities = [e for _, e in self._equity_curve]
        returns = np.diff(equities) / np.array(equities[:-1])
        downside = returns[returns < 0]
        downside_std = np.std(downside)
        if downside_std == 0:
            return 0.0
        return float(np.mean(returns) / downside_std * np.sqrt(252))

    def max_drawdown(self) -> float:
        if not self._equity_curve:
            return 0.0
        equities = [e for _, e in self._equity_curve]
        peak = equities[0]
        max_dd = 0.0
        for e in equities:
            if e > peak:
                peak = e
            dd = (peak - e) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    def recovery_factor(self) -> float:
        net_profit = self._current_equity - self._initial_equity
        dd = self.max_drawdown() * self._initial_equity
        return net_profit / dd if dd > 0 else float("inf")

    def expectancy(self) -> float:
        """Expected value per trade in R-multiples."""
        if not self._trades:
            return 0.0
        return float(np.mean([t.r_multiple for t in self._trades]))

    def avg_trade_duration_hours(self) -> float:
        if not self._trades:
            return 0.0
        return float(np.mean([t.duration_hours for t in self._trades]))

    def total_trades(self) -> int:
        return len(self._trades)

    def net_pnl_usdt(self) -> float:
        return self._current_equity - self._initial_equity

    def roi_pct(self) -> float:
        return self.net_pnl_usdt() / self._initial_equity

    def summary(self) -> dict:
        return {
            "total_trades": self.total_trades(),
            "win_rate": round(self.win_rate(), 4),
            "win_rate_50": round(self.win_rate(last_n=50), 4),
            "profit_factor": round(self.profit_factor(), 3),
            "profit_factor_50": round(self.profit_factor(last_n=50), 3),
            "sharpe_ratio": round(self.sharpe_ratio(), 3),
            "sortino_ratio": round(self.sortino_ratio(), 3),
            "max_drawdown_pct": round(self.max_drawdown() * 100, 2),
            "recovery_factor": round(self.recovery_factor(), 3),
            "expectancy_r": round(self.expectancy(), 3),
            "avg_duration_h": round(self.avg_trade_duration_hours(), 2),
            "net_pnl_usdt": round(self.net_pnl_usdt(), 2),
            "roi_pct": round(self.roi_pct() * 100, 2),
            "current_equity": round(self._current_equity, 2),
        }
