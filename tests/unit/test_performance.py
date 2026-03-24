"""Unit tests for performance analytics."""
from datetime import datetime, timezone

import pytest

from analytics.performance import PerformanceTracker, TradeRecord


def make_trade(pnl: float, symbol: str = "BTC-USDT") -> TradeRecord:
    return TradeRecord(
        symbol=symbol,
        direction="LONG",
        strategy="alpha",
        entry_price=50_000.0,
        exit_price=50_000.0 + pnl,
        quantity=0.01,
        pnl_usdt=pnl,
        opened_at=datetime.now(timezone.utc),
    )


class TestPerformanceTracker:
    def setup_method(self):
        self.tracker = PerformanceTracker(initial_equity=10_000)

    def test_win_rate_all_wins(self):
        for _ in range(10):
            self.tracker.record_trade(make_trade(100))
        assert self.tracker.win_rate() == 1.0

    def test_win_rate_mixed(self):
        for _ in range(6):
            self.tracker.record_trade(make_trade(100))
        for _ in range(4):
            self.tracker.record_trade(make_trade(-100))
        assert abs(self.tracker.win_rate() - 0.6) < 1e-6

    def test_profit_factor(self):
        self.tracker.record_trade(make_trade(200))
        self.tracker.record_trade(make_trade(-100))
        assert abs(self.tracker.profit_factor() - 2.0) < 1e-6

    def test_profit_factor_no_losses(self):
        self.tracker.record_trade(make_trade(100))
        assert self.tracker.profit_factor() == float("inf")

    def test_max_drawdown(self):
        self.tracker.record_trade(make_trade(1000))   # equity = 11000
        self.tracker.record_trade(make_trade(-2000))  # equity = 9000
        dd = self.tracker.max_drawdown()
        # DD = (11000 - 9000) / 11000 ≈ 18.18%
        assert abs(dd - 2000 / 11000) < 1e-6

    def test_net_pnl(self):
        self.tracker.record_trade(make_trade(500))
        self.tracker.record_trade(make_trade(-200))
        assert abs(self.tracker.net_pnl_usdt() - 300) < 1e-6

    def test_total_trades(self):
        for _ in range(5):
            self.tracker.record_trade(make_trade(10))
        assert self.tracker.total_trades() == 5

    def test_summary_keys(self):
        self.tracker.record_trade(make_trade(100))
        summary = self.tracker.summary()
        required = [
            "total_trades", "win_rate", "profit_factor", "sharpe_ratio",
            "max_drawdown_pct", "net_pnl_usdt", "roi_pct"
        ]
        for key in required:
            assert key in summary

    def test_win_rate_last_n(self):
        for _ in range(5):
            self.tracker.record_trade(make_trade(100))
        for _ in range(5):
            self.tracker.record_trade(make_trade(-100))
        # Last 5 should all be losses
        assert self.tracker.win_rate(last_n=5) == 0.0
