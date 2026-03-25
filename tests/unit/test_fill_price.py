"""Tests for H-7: Fill Price Tracking."""
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.state_manager import (
    Position,
    StateManager,
    compute_slippage_bps,
    recalculate_sl_tp,
)
from exchange.order_manager import FillResult, Order, OrderManager


# ── Unit tests: slippage calculation ─────────────────────────────────────────


class TestSlippageCalculation:
    def test_compute_slippage_bps_positive(self):
        # fill higher than requested → slippage
        bps = compute_slippage_bps(fill_price=50_100, requested_price=50_000)
        assert abs(bps - 20.0) < 0.1  # 0.2% = 20 bps

    def test_compute_slippage_bps_negative(self):
        # fill lower than requested → still positive bps
        bps = compute_slippage_bps(fill_price=49_900, requested_price=50_000)
        assert abs(bps - 20.0) < 0.1

    def test_compute_slippage_bps_zero(self):
        bps = compute_slippage_bps(fill_price=50_000, requested_price=50_000)
        assert bps == 0.0

    def test_compute_slippage_bps_zero_requested(self):
        bps = compute_slippage_bps(fill_price=50_000, requested_price=0.0)
        assert bps == 0.0


# ── Unit tests: SL/TP recalculation ──────────────────────────────────────────


class TestRecalculateSLTP:
    def test_long_fill_higher_than_requested(self):
        """LONG: fill at 50100 instead of 50000 → SL and TP shift up by 100."""
        new_sl, new_tp = recalculate_sl_tp(
            fill_price=50_100,
            requested_price=50_000,
            original_sl=49_500,
            original_tp=51_500,
            position_side="LONG",
        )
        # SL distance was 500, so new SL = 50100 - 500 = 49600
        assert abs(new_sl - 49_600) < 0.01
        # TP distance was 1500, so new TP = 50100 + 1500 = 51600
        assert abs(new_tp - 51_600) < 0.01

    def test_long_fill_lower_than_requested(self):
        """LONG: fill at 49900 instead of 50000 → SL and TP shift down."""
        new_sl, new_tp = recalculate_sl_tp(
            fill_price=49_900,
            requested_price=50_000,
            original_sl=49_500,
            original_tp=51_500,
            position_side="LONG",
        )
        assert abs(new_sl - 49_400) < 0.01
        assert abs(new_tp - 51_400) < 0.01

    def test_short_fill_lower_than_requested(self):
        """SHORT: fill at 49900 instead of 50000 → SL shifts down, TP shifts down."""
        new_sl, new_tp = recalculate_sl_tp(
            fill_price=49_900,
            requested_price=50_000,
            original_sl=50_500,
            original_tp=48_500,
            position_side="SHORT",
        )
        # SL distance was 500, new SL = 49900 + 500 = 50400
        assert abs(new_sl - 50_400) < 0.01
        # TP distance was 1500, new TP = 49900 - 1500 = 48400
        assert abs(new_tp - 48_400) < 0.01

    def test_no_slippage_no_change(self):
        """Fill at exactly requested price → SL/TP unchanged."""
        new_sl, new_tp = recalculate_sl_tp(
            fill_price=50_000,
            requested_price=50_000,
            original_sl=49_500,
            original_tp=51_500,
            position_side="LONG",
        )
        assert abs(new_sl - 49_500) < 0.01
        assert abs(new_tp - 51_500) < 0.01


# ── Unit tests: StateManager with fill price ─────────────────────────────────


class TestStateManagerFillPrice:
    def test_open_position_with_fill_uses_fill_price(self):
        sm = StateManager()
        pos = Position(
            symbol="BTC-USDT",
            position_side="LONG",
            entry_price=50_000,
            quantity=0.1,
            stop_loss=49_500,
            take_profit=51_500,
            strategy_name="alpha",
        )
        result = sm.open_position_with_fill(pos, fill_price=50_100, requested_price=50_000)

        # entry_price should be fill_price
        assert result.entry_price == 50_100
        assert result.fill_price == 50_100
        assert result.requested_price == 50_000
        # SL/TP recalculated
        assert abs(result.stop_loss - 49_600) < 0.01
        assert abs(result.take_profit - 51_600) < 0.01
        # Slippage recorded
        assert result.slippage_bps > 0
        # Position is in state
        assert sm.get_position("BTC-USDT", "LONG") is result

    def test_pnl_uses_fill_price_as_entry(self):
        sm = StateManager()
        pos = Position(
            symbol="ETH-USDT",
            position_side="LONG",
            entry_price=3_000,
            quantity=1.0,
            stop_loss=2_900,
            take_profit=3_300,
            strategy_name="beta",
        )
        result = sm.open_position_with_fill(pos, fill_price=3_010, requested_price=3_000)
        # PnL should be based on fill_price (3010), not requested (3000)
        pnl = result.unrealized_pnl(3_100)
        assert abs(pnl - 90.0) < 0.01  # (3100 - 3010) * 1.0 = 90


# ── Unit tests: OrderManager fill query ──────────────────────────────────────


class TestOrderManagerFillQuery:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.place_order = AsyncMock(return_value={"orderId": "12345", "status": "NEW"})
        client.get_order_detail = AsyncMock(return_value={
            "orderId": "12345",
            "status": "FILLED",
            "avgPrice": "50100.5",
            "executedQty": "0.1",
        })
        return client

    @pytest.mark.asyncio
    async def test_query_fill_returns_fill_result(self, mock_client):
        mgr = OrderManager(mock_client)
        fill = await mgr.query_fill("BTC-USDT", "12345")

        assert fill is not None
        assert fill.avg_price == 50100.5
        assert fill.executed_qty == 0.1
        assert fill.status == "FILLED"

    @pytest.mark.asyncio
    async def test_submit_market_order_with_fill(self, mock_client):
        mgr = OrderManager(mock_client)
        order = Order(
            symbol="BTC-USDT",
            side="BUY",
            position_side="LONG",
            order_type="MARKET",
            quantity=0.1,
        )
        result, fill = await mgr.submit_market_order_with_fill(order)

        assert result is not None
        assert fill is not None
        assert order.avg_price == 50100.5
        assert order.filled_qty == 0.1

    @pytest.mark.asyncio
    async def test_query_fill_cancelled_returns_none(self, mock_client):
        mock_client.get_order_detail = AsyncMock(return_value={
            "orderId": "12345",
            "status": "CANCELLED",
            "avgPrice": "0",
            "executedQty": "0",
        })
        mgr = OrderManager(mock_client)
        fill = await mgr.query_fill("BTC-USDT", "12345")
        assert fill is None


# ── Unit tests: PerformanceTracker slippage ──────────────────────────────────


class TestPerformanceSlippage:
    def test_slippage_stats_empty(self):
        from analytics.performance import PerformanceTracker
        tracker = PerformanceTracker(initial_equity=10_000)
        stats = tracker.slippage_stats()
        assert stats["count"] == 0
        assert stats["mean_bps"] == 0.0

    def test_slippage_stats_recorded(self):
        from analytics.performance import PerformanceTracker
        tracker = PerformanceTracker(initial_equity=10_000)
        tracker.record_slippage("BTC-USDT", 5.0)
        tracker.record_slippage("BTC-USDT", 10.0)
        tracker.record_slippage("BTC-USDT", 15.0)

        stats = tracker.slippage_stats("BTC-USDT")
        assert stats["count"] == 3
        assert abs(stats["mean_bps"] - 10.0) < 0.01
        assert abs(stats["median_bps"] - 10.0) < 0.01
        assert stats["p95_bps"] >= 14.0

    def test_slippage_in_summary(self):
        from analytics.performance import PerformanceTracker, TradeRecord
        tracker = PerformanceTracker(initial_equity=10_000)
        trade = TradeRecord(
            symbol="BTC-USDT", direction="LONG", strategy="alpha",
            entry_price=50_100, exit_price=50_200, quantity=0.1,
            pnl_usdt=10.0, opened_at=datetime.now(timezone.utc),
            slippage_bps=20.0, requested_price=50_000, fill_price=50_100,
        )
        tracker.record_trade(trade)
        summary = tracker.summary()
        assert "avg_slippage_bps" in summary
        assert "p95_slippage_bps" in summary
