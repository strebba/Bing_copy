"""
Tests for C-2: TP order placement on exchange and bracket lifecycle.

Covers:
  - Long/Short entry → SL and TP both placed on BingX
  - Entry failure → bracket orders NOT placed
  - SL triggered → TP cancelled
  - TP triggered → SL cancelled
  - No double-cancel after first trigger
  - Trailing stop: update_tp_price cancels old TP and places new one
  - Signal without valid tp_price → rejected by RiskEngine
  - Signal with low R:R → rejected by RiskEngine
  - Backtester _check_exit applies slippage on TP fills
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from exchange.order_manager import BracketIds, Order, OrderManager
from risk.drawdown_monitor import DrawdownMonitor
from risk.position_sizer import PositionSizer
from risk.risk_engine import RiskEngine
from strategy.base_strategy import Signal, SignalDirection


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def make_mock_client(
    entry_order_id: str = "ENTRY-001",
    sl_order_id: str = "SL-001",
    tp_order_id: str = "TP-001",
) -> MagicMock:
    """
    BingXClient mock whose place_order returns canned IDs keyed by order type.
    A fresh TAKE_PROFIT_MARKET order after a cancel gets a "TP-NEW" id so tests
    can verify the bracket is updated.
    """
    client = MagicMock()
    call_count = {"tp": 0}

    async def mock_place_order(**kwargs):
        otype = kwargs.get("order_type", "")
        if otype == "STOP_MARKET":
            return {"orderId": sl_order_id, "status": "NEW"}
        if otype == "TAKE_PROFIT_MARKET":
            call_count["tp"] += 1
            if call_count["tp"] == 1:
                return {"orderId": tp_order_id, "status": "NEW"}
            # Replacement TP gets a new id
            return {"orderId": f"TP-NEW-{call_count['tp']}", "status": "NEW"}
        # MARKET entry
        return {"orderId": entry_order_id, "status": "FILLED"}

    client.place_order = mock_place_order
    client.cancel_order = AsyncMock(return_value={"orderId": "cancelled"})
    client.get_open_orders = AsyncMock(return_value=[])
    return client


def make_long_signal(entry: float = 100.0, sl: float = 97.0, tp: float = 106.0) -> Signal:
    """Valid LONG signal — risk=3, reward=6 → R:R=2.0 ≥ 1.5."""
    return Signal(
        direction=SignalDirection.LONG,
        symbol="BTC-USDT",
        strategy_name="alpha",
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        risk_pct=0.01,
        confidence=0.8,
        trailing_activation_rr=1.0,
    )


def make_short_signal(entry: float = 100.0, sl: float = 103.0, tp: float = 95.5) -> Signal:
    """Valid SHORT signal — risk=3, reward=4.5 → R:R=1.5 ≥ 1.5."""
    return Signal(
        direction=SignalDirection.SHORT,
        symbol="BTC-USDT",
        strategy_name="alpha",
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        risk_pct=0.01,
        confidence=0.8,
        trailing_activation_rr=1.0,
    )


def make_risk_engine() -> RiskEngine:
    dd = DrawdownMonitor(initial_equity=10_000)
    sizer = PositionSizer()
    return RiskEngine(dd_monitor=dd, position_sizer=sizer)


APPROVE_KWARGS = dict(
    equity=10_000,
    volume_24h=50_000_000,   # well above MIN_LIQUIDITY_24H_USD (10M)
    current_spread_pct=0.0001,
    existing_positions_risk_usdt=0,
)


# ── Bracket placement ─────────────────────────────────────────────────────────

class TestBracketOrderPlacement:
    @pytest.mark.asyncio
    async def test_long_entry_places_sl_and_tp(self):
        """Opening a LONG must place entry + SL + TP orders on BingX."""
        client = make_mock_client(sl_order_id="SL-123", tp_order_id="TP-456")
        om = OrderManager(client)

        result = await om.place_entry_with_brackets(
            symbol="BTC-USDT",
            position_side="LONG",
            quantity=0.01,
            sl_price=97.0,
            tp_price=106.0,
        )

        assert result["entry_result"] is not None, "Entry order must succeed"
        assert result["sl_result"] is not None, "SL order must be placed"
        assert result["tp_result"] is not None, "TP order must be placed"

        bracket = om._brackets.get(("BTC-USDT", "LONG"))
        assert bracket is not None, "Bracket must be registered"
        assert bracket.sl_order_id == "SL-123"
        assert bracket.tp_order_id == "TP-456"

    @pytest.mark.asyncio
    async def test_short_entry_places_sl_and_tp(self):
        """Opening a SHORT must place entry + SL + TP orders on BingX."""
        client = make_mock_client(sl_order_id="SL-777", tp_order_id="TP-888")
        om = OrderManager(client)

        result = await om.place_entry_with_brackets(
            symbol="ETH-USDT",
            position_side="SHORT",
            quantity=0.1,
            sl_price=103.0,
            tp_price=95.5,
        )

        assert result["sl_result"] is not None
        assert result["tp_result"] is not None
        bracket = om._brackets[("ETH-USDT", "SHORT")]
        assert bracket.sl_order_id == "SL-777"
        assert bracket.tp_order_id == "TP-888"

    @pytest.mark.asyncio
    async def test_entry_failure_skips_brackets(self):
        """If the entry order fails, SL and TP must NOT be placed."""
        client = MagicMock()
        client.place_order = AsyncMock(side_effect=RuntimeError("Network error"))
        client.cancel_order = AsyncMock()
        client.get_open_orders = AsyncMock(return_value=[])
        om = OrderManager(client)

        result = await om.place_entry_with_brackets(
            symbol="BTC-USDT",
            position_side="LONG",
            quantity=0.01,
            sl_price=97.0,
            tp_price=106.0,
        )

        assert result["entry_result"] is None
        assert result["sl_result"] is None
        assert result["tp_result"] is None
        assert ("BTC-USDT", "LONG") not in om._brackets


# ── Bracket lifecycle ─────────────────────────────────────────────────────────

class TestBracketLifecycle:
    @pytest.mark.asyncio
    async def test_sl_triggered_cancels_tp(self):
        """When SL fires, the pending TP order must be cancelled."""
        client = make_mock_client(sl_order_id="SL-001", tp_order_id="TP-001")
        om = OrderManager(client)

        await om.place_entry_with_brackets(
            symbol="BTC-USDT", position_side="LONG",
            quantity=0.01, sl_price=97.0, tp_price=106.0,
        )

        cancelled = await om.on_sl_triggered("BTC-USDT", "LONG")

        assert cancelled is True
        client.cancel_order.assert_called_once_with("BTC-USDT", "TP-001")
        assert ("BTC-USDT", "LONG") not in om._brackets

    @pytest.mark.asyncio
    async def test_tp_triggered_cancels_sl(self):
        """When TP fires, the pending SL order must be cancelled."""
        client = make_mock_client(sl_order_id="SL-002", tp_order_id="TP-002")
        om = OrderManager(client)

        await om.place_entry_with_brackets(
            symbol="BTC-USDT", position_side="LONG",
            quantity=0.01, sl_price=97.0, tp_price=106.0,
        )

        cancelled = await om.on_tp_triggered("BTC-USDT", "LONG")

        assert cancelled is True
        client.cancel_order.assert_called_once_with("BTC-USDT", "SL-002")
        assert ("BTC-USDT", "LONG") not in om._brackets

    @pytest.mark.asyncio
    async def test_no_double_cancel_after_sl(self):
        """A second on_sl_triggered call on the same symbol returns False (no bracket)."""
        client = make_mock_client()
        om = OrderManager(client)

        await om.place_entry_with_brackets(
            symbol="BTC-USDT", position_side="LONG",
            quantity=0.01, sl_price=97.0, tp_price=106.0,
        )
        await om.on_sl_triggered("BTC-USDT", "LONG")

        result = await om.on_sl_triggered("BTC-USDT", "LONG")
        assert result is False

    @pytest.mark.asyncio
    async def test_no_double_cancel_after_tp(self):
        """A second on_tp_triggered call on the same symbol returns False (no bracket)."""
        client = make_mock_client()
        om = OrderManager(client)

        await om.place_entry_with_brackets(
            symbol="BTC-USDT", position_side="LONG",
            quantity=0.01, sl_price=97.0, tp_price=106.0,
        )
        await om.on_tp_triggered("BTC-USDT", "LONG")

        result = await om.on_tp_triggered("BTC-USDT", "LONG")
        assert result is False


# ── Trailing stop ─────────────────────────────────────────────────────────────

class TestTrailingStop:
    @pytest.mark.asyncio
    async def test_update_tp_replaces_order(self):
        """
        After 1:1 R:R is reached, update_tp_price must cancel the old TP and
        place a new one with the trailing price.
        """
        client = make_mock_client(sl_order_id="SL-T1", tp_order_id="TP-T1")
        om = OrderManager(client)

        await om.place_entry_with_brackets(
            symbol="BTC-USDT", position_side="LONG",
            quantity=0.01, sl_price=97.0, tp_price=106.0,
        )

        success = await om.update_tp_price(
            symbol="BTC-USDT",
            position_side="LONG",
            new_tp_price=108.0,
            quantity=0.01,
        )

        assert success is True
        # Old TP must have been cancelled
        client.cancel_order.assert_called_with("BTC-USDT", "TP-T1")
        # Bracket record should hold the new TP id
        bracket = om._brackets.get(("BTC-USDT", "LONG"))
        assert bracket is not None
        assert bracket.tp_order_id != "TP-T1"
        assert bracket.tp_order_id is not None

    @pytest.mark.asyncio
    async def test_update_tp_no_bracket_returns_false(self):
        """update_tp_price on a symbol with no bracket must return False."""
        client = make_mock_client()
        om = OrderManager(client)

        result = await om.update_tp_price(
            symbol="BTC-USDT", position_side="LONG",
            new_tp_price=108.0, quantity=0.01,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_update_tp_quantity_delegates(self):
        """update_tp_quantity (partial take) should also update the bracket TP id."""
        client = make_mock_client(sl_order_id="SL-Q1", tp_order_id="TP-Q1")
        om = OrderManager(client)

        await om.place_entry_with_brackets(
            symbol="BTC-USDT", position_side="LONG",
            quantity=0.02, sl_price=97.0, tp_price=106.0,
        )

        success = await om.update_tp_quantity(
            symbol="BTC-USDT", position_side="LONG",
            new_quantity=0.01,   # 50 % partial take
            tp_price=106.0,
        )

        assert success is True
        bracket = om._brackets[("BTC-USDT", "LONG")]
        assert bracket.tp_order_id != "TP-Q1"


# ── Risk engine TP validation ─────────────────────────────────────────────────

class TestRiskEngineTPValidation:
    @pytest.mark.asyncio
    async def test_valid_long_signal_approved(self):
        """Valid LONG with TP above entry and R:R ≥ 1.5 must pass."""
        re = make_risk_engine()
        signal = make_long_signal(entry=100.0, sl=97.0, tp=106.0)  # R:R = 2.0

        approved, reason = await re.approve_signal(signal=signal, **APPROVE_KWARGS)

        assert approved, f"Expected approved but got: {reason}"

    @pytest.mark.asyncio
    async def test_valid_short_signal_approved(self):
        """Valid SHORT with TP below entry and R:R ≥ 1.5 must pass."""
        re = make_risk_engine()
        signal = make_short_signal(entry=100.0, sl=103.0, tp=95.5)  # R:R = 1.5

        approved, reason = await re.approve_signal(signal=signal, **APPROVE_KWARGS)

        assert approved, f"Expected approved but got: {reason}"

    @pytest.mark.asyncio
    async def test_long_tp_below_entry_rejected(self):
        """LONG signal where TP is below entry must be rejected."""
        re = make_risk_engine()
        signal = Signal(
            direction=SignalDirection.LONG,
            symbol="BTC-USDT",
            strategy_name="alpha",
            entry_price=100.0,
            stop_loss=97.0,
            take_profit=98.0,   # Wrong side
            risk_pct=0.01,
            confidence=0.8,
        )

        approved, reason = await re.approve_signal(signal=signal, **APPROVE_KWARGS)

        assert not approved
        assert "TP" in reason

    @pytest.mark.asyncio
    async def test_short_tp_above_entry_rejected(self):
        """SHORT signal where TP is above entry must be rejected."""
        re = make_risk_engine()
        signal = Signal(
            direction=SignalDirection.SHORT,
            symbol="BTC-USDT",
            strategy_name="alpha",
            entry_price=100.0,
            stop_loss=103.0,
            take_profit=102.0,  # Wrong side
            risk_pct=0.01,
            confidence=0.8,
        )

        approved, reason = await re.approve_signal(signal=signal, **APPROVE_KWARGS)

        assert not approved
        assert "TP" in reason

    @pytest.mark.asyncio
    async def test_signal_missing_tp_rejected(self):
        """Signal with take_profit=0 (effectively missing) must be rejected."""
        re = make_risk_engine()
        signal = Signal(
            direction=SignalDirection.LONG,
            symbol="BTC-USDT",
            strategy_name="alpha",
            entry_price=100.0,
            stop_loss=97.0,
            take_profit=0.0,    # Missing
            risk_pct=0.01,
            confidence=0.8,
        )

        approved, reason = await re.approve_signal(signal=signal, **APPROVE_KWARGS)

        assert not approved

    @pytest.mark.asyncio
    async def test_signal_low_rr_rejected(self):
        """Signal with R:R < 1.5 must be rejected."""
        re = make_risk_engine()
        signal = Signal(
            direction=SignalDirection.LONG,
            symbol="BTC-USDT",
            strategy_name="alpha",
            entry_price=100.0,
            stop_loss=98.0,    # risk = 2
            take_profit=102.5, # reward = 2.5 → R:R = 1.25
            risk_pct=0.01,
            confidence=0.8,
        )

        approved, reason = await re.approve_signal(signal=signal, **APPROVE_KWARGS)

        assert not approved
        assert "R:R" in reason or "ratio" in reason.lower()


# ── Backtester TP slippage ────────────────────────────────────────────────────

class TestBacktesterTPSlippage:
    """Verify _check_exit applies slippage consistently for both SL and TP fills."""

    def _make_backtester(self, slippage: float = 0.002):
        from backtest.backtester import Backtester, BacktestConfig
        cfg = BacktestConfig(slippage=slippage)
        bt = Backtester.__new__(Backtester)
        bt._cfg = cfg
        return bt

    def _make_trade(self, direction: str = "LONG"):
        from backtest.backtester import OpenTrade
        if direction == "LONG":
            return OpenTrade(
                symbol="BTC-USDT",
                direction="LONG",
                entry_price=100.0,
                stop_loss=97.0,   # below entry
                take_profit=106.0, # above entry
                quantity=0.01,
                strategy="alpha",
                opened_bar=0,
            )
        # SHORT: SL above entry, TP below entry
        return OpenTrade(
            symbol="BTC-USDT",
            direction="SHORT",
            entry_price=100.0,
            stop_loss=103.0,  # above entry
            take_profit=94.0,  # below entry
            quantity=0.01,
            strategy="alpha",
            opened_bar=0,
        )

    def test_long_tp_fill_below_tp_price(self):
        """LONG TP fill must be slightly below the trigger price (market slippage)."""
        bt = self._make_backtester(slippage=0.002)
        trade = self._make_trade("LONG")

        closed, exit_price, reason = bt._check_exit(
            trade, bar_high=107.0, bar_low=105.0, bar_close=106.5, bar_idx=5
        )

        assert closed is True
        assert reason == "take_profit"
        assert exit_price == pytest.approx(106.0 * (1 - 0.002))
        assert exit_price < 106.0

    def test_short_tp_fill_above_tp_price(self):
        """SHORT TP fill must be slightly above the trigger price (market slippage)."""
        bt = self._make_backtester(slippage=0.002)
        trade = self._make_trade("SHORT")

        closed, exit_price, reason = bt._check_exit(
            trade, bar_high=95.0, bar_low=93.0, bar_close=94.2, bar_idx=5
        )

        assert closed is True
        assert reason == "take_profit"
        assert exit_price == pytest.approx(94.0 * (1 + 0.002))
        assert exit_price > 94.0

    def test_long_sl_fill_below_sl_price(self):
        """LONG SL fill must be slightly below the stop price (market slippage)."""
        bt = self._make_backtester(slippage=0.002)
        trade = self._make_trade("LONG")

        closed, exit_price, reason = bt._check_exit(
            trade, bar_high=99.0, bar_low=96.5, bar_close=97.5, bar_idx=5
        )

        assert closed is True
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(97.0 * (1 - 0.002))
        assert exit_price < 97.0

    def test_short_sl_fill_above_sl_price(self):
        """SHORT SL fill must be slightly above the stop price (market slippage)."""
        bt = self._make_backtester(slippage=0.002)
        trade = self._make_trade("SHORT")

        closed, exit_price, reason = bt._check_exit(
            trade, bar_high=103.5, bar_low=102.0, bar_close=103.0, bar_idx=5
        )

        assert closed is True
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(103.0 * (1 + 0.002))
        assert exit_price > 103.0

    def test_no_exit_when_price_within_range(self):
        """Bar that doesn't touch SL or TP must leave the trade open."""
        bt = self._make_backtester()
        trade = self._make_trade("LONG")

        closed, _, reason = bt._check_exit(
            trade, bar_high=104.0, bar_low=99.0, bar_close=102.0, bar_idx=5
        )

        assert closed is False
        assert reason == "open"
