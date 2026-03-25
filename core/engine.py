"""
Trading Engine — main orchestrator.
Runs the event loop, processes signals, manages orders, enforces risk rules.
"""
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import settings
from config.pairs import get_pair_config
from core.event_bus import Event, EventBus, EventType
from core.state_manager import Position, StateManager
from data.feature_engine import enrich
from data.funding_rate import FundingRateTracker
from data.market_data import MarketDataManager
from data.orderbook import OrderBookProcessor
from exchange.bingx_client import BingXClient
from exchange.bingx_ws import BingXWebSocket, MarketDataStream
from exchange.order_manager import Order, OrderManager
from monitoring.alerting import TelegramAlerter
from risk.correlation_guard import CorrelationGuard
from risk.drawdown_monitor import DrawdownMonitor
from risk.emergency_stop import EmergencyStop
from risk.position_sizer import PositionSizer
from risk.risk_engine import RiskEngine
from strategy.base_strategy import Signal, SignalDirection
from strategy.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)

CANDLE_EVAL_INTERVAL_S = 60    # Evaluate signals every minute
RECONCILE_INTERVAL_S = 300     # Reconcile positions every 5 minutes
EQUITY_POLL_INTERVAL_S = 60    # Poll equity every minute


class TradingEngine:
    """
    Central coordinator that ties together all subsystems.
    Runs as a long-lived async process.
    """

    def __init__(self) -> None:
        # Exchange
        self._client = BingXClient()
        self._order_mgr = OrderManager(self._client)
        self._ws = BingXWebSocket()
        self._market_stream = MarketDataStream(self._ws)

        # Data
        self._market_data = MarketDataManager(self._client)
        self._orderbook = OrderBookProcessor()
        self._funding = FundingRateTracker(self._client)

        # Risk
        self._dd_monitor = DrawdownMonitor(settings.INITIAL_CAPITAL)
        self._sizer = PositionSizer()
        self._corr_guard = CorrelationGuard()
        self._risk_engine = RiskEngine(
            self._dd_monitor, self._sizer, correlation_guard=self._corr_guard
        )
        self._emergency = EmergencyStop()

        # Alerting
        self._alerter = TelegramAlerter()

        # Strategy
        self._portfolio = PortfolioManager(self._dd_monitor)

        # State
        self._state = StateManager()
        self._event_bus = EventBus()
        self._equity = settings.INITIAL_CAPITAL
        self._running = False

        # Performance tracking (simple rolling stats)
        self._win_count = 0
        self._loss_count = 0
        self._total_win_usdt = 0.0
        self._total_loss_usdt = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        logger.info("=== WAGMI Trading Engine Starting ===")
        logger.info("Mode: %s", "DEMO" if settings.DEMO_MODE else "LIVE")
        logger.info("Pairs: %s", settings.TRADING_PAIRS)

        self._running = True

        # Initialize market data
        await self._market_data.initialize(
            settings.TRADING_PAIRS, intervals=["1H", "4H"]
        )

        # Setup WebSocket subscriptions
        for symbol in settings.TRADING_PAIRS:
            self._market_stream.subscribe_kline(
                symbol, "1H", self._on_kline_update
            )
            self._market_stream.subscribe_depth(
                symbol, self._on_depth_update
            )

        # Subscribe event handlers
        self._event_bus.subscribe(EventType.CIRCUIT_BREAKER, self._on_circuit_breaker)
        self._event_bus.subscribe(EventType.EMERGENCY_STOP, self._on_emergency_stop)

        # Wire Telegram alerter to event bus
        self._alerter.subscribe_to_event_bus(
            self._event_bus,
            state_manager=self._state,
            dd_monitor=self._dd_monitor,
        )

        # Run all subsystems concurrently
        await asyncio.gather(
            self._ws.start(),
            self._funding.start(settings.TRADING_PAIRS),
            self._event_bus.process_events(),
            self._strategy_loop(),
            self._equity_loop(),
            self._reconcile_loop(),
            self._heartbeat_loop(),
            self._alerter.digest_loop(),
            self._alerter.bot_polling_loop(),
        )

    async def stop(self) -> None:
        self._running = False
        await self._ws.stop()
        await self._funding.stop()
        await self._client.close()
        await self._alerter.close()
        logger.info("Trading engine stopped")

    # ── Signal processing ─────────────────────────────────────────────────────

    async def _strategy_loop(self) -> None:
        """Periodic strategy evaluation loop."""
        while self._running:
            try:
                await self._evaluate_signals()
                await self._manage_open_positions()
            except Exception as exc:
                logger.error("Strategy loop error: %s", exc, exc_info=True)
            await asyncio.sleep(CANDLE_EVAL_INTERVAL_S)

    async def _evaluate_signals(self) -> None:
        if self._emergency.is_active:
            return

        df_by_symbol: Dict = {}
        for symbol in settings.TRADING_PAIRS:
            df = self._market_data.get_df(symbol, "1H")
            if df is not None and not df.empty:
                df_by_symbol[symbol] = enrich(df, "1H")

        if not df_by_symbol:
            return

        funding_rates = {s: self._funding.get_rate(s) for s in settings.TRADING_PAIRS}
        signals = self._portfolio.generate_signals(
            df_by_symbol, funding_rates=funding_rates
        )

        for signal in signals:
            await self._process_signal(signal)

    async def _process_signal(self, signal: Signal) -> None:
        """Pre-trade checks → size calculation → order submission."""
        symbol = signal.symbol

        # Spread check
        spread = self._orderbook.spread_pct(symbol)

        # Liquidity check from ticker
        try:
            ticker = await self._client.get_ticker(symbol)
            volume_24h = float(ticker.get("quoteVolume", 0))
        except Exception:
            volume_24h = 0.0

        # Full pre-trade check pipeline (includes correlation guard)
        approved, reason = await self._risk_engine.pre_trade_check(
            signal=signal,
            equity=self._equity,
            volume_24h=volume_24h,
            current_spread_pct=spread,
            existing_positions_risk_usdt=self._state.total_open_risk_usdt(),
            open_positions=self._state.all_positions(),
        )
        if not approved:
            logger.debug("Signal rejected (%s): %s", reason, symbol)
            return

        # Existing position in same direction?
        existing = self._state.get_position(symbol, signal.direction.value)
        if existing:
            logger.debug("Position already open: %s %s", symbol, signal.direction)
            return

        # Calculate position size
        win_rate = self._win_rate()
        risk_usdt = self._risk_engine.calculate_order_size(
            signal=signal, equity=self._equity, win_rate=win_rate
        )
        if risk_usdt <= 0:
            return

        # Convert risk to quantity
        try:
            pair_cfg = get_pair_config(symbol)
        except ValueError:
            logger.warning("No config for %s", symbol)
            return

        quantity = self._sizer.quantity_from_risk(
            risk_usdt, signal.entry_price, signal.stop_loss, pair_cfg.default_leverage
        )
        quantity = max(round(quantity, 3), pair_cfg.lot_size)

        # Set leverage
        try:
            await self._client.set_leverage(symbol, pair_cfg.default_leverage)
        except Exception as exc:
            logger.warning("Failed to set leverage for %s: %s", symbol, exc)

        # Place entry market order
        entry_side = "BUY" if signal.direction == SignalDirection.LONG else "SELL"
        entry_order = Order(
            symbol=symbol,
            side=entry_side,
            position_side=signal.direction.value,
            order_type="MARKET",
            quantity=quantity,
        )
        result = await self._order_mgr.submit_order(entry_order)
        if not result:
            return

        # Place stop-loss order
        sl_side = "SELL" if signal.direction == SignalDirection.LONG else "BUY"
        sl_order = Order(
            symbol=symbol,
            side=sl_side,
            position_side=signal.direction.value,
            order_type="STOP_MARKET",
            quantity=quantity,
            stop_price=signal.stop_loss,
            reduce_only=True,
        )
        sl_result = await self._order_mgr.submit_order(sl_order)

        # Track position in state
        pos = Position(
            symbol=symbol,
            position_side=signal.direction.value,
            entry_price=signal.entry_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy_name=signal.strategy_name,
            sl_order_id=sl_result.get("orderId") if sl_result else None,
        )
        self._state.open_position(pos)

        await self._event_bus.publish(Event(
            type=EventType.POSITION_OPENED,
            data={
                "symbol": symbol,
                "direction": signal.direction.value,
                "entry": signal.entry_price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "qty": quantity,
                "strategy": signal.strategy_name,
                "risk_usdt": risk_usdt,
            },
        ))

    # ── Position management ───────────────────────────────────────────────────

    async def _manage_open_positions(self) -> None:
        """Check trailing stops, time stops, partial profits."""
        for pos in self._state.all_positions():
            try:
                df = self._market_data.get_df(pos.symbol, "1H")
                if df is None or df.empty:
                    continue
                current_price = float(df["close"].iloc[-1])

                # Time stop
                if self._risk_engine.check_time_stop(pos.open_hours):
                    logger.info(
                        "Time stop: closing %s %s after %.1fh",
                        pos.symbol, pos.position_side, pos.open_hours,
                    )
                    await self._close_position(pos, current_price, "time_stop")
                    continue

                # Trailing stop
                from strategy.indicators import atr as calc_atr
                atr_vals = calc_atr(df["high"], df["low"], df["close"], 14)
                atr_val = float(atr_vals.iloc[-1])

                new_sl = self._risk_engine.check_trailing_stop(
                    pos.position_side,
                    pos.entry_price,
                    current_price,
                    atr_val,
                    partial_profit_taken=pos.partial_profit_taken,
                )
                if new_sl:
                    self._state.update_stop_loss(pos.symbol, pos.position_side, new_sl)

            except Exception as exc:
                logger.error("Position management error for %s: %s", pos.symbol, exc)

    async def _close_position(
        self, pos: Position, current_price: float, reason: str
    ) -> None:
        result = await self._order_mgr.close_position(
            pos.symbol, pos.position_side, pos.quantity
        )
        if result:
            pnl = pos.unrealized_pnl(current_price)
            if pnl > 0:
                self._win_count += 1
                self._total_win_usdt += pnl
            else:
                self._loss_count += 1
                self._total_loss_usdt += abs(pnl)

            self._state.close_position(pos.symbol, pos.position_side)
            await self._event_bus.publish(Event(
                type=EventType.POSITION_CLOSED,
                data={
                    "symbol": pos.symbol,
                    "direction": pos.position_side,
                    "pnl": pnl,
                    "reason": reason,
                },
            ))

    # ── Background loops ──────────────────────────────────────────────────────

    async def _equity_loop(self) -> None:
        _prev_cb_level = None
        while self._running:
            try:
                data = await self._client.get_balance()
                balance = data.get("data", {}).get("balance", {})
                equity = float(balance.get("equity", self._equity))
                self._equity = equity
                level = self._dd_monitor.update(equity)

                # Publish circuit breaker event on level change
                if level != _prev_cb_level:
                    _prev_cb_level = level
                    await self._event_bus.publish(Event(
                        EventType.CIRCUIT_BREAKER,
                        {
                            "level": level,
                            "dd_pct": abs(self._dd_monitor.current_drawdown() * 100),
                        },
                    ))

                if self._dd_monitor.requires_emergency_close():
                    self._emergency.activate("Max drawdown exceeded")
                    await self._event_bus.publish(
                        Event(EventType.EMERGENCY_STOP, {"reason": "max_dd"})
                    )
            except Exception as exc:
                logger.error("Equity poll failed: %s", exc)
            await asyncio.sleep(EQUITY_POLL_INTERVAL_S)

    async def _reconcile_loop(self) -> None:
        """Reconcile local state with exchange positions every 5 minutes."""
        while self._running:
            await asyncio.sleep(RECONCILE_INTERVAL_S)
            try:
                exchange_positions = await self._client.get_positions()
                exchange_syms = {
                    f"{p['symbol']}_{p['positionSide']}"
                    for p in exchange_positions
                    if abs(float(p.get("positionAmt", 0))) > 0
                }
                local_syms = {
                    f"{pos.symbol}_{pos.position_side}"
                    for pos in self._state.all_positions()
                }
                ghost = local_syms - exchange_syms
                for key in ghost:
                    sym, side = key.rsplit("_", 1)
                    self._state.close_position(sym, side)
                    logger.warning("Ghost position reconciled: %s", key)
            except Exception as exc:
                logger.error("Reconcile failed: %s", exc)

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(settings.HEARTBEAT_INTERVAL_S)
            await self._event_bus.publish(
                Event(EventType.HEARTBEAT, {"ts": time.time()})
            )

    # ── WebSocket callbacks ───────────────────────────────────────────────────

    def _on_kline_update(self, data: dict) -> None:
        """Handle incoming kline WebSocket message."""
        try:
            k = data.get("data", {}).get("k", {})
            if not k.get("x", False):   # Not a closed candle
                return
            symbol = data.get("data", {}).get("s", "")
            interval = k.get("i", "1H").upper()
            self._market_data.update_candle(symbol, interval, k)
            self._event_bus.publish_nowait(
                Event(EventType.CANDLE_CLOSE, {"symbol": symbol, "interval": interval})
            )
        except Exception as exc:
            logger.error("Kline callback error: %s", exc)

    def _on_depth_update(self, data: dict) -> None:
        try:
            depth_data = data.get("data", {})
            symbol = data.get("dataType", "").split("@")[0]
            self._orderbook.update(symbol, depth_data)
            self._event_bus.publish_nowait(
                Event(EventType.DEPTH_UPDATE, {"symbol": symbol})
            )
        except Exception as exc:
            logger.error("Depth callback error: %s", exc)

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _on_circuit_breaker(self, event: Event) -> None:
        level = event.data.get("level")
        logger.warning("Circuit breaker event: %s", level)

    async def _on_emergency_stop(self, event: Event) -> None:
        logger.critical("EMERGENCY STOP: %s", event.data.get("reason"))
        exchange_positions = await self._client.get_positions()
        await self._emergency.close_all_positions(self._order_mgr, exchange_positions)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _win_rate(self) -> float:
        total = self._win_count + self._loss_count
        if total == 0:
            return 0.55   # Prior
        return self._win_count / total
