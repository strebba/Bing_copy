"""
Prometheus metrics exporter for the trading bot.
"""
import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — metrics disabled")


class Metrics:
    """Wraps Prometheus gauges and counters. No-ops if prometheus not installed."""

    def __init__(self) -> None:
        if not PROMETHEUS_AVAILABLE:
            self._enabled = False
            return
        self._enabled = True

        self.equity = Gauge("wagmi_equity_usdt", "Current account equity in USDT")
        self.drawdown_pct = Gauge("wagmi_drawdown_pct", "Current drawdown percentage")
        self.open_positions = Gauge("wagmi_open_positions", "Number of open positions")
        self.daily_pnl = Gauge("wagmi_daily_pnl_usdt", "Daily realized PnL in USDT")
        self.win_rate = Gauge("wagmi_win_rate", "Rolling win rate")
        self.profit_factor = Gauge("wagmi_profit_factor", "Rolling profit factor")
        self.sharpe = Gauge("wagmi_sharpe_ratio", "Annualized Sharpe ratio")

        self.orders_placed = Counter("wagmi_orders_placed_total", "Total orders placed", ["symbol", "side"])
        self.orders_filled = Counter("wagmi_orders_filled_total", "Total orders filled", ["symbol"])
        self.circuit_breaker_triggers = Counter("wagmi_circuit_breaker_total", "Circuit breaker triggers", ["level"])

        self.api_latency = Histogram(
            "wagmi_api_latency_seconds",
            "BingX API request latency",
            ["endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        )

    def start_server(self, port: int = 8000) -> None:
        if self._enabled:
            start_http_server(port)
            logger.info("Prometheus metrics server started on port %d", port)

    def update_equity(self, value: float) -> None:
        if self._enabled:
            self.equity.set(value)

    def update_drawdown(self, value: float) -> None:
        if self._enabled:
            self.drawdown_pct.set(value)

    def update_open_positions(self, count: int) -> None:
        if self._enabled:
            self.open_positions.set(count)

    def update_performance(self, win_rate: float, pf: float, sharpe: float) -> None:
        if self._enabled:
            self.win_rate.set(win_rate)
            self.profit_factor.set(pf)
            self.sharpe.set(sharpe)

    def inc_orders(self, symbol: str, side: str) -> None:
        if self._enabled:
            self.orders_placed.labels(symbol=symbol, side=side).inc()


# Global singleton
METRICS = Metrics()
