"""
Central configuration for WAGMI Copy Trading Bot.
All environment variables are loaded here.
"""
import os
from dotenv import load_dotenv

load_dotenv()


# ── BingX API ────────────────────────────────────────────────────────────────
BINGX_API_KEY = os.getenv("BINGX_API_KEY", "")
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET", "")
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

BINGX_BASE_URL = (
    "https://open-api-vst.bingx.com" if DEMO_MODE else "https://open-api.bingx.com"
)
BINGX_WS_URL = "wss://open-api-ws.bingx.com/market"

BINGX_ENDPOINTS = {
    "place_order": "/openApi/swap/v2/trade/order",
    "cancel_order": "/openApi/swap/v2/trade/cancelOrder",
    "positions": "/openApi/swap/v2/user/positions",
    "balance": "/openApi/swap/v2/user/balance",
    "klines": "/openApi/swap/v2/quote/klines",
    "ticker": "/openApi/swap/v2/quote/ticker",
    "depth": "/openApi/swap/v2/quote/depth",
    "funding_rate": "/openApi/swap/v2/quote/premiumIndex",
    "open_orders": "/openApi/swap/v2/trade/openOrders",
    "order_detail": "/openApi/swap/v2/trade/order",
    "leverage": "/openApi/swap/v2/trade/leverage",
    "margin_type": "/openApi/swap/v2/trade/marginType",
}

BINGX_RATE_LIMITS = {
    "orders_per_second": 10,
    "requests_per_10s": 2000,
    "cancellation_rate_limit": 0.99,
}

# ── Trading Parameters ────────────────────────────────────────────────────────
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "2000"))
MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "5"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))   # 1 %
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "5"))

TRADING_PAIRS = os.getenv(
    "TRADING_PAIRS", "BTC-USDT,ETH-USDT,SOL-USDT"
).split(",")

STRATEGY_WEIGHTS = {
    "alpha": float(os.getenv("STRATEGY_WEIGHT_ALPHA", "0.4")),
    "beta": float(os.getenv("STRATEGY_WEIGHT_BETA", "0.3")),
    "gamma": float(os.getenv("STRATEGY_WEIGHT_GAMMA", "0.3")),
}

# ── Risk Limits ───────────────────────────────────────────────────────────────
MAX_SINGLE_POSITION_PCT = 0.03   # 3 % equity
DAILY_LOSS_LIMIT_PCT = 0.03      # 3 % → halt 24 h
WEEKLY_LOSS_LIMIT_PCT = 0.05     # 5 % → halt 48 h
DD_SOFT_LIMIT_PCT = 0.10         # 10 % → reduce size 50 %
DD_HARD_LIMIT_PCT = 0.15         # 15 % → halt new trades
DD_EMERGENCY_PCT = 0.20          # 20 % → close all
MAX_OPEN_RISK_PCT = 0.05         # 5 % total open risk
MAX_CORR_NEW_POSITION = 0.6      # Correlation guard
MIN_LIQUIDITY_24H_USD = 10_000_000  # $10 M
MAX_SPREAD_PCT = 0.0005          # 0.05 %

# ── Copy Trading ──────────────────────────────────────────────────────────────
PROFIT_SHARING_RATE = float(os.getenv("PROFIT_SHARING_RATE", "0.10"))  # 10 %
MIN_ACCOUNT_BALANCE = 110        # USDT

# ── Telegram Alerting ─────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Database / Cache ──────────────────────────────────────────────────────────
POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN", "postgresql://wagmi:wagmi@localhost:5432/wagmi_ct"
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "wagmi")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "trading")

# ── Monitoring ────────────────────────────────────────────────────────────────
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8000"))
HEARTBEAT_INTERVAL_S = 30
STALE_DATA_THRESHOLD_S = 5

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/wagmi_ct.log")
