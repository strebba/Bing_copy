# WAGMI Copy Trading Bot

Automated copy trading bot for BingX Perpetual Futures with multi-strategy portfolio, risk management, and backtesting.

## Features

- **Multi-Strategy Portfolio**: Alpha (Momentum Reversal), Beta (Mean Reversion), Gamma (Breakout Volatility)
- **Regime Detection**: Automatic market regime switching using Hurst Exponent and ATR percentile
- **Risk Management**: 4-level defense system with circuit breaker (5/10/15/20% drawdown)
- **Position Sizing**: Fractional Kelly Criterion + Fixed Fractional
- **Backtesting**: Walk-forward analysis + Monte Carlo simulation (10,000 runs)
- **Monitoring**: Prometheus metrics + Grafana dashboard + Telegram alerts
- **Docker Deployment**: Full stack with Redis, PostgreSQL, Prometheus, Grafana

## Architecture

```
wagmi-ct-bot/
├── config/          # Configuration (settings, pairs, strategies)
├── core/            # Trading engine, event bus, state manager
├── exchange/        # BingX API client (REST + WebSocket)
├── strategy/        # Trading strategies (Alpha, Beta, Gamma)
├── risk/            # Risk engine, drawdown monitor, position sizer
├── data/            # Market data, orderbook, feature engineering
├── analytics/       # Performance tracking, trade journal
├── backtest/        # Backtester, walk-forward, Monte Carlo
├── monitoring/      # Health checks, alerting, metrics
├── tests/           # Unit + integration tests
├── scripts/         # Setup, data download, optimization
└── docker/          # Dockerfile, docker-compose
```

## Requirements

- Python 3.9+
- Docker + Docker Compose
- BingX API Key (with Read + Trade permissions)

## Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd wagmi-ct-bot
cp .env.example .env
```

### 2. Configure Environment

Edit `.env`:

```env
# BingX API (get from https://www.bingx.com/)
BINGX_API_KEY=your_api_key
BINGX_API_SECRET=your_api_secret

# Trading Mode
DEMO_MODE=true  # Set false for live trading

# Initial Capital
INITIAL_CAPITAL=2000
MAX_LEVERAGE=5
RISK_PER_TRADE=0.01

# Trading Pairs
TRADING_PAIRS=BTC-USDT,ETH-USDT,SOL-USDT

# Telegram Alerts (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

### 3. Run with Docker

```bash
cd docker
docker-compose up -d
```

Services:
- Bot: http://localhost:8000/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/wagmi_grafana)

### 4. Run Locally (Development)

```bash
pip install -r requirements.txt
python main.py
```

## Testing

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=. --cov-report=html
```

Current: **313 tests passed**

## Strategy Details

### Alpha — Momentum Reversal (40%)
- Entry: RSI divergence + MACD slope + volume confirmation
- Stop: 1.5 × ATR
- Target: 2.5 × ATR (R:R 1:1.67)
- Filtro: EMA 50/200 trend context, ADX > 20

### Beta — Mean Reversion (30%)
- Entry: Bollinger Bands 2.5σ + Z-Score > 2.0 + RSI extrema
- Stop: Beyond 3.0σ
- Target: BB midline (mean reversion)
- Filtro: Hurst < 0.45 (mean-reverting regime)

### Gamma — Breakout Volatility (30%)
- Entry: Donchian breakout + volume spike + BBW compression
- Stop: Center of compression range
- Target: Trailing stop 2× ATR after 1:1
- Filtro: ATR below 20th percentile for 10+ periods

## Risk Management

| Level | Drawdown | Action |
|-------|----------|--------|
| 1 | -5% | Reduce size 25% |
| 2 | -10% | Reduce size 50%, 12h cooldown |
| 3 | -15% | Halt new trades, 48h cooldown |
| 4 | -20% | Emergency close all, 1 week cooldown |

Additional limits:
- Max 5 open positions
- Max 3% equity per position
- Max 5% total open risk
- Daily loss limit: 3%
- Weekly loss limit: 5%

## Backtesting

```bash
# Download historical data
python scripts/download_data.py

# Run backtest
python scripts/run_backtest.py

# Walk-forward analysis
python -c "from backtest.walk_forward import WalkForwardAnalyzer; ..."

# Monte Carlo stress test
python -c "from backtest.monte_carlo import MonteCarloSimulator; ..."
```

## Deployment Checklist

- [ ] BingX account verified (KYC)
- [ ] API keys generated (Read + Trade, NO Withdraw)
- [ ] IP whitelisting configured
- [ ] VPS provisioned (Hetzner/Contabo recommended)
- [ ] Docker images built and tested
- [ ] Monitoring configured
- [ ] Runbook for incidents written

## Performance Targets

| Metric | Minimum | Target |
|--------|---------|--------|
| Win Rate | 55% | 60% |
| Profit Factor | 1.3 | 1.6 |
| Sharpe Ratio | 1.5 | 2.0 |
| Max Drawdown | 20% | 15% |
| Recovery Factor | 2.0 | 3.0 |

## Copy Trading (BingX 2.0)

Requirements for trader profile:
- Account balance ≥ 110 USDT
- Consistent track record
- Drawdown < 20%

Revenue model:
- Profit sharing: 8-10%
- Referral commission: 0.05% × volume

## License

MIT
