# Crypto Trading Bot - Bybit Futures Supply & Demand Strategy

A professional-grade automated cryptocurrency trading bot that operates on Bybit futures markets using a Supply & Demand zone strategy. The bot features Telegram integration for control and notifications, comprehensive backtesting, and 24/7 server deployment capabilities.

## Features

### Core Trading Features
- **Supply & Demand Strategy**: Automated detection and trading of supply/demand zones
- **Bybit Futures Integration**: Full support for USDT perpetual futures
- **Multi-Symbol Trading**: Monitor and trade multiple cryptocurrency pairs simultaneously
- **Risk Management**: Configurable position sizing, stop-loss, take-profit, and trailing stops
- **Margin & Leverage Control**: Per-symbol margin mode (cross/isolated) and leverage settings

### Telegram Bot Interface
- Start/stop trading with simple commands
- Real-time position monitoring
- Trade notifications and alerts
- Backtesting directly from chat
- Risk and strategy configuration
- Daily performance summaries

### Advanced Features
- **Backtesting Engine**: Test strategies on historical data with detailed metrics
- **Zone Scoring**: Probability-based zone evaluation (0-100 score)
- **Multi-Timeframe Analysis**: Zone detection across different timeframes
- **Partial Profit Taking**: Automatic TP1/TP2 management
- **Break-even Management**: Move stop to entry after TP1
- **Emergency Stop**: Instantly close all positions

### Technical Infrastructure
- FastAPI webhook for Telegram integration
- PostgreSQL database for trade history
- Redis for caching and task queuing
- Docker containerization
- Prometheus metrics and Grafana dashboards
- Structured logging with trade journals

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL
- Redis
- Bybit account with API keys
- Telegram Bot Token

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd crypto_trading_bot
```

2. Create environment file:
```bash
cp .env.example .env
```

3. Configure your `.env` file:
```env
# Bybit API (Get from https://www.bybit.com/app/user/api-management)
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET=true  # Start with testnet!

# Telegram Bot (Get from @BotFather)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_ALLOWED_CHAT_IDS=your_chat_id

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/crypto_bot
REDIS_URL=redis://localhost:6379/0

# Trading Settings
DEFAULT_RISK_PERCENT=1.0
MAX_CONCURRENT_POSITIONS=5
DEFAULT_LEVERAGE=3
```

### Docker Deployment (Recommended)

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f bot

# Stop services
docker-compose down
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from src.db.database import init_db; import asyncio; asyncio.run(init_db())"

# Run the bot
python -m src.main
```

## Usage

### Telegram Commands

#### Basic Controls
- `/start` - Initialize bot and show welcome message
- `/enable` - Start automated trading
- `/disable` - Stop automated trading
- `/status` - Show current bot status
- `/help` - List all commands

#### Configuration
- `/symbols` - Manage traded symbols
- `/margin BTCUSDT isolated` - Set margin mode
- `/leverage BTCUSDT 5` - Set leverage
- `/risk` - Configure risk parameters
- `/strategy` - View strategy settings

#### Analysis
- `/positions` - View open positions
- `/backtest BTCUSDT 15m 30` - Run backtest
- `/logs` - View recent activity

### API Endpoints

The bot exposes a REST API for programmatic control:

- `GET /health` - Health check
- `GET /api/status` - Bot status
- `POST /api/enable` - Enable trading
- `POST /api/disable` - Disable trading
- `POST /api/emergency_stop` - Close all positions
- `GET /api/positions` - Get open positions
- `GET /api/zones/{symbol}` - Get active zones
- `GET /metrics` - Prometheus metrics

## Strategy Configuration

Edit strategy parameters in `.env` or via Telegram:

```env
# Supply & Demand Settings
SD_MIN_BASE_CANDLES=3          # Minimum candles for consolidation
SD_MAX_BASE_CANDLES=10         # Maximum candles for consolidation
SD_DEPARTURE_ATR_MULTIPLIER=2.0 # Strength of breakout required
SD_ZONE_BUFFER_PERCENT=0.2     # Stop loss buffer beyond zone
SD_MAX_ZONE_TOUCHES=3          # Max times zone can be tested
SD_MIN_ZONE_SCORE=60.0         # Minimum score to trade zone

# Risk Management
TP1_RISK_RATIO=1.0             # First take profit at 1:1 R:R
TP2_RISK_RATIO=2.0             # Second take profit at 1:2 R:R
PARTIAL_TP1_PERCENT=50.0       # Close 50% at TP1
MOVE_STOP_TO_BREAKEVEN_AT_TP1=true
USE_TRAILING_STOP=true
```

## Backtesting

Run backtests via Telegram or API:

```bash
# Via Telegram
/backtest BTCUSDT 15m 30

# Returns:
# - Win rate, profit factor, Sharpe ratio
# - Zone performance statistics
# - Probability scores by zone quality
# - Equity curve chart
```

## Monitoring

### Grafana Dashboard
Access at `http://localhost:3000` (default: admin/admin)

Metrics include:
- PnL curves
- Win rate trends
- Position exposure
- API latency
- Zone detection rates

### Logs
- Main log: `logs/bot_YYYYMMDD.log`
- Error log: `logs/errors_YYYYMMDD.log`
- Trade journal: `logs/trades_YYYYMMDD.log`

## Safety Features

1. **Start with Testnet**: Always test on Bybit Testnet first
2. **Risk Limits**: Configurable max daily loss and position limits
3. **Emergency Stop**: Instant position closure via Telegram or API
4. **User Allowlist**: Only authorized Telegram users can control bot
5. **Secure Secrets**: API keys stored encrypted, never logged

## Development

### Project Structure
```
crypto_trading_bot/
├── src/
│   ├── api/           # Bybit API client
│   ├── strategy/      # Supply & Demand strategy
│   ├── telegram/      # Telegram bot interface
│   ├── trading/       # Order management
│   ├── backtesting/   # Backtest engine
│   ├── db/           # Database models
│   ├── utils/        # Utilities
│   └── main.py       # FastAPI application
├── docker/           # Docker configs
├── tests/           # Test suite
└── requirements.txt
```

### Testing
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Deployment Checklist

- [ ] Generate Bybit API keys (with trading permissions only)
- [ ] Create Telegram bot via @BotFather
- [ ] Configure `.env` with production values
- [ ] Test on Bybit Testnet first
- [ ] Set up SSL certificates for webhook
- [ ] Configure firewall rules
- [ ] Set up monitoring alerts
- [ ] Create backup strategy
- [ ] Document emergency procedures

## Support & Resources

- **Bybit API Docs**: https://bybit-exchange.github.io/docs/
- **Telegram Bot API**: https://core.telegram.org/bots/api
- **Supply & Demand Guide**: See strategy documentation

## Disclaimer

**IMPORTANT**: Cryptocurrency futures trading involves substantial risk of loss. This bot is provided as-is without any guarantees. Always:

1. Start with Testnet
2. Use small position sizes
3. Never risk more than you can afford to lose
4. Monitor the bot regularly
5. Have an emergency plan

The authors are not responsible for any financial losses incurred through use of this software.

## License

MIT License - See LICENSE file for details