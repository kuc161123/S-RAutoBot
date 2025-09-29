# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An ML-enhanced automated trading bot for Bybit perpetual futures that monitors 263 trading pairs simultaneously. Uses pullback strategies with pivot-based support/resistance detection, market structure analysis, and machine learning signal scoring for trade execution with strict risk management.

## Key Architecture

### Core Components
- **live_bot.py**: Main entry point, orchestrates all components, manages WebSocket connections
- **strategy_pullback_ml_learning.py**: Active ML-enhanced pullback strategy with adaptive learning
- **ml_signal_scorer_immediate.py**: ML scoring engine with 70% threshold for signal quality
- **phantom_trade_tracker.py**: Tracks all signals (executed and rejected) for ML learning
- **multi_websocket_handler.py**: Manages multiple WebSocket connections for 263 symbols
- **position_mgr.py**: Position tracking, risk management, one-per-symbol limit
- **broker_bybit.py**: Bybit API wrapper for order execution with Partial mode
- **telegram_bot.py**: Telegram interface with 40+ commands for monitoring/control
- **candle_storage_postgres.py**: Persistent OHLCV storage (PostgreSQL/SQLite fallback)
- **trade_tracker_postgres.py**: Trade history and statistics with PostgreSQL persistence
- **multi_timeframe_sr.py**: HTF support/resistance analysis for better levels

### ML System Components
- **ml_signal_scorer_immediate.py**: Immediate ML scoring (active from day 1)
- **phantom_trade_tracker.py**: Shadow tracking for learning from all signals
- **ml_evolution_system.py**: Symbol-specific models (shadow mode, learning only)
- **symbol_data_collector.py**: Collects data for future ML improvements
- **symbol_clustering.py**: Groups symbols by behavior for better ML performance

### Data Flow
1. Multi-WebSocket streams → 263 symbols' live klines → frames dict
2. Strategy analyzes 200+ candles → detects pullback signals
3. ML scorer evaluates signal → 34 features → score 0-100
4. Phantom tracker records all signals for learning
5. Position manager checks risk rules → executes if score ≥ 70
6. Broker executes orders → TP/SL in Partial mode
7. Telegram bot sends notifications and accepts commands

## Development Commands

### Running the Bot
```bash
# Install dependencies
pip install -r requirements.txt

# Run main bot with ML scoring (production)
python live_bot.py

# Run with custom risk amount
RISK_USD=10 python live_bot.py

# Run with percentage risk
RISK_PERCENT=3 python live_bot.py
```

### Testing and Development
```bash
# Test ML integration without trading
python test_ml_integration.py

# Test phantom trade collection
python test_phantom_collection.py

# Test enhanced strategy components
python test_enhanced_strategy.py

# Check all symbols connectivity
python check_all_symbols.py

# Verify symbol specifications
python fetch_symbol_specs.py
```

### ML System Management
```bash
# Clear ML Redis data (careful!)
python clear_ml_redis.py

# Force reset ML models
python force_reset_ml.py

# Check ML state and performance
python check_ml_state.py

# Reset trading statistics
python reset_all_stats.py
```

### Symbol Management
```bash
# Update symbols by market cap
python get_top_50_marketcap.py    # Fetches top 50
python get_top_100_marketcap.py   # Fetches top 100
python get_top_250_marketcap.py   # Fetches top 250
python update_by_marketcap.py     # Updates config.yaml

# Test symbol connectivity
python test_250_symbols.py
```

### Environment Setup
Required environment variables (create .env file from .env.example):
- `BYBIT_API_KEY`: Bybit API key with trading permissions
- `BYBIT_API_SECRET`: Bybit API secret
- `TELEGRAM_BOT_TOKEN`: Telegram bot token from BotFather
- `TELEGRAM_CHAT_ID`: Your Telegram chat ID
- `DATABASE_URL`: PostgreSQL connection string (optional, uses SQLite if not set)
- `REDIS_URL`: Redis connection for ML data (optional, uses local Redis)
- `RISK_USD`: Override risk amount in USD (optional)
- `RISK_PERCENT`: Override risk as percentage (optional)

### Database Operations
- Candles auto-save every 15 minutes
- Auto-cleanup of data older than 30 days
- PostgreSQL recommended for production (Railway)
- SQLite fallback for local development

## Trading Strategy Configuration

Key parameters in config.yaml:
- `timeframe`: 15 (15-minute candles, fixed)
- `risk_usd`: Fixed USD risk per trade (default 10)
- `risk_percent`: Risk as % of balance (default 3%)
- `rr`: Risk/Reward ratio (2.5 after fees = 2.67 gross)
- `use_ema`: EMA filter (disabled by default)
- `sl_buf_atr`: ATR buffer multiplier (1.0-1.5 based on volatility)
- `both_hit_rule`: SL_FIRST or TP_FIRST when both hit
- `ml_threshold`: Minimum ML score to take trade (70)
- `htf_timeframe`: Higher timeframe for S/R (4H = 240)

### Current Strategy Settings
- **Active Strategy**: strategy_pullback_ml_learning.py
- **ML Scoring**: Immediate scorer with 70% threshold
- **Pullback Requirements**: 2+ confirmation candles
- **S/R Zones**: 0.3% around levels
- **Stop Loss**: Hybrid method (most conservative of 3 calculations)
- **ML Features**: 34 total (22 technical + 12 cluster/MTF)

## Deployment

### Railway Production
```bash
# Push to GitHub
git add . && git commit -m "Update"
git push origin main

# Railway auto-deploys from main branch
# Uses Dockerfile with start.py wrapper
```

### Local Development
```bash
# Setup environment
cp .env.example .env
# Edit .env with your credentials

# Install dependencies
pip install -r requirements.txt

# Run bot
python live_bot.py

# Monitor via Telegram
```

### Docker Deployment
```bash
# Build image
docker build -t trading-bot .

# Run container
docker run -d \
  --env-file .env \
  --name trading-bot \
  trading-bot
```

## Important Notes

### Trading Behavior
- One position per symbol maximum
- No position flipping (closes before new entry)
- Bybit Partial mode for better fills
- Leverage auto-set based on symbol specs
- All timestamps in UTC

### ML System
- Immediate scorer active from day 1
- Phantom tracking for all signals
- Retrains every 100 trades
- Symbol clustering for better performance
- Evolution system in shadow mode (learning only)

### Safety Features
- Multi-WebSocket with auto-reconnection
- Order retry with exponential backoff
- Position recovery on restart
- Graceful shutdown (SIGINT/SIGTERM)
- Comprehensive error logging

## Monitoring

### Key Telegram Commands
```
Position Management:
/status - View all open positions
/panic_close [symbol] - Emergency close
/close_all - Close all positions

Risk Control:
/set_risk [amount] - Set USD risk
/set_risk_percent [%] - Set % risk
/get_risk - View current settings

Performance:
/stats - Overall statistics
/ml_stats - ML system metrics
/ml_rankings - Symbol performance
/phantom_stats - Shadow tracking

System:
/balance - Account balance
/symbols - Active symbols list
/help - All commands
/restart - Restart bot
```