# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a professional automated trading bot for Bybit perpetual futures that uses pivot-based support/resistance levels and market structure analysis. The bot identifies trends using Higher Highs/Higher Lows patterns and executes trades on breakouts with proper risk management.

## Key Architecture

### Core Components
- **live_bot.py**: Main entry point, WebSocket connection manager, orchestrates all components
- **strategy.py**: Trading logic - pivot detection, market structure analysis, signal generation
- **position_mgr.py**: Position tracking and management
- **broker_bybit.py**: Bybit API wrapper for order execution
- **telegram_bot.py**: Telegram interface for monitoring and control
- **candle_storage_postgres.py**: Persistent storage for OHLCV data (PostgreSQL/SQLite)
- **trade_tracker.py**: Trade history and statistics tracking

### Data Flow
1. WebSocket streams live klines from Bybit → stored in frames dict
2. Strategy analyzes last 200+ candles for signals
3. Position manager checks risk rules before entry
4. Broker executes orders with TP/SL in Partial mode
5. Telegram bot sends notifications and accepts commands

## Development Commands

### Running the Bot
```bash
# Install dependencies
pip install -r requirements.txt

# Run the bot (requires configured environment variables)
python live_bot.py

# For testing with smaller risk
RISK_USD=1 python live_bot.py
```

### Environment Setup
Required environment variables (use .env file or export):
- `BYBIT_API_KEY`: Bybit API key with trading permissions
- `BYBIT_API_SECRET`: Bybit API secret
- `TELEGRAM_BOT_TOKEN`: Telegram bot token from BotFather
- `TELEGRAM_CHAT_ID`: Your Telegram chat ID
- `DATABASE_URL`: PostgreSQL connection string (optional, uses SQLite if not set)

### Testing
No formal test suite currently. For testing:
1. Use Bybit testnet by changing base_url in config.yaml
2. Start with minimal risk_usd (e.g., $1-2)
3. Monitor via Telegram commands and logs

### Common Tasks

**Update trading symbols:**
```bash
# Fetch top 50 by market cap
python get_top_50_marketcap.py

# Update config.yaml with new symbols
python update_by_marketcap.py
```

**Check symbol specifications:**
```bash
python fetch_symbol_specs.py
```

**Database operations:**
- Candles auto-save every 2 minutes
- Auto-cleanup of data older than 30 days
- PostgreSQL recommended for production (Railway deployment)

## Trading Strategy Configuration

Key parameters in config.yaml:
- `timeframe`: Candle period (15 = 15 minutes recommended)
- `risk_usd`: Fixed USD risk per trade
- `rr`: Risk/Reward ratio (default 2.0)
- `use_ema`: Enable 200 EMA trend filter
- `sl_buf_atr`: ATR multiplier for stop loss buffer
- `both_hit_rule`: SL_FIRST or TP_FIRST when both hit same candle

## Deployment

### Railway
1. Push to GitHub
2. Connect Railway to repo
3. Add PostgreSQL database
4. Set environment variables
5. Deploy (uses Dockerfile)

### Local Development
1. Create .env file with credentials
2. Run `python live_bot.py`
3. Monitor logs and Telegram

## Important Notes

- Bot uses Bybit Partial mode for better TP fills
- One position per symbol maximum
- Automatic position flipping on opposite signals
- WebSocket auto-reconnects on disconnect
- All times in UTC
- Leverage is automatically set based on symbol specifications

## Error Handling

The bot includes comprehensive error handling:
- WebSocket reconnection with exponential backoff
- Order retry logic with delays
- Telegram command error reporting
- Database connection pooling and retries
- Graceful shutdown on SIGINT/SIGTERM

## Monitoring

Use Telegram commands:
- `/status` - View open positions
- `/balance` - Check account balance
- `/stats` - Trading statistics
- `/panic_close [symbol]` - Emergency close
- `/set_risk [amount]` - Adjust risk per trade