# Crypto Trading Bot v2.0

A simple, clean, and efficient cryptocurrency trading bot for Bybit exchange.

## Features

- ✅ Simple RSI + MACD strategy
- ✅ Real-time WebSocket data streaming
- ✅ Automatic position management
- ✅ Risk management (1% per trade default)
- ✅ Telegram notifications and control
- ✅ Fast startup (<30 seconds)
- ✅ Clean architecture
- ✅ Testnet support

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `env.example` to `.env` and fill in your credentials:

```bash
cp env.example .env
```

Edit `.env` with your API keys:
- Get Bybit API keys from: https://testnet.bybit.com/user/api-management (for testnet)
- Create a Telegram bot via @BotFather on Telegram
- Get your Telegram chat ID via @userinfobot

### 3. Run the Bot

```bash
python main.py
```

## Architecture

```
crypto_bot/
├── exchange/         # Exchange connectivity
├── strategy/         # Trading strategies
├── trading/          # Order and position management
├── telegram/         # Telegram bot interface
└── utils/           # Utilities and helpers
```

## Configuration

The bot uses these default settings (configurable in config.py):

- **Initial Symbols**: Top 10 cryptocurrencies
- **Risk per Trade**: 1% of balance
- **Max Positions**: 10
- **Leverage**: 10x
- **Strategy**: RSI (14) + MACD (12,26,9)
- **Scan Interval**: 60 seconds

## Trading Strategy

The bot uses a simple but effective strategy:

**BUY Signals** (need 2+ conditions):
- RSI < 30 (oversold)
- MACD bullish crossover
- Bollinger Band lower bounce
- Above SMA 200 (trend filter)

**Risk Management**:
- Stop Loss: 2 ATR
- Take Profit: 4 ATR (2:1 risk/reward)
- Position Sizing: Based on 1% account risk

## Telegram Commands

- `/start` - Start the bot
- `/stop` - Stop trading
- `/status` - Current status
- `/positions` - Open positions
- `/balance` - Account balance
- `/stats` - Trading statistics
- `/help` - Help message

## Performance

- **Startup Time**: <30 seconds
- **Memory Usage**: <200MB
- **CPU Usage**: <5%
- **Symbols Supported**: 250+ (configurable)

## Safety Features

- Testnet mode by default
- Maximum position limits
- Stop loss on every trade
- Daily loss limits (configurable)
- Graceful shutdown

## Troubleshooting

1. **Bot won't start**: Check your .env file and API credentials
2. **No signals**: Ensure market conditions meet strategy criteria
3. **Connection issues**: Check internet connection and Bybit status
4. **Telegram not working**: Verify bot token and chat IDs

## Development

To add more symbols, edit `config.py`:

```python
initial_symbols: List[str] = Field(
    default=["BTCUSDT", "ETHUSDT", ...]  # Add more symbols
)
```

To modify strategy parameters:

```python
rsi_oversold: float = Field(30.0)  # Change RSI levels
rsi_overbought: float = Field(70.0)
```

## Disclaimer

This bot is for educational purposes. Trading cryptocurrencies carries risk. Always test on testnet first and never trade with funds you cannot afford to lose.

## License

MIT License - Feel free to modify and use as needed.