# Support/Resistance + Market Structure Trading Bot

A professional automated trading bot for Bybit that uses pivot-based support/resistance levels and market structure analysis to execute trades with proper risk management.

## Features

- **Market Structure Analysis**: Identifies trends using Higher Highs/Higher Lows and Lower Highs/Lower Lows
- **Support/Resistance Detection**: Uses pivot points to identify key levels
- **Risk Management**: Fixed USD risk per trade with configurable R:R ratio
- **Real-time WebSocket**: Streams live kline data from Bybit
- **Telegram Integration**: Control and monitor bot via Telegram commands
- **Position Management**: One position per symbol with automatic flipping
- **Safety Features**: Panic close, configurable risk limits, proper error handling

## Strategy Overview

The bot identifies market structure trends and enters positions on breakouts:
- **Long Entry**: When price breaks above resistance in an uptrend (HH + HL)
- **Short Entry**: When price breaks below support in a downtrend (LH + LL)
- **Stop Loss**: Placed at opposite swing level with ATR buffer
- **Take Profit**: Fixed R:R ratio (default 1:2)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd AutoTrading-Bot
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the bot:
- Copy `config.yaml.example` to `config.yaml`
- Add your Bybit API credentials
- Add your Telegram bot token and chat ID
- Adjust trading parameters as needed

5. Run the bot:
```bash
python live_bot.py
```

## Configuration

### Environment Variables
The bot supports environment variables for sensitive data:
- `BYBIT_API_KEY`: Your Bybit API key
- `BYBIT_API_SECRET`: Your Bybit API secret
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token
- `TELEGRAM_CHAT_ID`: Your Telegram chat ID

### Trading Parameters
Edit `config.yaml` to adjust:
- `symbols`: List of trading pairs
- `timeframe`: Candle timeframe (1, 3, 5, 15, 60, 240, 1440)
- `risk_usd`: Risk amount per trade in USD
- `rr`: Risk/Reward ratio
- `use_ema`: Enable EMA filter
- `use_vol`: Enable volume filter

## Telegram Commands

- `/start` - Initialize bot
- `/help` - Show available commands
- `/status` - Display open positions
- `/balance` - Show account balance
- `/set_risk [amount]` - Set risk per trade
- `/panic_close [symbol]` - Emergency close position

## Safety Features

1. **Risk Limits**: Maximum risk per trade is configurable
2. **Position Limits**: One position per symbol
3. **Panic Close**: Emergency position closure via Telegram
4. **Error Handling**: Comprehensive error catching and logging
5. **Reconnection**: Automatic WebSocket reconnection

## Project Structure

```
├── live_bot.py         # Main bot entry point
├── strategy.py         # Trading strategy logic
├── position_mgr.py     # Position management
├── sizer.py           # Position sizing calculator
├── broker_bybit.py    # Bybit API wrapper
├── telegram_bot.py    # Telegram bot interface
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
└── README.md         # Documentation
```

## Deployment

### Railway
1. Push code to GitHub
2. Connect Railway to your GitHub repo
3. Set environment variables in Railway
4. Deploy

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "live_bot.py"]
```

## Risk Warning

**IMPORTANT**: Trading cryptocurrencies involves substantial risk of loss. This bot is for educational purposes. Always:
- Test with small amounts first
- Use testnet for initial testing
- Never risk more than you can afford to lose
- Monitor the bot regularly
- Have emergency procedures in place

## Support

For issues or questions:
- Check the logs for error messages
- Review the configuration
- Ensure API keys have proper permissions
- Verify network connectivity

## License

MIT License - Use at your own risk

## Disclaimer

This software is provided "as is" without warranty of any kind. The authors are not responsible for any losses incurred through the use of this software. Always conduct your own research and risk assessment before trading.