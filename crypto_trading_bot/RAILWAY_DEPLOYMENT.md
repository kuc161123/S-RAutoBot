# Railway Deployment Guide

## Quick Deploy to Railway

### Step 1: Prepare Your Repository
Make sure your code is pushed to GitHub:
```bash
git add .
git commit -m "Add Railway configuration"
git push origin main
```

### Step 2: Deploy on Railway

1. Go to [Railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository: `AutoTrading-Bot`
5. Railway will automatically detect the configuration

### Step 3: Configure Environment Variables

In your Railway project dashboard:

1. Click on your service
2. Go to "Variables" tab
3. Click "Raw Editor"
4. Paste the following (with your actual values):

```env
# REQUIRED - Bybit API (Get from https://testnet.bybit.com/user/api-management)
BYBIT_API_KEY=your_actual_api_key
BYBIT_API_SECRET=your_actual_api_secret
BYBIT_TESTNET=true

# REQUIRED - Telegram Bot (Get from @BotFather on Telegram)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_IDS=[123456789]

# OPTIONAL - Trading Settings
RISK_PER_TRADE=0.01
MAX_POSITIONS=10
LEVERAGE=10
SCAN_INTERVAL=60
LOG_LEVEL=INFO
```

### Step 4: Deploy

1. Click "Deploy" button
2. Wait for the build to complete (2-3 minutes)
3. Check logs to ensure bot started successfully

### Step 5: Verify

1. Check Railway logs for:
   - "Trading bot initializing..."
   - "Connected to Bybit"
   - "Telegram bot started"

2. Send `/status` to your Telegram bot

### Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BYBIT_API_KEY` | ✅ Yes | - | Your Bybit API key |
| `BYBIT_API_SECRET` | ✅ Yes | - | Your Bybit API secret |
| `BYBIT_TESTNET` | ✅ Yes | true | Use testnet (true) or mainnet (false) |
| `TELEGRAM_BOT_TOKEN` | ✅ Yes | - | Telegram bot token from @BotFather |
| `TELEGRAM_CHAT_IDS` | ✅ Yes | - | Your Telegram chat ID (as array: [123456789]) |
| `RISK_PER_TRADE` | No | 0.01 | Risk per trade (1% = 0.01) |
| `MAX_POSITIONS` | No | 10 | Maximum concurrent positions |
| `LEVERAGE` | No | 10 | Trading leverage |
| `SCAN_INTERVAL` | No | 60 | Seconds between market scans |
| `LOG_LEVEL` | No | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Important Notes

1. **Start with TESTNET**: Always set `BYBIT_TESTNET=true` initially
2. **API Permissions**: Ensure your Bybit API key has:
   - Futures trading enabled
   - Spot & Margin Trading enabled
   - Read access
3. **Telegram Chat ID**: Get yours from @userinfobot on Telegram
4. **Memory**: The bot uses <200MB RAM, well within Railway's free tier

### Monitoring

View your bot's performance:

1. **Railway Dashboard**: Check logs and metrics
2. **Telegram Bot**: Send `/status` or `/positions`
3. **Resource Usage**: Monitor in Railway's Metrics tab

### Troubleshooting

#### Bot Won't Start
- Check environment variables are set correctly
- Verify API keys are valid
- Check logs for specific error messages

#### No Trading Signals
- Confirm market is open
- Check if using testnet (limited pairs)
- Verify symbols are active on Bybit

#### Telegram Not Working
- Verify bot token is correct
- Check chat ID is in array format: [123456789]
- Ensure bot is added to your chat

### Updating

To update your bot:

1. Make changes locally
2. Test thoroughly
3. Push to GitHub:
   ```bash
   git add .
   git commit -m "Update description"
   git push origin main
   ```
4. Railway will auto-deploy

### Cost

- **Free Tier**: 500 hours/month (enough for 24/7 operation)
- **Resource Usage**: ~200MB RAM, minimal CPU
- **Estimated Cost**: $0 (within free tier)

### Security

- Never commit API keys to GitHub
- Use Railway's environment variables
- Enable 2FA on all accounts
- Start with testnet for safety

### Support

- Railway Status: https://status.railway.app
- Bybit Status: https://www.bybit.com/service
- Bot Issues: Check logs in Railway dashboard