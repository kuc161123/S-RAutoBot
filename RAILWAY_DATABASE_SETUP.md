# Railway PostgreSQL Setup Guide

## Steps to Add PostgreSQL to Your Railway Project:

1. **In Railway Dashboard:**
   - Click "Add PostgreSQL" (as shown in your screenshot)
   - Railway will automatically provision a PostgreSQL database
   - It will automatically set the `DATABASE_URL` environment variable

2. **The bot will automatically:**
   - Detect the `DATABASE_URL` from Railway environment
   - Connect to PostgreSQL instead of local SQLite
   - Create all necessary tables on first run
   - Save and load candle data persistently

## Benefits of PostgreSQL on Railway:

✅ **Persistent Storage** - Data survives all deployments and restarts
✅ **No Data Loss** - Professional database with ACID compliance  
✅ **Automatic Backups** - Railway handles database backups
✅ **Scalable** - Can handle millions of candles
✅ **Shared Access** - Access from local dev and production

## How It Works:

1. When you add PostgreSQL in Railway, it provides a `DATABASE_URL` like:
   ```
   postgresql://user:password@host:port/database
   ```

2. The bot automatically:
   - Detects this URL from environment
   - Connects to PostgreSQL
   - Creates the `candles` table
   - Saves all OHLCV data

3. On restart:
   - Loads all historical candles instantly
   - No more waiting for data accumulation
   - Continues where it left off

## Local Development:

For local testing, the bot will automatically fall back to SQLite if no `DATABASE_URL` is set.

To use Railway's PostgreSQL locally:
1. Copy the `DATABASE_URL` from Railway variables
2. Add to your `.env` file:
   ```
   DATABASE_URL=postgresql://...
   ```

## Database Schema:

The bot creates a `candles` table with:
- `symbol` - Trading pair (e.g., BTCUSDT)
- `timestamp` - Unix timestamp in milliseconds
- `open`, `high`, `low`, `close` - Price data
- `volume` - Trading volume
- Indexed for fast queries on symbol and timestamp

## Monitoring:

The bot logs database statistics on startup:
- Number of symbols stored
- Total candles in database
- Database size in MB

## Maintenance:

- Auto-cleanup of candles older than 30 days (configurable)
- Auto-save every 2 minutes during operation
- Save on shutdown for data integrity