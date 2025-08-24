# Railway Deployment Setup

## Environment Variables Required in Railway

### 1. In your Railway project dashboard:

1. **Go to your Bot Service** → Variables tab
2. **Add these Reference Variables:**
   ```
   REDIS_URL=${{Redis.REDIS_URL}}
   DATABASE_URL=${{Postgres.DATABASE_URL}}
   ```

3. **Add these Raw Variables:**
   ```
   BYBIT_API_KEY=your_api_key_here
   BYBIT_API_SECRET=your_api_secret_here
   BYBIT_TESTNET=true
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_SECRET_TOKEN=your_webhook_secret
   TELEGRAM_ALLOWED_CHAT_IDS=123456789
   ```

### 2. Redis Service Setup

If Redis is showing connection errors:

1. **Check Redis Service** is running in Railway
2. **Internal URL Format:** `redis://default:[password]@redis.railway.internal:6379`
3. **Redis must be in same project** for internal networking to work

### 3. Common Issues & Solutions

#### Redis Connection Failed
- **Issue:** `Error 22 connecting to redis.railway.internal:6379`
- **Solution:** 
  - Redis service must be in the same Railway project
  - Use the Reference Variable: `${{Redis.REDIS_URL}}`
  - Internal URLs only work within Railway's network

#### If Redis Still Doesn't Connect
The bot will automatically fall back to in-memory queue, which works fine for:
- Single instance deployments
- Testing and development
- Low-volume trading

Redis is only needed for:
- Multi-instance deployments
- Signal persistence across restarts
- High-volume concurrent operations

### 4. Verify Setup

After deployment, check logs for:
```
✅ "Redis client connected successfully" - Redis is working
⚠️ "Redis not available - using in-memory queue" - Using fallback (still works!)
```

## Local Development

For local development, use `.env` file:
```bash
REDIS_URL=redis://localhost:6379/0
# Or leave empty to use in-memory queue
```

## Production Best Practices

1. **Always use Reference Variables** for service URLs in Railway
2. **Never hardcode** internal URLs - they change on redeploy
3. **Bot works without Redis** - it's an optional enhancement