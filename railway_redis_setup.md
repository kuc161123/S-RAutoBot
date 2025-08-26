# Railway Redis Setup

## Adding Redis to Your Railway Project

1. **In Railway Dashboard**:
   - Go to your project
   - Click "New" → "Database" → "Add Redis"
   - Redis will be automatically provisioned

2. **Connect to Your Service**:
   - Railway will automatically inject `REDIS_URL` environment variable
   - Format: `redis://default:password@redis.railway.internal:6379`

3. **Environment Variables**:
   Railway automatically provides:
   - `REDIS_URL` - Full connection URL
   - `REDISHOST` - Host (redis.railway.internal)  
   - `REDISPORT` - Port (6379)
   - `REDISUSER` - Username (default)
   - `REDISPASSWORD` - Password

4. **No Configuration Needed**:
   The bot will automatically use Railway's Redis when deployed!

## Current Status

Your bot is correctly configured to:
- Use Railway's Redis when deployed (via REDIS_URL)
- Fall back to in-memory queue if Redis isn't available
- This is working as expected!

## The "redis.railway.internal" Message

The message you see:
```
Redis connection failed (internal hostname may not be accessible locally) - using in-memory queue
```

This is NORMAL when:
- Running locally (your computer can't access Railway's internal network)
- Redis service hasn't been added to Railway yet

## To Enable Redis on Railway:

1. Go to your Railway project
2. Click "+ New" 
3. Select "Redis"
4. It will automatically connect to your app

That's it! No code changes needed.