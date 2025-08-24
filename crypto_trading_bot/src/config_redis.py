"""
Redis Configuration Helper
Provides different Redis connection options
"""

# Option 1: Local Redis (if you have Redis installed locally)
LOCAL_REDIS_URL = "redis://localhost:6379/0"

# Option 2: Redis Cloud (free tier available at https://redis.com/try-free/)
# REDIS_CLOUD_URL = "redis://default:YOUR_PASSWORD@YOUR_HOST:YOUR_PORT"

# Option 3: Railway Redis (if deployed on Railway)
# RAILWAY_REDIS_URL = "redis://default:PASSWORD@redis.railway.internal:6379"

# Option 4: Docker Redis (if running Redis in Docker)
# DOCKER_REDIS_URL = "redis://redis:6379/0"

# Option 5: No Redis (use in-memory queue)
NO_REDIS_URL = None

# To use Redis, add this to your .env file:
# REDIS_URL=redis://localhost:6379/0

# Or set it in your settings.py:
# redis_url = "redis://localhost:6379/0"