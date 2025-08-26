"""
Shared Redis Connection Manager
Provides a single Redis client instance for all components
"""
import asyncio
import structlog
try:
    import redis.asyncio as redis
except ImportError:
    redis = None

logger = structlog.get_logger(__name__)


class RedisManager:
    """
    Singleton Redis connection manager with connection pooling
    Provides connection pool for better performance under load
    """
    _instance = None
    _pool = None
    _client = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_client(self, redis_url: str = None) -> redis.Redis:
        """
        Get or create Redis client
        
        Args:
            redis_url: Redis connection URL
        
        Returns:
            Redis client instance or None if unavailable
        """
        async with self._lock:
            if self._client is not None:
                try:
                    # Test if connection is still alive
                    await self._client.ping()
                    return self._client
                except:
                    # Connection dead, recreate
                    logger.warning("Redis connection lost, reconnecting...")
                    self._client = None
            
            if not redis:
                logger.warning("Redis module not available")
                return None
            
            try:
                # Create connection pool if not exists
                if self._pool is None:
                    # Check for Railway internal Redis URL
                    import os
                    if not redis_url:
                        # Try to get from environment
                        redis_url = os.getenv('REDIS_URL')
                    
                    # Handle Railway internal URL when running locally
                    if redis_url and 'redis.railway.internal' in redis_url:
                        # Check if we're running locally (not on Railway)
                        import os
                        if not os.getenv('RAILWAY_ENVIRONMENT'):
                            logger.info("Railway internal Redis URL detected but not running on Railway - using in-memory queue")
                            return None
                    
                    # Skip localhost Redis in production
                    if redis_url and 'localhost' in redis_url:
                        import os
                        if os.getenv('ENVIRONMENT') == 'production':
                            logger.info("Skipping localhost Redis in production - using in-memory queue")
                            return None
                    
                    if redis_url:
                        logger.info(f"Attempting Redis connection to: {redis_url.split('@')[-1] if '@' in redis_url else redis_url}")
                        
                        # Create connection pool from URL
                        from redis.asyncio.connection import ConnectionPool
                        self._pool = ConnectionPool.from_url(
                            redis_url,
                            max_connections=20,  # Pool size
                            decode_responses=True,
                            socket_keepalive=True,
                            socket_keepalive_options={
                                1: 1,  # TCP_KEEPIDLE
                                2: 1,  # TCP_KEEPINTVL
                                3: 5,  # TCP_KEEPCNT
                            },
                            socket_connect_timeout=5,  # Add connection timeout
                            retry_on_timeout=True
                        )
                    else:
                        # No Redis URL provided - skip Redis
                        logger.info("No Redis URL configured - using in-memory queue")
                        return None
                
                # Create client from pool
                self._client = redis.Redis(connection_pool=self._pool)
                
                # Test connection
                await self._client.ping()
                logger.info("Redis client connected successfully")
                return self._client
                
            except Exception as e:
                error_msg = str(e)
                if 'redis.railway.internal' in (redis_url or ''):
                    logger.info("Redis connection failed (internal hostname may not be accessible locally) - using in-memory queue")
                else:
                    logger.warning(f"Redis connection failed: {error_msg} - using in-memory queue")
                self._client = None
                return None
    
    async def close(self):
        """Close Redis connection pool and client"""
        async with self._lock:
            if self._client:
                try:
                    await self._client.close()
                    logger.info("Redis client closed")
                except Exception as e:
                    logger.error(f"Error closing Redis client: {e}")
                finally:
                    self._client = None
            
            if self._pool:
                try:
                    await self._pool.disconnect()
                    logger.info("Redis connection pool closed")
                except Exception as e:
                    logger.error(f"Error closing Redis pool: {e}")
                finally:
                    self._pool = None
    
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        return self._client is not None


# Global instance
redis_manager = RedisManager()