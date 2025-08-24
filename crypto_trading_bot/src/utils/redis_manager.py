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
    Singleton Redis connection manager
    Ensures only one Redis client is created and shared across all components
    """
    _instance = None
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
                if redis_url:
                    self._client = redis.from_url(
                        redis_url, 
                        decode_responses=True,
                        socket_keepalive=True,
                        socket_keepalive_options={
                            1: 1,  # TCP_KEEPIDLE
                            2: 1,  # TCP_KEEPINTVL
                            3: 5,  # TCP_KEEPCNT
                        }
                    )
                else:
                    self._client = redis.Redis(
                        host='localhost',
                        port=6379,
                        db=0,
                        decode_responses=True,
                        socket_keepalive=True,
                        socket_keepalive_options={
                            1: 1,  # TCP_KEEPIDLE
                            2: 1,  # TCP_KEEPINTVL
                            3: 5,  # TCP_KEEPCNT
                        }
                    )
                
                # Test connection
                await self._client.ping()
                logger.info("Redis client connected successfully")
                return self._client
                
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._client = None
                return None
    
    async def close(self):
        """Close Redis connection"""
        async with self._lock:
            if self._client:
                try:
                    await self._client.close()
                    logger.info("Redis connection closed")
                except Exception as e:
                    logger.error(f"Error closing Redis connection: {e}")
                finally:
                    self._client = None
    
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        return self._client is not None


# Global instance
redis_manager = RedisManager()