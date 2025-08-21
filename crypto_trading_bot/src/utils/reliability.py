"""
Enhanced reliability features for production trading bot
"""
import asyncio
import time
from typing import Any, Callable, Optional, Dict, List
from functools import wraps
from datetime import datetime, timedelta
import structlog
from collections import deque
import traceback

logger = structlog.get_logger(__name__)

class RateLimiter:
    """Advanced rate limiting with burst protection"""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if rate limit would be exceeded"""
        async with self.lock:
            now = time.time()
            
            # Remove old requests outside time window
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()
            
            # Check if we're at limit
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = self.requests[0]
                wait_time = (oldest_request + self.time_window) - now
                
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    return await self.acquire()  # Retry
            
            # Add current request
            self.requests.append(now)

class ConnectionMonitor:
    """Monitor and maintain healthy connections"""
    
    def __init__(self):
        self.connections = {}
        self.last_heartbeat = {}
        self.reconnect_attempts = {}
        self.max_reconnect_attempts = 5
        self.heartbeat_interval = 30  # seconds
    
    async def register_connection(self, name: str, health_check: Callable):
        """Register a connection to monitor"""
        self.connections[name] = health_check
        self.last_heartbeat[name] = datetime.now()
        self.reconnect_attempts[name] = 0
        logger.info(f"Registered connection: {name}")
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                for name, health_check in self.connections.items():
                    try:
                        # Check connection health
                        is_healthy = await health_check()
                        
                        if is_healthy:
                            self.last_heartbeat[name] = datetime.now()
                            self.reconnect_attempts[name] = 0
                        else:
                            await self._handle_unhealthy_connection(name)
                            
                    except Exception as e:
                        logger.error(f"Error checking {name}: {e}")
                        await self._handle_unhealthy_connection(name)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Connection monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _handle_unhealthy_connection(self, name: str):
        """Handle unhealthy connection"""
        self.reconnect_attempts[name] += 1
        
        if self.reconnect_attempts[name] > self.max_reconnect_attempts:
            logger.critical(f"Connection {name} failed after {self.max_reconnect_attempts} attempts")
            # Send alert
            return
        
        logger.warning(f"Connection {name} unhealthy, attempt {self.reconnect_attempts[name]}")
        # Trigger reconnection logic

class DataValidator:
    """Validate data integrity and sanity"""
    
    @staticmethod
    def validate_price(price: float, symbol: str, reference_price: Optional[float] = None) -> bool:
        """Validate price is reasonable"""
        if price <= 0:
            logger.error(f"Invalid price {price} for {symbol}")
            return False
        
        # Check for extreme deviations if reference price available
        if reference_price:
            deviation = abs((price - reference_price) / reference_price)
            if deviation > 0.5:  # 50% deviation
                logger.warning(f"Extreme price deviation for {symbol}: {deviation:.2%}")
                return False
        
        return True
    
    @staticmethod
    def validate_order(order_data: Dict) -> bool:
        """Validate order parameters"""
        required_fields = ['symbol', 'side', 'qty', 'order_type']
        
        for field in required_fields:
            if field not in order_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate quantity
        if order_data['qty'] <= 0:
            logger.error(f"Invalid quantity: {order_data['qty']}")
            return False
        
        # Validate side
        if order_data['side'] not in ['Buy', 'Sell']:
            logger.error(f"Invalid side: {order_data['side']}")
            return False
        
        return True
    
    @staticmethod
    def validate_signal(signal: Any) -> bool:
        """Validate trading signal"""
        if not signal:
            return False
        
        # Check stop loss is reasonable
        if signal.side == 'Buy':
            if signal.stop_loss >= signal.entry_price:
                logger.error("Invalid Buy signal: stop loss above entry")
                return False
            if signal.take_profit_1 <= signal.entry_price:
                logger.error("Invalid Buy signal: TP below entry")
                return False
        else:
            if signal.stop_loss <= signal.entry_price:
                logger.error("Invalid Sell signal: stop loss below entry")
                return False
            if signal.take_profit_1 >= signal.entry_price:
                logger.error("Invalid Sell signal: TP above entry")
                return False
        
        # Check position size
        if signal.position_size <= 0:
            logger.error("Invalid signal: position size <= 0")
            return False
        
        return True

def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Advanced retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            delay = initial_delay
            last_exception = None
            
            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                    
                except exceptions as e:
                    attempt += 1
                    last_exception = e
                    
                    if attempt >= max_attempts:
                        logger.error(
                            f"Max retry attempts ({max_attempts}) reached for {func.__name__}",
                            error=str(e),
                            traceback=traceback.format_exc()
                        )
                        raise
                    
                    # Calculate next delay
                    delay = min(delay * exponential_base, max_delay)
                    
                    # Add jitter to prevent thundering herd
                    jitter = delay * 0.1
                    actual_delay = delay + (asyncio.get_event_loop().time() % jitter)
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}, "
                        f"retrying in {actual_delay:.2f}s",
                        error=str(e)
                    )
                    
                    await asyncio.sleep(actual_delay)
            
            # Should never reach here
            raise last_exception
        
        return wrapper
    return decorator

class ErrorRecovery:
    """Centralized error recovery management"""
    
    def __init__(self):
        self.error_counts = {}
        self.recovery_strategies = {}
        self.circuit_breakers = {}
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register a recovery strategy for specific error type"""
        self.recovery_strategies[error_type] = strategy
        logger.info(f"Registered recovery strategy for {error_type}")
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle error with appropriate recovery strategy"""
        error_type = type(error).__name__
        
        # Track error frequency
        if error_type not in self.error_counts:
            self.error_counts[error_type] = deque(maxlen=100)
        self.error_counts[error_type].append(datetime.now())
        
        # Check if circuit breaker should trip
        if self._should_trip_circuit_breaker(error_type):
            logger.critical(f"Circuit breaker tripped for {error_type}")
            return False
        
        # Apply recovery strategy
        if error_type in self.recovery_strategies:
            try:
                strategy = self.recovery_strategies[error_type]
                await strategy(error, context)
                logger.info(f"Recovery strategy applied for {error_type}")
                return True
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
                return False
        
        # Default handling
        logger.error(f"No recovery strategy for {error_type}: {error}")
        return False
    
    def _should_trip_circuit_breaker(self, error_type: str) -> bool:
        """Check if circuit breaker should trip"""
        if error_type not in self.error_counts:
            return False
        
        # Count recent errors (last 5 minutes)
        cutoff = datetime.now() - timedelta(minutes=5)
        recent_errors = [t for t in self.error_counts[error_type] if t > cutoff]
        
        # Trip if more than 10 errors in 5 minutes
        return len(recent_errors) > 10

class MemoryManager:
    """Monitor and manage memory usage"""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.cleanup_callbacks = []
    
    def register_cleanup(self, callback: Callable):
        """Register cleanup callback"""
        self.cleanup_callbacks.append(callback)
    
    async def monitor_memory(self):
        """Monitor memory usage and trigger cleanup if needed"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        while True:
            try:
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                if memory_mb > self.max_memory_mb:
                    logger.warning(f"Memory usage high: {memory_mb:.1f}MB")
                    
                    # Trigger cleanup callbacks
                    for callback in self.cleanup_callbacks:
                        try:
                            await callback()
                        except Exception as e:
                            logger.error(f"Cleanup callback failed: {e}")
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Check memory again
                    new_memory_mb = process.memory_info().rss / 1024 / 1024
                    logger.info(f"Memory after cleanup: {new_memory_mb:.1f}MB")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
                await asyncio.sleep(60)

class SafeExecutor:
    """Execute operations safely with comprehensive error handling"""
    
    @staticmethod
    async def safe_execute(
        operation: Callable,
        operation_name: str,
        fallback_value: Any = None,
        critical: bool = False
    ) -> Any:
        """Execute operation with safety checks"""
        try:
            # Pre-execution validation
            logger.debug(f"Executing {operation_name}")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                operation(),
                timeout=30.0  # 30 second timeout
            )
            
            # Post-execution validation
            if result is None and critical:
                logger.warning(f"{operation_name} returned None")
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"{operation_name} timed out")
            if critical:
                raise
            return fallback_value
            
        except Exception as e:
            logger.error(
                f"{operation_name} failed",
                error=str(e),
                traceback=traceback.format_exc()
            )
            if critical:
                raise
            return fallback_value

# Global instances
rate_limiter = RateLimiter(max_requests=100, time_window=60)
connection_monitor = ConnectionMonitor()
data_validator = DataValidator()
error_recovery = ErrorRecovery()
memory_manager = MemoryManager(max_memory_mb=512)
safe_executor = SafeExecutor()