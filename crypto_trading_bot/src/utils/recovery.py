"""
Automatic recovery and error handling utilities
"""
import asyncio
from typing import Optional, Callable, Any
from datetime import datetime, timedelta
import structlog
from functools import wraps

logger = structlog.get_logger(__name__)

class RecoveryManager:
    """Manages automatic recovery from failures"""
    
    def __init__(self):
        self.recovery_attempts = {}
        self.max_recovery_attempts = 3
        self.recovery_delay = 30  # seconds
        
    async def attempt_recovery(self, component: str, recovery_func: Callable) -> bool:
        """Attempt to recover a failed component"""
        if component not in self.recovery_attempts:
            self.recovery_attempts[component] = 0
        
        if self.recovery_attempts[component] >= self.max_recovery_attempts:
            logger.error(f"Max recovery attempts reached for {component}")
            return False
        
        self.recovery_attempts[component] += 1
        logger.info(f"Attempting recovery for {component} (attempt {self.recovery_attempts[component]})")
        
        try:
            await recovery_func()
            logger.info(f"Recovery successful for {component}")
            self.recovery_attempts[component] = 0  # Reset on success
            return True
        except Exception as e:
            logger.error(f"Recovery failed for {component}: {e}")
            await asyncio.sleep(self.recovery_delay * self.recovery_attempts[component])
            return False

def with_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for automatic retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Max retry attempts reached for {func.__name__}: {e}")
                        raise
                    
                    logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}")
                    logger.info(f"Retrying in {current_delay:.1f} seconds...")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator

class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
        
    def record_success(self):
        """Record a successful operation"""
        self.failure_count = 0
        self.is_open = False
        
    def record_failure(self):
        """Record a failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if not self.is_open:
            return True
        
        # Check if timeout has passed
        if self.last_failure_time:
            time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
            if time_since_failure >= self.timeout:
                logger.info("Circuit breaker timeout expired, attempting reset")
                self.is_open = False
                self.failure_count = 0
                return True
        
        return False

class TestMode:
    """Test mode for safe testing without real trading"""
    
    def __init__(self):
        self.enabled = False
        self.simulated_trades = []
        self.simulated_balance = 10000  # Starting balance for simulation
        
    def enable(self):
        """Enable test mode"""
        self.enabled = True
        logger.warning("TEST MODE ENABLED - No real trades will be executed")
        
    def disable(self):
        """Disable test mode"""
        self.enabled = False
        logger.info("Test mode disabled - Real trading enabled")
        
    async def simulate_order(self, order_data: dict) -> dict:
        """Simulate an order in test mode"""
        if not self.enabled:
            raise ValueError("Test mode is not enabled")
        
        # Simulate order execution
        simulated_order = {
            "order_id": f"TEST_{datetime.now().timestamp()}",
            "symbol": order_data.get("symbol"),
            "side": order_data.get("side"),
            "qty": order_data.get("qty"),
            "price": order_data.get("price", "MARKET"),
            "status": "FILLED",
            "timestamp": datetime.now().isoformat()
        }
        
        self.simulated_trades.append(simulated_order)
        logger.info(f"[TEST MODE] Simulated order: {simulated_order}")
        
        return simulated_order
    
    def get_test_summary(self) -> dict:
        """Get summary of test mode activities"""
        return {
            "enabled": self.enabled,
            "simulated_trades": len(self.simulated_trades),
            "simulated_balance": self.simulated_balance,
            "trades": self.simulated_trades[-10:]  # Last 10 trades
        }

# Global instances
recovery_manager = RecoveryManager()
test_mode = TestMode()