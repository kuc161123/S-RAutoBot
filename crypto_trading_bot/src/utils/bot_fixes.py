"""
Critical Bot Fixes and Improvements
Based on comprehensive analysis and best practices
"""
import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import structlog
from collections import deque
import numpy as np

logger = structlog.get_logger(__name__)

class RateLimitManager:
    """
    Enhanced rate limiting for Bybit V5 API
    - 600 requests per 5 seconds (120/second)
    - Connection limit: 500 per 5 minutes
    - WebSocket: 1000 connections per IP
    """
    
    def __init__(self):
        # Request tracking
        self.request_times = deque(maxlen=600)
        self.connection_times = deque(maxlen=500)
        
        # Bybit V5 limits
        self.max_requests_per_5s = 600
        self.max_connections_per_5m = 500
        
        # Backoff state
        self.backoff_until = None
        self.consecutive_errors = 0
        
    async def acquire_request(self):
        """Acquire permission to make an API request"""
        now = time.time()
        
        # Check if we're in backoff
        if self.backoff_until and now < self.backoff_until:
            wait_time = self.backoff_until - now
            logger.warning(f"Rate limit backoff: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            
        # Clean old requests (older than 5 seconds)
        cutoff = now - 5
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()
        
        # Check rate limit
        if len(self.request_times) >= self.max_requests_per_5s:
            # Calculate wait time
            oldest = self.request_times[0]
            wait_time = (oldest + 5) - now + 0.1  # Add small buffer
            logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            
        # Record request
        self.request_times.append(now)
        
    def handle_rate_limit_error(self, error_code: int = None):
        """Handle rate limit error (403 or 429)"""
        self.consecutive_errors += 1
        
        # Exponential backoff: 10s, 20s, 40s, 80s, max 600s
        backoff_time = min(10 * (2 ** self.consecutive_errors), 600)
        self.backoff_until = time.time() + backoff_time
        
        logger.error(f"Rate limit hit! Backing off for {backoff_time}s")
        
    def reset_backoff(self):
        """Reset backoff on successful request"""
        self.consecutive_errors = 0
        self.backoff_until = None


class PositionSafetyManager:
    """
    Ensures one position per symbol and prevents duplicate orders
    """
    
    def __init__(self):
        self.active_positions: Dict[str, Dict] = {}
        self.pending_orders: Dict[str, List] = {}
        self.position_locks: Dict[str, asyncio.Lock] = {}
        
    async def can_open_position(self, symbol: str, side: str) -> bool:
        """Check if we can open a position for symbol"""
        
        # Get or create lock for symbol
        if symbol not in self.position_locks:
            self.position_locks[symbol] = asyncio.Lock()
            
        async with self.position_locks[symbol]:
            # Check active positions
            if symbol in self.active_positions:
                existing = self.active_positions[symbol]
                
                # Same direction - allow pyramid (if configured)
                if existing['side'] == side:
                    logger.info(f"Position already exists for {symbol} ({side})")
                    return False
                    
                # Opposite direction - need to close first
                logger.warning(f"Opposite position exists for {symbol}")
                return False
                
            # Check pending orders
            if symbol in self.pending_orders:
                pending = self.pending_orders[symbol]
                if any(o['side'] == side for o in pending):
                    logger.warning(f"Pending order already exists for {symbol} ({side})")
                    return False
                    
            return True
            
    def register_position(self, symbol: str, position_data: Dict):
        """Register an active position"""
        self.active_positions[symbol] = {
            'side': position_data['side'],
            'size': position_data['size'],
            'entry_price': position_data['entry_price'],
            'timestamp': datetime.now()
        }
        logger.info(f"Registered position for {symbol}: {position_data['side']}")
        
    def remove_position(self, symbol: str):
        """Remove position when closed"""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            logger.info(f"Removed position for {symbol}")


class MLDataValidator:
    """
    Validates ML training data and predictions
    """
    
    def __init__(self):
        self.min_samples = 50
        self.max_prediction_age = timedelta(minutes=5)
        self.confidence_threshold = 0.6
        
    def validate_training_data(self, data: List[Dict]) -> bool:
        """Validate ML training data quality"""
        
        if len(data) < self.min_samples:
            logger.warning(f"Insufficient training data: {len(data)} < {self.min_samples}")
            return False
            
        # Check for required fields
        required_fields = ['zone_type', 'volume_ratio', 'departure_strength', 'outcome']
        for sample in data:
            if not all(field in sample for field in required_fields):
                logger.error("Missing required fields in training data")
                return False
                
        # Check data distribution
        outcomes = [s['outcome'] for s in data]
        win_rate = sum(outcomes) / len(outcomes)
        
        if win_rate < 0.1 or win_rate > 0.9:
            logger.warning(f"Suspicious win rate in training data: {win_rate:.2%}")
            return False
            
        return True
        
    def validate_prediction(self, prediction: Dict) -> bool:
        """Validate ML prediction before using"""
        
        # Check confidence
        confidence = prediction.get('confidence', 0)
        if confidence < self.confidence_threshold:
            logger.info(f"Low confidence prediction: {confidence:.2%}")
            return False
            
        # Check prediction age
        timestamp = prediction.get('timestamp')
        if timestamp:
            age = datetime.now() - timestamp
            if age > self.max_prediction_age:
                logger.warning(f"Stale prediction: {age.total_seconds()}s old")
                return False
                
        # Sanity check values
        expected_profit = prediction.get('expected_profit', 0)
        if expected_profit < -50 or expected_profit > 100:
            logger.error(f"Invalid expected profit: {expected_profit}%")
            return False
            
        return True


class DatabaseConnectionPool:
    """
    Manages database connections with retry and pooling
    """
    
    def __init__(self, max_connections: int = 20):
        self.max_connections = max_connections
        self.connection_semaphore = asyncio.Semaphore(max_connections)
        self.retry_attempts = 3
        self.retry_delay = 1
        
    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute database operation with retry logic"""
        
        for attempt in range(self.retry_attempts):
            try:
                async with self.connection_semaphore:
                    result = await func(*args, **kwargs)
                    return result
                    
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    logger.warning(f"Database error (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"Database operation failed after {self.retry_attempts} attempts")
                    raise


class WebSocketReconnectManager:
    """
    Manages WebSocket connections with auto-reconnect
    """
    
    def __init__(self):
        self.reconnect_delay = 5
        self.max_reconnect_delay = 300
        self.connection_health = {}
        
    async def maintain_connection(self, ws_client, identifier: str):
        """Maintain WebSocket connection with auto-reconnect"""
        
        reconnect_count = 0
        
        while True:
            try:
                # Check connection health
                if not self.is_healthy(identifier):
                    logger.warning(f"WebSocket {identifier} unhealthy, reconnecting...")
                    await self.reconnect(ws_client, identifier)
                    
                await asyncio.sleep(30)  # Health check interval
                
            except Exception as e:
                logger.error(f"WebSocket {identifier} error: {e}")
                
                # Exponential backoff
                delay = min(self.reconnect_delay * (2 ** reconnect_count), 
                          self.max_reconnect_delay)
                await asyncio.sleep(delay)
                
                await self.reconnect(ws_client, identifier)
                reconnect_count += 1
                
    def is_healthy(self, identifier: str) -> bool:
        """Check if connection is healthy"""
        
        health = self.connection_health.get(identifier, {})
        
        # Check last message time
        last_message = health.get('last_message')
        if last_message:
            age = (datetime.now() - last_message).total_seconds()
            if age > 60:  # No message for 60 seconds
                return False
                
        # Check error count
        error_count = health.get('error_count', 0)
        if error_count > 5:
            return False
            
        return True
        
    async def reconnect(self, ws_client, identifier: str):
        """Reconnect WebSocket"""
        
        try:
            # Close existing connection
            if hasattr(ws_client, 'exit'):
                ws_client.exit()
                
            # Wait before reconnecting
            await asyncio.sleep(self.reconnect_delay)
            
            # Reconnect
            await ws_client.connect()
            
            # Reset health metrics
            self.connection_health[identifier] = {
                'last_message': datetime.now(),
                'error_count': 0
            }
            
            logger.info(f"WebSocket {identifier} reconnected successfully")
            
        except Exception as e:
            logger.error(f"Failed to reconnect {identifier}: {e}")
            raise


class OrderValidation:
    """
    Validates orders before submission
    """
    
    def __init__(self, instruments: Dict):
        self.instruments = instruments
        
    def validate_order(self, order: Dict) -> tuple[bool, str]:
        """Validate order parameters"""
        
        symbol = order.get('symbol')
        if not symbol or symbol not in self.instruments:
            return False, f"Invalid symbol: {symbol}"
            
        instrument = self.instruments[symbol]
        
        # Check quantity
        qty = float(order.get('qty', 0))
        qty_step = float(instrument['qty_step'])
        
        if qty <= 0:
            return False, "Quantity must be positive"
            
        # Check if quantity is valid step
        if qty % qty_step != 0:
            return False, f"Quantity must be multiple of {qty_step}"
            
        # Check minimum notional
        price = float(order.get('price', 0))
        if price > 0:
            notional = qty * price
            min_notional = float(instrument.get('min_notional', 0))
            if notional < min_notional:
                return False, f"Order value {notional} below minimum {min_notional}"
                
        # Check leverage
        leverage = order.get('leverage')
        if leverage:
            max_leverage = float(instrument['max_leverage'])
            min_leverage = float(instrument['min_leverage'])
            
            if leverage < min_leverage or leverage > max_leverage:
                return False, f"Leverage must be between {min_leverage} and {max_leverage}"
                
        return True, "Valid"


class SystemHealthMonitor:
    """
    Monitors system health and performance
    """
    
    def __init__(self):
        self.metrics = {
            'api_latency': deque(maxlen=100),
            'order_success_rate': deque(maxlen=100),
            'position_sync_errors': 0,
            'ml_prediction_accuracy': deque(maxlen=50)
        }
        self.alert_thresholds = {
            'api_latency': 2000,  # ms
            'order_success_rate': 0.8,
            'position_sync_errors': 5
        }
        
    def record_api_latency(self, latency_ms: float):
        """Record API latency"""
        self.metrics['api_latency'].append(latency_ms)
        
        # Check threshold
        avg_latency = np.mean(self.metrics['api_latency'])
        if avg_latency > self.alert_thresholds['api_latency']:
            logger.warning(f"High API latency: {avg_latency:.0f}ms")
            
    def record_order_result(self, success: bool):
        """Record order execution result"""
        self.metrics['order_success_rate'].append(1 if success else 0)
        
        # Check success rate
        if len(self.metrics['order_success_rate']) >= 10:
            success_rate = np.mean(self.metrics['order_success_rate'])
            if success_rate < self.alert_thresholds['order_success_rate']:
                logger.error(f"Low order success rate: {success_rate:.1%}")
                
    def check_system_health(self) -> Dict:
        """Get overall system health status"""
        
        health_status = {
            'healthy': True,
            'warnings': [],
            'errors': []
        }
        
        # Check API latency
        if self.metrics['api_latency']:
            avg_latency = np.mean(self.metrics['api_latency'])
            if avg_latency > self.alert_thresholds['api_latency']:
                health_status['warnings'].append(f"High API latency: {avg_latency:.0f}ms")
                
        # Check order success rate
        if len(self.metrics['order_success_rate']) >= 10:
            success_rate = np.mean(self.metrics['order_success_rate'])
            if success_rate < self.alert_thresholds['order_success_rate']:
                health_status['errors'].append(f"Low order success rate: {success_rate:.1%}")
                health_status['healthy'] = False
                
        # Check position sync errors
        if self.metrics['position_sync_errors'] > self.alert_thresholds['position_sync_errors']:
            health_status['errors'].append(f"Position sync errors: {self.metrics['position_sync_errors']}")
            health_status['healthy'] = False
            
        return health_status


# Global instances
rate_limiter = RateLimitManager()
position_safety = PositionSafetyManager()
ml_validator = MLDataValidator()
db_pool = DatabaseConnectionPool()
ws_manager = WebSocketReconnectManager()
health_monitor = SystemHealthMonitor()