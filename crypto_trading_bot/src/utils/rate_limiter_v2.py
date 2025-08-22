"""
Enhanced Rate Limiter with sliding window, request queue, and circuit breaker
Implements Bybit V5 API limits: 600 requests per 5 seconds
"""
import asyncio
import time
from typing import Dict, Optional, Any, Callable
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
import structlog
from dataclasses import dataclass
import heapq

logger = structlog.get_logger(__name__)

class Priority(Enum):
    """Request priority levels"""
    CRITICAL = 1  # Order placement/cancellation
    HIGH = 2      # Position management
    MEDIUM = 3    # Market data
    LOW = 4       # Historical data

@dataclass
class Request:
    """Request wrapper with priority"""
    priority: Priority
    timestamp: float
    func: Callable
    args: tuple
    kwargs: dict
    future: asyncio.Future
    
    def __lt__(self, other):
        # For heap comparison - lower priority value = higher priority
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery

class EnhancedRateLimiter:
    """
    Production-grade rate limiter with:
    - Sliding window algorithm
    - Priority queue for requests
    - Circuit breaker pattern
    - Adaptive throttling
    - Multiple rate limit tiers
    """
    
    def __init__(self):
        # Bybit V5 limits
        self.limits = {
            'default': {'requests': 600, 'window': 5},  # 600 req/5s
            'order': {'requests': 100, 'window': 5},    # Order operations
            'websocket': {'requests': 500, 'window': 300}  # 500 connections/5min
        }
        
        # Sliding window tracking
        self.request_times: Dict[str, deque] = {
            'default': deque(),
            'order': deque(),
            'websocket': deque()
        }
        
        # Request queue with priority
        self.request_queue: list = []  # Min heap
        self.queue_lock = asyncio.Lock()
        self.max_queue_size = 1000
        
        # Circuit breaker
        self.circuit_state = CircuitState.CLOSED
        self.circuit_failures = 0
        self.circuit_threshold = 5
        self.circuit_timeout = 60  # seconds
        self.circuit_half_open_requests = 0
        self.circuit_test_limit = 3
        self.last_circuit_open = None
        
        # Adaptive throttling
        self.response_times = deque(maxlen=100)
        self.error_counts = deque(maxlen=100)
        self.adaptive_delay = 0.0
        self.min_delay = 0.01  # 10ms minimum
        self.max_delay = 1.0   # 1s maximum
        
        # Statistics
        self.total_requests = 0
        self.blocked_requests = 0
        self.queued_requests = 0
        self.circuit_trips = 0
        
        # Background processor
        self.processor_task = None
        self.running = False
        
    async def start(self):
        """Start the rate limiter background processor"""
        if not self.running:
            self.running = True
            self.processor_task = asyncio.create_task(self._process_queue())
            logger.info("Rate limiter started")
    
    async def stop(self):
        """Stop the rate limiter"""
        self.running = False
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Rate limiter stopped")
    
    async def acquire(
        self,
        category: str = 'default',
        priority: Priority = Priority.MEDIUM,
        timeout: float = 30.0
    ) -> bool:
        """
        Acquire permission to make a request
        
        Args:
            category: Rate limit category
            priority: Request priority
            timeout: Maximum wait time
            
        Returns:
            True if request can proceed, False if blocked
        """
        self.total_requests += 1
        
        # Check circuit breaker
        if not self._check_circuit():
            self.blocked_requests += 1
            logger.warning("Request blocked by circuit breaker")
            return False
        
        # Check rate limit
        now = time.time()
        if self._can_proceed(category, now):
            self._record_request(category, now)
            
            # Add adaptive delay
            if self.adaptive_delay > 0:
                await asyncio.sleep(self.adaptive_delay)
            
            return True
        
        # Queue the request if under limit
        if len(self.request_queue) >= self.max_queue_size:
            self.blocked_requests += 1
            logger.warning(f"Request queue full ({self.max_queue_size})")
            return False
        
        # Create future for queued request
        future = asyncio.Future()
        request = Request(
            priority=priority,
            timestamp=now,
            func=None,  # Not used for simple acquire
            args=(),
            kwargs={'category': category},
            future=future
        )
        
        async with self.queue_lock:
            heapq.heappush(self.request_queue, request)
            self.queued_requests += 1
        
        # Wait for request to be processed
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Rate limit acquire timeout after {timeout}s")
            return False
    
    async def execute_with_limit(
        self,
        func: Callable,
        *args,
        category: str = 'default',
        priority: Priority = Priority.MEDIUM,
        timeout: float = 30.0,
        **kwargs
    ) -> Any:
        """
        Execute a function with rate limiting
        
        Args:
            func: Async function to execute
            category: Rate limit category
            priority: Request priority
            timeout: Maximum wait time
            
        Returns:
            Function result
        """
        # Acquire rate limit
        if not await self.acquire(category, priority, timeout):
            raise Exception("Rate limit exceeded")
        
        # Execute function with timing
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            
            # Record success
            response_time = time.time() - start_time
            self._record_response(response_time, success=True)
            
            return result
            
        except Exception as e:
            # Record failure
            response_time = time.time() - start_time
            self._record_response(response_time, success=False)
            
            # Check for rate limit errors
            if self._is_rate_limit_error(e):
                self._handle_rate_limit_error()
            
            raise
    
    def _can_proceed(self, category: str, now: float) -> bool:
        """Check if request can proceed under rate limit"""
        limit_config = self.limits.get(category, self.limits['default'])
        window = limit_config['window']
        max_requests = limit_config['requests']
        
        # Clean old requests outside window
        request_times = self.request_times[category]
        cutoff = now - window
        
        while request_times and request_times[0] < cutoff:
            request_times.popleft()
        
        # Check if under limit
        return len(request_times) < max_requests
    
    def _record_request(self, category: str, timestamp: float):
        """Record a request in the sliding window"""
        self.request_times[category].append(timestamp)
    
    async def _process_queue(self):
        """Background task to process queued requests"""
        while self.running:
            try:
                # Check if there are queued requests
                if not self.request_queue:
                    await asyncio.sleep(0.01)  # 10ms check interval
                    continue
                
                # Process highest priority request
                async with self.queue_lock:
                    if not self.request_queue:
                        continue
                    
                    # Peek at next request
                    request = self.request_queue[0]
                    category = request.kwargs.get('category', 'default')
                    
                    # Check if it can proceed
                    now = time.time()
                    if self._can_proceed(category, now):
                        # Remove from queue
                        heapq.heappop(self.request_queue)
                        
                        # Record request
                        self._record_request(category, now)
                        
                        # Complete the future
                        if not request.future.done():
                            request.future.set_result(True)
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(1)
    
    def _check_circuit(self) -> bool:
        """Check circuit breaker state"""
        if self.circuit_state == CircuitState.CLOSED:
            return True
        
        elif self.circuit_state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_circuit_open:
                elapsed = (datetime.now() - self.last_circuit_open).total_seconds()
                if elapsed >= self.circuit_timeout:
                    # Move to half-open state
                    self.circuit_state = CircuitState.HALF_OPEN
                    self.circuit_half_open_requests = 0
                    logger.info("Circuit breaker moved to HALF_OPEN state")
                    return True
            return False
        
        elif self.circuit_state == CircuitState.HALF_OPEN:
            # Allow limited requests for testing
            if self.circuit_half_open_requests < self.circuit_test_limit:
                self.circuit_half_open_requests += 1
                return True
            return False
        
        return False
    
    def _handle_rate_limit_error(self):
        """Handle rate limit error (403/429)"""
        self.circuit_failures += 1
        
        if self.circuit_failures >= self.circuit_threshold:
            # Trip the circuit
            self.circuit_state = CircuitState.OPEN
            self.last_circuit_open = datetime.now()
            self.circuit_trips += 1
            logger.error(f"Circuit breaker OPEN after {self.circuit_failures} failures")
            
            # Increase adaptive delay
            self.adaptive_delay = min(self.adaptive_delay * 2, self.max_delay)
    
    def _record_response(self, response_time: float, success: bool):
        """Record response metrics for adaptive throttling"""
        self.response_times.append(response_time)
        self.error_counts.append(0 if success else 1)
        
        if success and self.circuit_state == CircuitState.HALF_OPEN:
            # Successful request in half-open state
            self.circuit_failures = 0
            self.circuit_state = CircuitState.CLOSED
            logger.info("Circuit breaker CLOSED after successful test")
        
        # Adjust adaptive delay based on performance
        if len(self.response_times) >= 10:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            error_rate = sum(self.error_counts) / len(self.error_counts)
            
            if error_rate > 0.1:  # More than 10% errors
                # Increase delay
                self.adaptive_delay = min(
                    self.adaptive_delay + 0.01,
                    self.max_delay
                )
            elif error_rate < 0.01 and avg_response_time < 0.2:  # Good performance
                # Decrease delay
                self.adaptive_delay = max(
                    self.adaptive_delay - 0.001,
                    self.min_delay
                )
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is rate limit related"""
        error_str = str(error).lower()
        return any(indicator in error_str for indicator in [
            '403', '429', 'rate limit', 'too many requests',
            'too frequent', 'exceeded', 'throttl'
        ])
    
    def reset(self):
        """Reset rate limiter state"""
        for queue in self.request_times.values():
            queue.clear()
        self.request_queue.clear()
        self.circuit_state = CircuitState.CLOSED
        self.circuit_failures = 0
        self.adaptive_delay = self.min_delay
        logger.info("Rate limiter reset")
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        return {
            'total_requests': self.total_requests,
            'blocked_requests': self.blocked_requests,
            'queued_requests': self.queued_requests,
            'queue_size': len(self.request_queue),
            'circuit_state': self.circuit_state.value,
            'circuit_trips': self.circuit_trips,
            'adaptive_delay': self.adaptive_delay,
            'windows': {
                category: len(times)
                for category, times in self.request_times.items()
            }
        }
    
    async def wait_for_capacity(self, category: str = 'default') -> float:
        """
        Calculate and wait for rate limit capacity
        
        Returns:
            Wait time in seconds
        """
        limit_config = self.limits.get(category, self.limits['default'])
        window = limit_config['window']
        max_requests = limit_config['requests']
        
        request_times = self.request_times[category]
        
        if len(request_times) >= max_requests:
            # Calculate wait time
            oldest_request = request_times[0]
            wait_time = (oldest_request + window) - time.time()
            
            if wait_time > 0:
                logger.info(f"Waiting {wait_time:.2f}s for rate limit capacity")
                await asyncio.sleep(wait_time)
                return wait_time
        
        return 0.0

# Global rate limiter instance
rate_limiter_v2 = EnhancedRateLimiter()