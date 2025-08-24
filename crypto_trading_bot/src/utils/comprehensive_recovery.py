"""
Comprehensive Error Recovery System
Ensures bot resilience and automatic recovery from failures
"""
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import structlog
import traceback
from dataclasses import dataclass, field
from enum import Enum

logger = structlog.get_logger(__name__)


class RecoveryAction(Enum):
    """Types of recovery actions"""
    RETRY = "retry"
    RECONNECT = "reconnect"
    RESTART = "restart"
    SKIP = "skip"
    ALERT = "alert"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class ErrorContext:
    """Context for an error occurrence"""
    error_type: str
    error_message: str
    component: str
    timestamp: datetime
    stack_trace: str
    retry_count: int = 0
    recovery_attempts: List[str] = field(default_factory=list)


class ComprehensiveRecoveryManager:
    """
    Manages comprehensive error recovery for all bot components
    """
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, Callable] = {}
        self.component_health: Dict[str, bool] = {}
        self.max_retries = 3
        self.retry_delays = [1, 5, 15]  # Exponential backoff
        self.emergency_stop_threshold = 10  # Errors in 1 minute
        self.circuit_breakers: Dict[str, Dict] = {}
        
    async def handle_error(
        self,
        error: Exception,
        component: str,
        context: Dict[str, Any] = None
    ) -> RecoveryAction:
        """
        Main error handler with intelligent recovery decision
        """
        try:
            # Create error context
            error_ctx = ErrorContext(
                error_type=type(error).__name__,
                error_message=str(error),
                component=component,
                timestamp=datetime.now(),
                stack_trace=traceback.format_exc()
            )
            
            self.error_history.append(error_ctx)
            
            # Check for emergency stop condition
            if self._should_emergency_stop():
                logger.critical(f"Emergency stop triggered: too many errors")
                return RecoveryAction.EMERGENCY_STOP
            
            # Determine recovery action based on error type
            action = self._determine_recovery_action(error, component)
            
            # Execute recovery
            success = await self._execute_recovery(action, component, error_ctx)
            
            if not success and error_ctx.retry_count < self.max_retries:
                # Retry with backoff
                delay = self.retry_delays[min(error_ctx.retry_count, len(self.retry_delays) - 1)]
                await asyncio.sleep(delay)
                error_ctx.retry_count += 1
                return await self.handle_error(error, component, context)
            
            return action
            
        except Exception as e:
            logger.critical(f"Recovery system failed: {e}")
            return RecoveryAction.EMERGENCY_STOP
    
    def _determine_recovery_action(self, error: Exception, component: str) -> RecoveryAction:
        """
        Determine appropriate recovery action based on error type
        """
        error_type = type(error).__name__
        
        # Connection errors - reconnect
        if any(x in str(error).lower() for x in ['connection', 'timeout', 'refused']):
            return RecoveryAction.RECONNECT
        
        # Rate limit errors - wait and retry
        if any(x in str(error).lower() for x in ['rate limit', '429', 'too many']):
            return RecoveryAction.RETRY
        
        # Database errors - restart connection
        if 'database' in component.lower() or 'db' in component.lower():
            return RecoveryAction.RESTART
        
        # WebSocket errors - reconnect
        if 'websocket' in component.lower() or 'ws' in component.lower():
            return RecoveryAction.RECONNECT
        
        # Critical errors - alert and stop
        if any(x in error_type.lower() for x in ['critical', 'fatal']):
            return RecoveryAction.EMERGENCY_STOP
        
        # Default - retry
        return RecoveryAction.RETRY
    
    async def _execute_recovery(
        self,
        action: RecoveryAction,
        component: str,
        error_ctx: ErrorContext
    ) -> bool:
        """
        Execute the recovery action
        """
        try:
            logger.info(f"Executing {action.value} recovery for {component}")
            
            if action == RecoveryAction.RETRY:
                # Simple retry after delay
                return True
            
            elif action == RecoveryAction.RECONNECT:
                # Reconnect component
                if component in self.recovery_strategies:
                    return await self.recovery_strategies[component]()
                return False
            
            elif action == RecoveryAction.RESTART:
                # Restart component
                logger.info(f"Restarting {component}")
                if f"{component}_restart" in self.recovery_strategies:
                    return await self.recovery_strategies[f"{component}_restart"]()
                return False
            
            elif action == RecoveryAction.SKIP:
                # Skip and continue
                logger.warning(f"Skipping error in {component}")
                return True
            
            elif action == RecoveryAction.ALERT:
                # Send alert
                logger.error(f"Alert sent for {component} error")
                return True
            
            elif action == RecoveryAction.EMERGENCY_STOP:
                # Emergency stop
                logger.critical(f"Emergency stop for {component}")
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            return False
    
    def _should_emergency_stop(self) -> bool:
        """
        Check if we should trigger emergency stop
        """
        recent_errors = [
            e for e in self.error_history
            if (datetime.now() - e.timestamp).total_seconds() < 60
        ]
        return len(recent_errors) >= self.emergency_stop_threshold
    
    def register_recovery_strategy(self, component: str, strategy: Callable):
        """
        Register a recovery strategy for a component
        """
        self.recovery_strategies[component] = strategy
    
    def is_circuit_open(self, component: str) -> bool:
        """
        Check if circuit breaker is open for component
        """
        if component not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[component]
        if breaker['state'] == 'open':
            # Check if cooldown period has passed
            if (datetime.now() - breaker['opened_at']).total_seconds() > breaker['cooldown']:
                # Try to close circuit
                breaker['state'] = 'half_open'
                return False
            return True
        return False
    
    def open_circuit(self, component: str, cooldown: int = 60):
        """
        Open circuit breaker for component
        """
        self.circuit_breakers[component] = {
            'state': 'open',
            'opened_at': datetime.now(),
            'cooldown': cooldown
        }
        logger.warning(f"Circuit breaker opened for {component} (cooldown: {cooldown}s)")
    
    def close_circuit(self, component: str):
        """
        Close circuit breaker for component
        """
        if component in self.circuit_breakers:
            self.circuit_breakers[component]['state'] = 'closed'
            logger.info(f"Circuit breaker closed for {component}")
    
    def get_health_status(self) -> Dict:
        """
        Get overall health status
        """
        recent_errors = [
            e for e in self.error_history
            if (datetime.now() - e.timestamp).total_seconds() < 300
        ]
        
        error_rate = len(recent_errors) / 5  # Errors per minute
        
        return {
            'healthy': error_rate < 1,
            'error_rate': error_rate,
            'recent_errors': len(recent_errors),
            'circuit_breakers': {
                k: v['state'] for k, v in self.circuit_breakers.items()
            },
            'component_health': self.component_health
        }
    
    async def periodic_health_check(self):
        """
        Periodic health check and recovery
        """
        while True:
            try:
                # Check component health
                for component, healthy in self.component_health.items():
                    if not healthy and component in self.recovery_strategies:
                        logger.info(f"Attempting recovery for unhealthy {component}")
                        success = await self.recovery_strategies[component]()
                        if success:
                            self.component_health[component] = True
                
                # Clean old error history
                cutoff = datetime.now() - timedelta(hours=1)
                self.error_history = [
                    e for e in self.error_history
                    if e.timestamp > cutoff
                ]
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(60)


# Global instance
recovery_manager = ComprehensiveRecoveryManager()


# Decorator for automatic error recovery
def with_recovery(component: str):
    """
    Decorator to add automatic error recovery to async functions
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in {component}: {e} (attempt {attempt + 1}/{max_attempts})")
                    
                    if attempt < max_attempts - 1:
                        action = await recovery_manager.handle_error(e, component)
                        if action == RecoveryAction.EMERGENCY_STOP:
                            raise
                        
                        # Wait before retry
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise
            
        return wrapper
    return decorator