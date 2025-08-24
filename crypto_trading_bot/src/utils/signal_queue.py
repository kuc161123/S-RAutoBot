"""
Redis-backed Signal Queue
Ensures reliable signal delivery between components
"""
import json
import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime
from dataclasses import asdict
import structlog
from .redis_manager import redis_manager

logger = structlog.get_logger(__name__)


class SignalQueue:
    """
    Persistent signal queue using Redis
    Ensures signals are not lost between generator and executor
    """
    
    def __init__(self):
        self.redis_client = None
        self.queue_key = "trading:signal_queue"
        self.processing_key = "trading:signal_processing"
        self.completed_key = "trading:signal_completed"
        self.failed_key = "trading:signal_failed"
        self.signal_ttl = 300  # 5 minutes TTL for signals
        self.in_memory_queue = None
        
    async def connect(self, redis_url: str = None):
        """
        Connect to Redis using shared manager
        """
        if self.redis_client:
            return
        
        # Get client from shared manager
        self.redis_client = await redis_manager.get_client(redis_url)
        
        if not self.redis_client:
            logger.warning("Redis not available - using in-memory queue")
            self.in_memory_queue = asyncio.Queue()
        else:
            logger.info("Signal queue connected to Redis via shared manager")
    
    async def push(self, signal: Any) -> bool:
        """
        Push signal to queue
        
        Args:
            signal: Signal (dict or dataclass) to queue
        
        Returns:
            Success status
        """
        try:
            # Convert dataclass to dict if needed
            if hasattr(signal, '__dataclass_fields__'):
                signal_dict = asdict(signal)
            elif isinstance(signal, dict):
                signal_dict = signal.copy()
            else:
                signal_dict = dict(signal)
            
            # Add metadata
            signal_dict['queued_at'] = datetime.utcnow().isoformat()
            if 'signal_id' not in signal_dict:
                signal_dict['signal_id'] = f"{signal_dict.get('symbol')}_{datetime.utcnow().timestamp()}"
            
            if self.redis_client:
                # Push to Redis queue
                signal_json = json.dumps(signal_dict)
                await self.redis_client.lpush(self.queue_key, signal_json)
                
                # Set expiry on the queue
                await self.redis_client.expire(self.queue_key, self.signal_ttl)
                
                logger.info(f"Signal queued for {signal_dict.get('symbol')}: {signal_dict.get('signal_id')}")
                return True
            else:
                # Use in-memory queue
                await self.in_memory_queue.put(signal_dict)
                logger.info(f"Signal queued in memory for {signal_dict.get('symbol')}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to queue signal: {e}")
            return False
    
    async def pop(self, timeout: int = 1) -> Optional[Dict[str, Any]]:
        """
        Pop signal from queue (blocking)
        
        Args:
            timeout: Blocking timeout in seconds
        
        Returns:
            Signal dictionary or None
        """
        try:
            if self.redis_client:
                # Blocking pop from Redis with timeout
                result = await self.redis_client.brpop(self.queue_key, timeout=timeout)
                
                if result:
                    _, signal_json = result
                    signal = json.loads(signal_json)
                    
                    # Move to processing set
                    await self.redis_client.sadd(self.processing_key, signal['signal_id'])
                    
                    logger.info(f"Signal popped: {signal.get('signal_id')}")
                    return signal
                return None
            else:
                # Use in-memory queue
                try:
                    signal = await asyncio.wait_for(
                        self.in_memory_queue.get(),
                        timeout=timeout
                    )
                    logger.info(f"Signal popped from memory: {signal.get('symbol')}")
                    return signal
                except asyncio.TimeoutError:
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to pop signal: {e}")
            return None
    
    async def mark_completed(self, signal_id: str, result: Dict = None):
        """
        Mark signal as completed
        
        Args:
            signal_id: Signal identifier
            result: Execution result
        """
        try:
            if self.redis_client:
                # Remove from processing
                await self.redis_client.srem(self.processing_key, signal_id)
                
                # Add to completed with result
                completed_data = {
                    'signal_id': signal_id,
                    'completed_at': datetime.utcnow().isoformat(),
                    'result': result
                }
                await self.redis_client.hset(
                    self.completed_key,
                    signal_id,
                    json.dumps(completed_data)
                )
                
                # Expire completed signals after 1 hour
                await self.redis_client.expire(self.completed_key, 3600)
                
            logger.info(f"Signal completed: {signal_id}")
            
        except Exception as e:
            logger.error(f"Failed to mark signal completed: {e}")
    
    async def mark_failed(self, signal_id: str, error: str):
        """
        Mark signal as failed
        
        Args:
            signal_id: Signal identifier
            error: Error message
        """
        try:
            if self.redis_client:
                # Remove from processing
                await self.redis_client.srem(self.processing_key, signal_id)
                
                # Add to failed with error
                failed_data = {
                    'signal_id': signal_id,
                    'failed_at': datetime.utcnow().isoformat(),
                    'error': error
                }
                await self.redis_client.hset(
                    self.failed_key,
                    signal_id,
                    json.dumps(failed_data)
                )
                
                # Expire failed signals after 1 hour
                await self.redis_client.expire(self.failed_key, 3600)
                
            logger.error(f"Signal failed: {signal_id} - {error}")
            
        except Exception as e:
            logger.error(f"Failed to mark signal as failed: {e}")
    
    async def get_queue_size(self) -> int:
        """
        Get current queue size
        
        Returns:
            Number of signals in queue
        """
        try:
            if self.redis_client:
                return await self.redis_client.llen(self.queue_key)
            else:
                return self.in_memory_queue.qsize() if hasattr(self, 'in_memory_queue') else 0
                
        except Exception as e:
            logger.error(f"Failed to get queue size: {e}")
            return 0
    
    async def get_processing_count(self) -> int:
        """
        Get number of signals currently being processed
        
        Returns:
            Number of signals in processing
        """
        try:
            if self.redis_client:
                return await self.redis_client.scard(self.processing_key)
            return 0
            
        except Exception as e:
            logger.error(f"Failed to get processing count: {e}")
            return 0
    
    async def clear_queue(self):
        """
        Clear all signals from queue (use with caution)
        """
        try:
            if self.redis_client:
                await self.redis_client.delete(self.queue_key)
            elif hasattr(self, 'in_memory_queue'):
                while not self.in_memory_queue.empty():
                    try:
                        self.in_memory_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            
            logger.warning("Signal queue cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear queue: {e}")
    
    async def recover_processing(self):
        """
        Recover signals that were being processed (e.g., after crash)
        Move them back to queue
        """
        try:
            if self.redis_client:
                processing = await self.redis_client.smembers(self.processing_key)
                
                for signal_id in processing:
                    # Could retrieve original signal from a backup store
                    # For now, just log
                    logger.warning(f"Found orphaned processing signal: {signal_id}")
                    
                # Clear processing set
                await self.redis_client.delete(self.processing_key)
                
        except Exception as e:
            logger.error(f"Failed to recover processing signals: {e}")
    
    async def get_stats(self) -> Dict:
        """
        Get queue statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            stats = {
                'queue_size': await self.get_queue_size(),
                'processing': await self.get_processing_count(),
                'completed': 0,
                'failed': 0
            }
            
            if self.redis_client:
                stats['completed'] = await self.redis_client.hlen(self.completed_key)
                stats['failed'] = await self.redis_client.hlen(self.failed_key)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {'error': str(e)}


# Global instance
signal_queue = SignalQueue()