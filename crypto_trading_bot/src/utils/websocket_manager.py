"""
Robust WebSocket Manager with auto-reconnect and health monitoring
Implements best practices for 2024: ping/pong, exponential backoff, multi-connection
"""
import asyncio
import json
import time
from typing import Dict, Optional, Callable, Any, List
from datetime import datetime, timedelta
from enum import Enum
import structlog
from pybit.unified_trading import WebSocket
import websockets
from collections import deque

logger = structlog.get_logger(__name__)

class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"

class WebSocketManager:
    """
    Production-grade WebSocket manager with:
    - Automatic reconnection with exponential backoff
    - Ping/pong heartbeat monitoring
    - Multiple connection management
    - Connection health tracking
    - Message queue with overflow protection
    """
    
    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        
        # Connection management
        self.connections: Dict[str, Dict] = {}
        self.connection_states: Dict[str, ConnectionState] = {}
        self.reconnect_attempts: Dict[str, int] = {}
        
        # Configuration
        self.ping_interval = 30  # seconds
        self.pong_timeout = 10  # seconds
        self.max_reconnect_attempts = 10
        self.initial_reconnect_delay = 2  # seconds
        self.max_reconnect_delay = 300  # 5 minutes
        
        # Message handling
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.message_queues: Dict[str, deque] = {}
        self.max_queue_size = 10000
        
        # Health monitoring
        self.connection_health: Dict[str, Dict] = {}
        self.last_ping_times: Dict[str, datetime] = {}
        self.last_pong_times: Dict[str, datetime] = {}
        self.message_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        
        # Tasks
        self.monitor_tasks: Dict[str, asyncio.Task] = {}
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}
        
    async def create_connection(
        self,
        connection_id: str,
        channel_type: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        topics: List[str] = None
    ) -> bool:
        """
        Create a new WebSocket connection with monitoring
        
        Args:
            connection_id: Unique identifier for connection
            channel_type: 'public', 'private', or 'trade'
            api_key: API key for private connections
            api_secret: API secret for private connections
            topics: List of topics to subscribe
        """
        try:
            logger.info(f"Creating WebSocket connection: {connection_id} ({channel_type})")
            
            # Set initial state
            self.connection_states[connection_id] = ConnectionState.CONNECTING
            self.reconnect_attempts[connection_id] = 0
            
            # Initialize message queue
            self.message_queues[connection_id] = deque(maxlen=self.max_queue_size)
            
            # Create WebSocket client based on type
            if channel_type == 'public':
                ws_client = WebSocket(
                    testnet=self.testnet,
                    channel_type="linear"
                )
            elif channel_type == 'private':
                if not api_key or not api_secret:
                    raise ValueError("API credentials required for private connection")
                ws_client = WebSocket(
                    testnet=self.testnet,
                    channel_type="private",
                    api_key=api_key,
                    api_secret=api_secret
                )
            elif channel_type == 'trade':
                if not api_key or not api_secret:
                    raise ValueError("API credentials required for trade connection")
                ws_client = WebSocket(
                    testnet=self.testnet,
                    channel_type="spot",  # For order execution
                    api_key=api_key,
                    api_secret=api_secret
                )
            else:
                raise ValueError(f"Invalid channel type: {channel_type}")
            
            # Store connection info
            self.connections[connection_id] = {
                'client': ws_client,
                'channel_type': channel_type,
                'api_key': api_key,
                'api_secret': api_secret,
                'topics': topics or [],
                'created_at': datetime.now()
            }
            
            # Set up message handler
            ws_client.order_stream = lambda msg: asyncio.create_task(
                self._handle_message(connection_id, msg)
            )
            
            # Connect
            await self._connect(connection_id)
            
            # Subscribe to topics
            if topics:
                await self._subscribe_topics(connection_id, topics)
            
            # Start monitoring tasks
            self.monitor_tasks[connection_id] = asyncio.create_task(
                self._monitor_connection(connection_id)
            )
            self.heartbeat_tasks[connection_id] = asyncio.create_task(
                self._heartbeat_loop(connection_id)
            )
            
            # Update state
            self.connection_states[connection_id] = ConnectionState.CONNECTED
            self._update_health(connection_id, 'connected', True)
            
            logger.info(f"WebSocket connection established: {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create connection {connection_id}: {e}")
            self.connection_states[connection_id] = ConnectionState.FAILED
            self.error_counts[connection_id] = self.error_counts.get(connection_id, 0) + 1
            return False
    
    async def _connect(self, connection_id: str):
        """Establish WebSocket connection"""
        conn_info = self.connections.get(connection_id)
        if not conn_info:
            return
        
        ws_client = conn_info['client']
        
        # PyBit WebSocket doesn't have async connect, it auto-connects
        # We'll verify connection through ping
        self.last_ping_times[connection_id] = datetime.now()
        
    async def _monitor_connection(self, connection_id: str):
        """
        Monitor connection health and trigger reconnection if needed
        """
        while connection_id in self.connections:
            try:
                # Check connection state
                state = self.connection_states.get(connection_id)
                
                if state == ConnectionState.CONNECTED:
                    # Check if connection is healthy
                    if not self._is_connection_healthy(connection_id):
                        logger.warning(f"Connection {connection_id} unhealthy, reconnecting...")
                        await self._reconnect(connection_id)
                
                elif state == ConnectionState.FAILED:
                    # Check if we should retry
                    if self.reconnect_attempts[connection_id] < self.max_reconnect_attempts:
                        await asyncio.sleep(self._get_reconnect_delay(connection_id))
                        await self._reconnect(connection_id)
                    else:
                        logger.error(f"Connection {connection_id} failed after max attempts")
                        break
                
                # Check message queue size
                queue_size = len(self.message_queues.get(connection_id, []))
                if queue_size > self.max_queue_size * 0.9:
                    logger.warning(f"Message queue near capacity for {connection_id}: {queue_size}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitor error for {connection_id}: {e}")
                await asyncio.sleep(5)
    
    async def _heartbeat_loop(self, connection_id: str):
        """
        Send periodic ping messages to keep connection alive
        """
        while connection_id in self.connections:
            try:
                state = self.connection_states.get(connection_id)
                
                if state == ConnectionState.CONNECTED:
                    # Send ping
                    await self._send_ping(connection_id)
                    
                    # Wait for pong
                    await asyncio.sleep(self.pong_timeout)
                    
                    # Check if pong received
                    last_pong = self.last_pong_times.get(connection_id)
                    last_ping = self.last_ping_times.get(connection_id)
                    
                    if last_pong and last_ping:
                        if (datetime.now() - last_pong).total_seconds() > self.pong_timeout:
                            logger.warning(f"No pong received for {connection_id}")
                            self._update_health(connection_id, 'pong_timeout', False)
                
                await asyncio.sleep(self.ping_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat error for {connection_id}: {e}")
                await asyncio.sleep(self.ping_interval)
    
    async def _send_ping(self, connection_id: str):
        """Send ping message"""
        try:
            conn_info = self.connections.get(connection_id)
            if conn_info:
                # For PyBit, we can send a custom ping or use built-in
                # Most exchanges respond to {"op": "ping"}
                ws_client = conn_info['client']
                
                # Record ping time
                self.last_ping_times[connection_id] = datetime.now()
                
                # PyBit handles ping internally, but we track timing
                self._update_health(connection_id, 'ping_sent', True)
                
        except Exception as e:
            logger.error(f"Failed to send ping to {connection_id}: {e}")
            self.error_counts[connection_id] = self.error_counts.get(connection_id, 0) + 1
    
    async def _handle_message(self, connection_id: str, message: Dict):
        """
        Handle incoming WebSocket message
        """
        try:
            # Update message count
            self.message_counts[connection_id] = self.message_counts.get(connection_id, 0) + 1
            
            # Check for pong
            if message.get('op') == 'pong' or message.get('ret_msg') == 'pong':
                self.last_pong_times[connection_id] = datetime.now()
                self._update_health(connection_id, 'pong_received', True)
                return
            
            # Add to queue
            queue = self.message_queues.get(connection_id, deque(maxlen=self.max_queue_size))
            queue.append({
                'timestamp': datetime.now(),
                'message': message
            })
            
            # Update health
            self._update_health(connection_id, 'last_message', datetime.now())
            
            # Call registered handlers
            handlers = self.message_handlers.get(connection_id, [])
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Handler error for {connection_id}: {e}")
            
        except Exception as e:
            logger.error(f"Message handling error for {connection_id}: {e}")
            self.error_counts[connection_id] = self.error_counts.get(connection_id, 0) + 1
    
    async def _reconnect(self, connection_id: str):
        """
        Reconnect WebSocket with exponential backoff
        """
        try:
            logger.info(f"Reconnecting {connection_id}...")
            
            # Update state
            self.connection_states[connection_id] = ConnectionState.RECONNECTING
            self.reconnect_attempts[connection_id] += 1
            
            # Get connection info
            conn_info = self.connections.get(connection_id)
            if not conn_info:
                return
            
            # Close existing connection
            try:
                ws_client = conn_info['client']
                if hasattr(ws_client, 'exit'):
                    ws_client.exit()
            except:
                pass
            
            # Wait with exponential backoff
            delay = self._get_reconnect_delay(connection_id)
            logger.info(f"Waiting {delay}s before reconnecting {connection_id}")
            await asyncio.sleep(delay)
            
            # Create new connection
            success = await self.create_connection(
                connection_id=connection_id,
                channel_type=conn_info['channel_type'],
                api_key=conn_info.get('api_key'),
                api_secret=conn_info.get('api_secret'),
                topics=conn_info.get('topics')
            )
            
            if success:
                self.reconnect_attempts[connection_id] = 0
                logger.info(f"Successfully reconnected {connection_id}")
            else:
                self.connection_states[connection_id] = ConnectionState.FAILED
                
        except Exception as e:
            logger.error(f"Reconnection failed for {connection_id}: {e}")
            self.connection_states[connection_id] = ConnectionState.FAILED
    
    def _get_reconnect_delay(self, connection_id: str) -> float:
        """Calculate exponential backoff delay"""
        attempts = self.reconnect_attempts.get(connection_id, 0)
        delay = min(
            self.initial_reconnect_delay * (2 ** attempts),
            self.max_reconnect_delay
        )
        return delay
    
    def _is_connection_healthy(self, connection_id: str) -> bool:
        """Check if connection is healthy"""
        health = self.connection_health.get(connection_id, {})
        
        # Check last message time
        last_message = health.get('last_message')
        if last_message:
            if isinstance(last_message, datetime):
                age = (datetime.now() - last_message).total_seconds()
                if age > 60:  # No message for 60 seconds
                    return False
        
        # Check error rate
        error_count = self.error_counts.get(connection_id, 0)
        message_count = self.message_counts.get(connection_id, 1)
        error_rate = error_count / max(message_count, 1)
        if error_rate > 0.1:  # More than 10% errors
            return False
        
        # Check pong timeout
        if health.get('pong_timeout'):
            return False
        
        return True
    
    def _update_health(self, connection_id: str, key: str, value: Any):
        """Update connection health metrics"""
        if connection_id not in self.connection_health:
            self.connection_health[connection_id] = {}
        self.connection_health[connection_id][key] = value
    
    async def _subscribe_topics(self, connection_id: str, topics: List[str]):
        """Subscribe to WebSocket topics"""
        try:
            conn_info = self.connections.get(connection_id)
            if not conn_info:
                return
            
            ws_client = conn_info['client']
            
            # Subscribe based on channel type
            if conn_info['channel_type'] == 'public':
                # Public topics like orderbook, trades, kline
                for topic in topics:
                    if 'orderbook' in topic:
                        ws_client.orderbook_stream(50, topic.split('.')[-1], self._handle_message)
                    elif 'trade' in topic:
                        ws_client.trade_stream(topic.split('.')[-1], self._handle_message)
                    elif 'kline' in topic:
                        parts = topic.split('.')
                        ws_client.kline_stream(parts[-1], parts[-2], self._handle_message)
            
            elif conn_info['channel_type'] == 'private':
                # Private topics like position, order, execution
                if 'position' in topics:
                    ws_client.position_stream(self._handle_message)
                if 'order' in topics:
                    ws_client.order_stream(self._handle_message)
                if 'execution' in topics:
                    ws_client.execution_stream(self._handle_message)
                if 'wallet' in topics:
                    ws_client.wallet_stream(self._handle_message)
            
            logger.info(f"Subscribed to topics for {connection_id}: {topics}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe topics for {connection_id}: {e}")
    
    def register_handler(self, connection_id: str, handler: Callable):
        """Register a message handler for a connection"""
        if connection_id not in self.message_handlers:
            self.message_handlers[connection_id] = []
        self.message_handlers[connection_id].append(handler)
    
    def get_connection_status(self, connection_id: str) -> Dict:
        """Get detailed connection status"""
        return {
            'state': self.connection_states.get(connection_id, ConnectionState.DISCONNECTED).value,
            'health': self.connection_health.get(connection_id, {}),
            'reconnect_attempts': self.reconnect_attempts.get(connection_id, 0),
            'message_count': self.message_counts.get(connection_id, 0),
            'error_count': self.error_counts.get(connection_id, 0),
            'queue_size': len(self.message_queues.get(connection_id, []))
        }
    
    def get_all_statuses(self) -> Dict[str, Dict]:
        """Get status of all connections"""
        return {
            conn_id: self.get_connection_status(conn_id)
            for conn_id in self.connections
        }
    
    async def close_connection(self, connection_id: str):
        """Close a specific connection"""
        try:
            # Cancel monitoring tasks
            if connection_id in self.monitor_tasks:
                self.monitor_tasks[connection_id].cancel()
            if connection_id in self.heartbeat_tasks:
                self.heartbeat_tasks[connection_id].cancel()
            
            # Close WebSocket
            conn_info = self.connections.get(connection_id)
            if conn_info:
                ws_client = conn_info['client']
                if hasattr(ws_client, 'exit'):
                    ws_client.exit()
            
            # Clean up
            self.connections.pop(connection_id, None)
            self.connection_states.pop(connection_id, None)
            self.message_queues.pop(connection_id, None)
            
            logger.info(f"Closed connection: {connection_id}")
            
        except Exception as e:
            logger.error(f"Error closing connection {connection_id}: {e}")
    
    async def close_all(self):
        """Close all connections"""
        for connection_id in list(self.connections.keys()):
            await self.close_connection(connection_id)

# Global WebSocket manager instance
ws_manager = WebSocketManager()