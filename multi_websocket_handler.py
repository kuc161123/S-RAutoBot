"""
Multi-WebSocket Handler for Bybit
Handles >200 symbol subscriptions by splitting across multiple connections
"""
import asyncio
import json
import logging
import websockets
from typing import List, AsyncGenerator, Tuple

logger = logging.getLogger(__name__)

class MultiWebSocketHandler:
    """Handles multiple WebSocket connections for large symbol lists"""
    
    MAX_SUBS_PER_CONNECTION = 190  # Keep under 200 limit with buffer
    
    def __init__(self, ws_url: str, running_flag):
        self.ws_url = ws_url
        self.running = running_flag
        self.connections = []
        
    async def multi_kline_stream(self, topics: List[str]) -> AsyncGenerator[Tuple[str, dict], None]:
        """
        Stream klines from multiple WebSocket connections
        Splits topics across connections if >190 topics
        """
        # Split topics into chunks
        topic_chunks = []
        for i in range(0, len(topics), self.MAX_SUBS_PER_CONNECTION):
            chunk = topics[i:i + self.MAX_SUBS_PER_CONNECTION]
            topic_chunks.append(chunk)
        
        logger.info(f"Splitting {len(topics)} topics across {len(topic_chunks)} WebSocket connections")
        
        # Create queues for each connection
        queues = [asyncio.Queue() for _ in topic_chunks]
        
        # Start connection tasks
        tasks = []
        for idx, (chunk, queue) in enumerate(zip(topic_chunks, queues)):
            task = asyncio.create_task(
                self._single_connection_handler(chunk, queue, idx + 1)
            )
            tasks.append(task)
        
        try:
            # Merge streams from all connections
            while self.running:
                # Check all queues for data
                for queue_idx, queue in enumerate(queues):
                    try:
                        # Non-blocking get
                        sym, kline = await asyncio.wait_for(queue.get(), timeout=0.01)
                        yield sym, kline
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error reading from queue {queue_idx}: {e}")
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.001)
                
        finally:
            # Cancel all connection tasks
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _single_connection_handler(self, topics: List[str], queue: asyncio.Queue, conn_id: int):
        """Handle a single WebSocket connection with its subset of topics"""
        sub = {"op": "subscribe", "args": [f"kline.{t}" for t in topics]}
        
        while self.running:
            try:
                logger.info(f"[WS-{conn_id}] Connecting with {len(topics)} topics...")
                
                async with websockets.connect(
                    self.ws_url, 
                    ping_interval=20, 
                    ping_timeout=10
                ) as ws:
                    
                    await ws.send(json.dumps(sub))
                    logger.info(f"[WS-{conn_id}] Subscribed to {len(topics)} topics")
                    
                    while self.running:
                        try:
                            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                            
                            if msg.get("success") == False:
                                logger.error(f"[WS-{conn_id}] Subscription failed: {msg}")
                                continue
                            
                            topic = msg.get("topic", "")
                            if topic.startswith("kline."):
                                sym = topic.split(".")[-1]
                                for k in msg.get("data", []):
                                    await queue.put((sym, k))
                                    
                        except asyncio.TimeoutError:
                            await ws.ping()
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning(f"[WS-{conn_id}] Connection closed, reconnecting...")
                            break
                            
            except Exception as e:
                logger.error(f"[WS-{conn_id}] Connection error: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting