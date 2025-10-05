"""
Multi-WebSocket Handler for Bybit
Handles >200 symbol subscriptions by splitting across multiple connections
"""
import asyncio
import json
import logging
import websockets
import time
from typing import List, AsyncGenerator, Tuple

logger = logging.getLogger(__name__)

class MultiWebSocketHandler:
    """Handles multiple WebSocket connections for large symbol lists"""
    
    MAX_SUBS_PER_CONNECTION = 190  # Keep under 200 limit with buffer
    
    def __init__(self, ws_url: str, running_flag):
        self.ws_url = ws_url
        # Accept either a boolean or an object with a 'running' attribute
        self._running_flag = running_flag
        self.connections = []

    def _is_running(self) -> bool:
        try:
            if isinstance(self._running_flag, bool):
                return self._running_flag
            # Object with a 'running' attribute
            if hasattr(self._running_flag, 'running'):
                return bool(getattr(self._running_flag, 'running'))
            # Fallback to truthiness
            return bool(self._running_flag)
        except Exception:
            return True
        
    async def multi_kline_stream(self, topics: List[str]) -> AsyncGenerator[Tuple[str, dict], None]:
        """
        Stream klines from multiple WebSocket connections
        Splits topics across connections if >190 topics
        """
        # Remove duplicates while preserving order
        seen = set()
        unique_topics = []
        for topic in topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)
        
        if len(unique_topics) != len(topics):
            logger.info(f"Removed {len(topics) - len(unique_topics)} duplicate topics")
        
        # Split topics into chunks
        topic_chunks = []
        for i in range(0, len(unique_topics), self.MAX_SUBS_PER_CONNECTION):
            chunk = unique_topics[i:i + self.MAX_SUBS_PER_CONNECTION]
            topic_chunks.append(chunk)
        
        logger.info(f"Splitting {len(unique_topics)} unique topics across {len(topic_chunks)} WebSocket connections")
        
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
            while self._is_running():
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
        
        backoff = 2.0  # seconds
        max_backoff = 30.0
        while self._is_running():
            try:
                logger.info(f"[WS-{conn_id}] Connecting with {len(topics)} topics...")
                
                async with websockets.connect(
                    self.ws_url, 
                    ping_interval=20, 
                    ping_timeout=10
                ) as ws:
                    # Reset backoff on successful connection
                    backoff = 2.0
                    
                    await ws.send(json.dumps(sub))
                    logger.info(f"[WS-{conn_id}] Subscribed to {len(topics)} topics")
                    
                    timeouts = 0
                    last_msg_ts = time.monotonic()
                    last_warn_ts = 0.0
                    while self._is_running():
                        try:
                            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                            timeouts = 0  # reset timeout counter on message
                            last_msg_ts = time.monotonic()
                            
                            if msg.get("success") == False:
                                # Check if it's a duplicate subscription error
                                if "already subscribed" in msg.get("ret_msg", ""):
                                    logger.debug(f"[WS-{conn_id}] Already subscribed to topic, continuing...")
                                else:
                                    logger.error(f"[WS-{conn_id}] Subscription failed: {msg}")
                                continue
                            
                            topic = msg.get("topic", "")
                            if topic.startswith("kline."):
                                sym = topic.split(".")[-1]
                                for k in msg.get("data", []):
                                    await queue.put((sym, k))
                                    
                        except asyncio.TimeoutError:
                            timeouts += 1
                            try:
                                await ws.ping()
                            except Exception:
                                logger.warning(f"[WS-{conn_id}] Ping failed after timeout, reconnecting...")
                                break
                            # Stale feed warning if no data for > 60 seconds
                            now = time.monotonic()
                            if now - last_msg_ts > 60 and (now - last_warn_ts > 60):
                                logger.warning(f"[WS-{conn_id}] No data received for {int(now - last_msg_ts)}s; monitoring…")
                                last_warn_ts = now
                            # If repeated timeouts, force reconnect
                            if timeouts >= 3:
                                logger.warning(f"[WS-{conn_id}] Repeated timeouts ({timeouts}), reconnecting...")
                                break
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning(f"[WS-{conn_id}] Connection closed, reconnecting...")
                            break
                            
            except Exception as e:
                logger.error(f"[WS-{conn_id}] Connection error: {e}")
                # Exponential backoff with jitter
                import random
                sleep_s = min(max_backoff, backoff * (1.0 + random.uniform(-0.2, 0.2)))
                await asyncio.sleep(sleep_s)
                backoff = min(max_backoff, backoff * 1.6)
