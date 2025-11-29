from __future__ import annotations
"""
Multi-WebSocket Handler for Bybit
Handles >200 symbol subscriptions by splitting across multiple connections
"""
import asyncio
import json
import logging
import websockets
import time
import socket
from urllib.parse import urlparse
from typing import List, AsyncGenerator, Tuple, Optional

logger = logging.getLogger(__name__)

class MultiWebSocketHandler:
    """Handles multiple WebSocket connections for large symbol lists"""
    
    MAX_SUBS_PER_CONNECTION = 190  # Keep under 200 limit with buffer
    
    def __init__(self, ws_url: str, running_flag, alt_ws_url: Optional[str] = None, use_alt_on_fail: bool = False):
        self.ws_url = ws_url
        self._ws_alt_url = alt_ws_url
        self._use_alt_on_fail = bool(use_alt_on_fail and bool(alt_ws_url))
        self._ws_idx = 0  # 0=primary, 1=alt
        # Failure tracking and ALT cool-off policy
        self._primary_fail_count = 0
        self._primary_hold_until = 0.0  # epoch seconds until which we stick to ALT
        # 10–15 minutes cool-off; default to 12 minutes
        self._alt_cooldown_sec = 12 * 60
        # Accept either a boolean or an object with a 'running' attribute
        self._running_flag = running_flag
        self.connections = []
        # Optional per-message kline trace: read from owner if available
        try:
            if hasattr(running_flag, '_ws_kline_trace'):
                self._kline_trace = bool(getattr(running_flag, '_ws_kline_trace'))
            else:
                self._kline_trace = False
        except Exception:
            self._kline_trace = False

        # DNS pre-resolution cache and per-host IP rotors
        self._dns_cache: dict[str, list[str]] = {}
        self._ip_rotate_idx: dict[str, int] = {}
        try:
            self._resolve_ws_hosts()
        except Exception:
            pass

    def _parse_ws_url(self, url: str) -> tuple[str, str, str]:
        """Return (scheme, host, path_and_query) for a WS URL."""
        try:
            p = urlparse(url)
            scheme = p.scheme or 'wss'
            host = p.hostname or ''
            path = p.path or '/'
            if p.query:
                path = path + '?' + p.query
            return scheme, host, path
        except Exception:
            return 'wss', '', '/'

    def _resolve_ws_hosts(self):
        """Pre-resolve primary and ALT WS hosts and populate IP rotor state."""
        hosts = []
        try:
            _, h0, _ = self._parse_ws_url(self.ws_url)
            if h0:
                hosts.append(h0)
        except Exception:
            pass
        try:
            if self._ws_alt_url:
                _, h1, _ = self._parse_ws_url(self._ws_alt_url)
                if h1 and h1 not in hosts:
                    hosts.append(h1)
        except Exception:
            pass
        for host in hosts:
            try:
                infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
                ips = []
                for info in infos:
                    try:
                        ip = info[4][0]
                        if ip and ip not in ips:
                            ips.append(ip)
                    except Exception:
                        continue
                if ips:
                    self._dns_cache[host] = ips
                    if host not in self._ip_rotate_idx:
                        self._ip_rotate_idx[host] = 0
            except Exception:
                continue

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
        # Derive a sensible recv timeout based on the largest timeframe in this connection.
        # Bybit kline streams can be silent for the entire bar; using a fixed 90s timeout
        # causes unnecessary reconnects for 3m/15m topics. Compute per-connection timeout
        # as max(120s, max_tf_minutes*60 + 30s).
        def _compute_timeout() -> float:
            try:
                tf_secs = 0
                for t in topics:
                    try:
                        tf = t.split(".")[0]
                        tf_i = int(tf)
                        tf_secs = max(tf_secs, tf_i * 60)
                    except Exception:
                        # Non-integer timeframe (e.g., '1') fallback handled by default
                        continue
                # Default to 60s if nothing parsed
                base = tf_secs if tf_secs > 0 else 60
                return max(120.0, float(base + 30))
            except Exception:
                return 120.0

        while self._is_running():
            try:
                # Select URL (toggle after failures when enabled) with ALT cool-off
                now = time.time()
                force_alt = self._use_alt_on_fail and (self._primary_hold_until > now)
                use_alt = False
                if self._use_alt_on_fail and self._ws_alt_url:
                    if force_alt:
                        use_alt = True
                    else:
                        use_alt = (self._ws_idx == 1)
                url = self._ws_alt_url if use_alt else self.ws_url
                which = 'ALT' if use_alt else 'PRIMARY'
                logger.info(f"[WS-{conn_id}] Connecting with {len(topics)} topics ({which})…")
                
                # Tune timeouts: derive open_timeout relative to expected cadence; pin a minimum of 25s
                _ot = max(25.0, min(40.0, _compute_timeout() / 3.0))

                # Resolve candidate IPs for this host (if available)
                scheme, host, path = self._parse_ws_url(url)
                candidates = []
                ips = list(self._dns_cache.get(host, []) or [])
                if ips:
                    start = int(self._ip_rotate_idx.get(host, 0)) % len(ips)
                    for i in range(len(ips)):
                        ip = ips[(start + i) % len(ips)]
                        candidates.append((f"{scheme}://{ip}{path}", host, ip))
                # Always include the original host as last fallback
                candidates.append((url, host, None))

                ws = None
                last_error = None
                for cand_url, sni_host, ip in candidates:
                    try:
                        # Attempt connection; provide server_hostname for SNI when using IP
                        try:
                            ws_cm = websockets.connect(
                                cand_url,
                                ping_interval=30,
                                ping_timeout=40,
                                open_timeout=_ot,
                                close_timeout=10,
                                server_hostname=sni_host
                            )
                        except TypeError:
                            # Older websockets doesn't support server_hostname
                            ws_cm = websockets.connect(
                                cand_url,
                                ping_interval=30,
                                ping_timeout=40,
                                open_timeout=_ot,
                                close_timeout=10
                            )
                        async with ws_cm as ws:
                            # Mark connected
                            try:
                                if hasattr(self._running_flag, '__dict__'):
                                    import time as _t
                                    setattr(self._running_flag, '_ws_connected', True)
                                    setattr(self._running_flag, '_ws_last_msg_ts', _t.time())
                            except Exception:
                                pass
                            # Reset backoff on successful connection
                            backoff = 2.0
                            # Reset primary failure counters on success of PRIMARY
                            if not use_alt:
                                self._primary_fail_count = 0
                                self._primary_hold_until = 0.0
                            # Advance rotor on success for IP-based connects
                            try:
                                if ip is not None:
                                    self._ip_rotate_idx[host] = (start + 1) % max(1, len(ips))
                            except Exception:
                                pass
                            
                            await ws.send(json.dumps(sub))
                            logger.info(f"[WS-{conn_id}] Subscribed to {len(topics)} topics")
                            
                            timeouts = 0
                            last_msg_ts = time.monotonic()
                            last_warn_ts = 0.0
                            # Compute connection-specific timeout once per connect
                            recv_timeout = _compute_timeout()
                            # Publish expected interval to owner so health monitor can scale thresholds
                            try:
                                if hasattr(self._running_flag, '__dict__'):
                                    setattr(self._running_flag, '_ws_expected_interval', float(recv_timeout))
                            except Exception:
                                pass
                            while self._is_running():
                                try:
                                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=recv_timeout))
                                    timeouts = 0  # reset timeout counter on message
                                    last_msg_ts = time.monotonic()
                                    # Publish heartbeat timestamp to owner for network health monitor
                                    try:
                                        if hasattr(self._running_flag, '__dict__'):
                                            import time as _t
                                            setattr(self._running_flag, '_ws_last_msg_ts', _t.time())
                                    except Exception:
                                        pass
                                    
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
                                            # Optional trace: log first few messages per connection; if tracing enabled, log all
                                            try:
                                                if not hasattr(self, '_first_logs'):
                                                    self._first_logs = {}
                                                cnt = int(self._first_logs.get(conn_id, 0))
                                                if self._kline_trace or cnt < 5:
                                                    cfm = bool(k.get('confirm', False))
                                                    o = k.get('open'); h = k.get('high'); l = k.get('low'); c = k.get('close'); v = k.get('volume')
                                                    ts0 = k.get('start')
                                                    logger.info(f"[WS-{conn_id} KLINE] {sym} confirm={cfm} o={o} h={h} l={l} c={c} v={v} ts={ts0}")
                                                    self._first_logs[conn_id] = cnt + 1
                                            except Exception:
                                                pass
                                            await queue.put((sym, k))
                                            
                                except asyncio.TimeoutError:
                                    # No message within recv_timeout. Send an explicit ping to keepalive.
                                    timeouts += 1
                                    try:
                                        await ws.ping()
                                    except Exception:
                                        logger.warning(f"[WS-{conn_id}] Ping failed after timeout, reconnecting...")
                                        break
                                    # Stale feed warning if no data for > recv_timeout
                                    now = time.monotonic()
                                    if now - last_msg_ts > recv_timeout and (now - last_warn_ts > 60):
                                        logger.warning(f"[WS-{conn_id}] No data received for {int(now - last_msg_ts)}s (timeout={int(recv_timeout)}s); monitoring…")
                                        last_warn_ts = now
                                    # If severely idle (3x timeout), reconnect to refresh subscription
                                    if (now - last_msg_ts) > (3 * recv_timeout):
                                        logger.warning(f"[WS-{conn_id}] Prolonged idle ({int(now - last_msg_ts)}s), reconnecting...")
                                        break
                                except websockets.exceptions.ConnectionClosed:
                                    # Throttle warning frequency; log INFO if recent
                                    logger.info(f"[WS-{conn_id}] Connection closed, reconnecting…")
                                    break
                                except asyncio.CancelledError:
                                    logger.debug(f"[WS-{conn_id}] Task cancelled")
                                    raise
                            # Connected loop ended; break candidates loop to retry fresh
                            break
                    except Exception as _c_err:
                        last_error = _c_err
                        continue

                # If we exhausted all candidates without establishing a running loop, raise to outer handler
                if ws is None:
                    raise last_error if last_error else RuntimeError('ws_connect_failed')
                    # Mark connected
                    try:
                        if hasattr(self._running_flag, '__dict__'):
                            import time as _t
                            setattr(self._running_flag, '_ws_connected', True)
                            setattr(self._running_flag, '_ws_last_msg_ts', _t.time())
                    except Exception:
                        pass
                    # Reset backoff on successful connection
                    backoff = 2.0
                    
                    await ws.send(json.dumps(sub))
                    logger.info(f"[WS-{conn_id}] Subscribed to {len(topics)} topics")
                    
                    timeouts = 0
                    last_msg_ts = time.monotonic()
                    last_warn_ts = 0.0
                    # Compute connection-specific timeout once per connect
                    recv_timeout = _compute_timeout()
                    # Publish expected interval to owner so health monitor can scale thresholds
                    try:
                        if hasattr(self._running_flag, '__dict__'):
                            setattr(self._running_flag, '_ws_expected_interval', float(recv_timeout))
                    except Exception:
                        pass
                    while self._is_running():
                        try:
                            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=recv_timeout))
                            timeouts = 0  # reset timeout counter on message
                            last_msg_ts = time.monotonic()
                            # Publish heartbeat timestamp to owner for network health monitor
                            try:
                                if hasattr(self._running_flag, '__dict__'):
                                    import time as _t
                                    setattr(self._running_flag, '_ws_last_msg_ts', _t.time())
                            except Exception:
                                pass
                            
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
                                    # Optional trace: log first few messages per connection; if tracing enabled, log all
                                    try:
                                        if not hasattr(self, '_first_logs'):
                                            self._first_logs = {}
                                        cnt = int(self._first_logs.get(conn_id, 0))
                                        if self._kline_trace or cnt < 5:
                                            cfm = bool(k.get('confirm', False))
                                            o = k.get('open'); h = k.get('high'); l = k.get('low'); c = k.get('close'); v = k.get('volume')
                                            ts0 = k.get('start')
                                            logger.info(f"[WS-{conn_id} KLINE] {sym} confirm={cfm} o={o} h={h} l={l} c={c} v={v} ts={ts0}")
                                            self._first_logs[conn_id] = cnt + 1
                                    except Exception:
                                        pass
                                    await queue.put((sym, k))
                                    
                        except asyncio.TimeoutError:
                            # No message within recv_timeout. Send an explicit ping to keepalive.
                            timeouts += 1
                            try:
                                await ws.ping()
                            except Exception:
                                logger.warning(f"[WS-{conn_id}] Ping failed after timeout, reconnecting...")
                                break
                            # Stale feed warning if no data for > recv_timeout
                            now = time.monotonic()
                            if now - last_msg_ts > recv_timeout and (now - last_warn_ts > 60):
                                logger.warning(f"[WS-{conn_id}] No data received for {int(now - last_msg_ts)}s (timeout={int(recv_timeout)}s); monitoring…")
                                last_warn_ts = now
                            # If severely idle (3x timeout), reconnect to refresh subscription
                            if (now - last_msg_ts) > (3 * recv_timeout):
                                logger.warning(f"[WS-{conn_id}] Prolonged idle ({int(now - last_msg_ts)}s), reconnecting...")
                                break
                        except websockets.exceptions.ConnectionClosed:
                            # Throttle warning frequency; log INFO if recent
                            logger.info(f"[WS-{conn_id}] Connection closed, reconnecting…")
                            break
                        except asyncio.CancelledError:
                            logger.debug(f"[WS-{conn_id}] Task cancelled")
                            raise
                            
            except asyncio.CancelledError:
                logger.debug(f"[WS-{conn_id}] Handler cancelled")
                break
            except Exception as e:
                import traceback
                # Downgrade to WARNING to reduce noise during transient network issues; reconnect logic follows
                logger.warning(f"[WS-{conn_id}] Connection error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                # Track primary failures and flip/hold ALT when threshold reached
                if self._use_alt_on_fail:
                    # Detect which endpoint was attempted last (approximate via index)
                    failed_primary = (which == 'PRIMARY')
                    if failed_primary:
                        self._primary_fail_count += 1
                        if self._primary_fail_count >= 3:
                            self._primary_hold_until = time.time() + max(600, self._alt_cooldown_sec)
                            self._ws_idx = 1  # force ALT next
                            logger.warning(f"[WS-{conn_id}] Primary failed {self._primary_fail_count}× — sticking to ALT for {int(self._primary_hold_until - time.time())}s")
                        else:
                            self._ws_idx = 1  # try ALT next
                    else:
                        # On ALT failure, try PRIMARY next when not in hold window
                        if time.time() >= self._primary_hold_until:
                            self._ws_idx = 0
                    try:
                        which2 = 'ALT' if self._ws_idx == 1 else 'PRIMARY'
                        logger.info(f"[WS-{conn_id}] Switching WS endpoint to {which2} and backing off")
                    except Exception:
                        pass
                # Exponential backoff with jitter
                import random
                # Mark disconnected while backing off
                try:
                    if hasattr(self._running_flag, '__dict__'):
                        setattr(self._running_flag, '_ws_connected', False)
                except Exception:
                    pass
                sleep_s = min(max_backoff, backoff * (1.0 + random.uniform(-0.2, 0.2)))
                await asyncio.sleep(sleep_s)
                backoff = min(max_backoff, backoff * 1.6)
