"""
Enhanced Bybit V5 API Client with proper rate limiting and error handling
"""
import asyncio
import time
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime, timedelta
import structlog
from pybit.unified_trading import HTTP, WebSocket
import pandas as pd

from ..config import settings
from ..utils.bot_fixes import (
    rate_limiter, 
    position_safety,
    ws_manager,
    health_monitor,
    OrderValidation
)
from ..utils.rounding import round_to_tick, round_to_qty_step

logger = structlog.get_logger(__name__)

class EnhancedBybitClient:
    """
    Enhanced Bybit client with:
    - Proper rate limiting (600 req/5s)
    - Error handling and retry logic
    - Position safety checks
    - WebSocket auto-reconnect
    - Health monitoring
    """
    
    def __init__(self):
        self.testnet = settings.bybit_testnet
        self.api_key = settings.bybit_api_key
        self.api_secret = settings.bybit_api_secret
        
        # HTTP client
        self.http_client = HTTP(
            testnet=self.testnet,
            api_key=self.api_key,
            api_secret=self.api_secret,
            recv_window=settings.bybit_recv_window
        )
        
        # WebSocket clients
        self.public_ws = None
        self.private_ws = None
        
        # Caches
        self.instruments: Dict[str, Dict] = {}
        self.positions: Dict[str, Dict] = {}
        self.orders: Dict[str, Dict] = {}
        
        # Validation
        self.order_validator = None
        
        # Stats
        self.request_count = 0
        self.error_count = 0
        
        # Account type cache
        self.is_unified_account = None
        
    async def initialize(self):
        """Initialize with proper error handling"""
        try:
            # Fetch instruments with rate limiting
            await self._fetch_all_instruments()
            
            # Initialize order validator
            self.order_validator = OrderValidation(self.instruments)
            
            # Connect WebSockets with auto-reconnect
            await self._connect_websockets()
            
            # Start monitoring tasks
            asyncio.create_task(self._monitor_connections())
            
            logger.info(f"Enhanced Bybit client initialized with {len(self.instruments)} instruments")
            
        except Exception as e:
            logger.error(f"Failed to initialize Bybit client: {e}")
            raise
            
    async def _fetch_all_instruments(self):
        """Fetch instruments with proper pagination and rate limiting"""
        
        instruments = []
        cursor = None
        
        while True:
            try:
                # Rate limit check
                await rate_limiter.acquire_request()
                
                params = {
                    "category": "linear",
                    "limit": 1000
                }
                if cursor:
                    params["cursor"] = cursor
                
                # Make request with timing
                start = time.time()
                response = self.http_client.get_instruments_info(**params)
                latency = (time.time() - start) * 1000
                
                # Record metrics
                health_monitor.record_api_latency(latency)
                self.request_count += 1
                
                # Check response
                if response["retCode"] != 0:
                    raise Exception(f"API error: {response['retMsg']}")
                
                # Reset rate limit backoff on success
                rate_limiter.reset_backoff()
                
                # Process results
                items = response["result"]["list"]
                instruments.extend(items)
                
                cursor = response["result"].get("nextPageCursor")
                if not cursor:
                    break
                    
            except Exception as e:
                self.error_count += 1
                
                # Check for rate limit error
                if "403" in str(e) or "429" in str(e) or "too frequent" in str(e):
                    rate_limiter.handle_rate_limit_error()
                    await asyncio.sleep(10)  # Wait before retry
                    continue
                    
                logger.error(f"Error fetching instruments: {e}")
                raise
                
        # Process instruments
        for inst in instruments:
            if inst["status"] == "Trading":
                symbol = inst["symbol"]
                self.instruments[symbol] = {
                    "symbol": symbol,
                    "tick_size": Decimal(inst["priceFilter"]["tickSize"]),
                    "qty_step": Decimal(inst["lotSizeFilter"]["qtyStep"]),
                    "min_qty": Decimal(inst["lotSizeFilter"]["minOrderQty"]),
                    "max_qty": Decimal(inst["lotSizeFilter"]["maxOrderQty"]),
                    "min_notional": Decimal(inst["lotSizeFilter"].get("minNotionalValue", "0")),
                    "max_leverage": Decimal(inst["leverageFilter"]["maxLeverage"]),
                    "min_leverage": Decimal(inst["leverageFilter"]["minLeverage"]),
                    "leverage_step": Decimal(inst["leverageFilter"]["leverageStep"]),
                    "base_coin": inst["baseCoin"],
                    "quote_coin": inst["quoteCoin"]
                }
                
        logger.info(f"Loaded {len(self.instruments)} tradeable instruments")
        
    async def _connect_websockets(self):
        """Connect WebSockets with auto-reconnect"""
        
        try:
            # Public WebSocket
            self.public_ws = WebSocket(
                testnet=self.testnet,
                channel_type="linear"
            )
            
            # Private WebSocket
            self.private_ws = WebSocket(
                testnet=self.testnet,
                channel_type="private",
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            
            # Note: Handlers should be set via callbacks in subscription methods
            # not by overriding stream methods
            
            # Start auto-reconnect monitors
            asyncio.create_task(ws_manager.maintain_connection(self.public_ws, "public"))
            asyncio.create_task(ws_manager.maintain_connection(self.private_ws, "private"))
            
            logger.info("WebSocket connections established")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            # Continue without WebSocket
            
    def _handle_public_ws_message(self, message):
        """Handle public WebSocket messages"""
        try:
            ws_manager.connection_health['public'] = {
                'last_message': datetime.now(),
                'error_count': 0
            }
            # Process message
            logger.debug(f"Public WS: {message.get('topic', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error handling public WS message: {e}")
            health = ws_manager.connection_health.get('public', {})
            health['error_count'] = health.get('error_count', 0) + 1
            
    def _handle_private_ws_message(self, message):
        """Handle private WebSocket messages"""
        try:
            ws_manager.connection_health['private'] = {
                'last_message': datetime.now(),
                'error_count': 0
            }
            
            topic = message.get('topic', '')
            
            # Update position cache
            if 'position' in topic:
                self._update_position_cache(message.get('data', []))
                
            # Update order cache
            elif 'order' in topic:
                self._update_order_cache(message.get('data', []))
                
        except Exception as e:
            logger.error(f"Error handling private WS message: {e}")
            health = ws_manager.connection_health.get('private', {})
            health['error_count'] = health.get('error_count', 0) + 1
            
    def _update_position_cache(self, positions: List[Dict]):
        """Update local position cache"""
        for pos in positions:
            symbol = pos.get('symbol')
            if symbol:
                if float(pos.get('size', 0)) > 0:
                    self.positions[symbol] = pos
                    position_safety.register_position(symbol, {
                        'side': pos.get('side'),
                        'size': float(pos.get('size')),
                        'entry_price': float(pos.get('avgPrice', 0))
                    })
                else:
                    # Position closed
                    if symbol in self.positions:
                        del self.positions[symbol]
                    position_safety.remove_position(symbol)
                    
    def _update_order_cache(self, orders: List[Dict]):
        """Update local order cache"""
        for order in orders:
            order_id = order.get('orderId')
            if order_id:
                status = order.get('orderStatus')
                if status in ['Filled', 'Cancelled', 'Rejected']:
                    # Remove from cache
                    if order_id in self.orders:
                        del self.orders[order_id]
                else:
                    # Update cache
                    self.orders[order_id] = order
                    
    async def _monitor_connections(self):
        """Monitor connection health"""
        
        while True:
            try:
                # Check system health
                health_status = health_monitor.check_system_health()
                
                if not health_status['healthy']:
                    logger.error(f"System unhealthy: {health_status}")
                    
                # Log stats
                if self.request_count > 0:
                    error_rate = self.error_count / self.request_count
                    if error_rate > 0.1:
                        logger.warning(f"High error rate: {error_rate:.1%}")
                        
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(60)
                
    # Enhanced API methods with proper error handling
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """Get klines with rate limiting and error handling"""
        
        try:
            # Rate limit
            await rate_limiter.acquire_request()
            
            # Make request
            start = time.time()
            response = self.http_client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            latency = (time.time() - start) * 1000
            
            # Record metrics
            health_monitor.record_api_latency(latency)
            
            if response["retCode"] != 0:
                raise Exception(f"API error: {response['retMsg']}")
                
            # Reset rate limit backoff
            rate_limiter.reset_backoff()
            
            # Convert to DataFrame
            klines = response["result"]["list"]
            if not klines:
                return pd.DataFrame()
                
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df = df.astype({
                'timestamp': 'int64',
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64',
                'turnover': 'float64'
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            if "403" in str(e) or "429" in str(e):
                rate_limiter.handle_rate_limit_error()
                
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()
            
    async def place_order(self, **kwargs) -> Optional[str]:
        """Place order with validation and safety checks"""
        
        symbol = kwargs.get('symbol')
        side = kwargs.get('side')
        
        try:
            # Round quantity to valid step before validation
            if 'qty' in kwargs and symbol in self.instruments:
                from ..utils.rounding import round_to_qty_step
                instrument = self.instruments[symbol]
                qty_step = float(instrument.get('qty_step', 0.001))
                original_qty = float(kwargs['qty'])
                rounded_qty = round_to_qty_step(original_qty, qty_step)
                
                # Check minimum notional value (5 USDT for most symbols)
                min_notional = float(instrument.get('min_notional', 5.0))
                
                # Get current price for notional calculation
                # For market orders, we need to estimate the notional value
                if 'price' in kwargs:
                    price = float(kwargs['price'])
                else:
                    # Try to get last price from ticker
                    ticker_info = await self.get_symbol_info(symbol)
                    if ticker_info:
                        price = float(ticker_info.get('lastPrice', 0))
                    else:
                        price = 0
                
                if price > 0:
                    notional = rounded_qty * price
                    if notional < min_notional:
                        # Adjust quantity to meet minimum notional
                        adjusted_qty = min_notional / price * 1.1  # Add 10% buffer
                        adjusted_qty = round_to_qty_step(adjusted_qty, qty_step)
                        logger.warning(f"Quantity {rounded_qty} (notional: ${notional:.2f}) below minimum ${min_notional}. Adjusted to {adjusted_qty}")
                        rounded_qty = adjusted_qty
                
                kwargs['qty'] = str(rounded_qty)  # Bybit API expects string
                
                if original_qty != rounded_qty:
                    logger.info(f"Adjusted quantity from {original_qty} to {rounded_qty} (step: {qty_step}, min notional: ${min_notional})")
            
            # Safety check - one position per symbol
            if not await position_safety.can_open_position(symbol, side):
                logger.warning(f"Cannot open position for {symbol} - position already exists")
                return None
                
            # Validate order
            is_valid, error_msg = self.order_validator.validate_order(kwargs)
            if not is_valid:
                logger.error(f"Order validation failed: {error_msg}")
                return None
                
            # Rate limit
            await rate_limiter.acquire_request()
            
            # Ensure category is set for V5 API
            if 'category' not in kwargs:
                kwargs['category'] = 'linear'  # For USDT perpetuals
            
            # Ensure orderType is correctly formatted
            if 'order_type' in kwargs:
                kwargs['orderType'] = kwargs.pop('order_type')
            
            # Ensure timeInForce is correctly formatted
            if 'time_in_force' in kwargs:
                kwargs['timeInForce'] = kwargs.pop('time_in_force')
                
            # Ensure reduceOnly is correctly formatted
            if 'reduce_only' in kwargs:
                kwargs['reduceOnly'] = kwargs.pop('reduce_only')
                
            # Ensure closeOnTrigger is correctly formatted
            if 'close_on_trigger' in kwargs:
                kwargs['closeOnTrigger'] = kwargs.pop('close_on_trigger')
            
            # Place order
            start = time.time()
            response = self.http_client.place_order(**kwargs)
            latency = (time.time() - start) * 1000
            
            # Record metrics
            health_monitor.record_api_latency(latency)
            
            if response["retCode"] != 0:
                health_monitor.record_order_result(False)
                raise Exception(f"Order failed: {response['retMsg']}")
                
            # Success
            health_monitor.record_order_result(True)
            rate_limiter.reset_backoff()
            
            order_id = response["result"]["orderId"]
            
            # Cache order
            self.orders[order_id] = {
                'symbol': symbol,
                'side': side,
                'order_id': order_id,
                'status': 'New'
            }
            
            logger.info(f"Order placed: {order_id} for {symbol}")
            return order_id
            
        except Exception as e:
            if "403" in str(e) or "429" in str(e):
                rate_limiter.handle_rate_limit_error()
                
            logger.error(f"Error placing order: {e}")
            return None
            
    async def get_positions(self) -> List[Dict]:
        """Get positions with caching"""
        
        try:
            # Try cache first
            if self.positions:
                return list(self.positions.values())
                
            # Fetch from API
            await rate_limiter.acquire_request()
            
            response = self.http_client.get_positions(
                category="linear",
                settleCoin="USDT"
            )
            
            if response["retCode"] != 0:
                raise Exception(f"API error: {response['retMsg']}")
                
            positions = response["result"]["list"]
            
            # Update cache
            self._update_position_cache(positions)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
            
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage with validation"""
        
        try:
            # Validate leverage
            if symbol not in self.instruments:
                logger.warning(f"Symbol {symbol} not available on Bybit, skipping leverage setting")
                return False
                
            inst = self.instruments[symbol]
            max_lev = int(inst['max_leverage'])
            min_lev = int(inst['min_leverage'])
            
            leverage = max(min_lev, min(max_lev, leverage))
            
            # Rate limit
            await rate_limiter.acquire_request()
            
            response = self.http_client.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            
            if response["retCode"] != 0:
                # Error code 110043 means leverage is already set to this value - this is OK
                if response["retCode"] == 110043 or "not modified" in response.get("retMsg", "").lower():
                    logger.debug(f"Leverage already set to {leverage}x for {symbol}")
                    return True
                logger.error(f"Set leverage failed for {symbol}: {response['retMsg']}")
                return False
                
            logger.info(f"Leverage set to {leverage}x for {symbol}")
            return True
            
        except Exception as e:
            # Check if it's a "leverage not modified" error in the exception message
            if "110043" in str(e) or "leverage not modified" in str(e).lower():
                logger.debug(f"Leverage already set to {leverage}x for {symbol}")
                return True
            logger.error(f"Error setting leverage for {symbol}: {e}")
            return False
            
    async def set_margin_mode(self, symbol: str, mode: str) -> bool:
        """Set margin mode (cross/isolated)"""
        
        try:
            # Check if we're using a unified account
            if self.is_unified_account is None:
                await self._check_account_type()
            
            # Unified accounts don't support margin mode switching (always cross)
            if self.is_unified_account:
                logger.debug(f"Unified account detected - margin mode is always cross for {symbol}")
                return True
            
            # Convert mode
            trade_mode = 0 if mode.lower() == "cross" else 1
            
            # Rate limit
            await rate_limiter.acquire_request()
            
            response = self.http_client.switch_margin_mode(
                category="linear",
                symbol=symbol,
                tradeMode=trade_mode,
                buyLeverage=str(settings.default_leverage),
                sellLeverage=str(settings.default_leverage)
            )
            
            if response["retCode"] != 0:
                # Error code 100028 means unified account (can't switch margin mode)
                if response["retCode"] == 100028:
                    self.is_unified_account = True
                    logger.debug(f"Unified account confirmed - margin mode is always cross for {symbol}")
                    return True
                # Already in this mode is OK
                if "not modified" in response.get("retMsg", "").lower():
                    return True
                logger.error(f"Set margin mode failed for {symbol}: {response['retMsg']}")
                return False
                
            logger.info(f"Margin mode set to {mode} for {symbol}")
            return True
            
        except Exception as e:
            # Check if it's a unified account error in the exception message
            if "100028" in str(e) or "unified account" in str(e).lower():
                if self.is_unified_account is None:
                    self.is_unified_account = True
                logger.debug(f"Unified account - margin mode is always cross for {symbol}")
                return True
            logger.error(f"Error setting margin mode for {symbol}: {e}")
            return False
            
    async def _check_account_type(self):
        """Check if this is a unified account"""
        try:
            # Try to get unified account info
            response = self.http_client.get_wallet_balance(
                accountType="UNIFIED"
            )
            
            if response["retCode"] == 0:
                self.is_unified_account = True
                logger.info("Unified account detected - margin mode is always cross")
            else:
                self.is_unified_account = False
                logger.info("Standard account detected - margin mode switching supported")
                
        except Exception as e:
            # Default to assuming unified account to avoid errors
            self.is_unified_account = True
            logger.warning(f"Could not determine account type, assuming unified: {e}")
    
    async def get_account_info(self) -> Dict:
        """Get account info with caching"""
        
        try:
            await rate_limiter.acquire_request()
            
            response = self.http_client.get_wallet_balance(
                accountType="UNIFIED" if not self.testnet else "CONTRACT",
                coin="USDT"
            )
            
            if response["retCode"] != 0:
                raise Exception(f"API error: {response['retMsg']}")
                
            # Extract balance info
            result = response["result"]["list"][0] if response["result"]["list"] else {}
            
            return {
                'totalWalletBalance': result.get('totalWalletBalance', '0'),
                'totalAvailableBalance': result.get('totalAvailableBalance', '0'),
                'totalMarginBalance': result.get('totalMarginBalance', '0'),
                'totalInitialMargin': result.get('totalInitialMargin', '0'),
                'totalMaintenanceMargin': result.get('totalMaintenanceMargin', '0')
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {'totalWalletBalance': '0'}
            
    def get_instrument(self, symbol: str) -> Optional[Dict]:
        """Get instrument info"""
        return self.instruments.get(symbol)
    
    async def get_active_symbols(self) -> List[str]:
        """Get all active trading symbols"""
        try:
            # Check if instruments are loaded
            if not self.instruments:
                logger.warning("Instruments not loaded, returning default symbols")
                from ..config import settings
                return settings.default_symbols[:30]
            
            # Return all symbols from instruments that are actively trading
            active_symbols = []
            for symbol, info in self.instruments.items():
                # Check if it's a USDT perpetual and actively trading
                status = info.get('status') or info.get('trading_status') or 'Trading'
                if status == 'Trading' and symbol.endswith('USDT'):
                    active_symbols.append(symbol)
            
            if not active_symbols:
                logger.warning("No active symbols found in instruments, using defaults")
                from ..config import settings
                return settings.default_symbols[:30]
            
            # Sort by symbol name for consistency
            active_symbols.sort()
            logger.info(f"Found {len(active_symbols)} active USDT perpetual symbols")
            return active_symbols
            
        except Exception as e:
            logger.error(f"Error getting active symbols: {e}")
            # Return default symbols as fallback
            from ..config import settings
            return settings.default_symbols[:30]
    
    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed symbol information including 24h stats"""
        try:
            # Get ticker info for volume and price data
            response = self.http_client.get_tickers(
                category="linear",
                symbol=symbol
            )
            
            if response['retCode'] == 0 and response['result']['list']:
                ticker = response['result']['list'][0]
                
                # Combine with instrument info
                instrument = self.instruments.get(symbol, {})
                
                return {
                    'symbol': symbol,
                    'status': instrument.get('status', 'Trading'),
                    'min_qty': instrument.get('min_qty', 0),
                    'max_qty': instrument.get('max_qty', 0),
                    'qty_step': instrument.get('qty_step', 0),
                    'tick_size': instrument.get('tick_size', 0),
                    'turnover24h': float(ticker.get('turnover24h', 0)),  # 24h volume in USDT
                    'volume24h': float(ticker.get('volume24h', 0)),  # 24h volume in contracts
                    'price24hPcnt': float(ticker.get('price24hPcnt', 0)),  # 24h price change %
                    'lastPrice': float(ticker.get('lastPrice', 0)),
                    'bid1Price': float(ticker.get('bid1Price', 0)),
                    'ask1Price': float(ticker.get('ask1Price', 0)),
                    'highPrice24h': float(ticker.get('highPrice24h', 0)),
                    'lowPrice24h': float(ticker.get('lowPrice24h', 0))
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    # WebSocket subscription methods
    async def subscribe_orderbook(self, symbol: str, callback: Any):
        """Subscribe to orderbook updates"""
        try:
            if self.public_ws:
                self.public_ws.orderbook_stream(
                    depth=25,
                    symbol=symbol,
                    callback=callback
                )
                logger.debug(f"Subscribed to orderbook for {symbol}")
        except Exception as e:
            logger.error(f"Error subscribing to orderbook for {symbol}: {e}")
    
    async def subscribe_trades(self, symbol: str, callback: Any):
        """Subscribe to trade updates"""
        try:
            if self.public_ws:
                self.public_ws.trade_stream(
                    symbol=symbol,
                    callback=callback
                )
                logger.debug(f"Subscribed to trades for {symbol}")
        except Exception as e:
            logger.error(f"Error subscribing to trades for {symbol}: {e}")
    
    async def subscribe_klines(self, symbol: str, interval: str, callback: Any):
        """Subscribe to kline updates"""
        try:
            if self.public_ws:
                self.public_ws.kline_stream(
                    interval=interval,
                    symbol=symbol,
                    callback=callback
                )
                logger.debug(f"Subscribed to {interval}m klines for {symbol}")
        except Exception as e:
            logger.error(f"Error subscribing to klines for {symbol}: {e}")
    
    async def subscribe_positions(self, callback: Any):
        """Subscribe to position updates"""
        try:
            if self.private_ws:
                self.private_ws.position_stream(callback=callback)
                logger.debug("Subscribed to position updates")
        except Exception as e:
            logger.error(f"Error subscribing to positions: {e}")
    
    async def subscribe_orders(self, callback: Any):
        """Subscribe to order updates"""
        try:
            if self.private_ws:
                self.private_ws.order_stream(callback=callback)
                logger.debug("Subscribed to order updates")
        except Exception as e:
            logger.error(f"Error subscribing to orders: {e}")
    
    async def subscribe_executions(self, callback: Any):
        """Subscribe to execution updates"""
        try:
            if self.private_ws:
                self.private_ws.execution_stream(callback=callback)
                logger.debug("Subscribed to execution updates")
        except Exception as e:
            logger.error(f"Error subscribing to executions: {e}")