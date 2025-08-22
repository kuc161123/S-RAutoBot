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
            
            # Set up handlers
            self.public_ws.order_stream = self._handle_public_ws_message
            self.private_ws.order_stream = self._handle_private_ws_message
            
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