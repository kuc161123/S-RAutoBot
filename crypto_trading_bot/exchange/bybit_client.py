"""
Bybit exchange client for trading operations
"""
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Any
from pybit.unified_trading import HTTP, WebSocket
import structlog
from datetime import datetime
import time

logger = structlog.get_logger(__name__)

class BybitClient:
    """Simple and efficient Bybit client"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, config=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.config = config
        
        # Initialize HTTP client
        self.client = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )
        
        # WebSocket clients
        self.ws_public = None
        self.ws_private = None
        
        # Data storage
        self.kline_data: Dict[str, pd.DataFrame] = {}
        self.orderbook_data: Dict[str, dict] = {}
        self.positions: Dict[str, dict] = {}
        
        # Callbacks
        self.on_kline_update = None
        self.on_position_update = None
        
        # Event loop reference for WebSocket callbacks
        self.loop = None
        
        logger.info(f"Bybit client initialized ({'Testnet' if testnet else 'Mainnet'})")
    
    async def initialize(self, symbols: List[str]):
        """Initialize client with symbols"""
        try:
            # Store the event loop for WebSocket callbacks
            self.loop = asyncio.get_event_loop()
            
            # Test connection
            server_time = self.client.get_server_time()
            logger.info(f"Connected to Bybit. Server time: {server_time['result']['timeNano']}")
            
            # Fetch initial data for all symbols in parallel (skip invalid ones)
            tasks = [self.fetch_klines(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out failed symbols
            valid_symbols = []
            for symbol, result in zip(symbols, results):
                if not isinstance(result, Exception) and result is not None:
                    valid_symbols.append(symbol)
                else:
                    logger.warning(f"Skipping invalid symbol: {symbol}")
            
            symbols = valid_symbols
            
            # Start WebSocket connections
            await self.start_websockets(symbols)
            
            logger.info(f"Initialized with {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    async def fetch_klines(self, symbol: str, interval: str = None, limit: int = 200):
        """Fetch historical kline data"""
        try:
            # Use interval from config if not provided
            if interval is None:
                interval = self.config.scalp_timeframe if self.config else "5"
            
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if response['retCode'] == 0:
                klines = response['result']['list']
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                ])
                
                # Convert types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                df = df.sort_values('timestamp')
                df = df.reset_index(drop=True)
                
                self.kline_data[symbol] = df
                logger.debug(f"Fetched {len(df)} klines for {symbol}")
                return df
            else:
                logger.error(f"Failed to fetch klines for {symbol}: {response['retMsg']}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return None
    
    async def start_websockets(self, symbols: List[str]):
        """Start WebSocket connections with retry logic for server stability"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Public WebSocket for market data
                self.ws_public = WebSocket(
                    testnet=self.testnet,
                    channel_type="linear"
                )
                
                # Private WebSocket for account updates
                self.ws_private = WebSocket(
                    testnet=self.testnet,
                    channel_type="private",
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
                
                # Connection successful, exit retry loop
                break
                
            except Exception as e:
                retry_count += 1
                logger.error(f"WebSocket connection failed (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    await asyncio.sleep(5 * retry_count)  # Exponential backoff
                else:
                    raise Exception("Failed to establish WebSocket connection after multiple attempts")
        
        try:
            # Subscribe to kline streams using config timeframe
            interval = int(self.config.scalp_timeframe) if self.config else 5
            kline_streams = [f"kline.{interval}.{symbol}" for symbol in symbols]
            
            def handle_kline(message):
                """Handle kline updates"""
                try:
                    if 'data' in message:
                        data = message['data'][0] if isinstance(message['data'], list) else message['data']
                        symbol = message['topic'].split('.')[-1]
                        
                        # Update kline data
                        if symbol in self.kline_data:
                            new_row = pd.DataFrame([{
                                'timestamp': pd.to_datetime(int(data['timestamp']), unit='ms'),
                                'open': float(data['open']),
                                'high': float(data['high']),
                                'low': float(data['low']),
                                'close': float(data['close']),
                                'volume': float(data['volume'])
                            }])
                            
                            # Append or update last row
                            self.kline_data[symbol] = pd.concat([
                                self.kline_data[symbol], new_row
                            ]).drop_duplicates(subset=['timestamp'], keep='last').tail(500)
                            
                            # Trigger callback if set
                            if self.on_kline_update and self.loop:
                                # Schedule the coroutine in the main event loop
                                asyncio.run_coroutine_threadsafe(
                                    self.on_kline_update(symbol, self.kline_data[symbol]), 
                                    self.loop
                                )
                                
                except Exception as e:
                    logger.error(f"Error handling kline update: {e}")
            
            def handle_position(message):
                """Handle position updates"""
                try:
                    if 'data' in message:
                        for position in message['data']:
                            symbol = position['symbol']
                            self.positions[symbol] = position
                            
                            # Trigger callback if set
                            if self.on_position_update and self.loop:
                                # Schedule the coroutine in the main event loop
                                asyncio.run_coroutine_threadsafe(
                                    self.on_position_update(symbol, position), 
                                    self.loop
                                )
                                
                except Exception as e:
                    logger.error(f"Error handling position update: {e}")
            
            # Subscribe to streams using config timeframe
            ws_interval = int(self.config.scalp_timeframe) if self.config else 5
            self.ws_public.kline_stream(
                interval=ws_interval,
                symbol=symbols,
                callback=handle_kline
            )
            
            self.ws_private.position_stream(callback=handle_position)
            
            logger.info(f"WebSocket connections started for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to start WebSockets: {e}")
    
    def get_account_balance(self) -> Optional[float]:
        """Get USDT balance"""
        try:
            # Try UNIFIED account first
            response = self.client.get_wallet_balance(
                accountType="UNIFIED"
            )
            
            if response['retCode'] == 0:
                # Debug log the full response
                logger.debug(f"Balance response: {response}")
                
                # Navigate the response structure correctly
                result = response.get('result', {})
                if 'list' in result and len(result['list']) > 0:
                    account_info = result['list'][0]
                    
                    # Try to get totalAvailableBalance first (best for derivatives)
                    available_balance = account_info.get('totalAvailableBalance')
                    if available_balance:
                        balance = float(available_balance)
                        logger.info(f"Account balance (available): ${balance:.2f}")
                        return balance
                    
                    # Otherwise check coin array
                    coins = account_info.get('coin', [])
                    for coin in coins:
                        if coin.get('coin') == 'USDT':
                            # Try different balance fields
                            for field in ['availableToWithdraw', 'walletBalance', 'free', 'equity', 'availableBalanceWithoutConvert']:
                                balance_str = coin.get(field)
                                if balance_str and balance_str != '' and balance_str != '0':
                                    try:
                                        balance = float(balance_str)
                                        if balance > 0:
                                            logger.info(f"Account balance (USDT from {field}): ${balance:.2f}")
                                            return balance
                                    except (ValueError, TypeError) as e:
                                        logger.debug(f"Could not convert {field}={balance_str} to float: {e}")
                                        continue
                            
                            # If no valid balance found in any field
                            logger.warning("USDT found but all balance fields are empty or zero")
                            return 0.0
                
                logger.warning("Could not find USDT balance in response")
                return 0.0
            else:
                logger.error(f"Failed to get balance: {response['retMsg']}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            logger.exception("Full traceback:")
            return None
    
    def place_order(self, symbol: str, side: str, qty: float, 
                   stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None) -> Optional[dict]:
        """Place a market order"""
        try:
            # Get appropriate quantity precision for symbol
            qty = self._format_quantity(symbol, qty)
            
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(qty),
                "timeInForce": "IOC",
                "positionIdx": 0  # One-way mode
            }
            
            logger.debug(f"Placing order: {symbol} {side} qty={qty}")
            
            # Add stop loss if provided (market order for immediate execution)
            if stop_loss:
                order_params["stopLoss"] = str(stop_loss)
                order_params["slOrderType"] = "Market"  # Stop loss as market order
                order_params["slTriggerBy"] = "LastPrice"
            
            # Add take profit if provided (limit order for better fill)
            if take_profit:
                order_params["takeProfit"] = str(take_profit)
                order_params["tpOrderType"] = "Limit"  # Take profit as limit order
                order_params["tpTriggerBy"] = "LastPrice"
                order_params["tpLimitPrice"] = str(take_profit)  # REQUIRED for limit TP
            
            response = self.client.place_order(**order_params)
            
            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                logger.info(f"Order placed: {symbol} {side} {qty} - ID: {order_id}")
                logger.info(f"TP: {take_profit} (Limit), SL: {stop_loss} (Market)")
                return response['result']
            else:
                logger.error(f"Failed to place order: {response['retMsg']}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def get_positions(self) -> List[dict]:
        """Get all open positions"""
        try:
            response = self.client.get_positions(
                category="linear",
                settleCoin="USDT"
            )
            
            if response['retCode'] == 0:
                positions = []
                for pos in response['result']['list']:
                    if float(pos['size']) > 0:
                        positions.append({
                            'symbol': pos['symbol'],
                            'side': pos['side'],
                            'size': float(pos['size']),
                            'entry_price': float(pos['avgPrice']),
                            'pnl': float(pos['unrealisedPnl']),
                            'pnl_percent': float(pos['unrealisedPnl']) / float(pos['positionValue']) * 100 if float(pos['positionValue']) > 0 else 0
                        })
                return positions
            else:
                logger.error(f"Failed to get positions: {response['retMsg']}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def close_position(self, symbol: str) -> bool:
        """Close a specific position"""
        try:
            # Get position details first
            positions = self.get_positions()
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position:
                logger.debug(f"No position found for {symbol}")
                return False
            
            # Format quantity properly
            qty = self._format_quantity(symbol, position['size'])
            
            # Place opposite order to close
            side = "Sell" if position['side'] == "Buy" else "Buy"
            
            response = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                timeInForce="IOC",
                positionIdx=0,
                reduceOnly=True
            )
            
            if response['retCode'] == 0:
                logger.info(f"Position closed: {symbol}")
                return True
            else:
                logger.error(f"Failed to close position: {response['retMsg']}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def _format_quantity(self, symbol: str, qty: float) -> float:
        """Format quantity based on symbol requirements - FROM OFFICIAL BYBIT API"""
        # Official Bybit quantity decimal rules (fetched from API)
        qty_rules = {
            # 0 decimals (integer only)
            "1000BONKUSDT": 0, "1000FLOKIUSDT": 0, "1000LUNCUSDT": 0,
            "1000PEPEUSDT": 0, "1000XECUSDT": 0, "ACHUSDT": 0,
            "ADAUSDT": 0, "ANKRUSDT": 0, "ARPAUSDT": 0,
            "BLURUSDT": 0, "CFXUSDT": 0, "CHZUSDT": 0, "CKBUSDT": 0,
            "CROUSDT": 0, "CTSIUSDT": 0, "DOGEUSDT": 0, "DOGSUSDT": 0,
            "GALAUSDT": 0, "GMTUSDT": 0, "HBARUSDT": 0, "IOTXUSDT": 0,
            "JASMYUSDT": 0, "JUPUSDT": 0, "KASUSDT": 0, "MAGICUSDT": 0,
            "MAVUSDT": 0, "MEMEUSDT": 0, "NOTUSDT": 0, "NTRNUSDT": 0,
            "PENDLEUSDT": 0, "PEOPLEUSDT": 0, "PHBUSDT": 0, "PYTHUSDT": 0,
            "RDNTUSDT": 0, "ROSEUSDT": 0, "RSRUSDT": 0, "SANDUSDT": 0,
            "SEIUSDT": 0, "SPELLUSDT": 0, "SUIUSDT": 0, "TRXUSDT": 0,
            "VETUSDT": 0, "WIFUSDT": 0, "XRPUSDT": 0, "ZRXUSDT": 0,
            "MATICUSDT": 0, "FETUSDT": 0, "STMXUSDT": 0, "ARKMUSDT": 0,
            "1INCHUSDT": 0, "ENJUSDT": 0,
            
            # 1 decimal
            "ACEUSDT": 1, "AIUSDT": 1, "ALGOUSDT": 1,
            "ALICEUSDT": 1, "APEUSDT": 1, "ARBUSDT": 1, "ARUSDT": 1,
            "ATOMUSDT": 1, "AVAXUSDT": 1, "AXSUSDT": 1, "BATUSDT": 1,
            "C98USDT": 1, "CELOUSDT": 1, "COREUSDT": 1, "CRVUSDT": 1,
            "CYBERUSDT": 1, "DOTUSDT": 1, "DYDXUSDT": 1,
            "ENSUSDT": 1, "ETCUSDT": 1, "FILUSDT": 1, "FLOWUSDT": 1,
            "GRTUSDT": 1, "HIGHUSDT": 1, "ICPUSDT": 1, "IMXUSDT": 1,
            "INJUSDT": 1, "IOTAUSDT": 1, "LDOUSDT": 1, "LINKUSDT": 1,
            "LRCUSDT": 1, "LTCUSDT": 1, "MANAUSDT": 1, "MANTAUSDT": 1,
            "MASKUSDT": 1, "MINAUSDT": 1, "MTLUSDT": 1, "NEARUSDT": 1,
            "OPUSDT": 1, "RENDERUSDT": 1, "RUNEUSDT": 1, "SNXUSDT": 1,
            "SOLUSDT": 1, "STXUSDT": 1, "SUSHIUSDT": 1, "SXPUSDT": 1,
            "THETAUSDT": 1, "TIAUSDT": 1, "TONUSDT": 1, "UMAUSDT": 1,
            "UNIUSDT": 1, "WLDUSDT": 1, "XTZUSDT": 1, "FTMUSDT": 1,
            "AGIXUSDT": 1, "OCEANUSDT": 1, "EOSUSDT": 1,
            
            # 2 decimals
            "AAVEUSDT": 2, "APTUSDT": 2, "BCHUSDT": 2, "BNBUSDT": 2,
            "COMPUSDT": 2, "EGLDUSDT": 2, "ETHUSDT": 2, "FXSUSDT": 2,
            "GMXUSDT": 2, "ILVUSDT": 2, "ORDIUSDT": 2, "QNTUSDT": 2,
            "SSVUSDT": 2, "MKRUSDT": 2,
            
            # 3 decimals
            "BTCUSDT": 3, "TAOUSDT": 3,
            
            # 4 decimals
            "YFIUSDT": 4,
            
            # Default for unknowns
            "default": 1
        }
        
        # Get precision for this symbol
        precision = qty_rules.get(symbol, qty_rules["default"])
        
        # For integer quantities
        if precision == 0:
            return float(int(qty))
        
        # For decimal quantities
        return round(qty, precision)
    
    def set_position_tp_sl(self, symbol: str, take_profit: float = None, stop_loss: float = None) -> bool:
        """Set or update TP/SL for an existing position"""
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "positionIdx": 0
            }
            
            if take_profit:
                params["takeProfit"] = str(take_profit)
                params["tpOrderType"] = "Limit"
                params["tpTriggerBy"] = "LastPrice"
                params["tpLimitPrice"] = str(take_profit)  # REQUIRED for limit TP
                params["tpslMode"] = "Partial"
                params["tpSize"] = "0"  # 0 means full position
                
            if stop_loss:
                params["stopLoss"] = str(stop_loss)
                params["slOrderType"] = "Market"
                params["slTriggerBy"] = "LastPrice"
                params["tpslMode"] = "Partial"
                params["slSize"] = "0"  # 0 means full position
            
            response = self.client.set_trading_stop(**params)
            
            if response['retCode'] == 0:
                logger.info(f"Updated TP/SL for {symbol} - TP: {take_profit} (Limit), SL: {stop_loss} (Market)")
                return True
            else:
                logger.error(f"Failed to set TP/SL: {response['retMsg']}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting TP/SL: {e}")
            return False
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol"""
        try:
            response = self.client.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            
            if response['retCode'] == 0:
                logger.info(f"Leverage set to {leverage}x for {symbol}")
                return True
            elif response['retCode'] == 110043:
                # Leverage already set correctly, not an error
                logger.debug(f"Leverage already set to {leverage}x for {symbol}")
                return True
            else:
                logger.error(f"Failed to set leverage: {response['retMsg']}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return False
    
    async def cleanup(self):
        """Clean up connections"""
        try:
            if self.ws_public:
                self.ws_public.exit()
            if self.ws_private:
                self.ws_private.exit()
            logger.info("Bybit client cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")