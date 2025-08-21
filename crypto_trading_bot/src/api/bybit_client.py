import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import time
from datetime import datetime, timedelta

from pybit.unified_trading import HTTP, WebSocket
import pandas as pd
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..utils.rounding import round_to_tick, round_to_qty_step

logger = structlog.get_logger(__name__)

class BybitClient:
    """Bybit V5 API client with WebSocket support"""
    
    def __init__(self):
        self.testnet = settings.bybit_testnet
        self.api_key = settings.bybit_api_key
        self.api_secret = settings.bybit_api_secret
        
        # HTTP client for REST API
        self.http_client = HTTP(
            testnet=self.testnet,
            api_key=self.api_key,
            api_secret=self.api_secret,
            recv_window=settings.bybit_recv_window
        )
        
        # WebSocket clients
        self.public_ws = None
        self.private_ws = None
        
        # Instrument cache
        self.instruments: Dict[str, Dict] = {}
        self.last_instrument_update = None
        
        # Position tracking
        self.positions: Dict[str, Dict] = {}
        self.orders: Dict[str, Dict] = {}
        
    async def initialize(self):
        """Initialize the client and fetch instruments"""
        await self.refresh_instruments()
        await self.connect_websockets()
        
    async def refresh_instruments(self):
        """Fetch and cache all linear futures instruments"""
        try:
            instruments = []
            cursor = None
            
            while True:
                params = {
                    "category": "linear",
                    "limit": 1000
                }
                if cursor:
                    params["cursor"] = cursor
                    
                response = self.http_client.get_instruments_info(**params)
                
                if response["retCode"] != 0:
                    raise Exception(f"Failed to fetch instruments: {response['retMsg']}")
                
                items = response["result"]["list"]
                instruments.extend(items)
                
                cursor = response["result"].get("nextPageCursor")
                if not cursor:
                    break
                    
                await asyncio.sleep(0.1)  # Rate limit respect
            
            # Process and cache instruments
            for inst in instruments:
                if inst["status"] == "Trading":
                    symbol = inst["symbol"]
                    self.instruments[symbol] = {
                        "symbol": symbol,
                        "tick_size": Decimal(inst["priceFilter"]["tickSize"]),
                        "qty_step": Decimal(inst["lotSizeFilter"]["qtyStep"]),
                        "min_notional": Decimal(inst["lotSizeFilter"].get("minNotionalValue", "0")),
                        "max_leverage": Decimal(inst["leverageFilter"]["maxLeverage"]),
                        "min_leverage": Decimal(inst["leverageFilter"]["minLeverage"]),
                        "leverage_step": Decimal(inst["leverageFilter"]["leverageStep"]),
                        "base_coin": inst["baseCoin"],
                        "quote_coin": inst["quoteCoin"],
                        "contract_type": inst["contractType"],
                        "delivery_time": inst.get("deliveryTime"),
                        "launch_time": inst.get("launchTime")
                    }
            
            self.last_instrument_update = datetime.now()
            logger.info(f"Loaded {len(self.instruments)} trading instruments")
            
        except Exception as e:
            logger.error(f"Error refreshing instruments: {e}")
            raise
    
    def get_instrument(self, symbol: str) -> Optional[Dict]:
        """Get instrument details for a symbol"""
        return self.instruments.get(symbol)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol"""
        try:
            instrument = self.get_instrument(symbol)
            if not instrument:
                raise ValueError(f"Unknown symbol: {symbol}")
            
            # Validate leverage
            min_lev = int(instrument["min_leverage"])
            max_lev = int(instrument["max_leverage"])
            step = int(instrument["leverage_step"])
            
            if leverage < min_lev or leverage > max_lev:
                raise ValueError(f"Leverage {leverage} out of range [{min_lev}, {max_lev}]")
            
            # Round to nearest step
            leverage = round(leverage / step) * step
            
            response = self.http_client.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            
            if response["retCode"] == 0:
                logger.info(f"Set leverage for {symbol} to {leverage}x")
                return True
            else:
                logger.error(f"Failed to set leverage: {response['retMsg']}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def set_margin_mode(self, symbol: str, mode: str, leverage: Optional[int] = None) -> bool:
        """Set margin mode (cross/isolated) for a symbol"""
        try:
            trade_mode = "0" if mode.lower() == "cross" else "1"
            
            params = {
                "category": "linear",
                "symbol": symbol,
                "tradeMode": trade_mode
            }
            
            if leverage:
                params["buyLeverage"] = str(leverage)
                params["sellLeverage"] = str(leverage)
            
            response = self.http_client.switch_margin_mode(**params)
            
            if response["retCode"] == 0:
                logger.info(f"Set margin mode for {symbol} to {mode}")
                return True
            else:
                logger.error(f"Failed to set margin mode: {response['retMsg']}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting margin mode: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def switch_position_mode(self, mode: str) -> bool:
        """Switch between one-way and hedge position mode"""
        try:
            # 0 = one-way, 3 = hedge
            position_mode = "0" if mode.lower() == "one_way" else "3"
            
            response = self.http_client.switch_position_mode(
                category="linear",
                mode=position_mode
            )
            
            if response["retCode"] == 0:
                logger.info(f"Switched position mode to {mode}")
                return True
            else:
                logger.error(f"Failed to switch position mode: {response['retMsg']}")
                return False
                
        except Exception as e:
            logger.error(f"Error switching position mode: {e}")
            raise
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """Get historical klines data"""
        try:
            response = self.http_client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if response["retCode"] != 0:
                raise Exception(f"Failed to get klines: {response['retMsg']}")
            
            klines = response["result"]["list"]
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            
            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = df[col].astype(float)
            
            df = df.sort_values("timestamp")
            df = df.set_index("timestamp")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching klines: {e}")
            raise
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reduce_only: bool = False,
        position_idx: Optional[int] = None
    ) -> Optional[str]:
        """Place an order"""
        try:
            instrument = self.get_instrument(symbol)
            if not instrument:
                raise ValueError(f"Unknown symbol: {symbol}")
            
            # Round qty and price
            qty = round_to_qty_step(qty, float(instrument["qty_step"]))
            
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(qty),
                "reduceOnly": reduce_only
            }
            
            if price and order_type == "Limit":
                price = round_to_tick(price, float(instrument["tick_size"]))
                params["price"] = str(price)
            
            if position_idx is not None:
                params["positionIdx"] = position_idx
            
            # Add stop loss and take profit
            if stop_loss:
                params["stopLoss"] = str(round_to_tick(stop_loss, float(instrument["tick_size"])))
            
            if take_profit:
                params["takeProfit"] = str(round_to_tick(take_profit, float(instrument["tick_size"])))
            
            response = self.http_client.place_order(**params)
            
            if response["retCode"] == 0:
                order_id = response["result"]["orderId"]
                logger.info(f"Placed {side} {order_type} order for {qty} {symbol}: {order_id}")
                return order_id
            else:
                logger.error(f"Failed to place order: {response['retMsg']}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        try:
            response = self.http_client.cancel_order(
                category="linear",
                symbol=symbol,
                orderId=order_id
            )
            
            if response["retCode"] == 0:
                logger.info(f"Cancelled order {order_id} for {symbol}")
                return True
            else:
                logger.error(f"Failed to cancel order: {response['retMsg']}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            raise
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get current positions"""
        try:
            params = {"category": "linear", "settleCoin": "USDT"}
            if symbol:
                params["symbol"] = symbol
            
            response = self.http_client.get_positions(**params)
            
            if response["retCode"] != 0:
                raise Exception(f"Failed to get positions: {response['retMsg']}")
            
            positions = response["result"]["list"]
            
            # Update cache
            for pos in positions:
                if float(pos["size"]) > 0:
                    self.positions[pos["symbol"]] = pos
                elif pos["symbol"] in self.positions:
                    del self.positions[pos["symbol"]]
            
            return positions
            
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise
    
    async def get_account_info(self) -> Dict:
        """Get account balance and margin info"""
        try:
            response = self.http_client.get_wallet_balance(
                accountType="UNIFIED" if not self.testnet else "CONTRACT"
            )
            
            if response["retCode"] != 0:
                raise Exception(f"Failed to get account info: {response['retMsg']}")
            
            return response["result"]["list"][0]
            
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            raise
    
    async def set_trading_stop(
        self,
        symbol: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        position_idx: Optional[int] = None
    ) -> bool:
        """Update stop loss, take profit, or trailing stop for a position"""
        try:
            instrument = self.get_instrument(symbol)
            if not instrument:
                raise ValueError(f"Unknown symbol: {symbol}")
            
            params = {
                "category": "linear",
                "symbol": symbol
            }
            
            if stop_loss:
                params["stopLoss"] = str(round_to_tick(stop_loss, float(instrument["tick_size"])))
            
            if take_profit:
                params["takeProfit"] = str(round_to_tick(take_profit, float(instrument["tick_size"])))
            
            if trailing_stop:
                params["trailingStop"] = str(trailing_stop)
            
            if position_idx is not None:
                params["positionIdx"] = position_idx
            
            response = self.http_client.set_trading_stop(**params)
            
            if response["retCode"] == 0:
                logger.info(f"Updated trading stops for {symbol}")
                return True
            else:
                logger.error(f"Failed to set trading stop: {response['retMsg']}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting trading stop: {e}")
            raise
    
    async def connect_websockets(self):
        """Connect to WebSocket streams"""
        try:
            # Public WebSocket for market data
            self.public_ws = WebSocket(
                testnet=self.testnet,
                channel_type="linear"
            )
            
            # Private WebSocket for account updates
            self.private_ws = WebSocket(
                testnet=self.testnet,
                channel_type="private",
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            
            logger.info("Connected to Bybit WebSocket streams")
            
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
            raise
    
    def subscribe_klines(self, symbol: str, interval: str, callback):
        """Subscribe to kline updates"""
        if self.public_ws:
            stream = f"kline.{interval}.{symbol}"
            self.public_ws.kline_stream(
                interval=interval,
                symbol=symbol,
                callback=callback
            )
            logger.info(f"Subscribed to {stream}")
    
    def subscribe_orderbook(self, symbol: str, callback):
        """Subscribe to orderbook updates"""
        if self.public_ws:
            self.public_ws.orderbook_stream(
                depth=25,
                symbol=symbol,
                callback=callback
            )
            logger.info(f"Subscribed to orderbook for {symbol}")
    
    def subscribe_positions(self, callback):
        """Subscribe to position updates"""
        if self.private_ws:
            self.private_ws.position_stream(callback=callback)
            logger.info("Subscribed to position updates")
    
    def subscribe_orders(self, callback):
        """Subscribe to order updates"""
        if self.private_ws:
            self.private_ws.order_stream(callback=callback)
            logger.info("Subscribed to order updates")
    
    def subscribe_executions(self, callback):
        """Subscribe to execution updates"""
        if self.private_ws:
            self.private_ws.execution_stream(callback=callback)
            logger.info("Subscribed to execution updates")
    
    async def close(self):
        """Clean up connections"""
        if self.public_ws:
            self.public_ws.exit()
        if self.private_ws:
            self.private_ws.exit()
        logger.info("Closed Bybit connections")