# Enhanced Bybit v5 REST wrapper with proper error handling
import time, hmac, hashlib, requests, json, os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class BybitConfig:
    base_url:str
    api_key:str
    api_secret:str

class Bybit:
    def __init__(self, cfg:BybitConfig):
        self.cfg = cfg
        self.session = requests.Session()
        
    def _ts(self) -> str:
        return str(int(time.time() * 1000))

    def _sign(self, ts:str, recv_window:str, body:str) -> str:
        # v5 signature: timestamp + api_key + recv_window + body
        msg = ts + self.cfg.api_key + recv_window + body
        return hmac.new(self.cfg.api_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

    def _request(self, method:str, path:str, params:dict=None) -> Dict[str, Any]:
        recv_window = "5000"
        ts = self._ts()
        
        if method == "POST":
            body = json.dumps(params or {}, separators=(",", ":"))
            query_string = ""
        else:
            body = ""
            # For GET requests, build query string
            if params:
                query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            else:
                query_string = ""
            
        # Sign with query string for GET, body for POST
        sign_payload = query_string if method == "GET" else body
        sign = self._sign(ts, recv_window, sign_payload)
        
        headers = {
            "X-BAPI-API-KEY": self.cfg.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-SIGN": sign,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }
        
        url = self.cfg.base_url + path
        if method == "GET" and query_string:
            url += "?" + query_string
        
        try:
            if method == "POST":
                r = self.session.post(url, headers=headers, data=body, timeout=15)
            else:
                r = self.session.get(url, headers=headers, timeout=15)
                
            r.raise_for_status()
            j = r.json()
            
            if str(j.get("retCode")) != "0":
                logger.error(f"Bybit API error: {j}")
                raise RuntimeError(f"Bybit error: {j.get('retMsg', 'Unknown error')}")
                
            return j
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def get_balance(self) -> Optional[float]:
        """Get USDT balance"""
        try:
            # Use UNIFIED account type (most common for retail)
            resp = self._request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
            if resp and resp.get("result"):
                for item in resp["result"]["list"]:
                    for coin in item.get("coin", []):
                        if coin["coin"] == "USDT":
                            # Try multiple balance fields
                            balance_str = coin.get("availableToWithdraw") or coin.get("walletBalance") or coin.get("equity") or "0"
                            
                            # Handle empty strings and convert to float
                            try:
                                if balance_str and balance_str != "":
                                    return float(balance_str)
                            except (ValueError, TypeError):
                                logger.debug(f"Could not parse balance: {balance_str}")
                                continue
            
            # If no USDT found, return None (non-critical)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            # Not critical - bot can continue without balance display
            return None

    def place_market(self, symbol:str, side:str, qty:float, reduce_only:bool=False) -> Dict[str, Any]:
        """Place market order"""
        data = {
            "category": "linear",
            "symbol": symbol,
            "side": side.capitalize(),   # "Buy" or "Sell"
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "IOC",
            "reduceOnly": reduce_only,
            "positionIdx": 0  # One-way mode
        }
        return self._request("POST", "/v5/order/create", data)

    def set_tpsl(self, symbol:str, take_profit:float, stop_loss:float) -> Dict[str, Any]:
        """Set position TP/SL with Limit TP and Market SL"""
        data = {
            "category": "linear",
            "symbol": symbol,
            "takeProfit": str(take_profit),
            "stopLoss": str(stop_loss),
            "tpTriggerBy": "LastPrice",
            "slTriggerBy": "LastPrice",
            "tpslMode": "Full",
            "tpOrderType": "Limit",     # Limit order for Take Profit (better fill)
            "slOrderType": "Market",     # Market order for Stop Loss (guaranteed fill)
            "positionIdx": 0
        }
        return self._request("POST", "/v5/position/trading-stop", data)
    
    def get_positions(self) -> list:
        """Get open positions"""
        try:
            resp = self._request("GET", "/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
            return resp["result"]["list"]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def cancel_all_orders(self, symbol:str=None) -> bool:
        """Cancel all orders for a symbol or all symbols"""
        try:
            data = {
                "category": "linear",
                "settleCoin": "USDT"  # Required for cancel-all
            }
            if symbol:
                data["symbol"] = symbol
            resp = self._request("POST", "/v5/order/cancel-all", data)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            # Not critical if no orders to cancel
            return True
    
    def get_klines(self, symbol:str, interval:str, limit:int=200) -> list:
        """Get historical kline/candlestick data"""
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,  # 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
                "limit": str(min(limit, 200))  # Max 200 per request
            }
            resp = self._request("GET", "/v5/market/kline", params)
            
            if resp and resp.get("result"):
                klines = resp["result"]["list"]
                # Bybit returns newest first, reverse for chronological order
                return list(reversed(klines))
            return []
            
        except Exception as e:
            logger.error(f"Failed to get klines for {symbol}: {e}")
            return []