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

    def _request(self, method:str, path:str, data:dict=None) -> Dict[str, Any]:
        recv_window = "5000"
        ts = self._ts()
        
        if method == "POST":
            body = json.dumps(data or {}, separators=(",", ":"))
        else:
            body = ""
            
        sign = self._sign(ts, recv_window, body)
        
        headers = {
            "X-BAPI-API-KEY": self.cfg.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-SIGN": sign,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }
        
        url = self.cfg.base_url + path
        
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
            resp = self._request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
            for coin in resp["result"]["list"][0]["coin"]:
                if coin["coin"] == "USDT":
                    return float(coin["walletBalance"])
            return None
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
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
        """Set position TP/SL"""
        data = {
            "category": "linear",
            "symbol": symbol,
            "takeProfit": str(take_profit),
            "stopLoss": str(stop_loss),
            "tpTriggerBy": "LastPrice",
            "slTriggerBy": "LastPrice",
            "tpslMode": "Full",
            "tpOrderType": "Market",
            "slOrderType": "Market",
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
            data = {"category": "linear"}
            if symbol:
                data["symbol"] = symbol
            resp = self._request("POST", "/v5/order/cancel-all", data)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return False