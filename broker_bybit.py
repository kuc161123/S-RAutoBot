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
        
        # Simple retry with exponential backoff for transient failures
        import random
        max_retries = 3
        backoff = 0.5
        last_err = None
        for attempt in range(max_retries):
            try:
                if method == "POST":
                    r = self.session.post(url, headers=headers, data=body, timeout=15)
                else:
                    r = self.session.get(url, headers=headers, timeout=15)

                r.raise_for_status()
                j = r.json()

                if str(j.get("retCode")) != "0":
                    # Treat specific non-critical cases as success
                    try:
                        if path == "/v5/position/set-leverage" and str(j.get("retCode")) == "110043":
                            logger.debug("Leverage not modified — already set; treating as success")
                            return j
                    except Exception:
                        pass
                    # Retry only for likely transient messages
                    msg = str(j.get('retMsg', '')).lower()
                    if any(k in msg for k in ["timeout", "limit", "too many", "system busy", "server error"]):
                        raise RuntimeError(msg)
                    logger.error(f"Bybit API error: {j}")
                    raise RuntimeError(f"Bybit error: {j.get('retMsg', 'Unknown error')}")

                return j

            except requests.exceptions.RequestException as e:
                last_err = e
                logger.warning(f"Bybit request attempt {attempt+1}/{max_retries} failed: {e}")
            except RuntimeError as e:
                last_err = e
                logger.warning(f"Bybit API attempt {attempt+1}/{max_retries} transient error: {e}")
            except Exception as e:
                last_err = e
                logger.warning(f"Unexpected Bybit error attempt {attempt+1}/{max_retries}: {e}")

            # Backoff with jitter before next try
            if attempt < max_retries - 1:
                sleep_s = backoff * (2 ** attempt) * (1.0 + random.uniform(-0.2, 0.2))
                time.sleep(max(0.2, min(5.0, sleep_s)))

        # All retries failed
        logger.error(f"Bybit request failed after {max_retries} attempts: {last_err}")
        if isinstance(last_err, Exception):
            raise last_err
        raise RuntimeError("Bybit request failed")

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

    def set_leverage(self, symbol:str, leverage:int) -> Dict[str, Any]:
        """Set leverage for a symbol"""
        try:
            data = {
                "category": "linear",
                "symbol": symbol,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage)
            }
            return self._request("POST", "/v5/position/set-leverage", data)
        except RuntimeError as e:
            # Check if it's the "leverage not modified" error (already set to requested value)
            if "leverage not modified" in str(e).lower():
                logger.debug(f"{symbol}: Leverage already set to {leverage}x")
                return {"result": "already_set"}  # Return success-like response
            else:
                logger.warning(f"Failed to set leverage for {symbol}: {e}")
                return None
        except Exception as e:
            logger.warning(f"Failed to set leverage for {symbol}: {e}")
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
    
    def get_position(self, symbol:str) -> Optional[Dict[str, Any]]:
        """Get current position for a symbol"""
        try:
            resp = self._request("GET", "/v5/position/list", {
                "category": "linear",
                "symbol": symbol
            })
            if resp and resp.get("result"):
                positions = resp["result"].get("list", [])
                if positions:
                    return positions[0]  # Return first position
            return None
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None

    def set_tpsl(self, symbol:str, take_profit:float, stop_loss:float, qty:float=None) -> Dict[str, Any]:
        """Set position TP/SL - Use Partial mode with Limit TP for better fills"""
        
        # If qty not provided, try to get current position size
        position_qty = qty
        if not position_qty:
            try:
                # Add small delay to ensure position is registered
                import time
                time.sleep(0.5)
                
                positions = self.get_positions()
                for pos in positions:
                    if pos.get("symbol") == symbol and float(pos.get("size", 0)) > 0:
                        position_qty = pos.get("size")
                        logger.info(f"Found position size for {symbol}: {position_qty}")
                        break
            except Exception as e:
                logger.warning(f"Could not get position size: {e}")
        
        # Use Partial mode with sizes if we have them
        if position_qty:
            data = {
                "category": "linear",
                "symbol": symbol,
                "takeProfit": str(take_profit),
                "stopLoss": str(stop_loss),
                "tpSize": str(position_qty),       # Required for Partial mode
                "slSize": str(position_qty),       # Required for Partial mode
                "tpLimitPrice": str(take_profit),  # Limit price for TP
                "tpTriggerBy": "LastPrice",
                "slTriggerBy": "LastPrice",
                "tpslMode": "Partial",             # Partial mode for better fills
                "tpOrderType": "Limit",            # Limit order for Take Profit
                "slOrderType": "Market",           # Market order for Stop Loss
                "positionIdx": 0
            }
        else:
            # Fallback: Use Full mode without sizes (simpler API)
            logger.warning(f"Using Full mode for {symbol} - no position size available")
            data = {
                "category": "linear",
                "symbol": symbol,
                "takeProfit": str(take_profit),
                "stopLoss": str(stop_loss),
                "tpTriggerBy": "LastPrice",
                "slTriggerBy": "LastPrice",
                "tpslMode": "Full",                # Full mode doesn't need sizes
                "tpOrderType": "Market",           # Market for both in Full mode
                "slOrderType": "Market",           # Market for both in Full mode
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
    
    def get_klines(self, symbol:str, interval:str, limit:int=200, start:Optional[int]=None, end:Optional[int]=None) -> list:
        """Get historical kline/candlestick data.

        Args:
            symbol: e.g., 'BTCUSDT'
            interval: '1','3','5','15','60','240','D',...
            limit: max records to return (Bybit caps at 200)
            start: optional start timestamp in ms
            end: optional end timestamp in ms
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,  # 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
                "limit": str(min(limit, 200))  # Max 200 per request
            }
            if start is not None:
                params["start"] = str(int(start))
            if end is not None:
                params["end"] = str(int(end))
            resp = self._request("GET", "/v5/market/kline", params)
            
            if resp and resp.get("result"):
                klines = resp["result"]["list"]
                # Bybit returns newest first, reverse for chronological order
                return list(reversed(klines))
            return []
            
        except Exception as e:
            logger.error(f"Failed to get klines for {symbol}: {e}")
            return []

    def get_api_key_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current API key, including expiration."""
        try:
            resp = self._request("GET", "/v5/user/query-api")
            if resp and resp.get("result"):
                api_keys = resp["result"].get("list", [])
                # Find the key that is currently being used
                for key_info in api_keys:
                    if key_info.get("apiKey") == self.cfg.api_key:
                        return key_info
            return None
        except Exception as e:
            logger.error(f"Failed to get API key info: {e}")
            return None
