from __future__ import annotations
# Enhanced Bybit v5 REST wrapper with proper error handling
import time, hmac, hashlib, requests, json, os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class BybitConfig:
    base_url: str
    api_key: str
    api_secret: str
    alt_base_url: Optional[str] = None
    use_alt_on_fail: bool = False

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
        recv_window = "20000" # Increased to handle time drift
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
        
        # Build ordered list of base URLs: primary first, then optional alternate
        bases = [self.cfg.base_url]
        if self.cfg.use_alt_on_fail and self.cfg.alt_base_url:
            try:
                # Avoid duplicates
                if self.cfg.alt_base_url not in bases:
                    bases.append(self.cfg.alt_base_url)
            except Exception:
                pass

        import random
        last_err = None
        for base_idx, base in enumerate(bases):
            # Compose URL for this base
            url = base + path
            if method == "GET" and query_string:
                url += "?" + query_string

            # Simple retry with exponential backoff for transient failures
            max_retries = 3
            backoff = 0.5
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
                    logger.warning(f"Bybit request attempt {attempt+1}/{max_retries} failed ({'ALT' if base_idx==1 else 'PRIMARY'}): {e}")
                except RuntimeError as e:
                    last_err = e
                    logger.warning(f"Bybit API attempt {attempt+1}/{max_retries} transient error ({'ALT' if base_idx==1 else 'PRIMARY'}): {e}")
                except Exception as e:
                    last_err = e
                    logger.warning(f"Unexpected Bybit error attempt {attempt+1}/{max_retries} ({'ALT' if base_idx==1 else 'PRIMARY'}): {e}")

                # Backoff with jitter before next try
                if attempt < max_retries - 1:
                    sleep_s = backoff * (2 ** attempt) * (1.0 + random.uniform(-0.2, 0.2))
                    time.sleep(max(0.2, min(5.0, sleep_s)))

            # If we exhausted retries on primary and have an alt, fall through to alt
            if base_idx == 0 and len(bases) > 1:
                logger.warning("Bybit primary base_url unreachable — attempting ALT base_url")

        # All bases failed
        logger.error(f"Bybit request failed across all endpoints: {last_err}")
        if isinstance(last_err, Exception):
            raise last_err
        raise RuntimeError("Bybit request failed")

    def get_instruments_info(self, category:str="linear", symbol:Optional[str]=None) -> list:
        """Get instrument info (tick size, lot size, max leverage, etc.)"""
        try:
            params = {"category": category, "limit": 1000}
            if symbol:
                params["symbol"] = symbol
            
            # Handle pagination if fetching all symbols
            all_items = []
            cursor = ""
            
            while True:
                if cursor:
                    params["cursor"] = cursor
                
                resp = self._request("GET", "/v5/market/instruments-info", params)
                
                if resp and resp.get("result"):
                    items = resp["result"].get("list", [])
                    all_items.extend(items)
                    cursor = resp["result"].get("nextPageCursor", "")
                    if not cursor or symbol: # If specific symbol, no need to paginate
                        break
                else:
                    break
                    
            return all_items
        except Exception as e:
            logger.error(f"Failed to get instruments info: {e}")
            return []

    def get_balance(self) -> Optional[float]:
        """Get USDT balance (robust): try UNIFIED → CONTRACT → SPOT; prefer equity."""
        try:
            account_types = ["UNIFIED", "CONTRACT", "SPOT"]
            for acct in account_types:
                try:
                    resp = self._request("GET", "/v5/account/wallet-balance", {"accountType": acct})
                except Exception as _e:
                    logger.debug(f"Wallet balance fetch failed for {acct}: {_e}")
                    continue
                try:
                    if not (resp and resp.get("result")):
                        continue
                    lists = resp["result"].get("list", [])
                    for item in lists:
                        for coin in item.get("coin", []):
                            if str(coin.get("coin")) == "USDT":
                                # Prefer equity → walletBalance → availableToWithdraw
                                fields = [coin.get("equity"), coin.get("walletBalance"), coin.get("availableToWithdraw")]
                                for val in fields:
                                    try:
                                        if val is not None and str(val) != "":
                                            v = float(val)
                                            # Accept first non-NaN number; keep searching only if zero across all fields
                                            if v != 0.0:
                                                return v
                                            zero_candidate = v
                                        else:
                                            continue
                                    except (ValueError, TypeError):
                                        continue
                                # If all fields parse but are 0.0, return zero and continue to next acct only if not found later
                                try:
                                    # Return the last parsed zero if we reached here
                                    return float(zero_candidate)  # may be 0.0
                                except Exception:
                                    return 0.0
                except Exception as _pe:
                    logger.debug(f"Parse wallet balance failed for {acct}: {_pe}")
                    continue
            # If no USDT found across account types
            return None
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None

    def set_leverage(self, symbol: str, leverage: int = 10) -> Dict[str, Any] | None:
        """Set leverage for a symbol - FIXED 10x only."""
        try:
            data = {
                "category": "linear",
                "symbol": symbol,
                "buyLeverage": "10",
                "sellLeverage": "10"
            }
            resp = self._request("POST", "/v5/position/set-leverage", data)
            return resp
        except RuntimeError as e:
            msg = str(e).lower()
            if "leverage not modified" in msg:
                return {"result": "already_set"}
            logger.warning(f"Failed to set leverage for {symbol}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to set leverage for {symbol}: {e}")
            return None
        return None
    
    def place_market(self, symbol:str, side:str, qty:float, reduce_only:bool=False) -> Dict[str, Any]:
        """Place market order"""
        # Convert 'long'/'short' to Bybit's 'Buy'/'Sell'
        bybit_side = "Buy" if side.lower() == "long" else "Sell"
        
        data = {
            "category": "linear",
            "symbol": symbol,
            "side": bybit_side,
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "IOC",
            "reduceOnly": reduce_only,
            "positionIdx": 0  # One-way mode
        }
        return self._request("POST", "/v5/order/create", data)
    
    def place_limit(self, symbol: str, side: str, qty: float, price: float, 
                    take_profit: float = None, stop_loss: float = None,
                    post_only: bool = False) -> Dict[str, Any]:
        """Place a limit order with optional TP/SL (bracket order).
        
        When TP/SL are provided, they are set atomically with the order.
        This means the position is protected from the instant it opens.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: 'long' or 'short'
            qty: Order quantity
            price: Limit price
            take_profit: Optional TP price (set at order creation for instant protection)
            stop_loss: Optional SL price (set at order creation for instant protection)
            post_only: If True, use PostOnly. If False, use GTC (recommended).
            
        Returns:
            API response dict with orderId if successful
        """
        bybit_side = "Buy" if side.lower() == "long" else "Sell"
        time_in_force = "PostOnly" if post_only else "GTC"
        
        data = {
            "category": "linear",
            "symbol": symbol,
            "side": bybit_side,
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(price),
            "timeInForce": time_in_force,
            "reduceOnly": False,  # Entry order, not reduce-only
            "positionIdx": 0  # One-way mode
        }
        
        # Add TP/SL as bracket order for instant protection
        if take_profit is not None:
            data["takeProfit"] = str(take_profit)
            data["tpTriggerBy"] = "LastPrice"
        
        if stop_loss is not None:
            data["stopLoss"] = str(stop_loss)
            data["slTriggerBy"] = "LastPrice"
        
        bracket_status = ""
        if take_profit or stop_loss:
            bracket_status = f" [BRACKET: TP={take_profit} SL={stop_loss}]"
        
        logger.info(f"📤 Placing {time_in_force} limit order: {symbol} {side} qty={qty} price={price}{bracket_status}")
        response = self._request("POST", "/v5/order/create", data)
        
        # Verify order status after placement
        if response and response.get('retCode') == 0:
            order_id = response.get('result', {}).get('orderId')
            if order_id:
                import time as time_module
                time_module.sleep(0.3)  # Brief delay for Bybit to process
                status = self.get_order_status(symbol, order_id)
                if status:
                    order_status = status.get('orderStatus', 'Unknown')
                    logger.info(f"📋 Order {order_id[:16]} status immediately after placement: {order_status}")
                    if order_status in ['Cancelled', 'Rejected', 'Deactivated']:
                        logger.warning(f"⚠️ Order was immediately {order_status}!")
                        response['_immediately_cancelled'] = True
                        response['_cancel_reason'] = order_status
        
        return response
    
    def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific order.
        
        Returns order info including:
        - orderStatus: 'New', 'PartiallyFilled', 'Filled', 'Cancelled', 'Rejected'
        - cumExecQty: Filled quantity
        - avgPrice: Average fill price
        
        Returns None if order not found.
        """
        try:
            resp = self._request("GET", "/v5/order/realtime", {
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id
            })
            
            if resp and resp.get("result"):
                orders = resp["result"].get("list", [])
                if orders:
                    return orders[0]
            
            # Order might be in history if already filled/cancelled
            resp = self._request("GET", "/v5/order/history", {
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id
            })
            
            if resp and resp.get("result"):
                orders = resp["result"].get("list", [])
                if orders:
                    return orders[0]
                    
            return None
        except Exception as e:
            logger.error(f"Failed to get order status for {symbol}/{order_id}: {e}")
            return None
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a specific order.
        
        Returns True if cancelled successfully or already cancelled.
        """
        try:
            data = {
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id
            }
            resp = self._request("POST", "/v5/order/cancel", data)
            
            if resp and resp.get("retCode") == 0:
                logger.info(f"✅ Cancelled order {order_id} for {symbol}")
                return True
            else:
                # Check if already cancelled/filled
                ret_msg = str(resp.get("retMsg", "")).lower() if resp else ""
                if "not exist" in ret_msg or "already" in ret_msg or "cancelled" in ret_msg:
                    logger.info(f"Order {order_id} already cancelled/filled")
                    return True
                logger.warning(f"Cancel order failed: {resp}")
                return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_executions(self, symbol: str, limit: int = 50, start_time: Optional[int] = None) -> list:
        """Fetch recent execution fills for a symbol (used to confirm position closes).

        Args:
            symbol: e.g., 'BTCUSDT'
            limit: number of rows (Bybit caps to 50)
            start_time: optional start timestamp (ms) for pagination

        Returns:
            List of execution records; empty list on error.
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "limit": str(min(max(1, limit), 50)),
            }
            if start_time is not None:
                params["startTime"] = str(int(start_time))
            resp = self._request("GET", "/v5/execution/list", params)
            if resp and resp.get("result"):
                return resp["result"].get("list", []) or []
            return []
        except Exception as e:
            logger.debug(f"Failed to get executions for {symbol}: {e}")
            return []
    
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
            # Fallback: Use Full mode without sizes (simpler API), but prefer Limit TP per requirements
            logger.warning(f"Using Full mode for {symbol} - no position size available (placing Limit TP)")
            data = {
                "category": "linear",
                "symbol": symbol,
                "takeProfit": str(take_profit),
                "stopLoss": str(stop_loss),
                "tpLimitPrice": str(take_profit),  # ensure Limit TP even in Full mode
                "tpTriggerBy": "LastPrice",
                "slTriggerBy": "LastPrice",
                "tpslMode": "Full",                # Full mode doesn't need sizes
                "tpOrderType": "Limit",            # Limit for TP (requested)
                "slOrderType": "Market",           # Market for SL
                "positionIdx": 0
            }
        
        return self._request("POST", "/v5/position/trading-stop", data)

    def set_sl_only(self, symbol: str, stop_loss: float, qty: float = None) -> Dict[str, Any]:
        """Set only Stop Loss for a position using trading-stop.

        Uses Partial mode when qty provided, otherwise Full mode.
        """
        try:
            if qty:
                data = {
                    "category": "linear",
                    "symbol": symbol,
                    "stopLoss": str(stop_loss),
                    "slSize": str(qty),
                    "slTriggerBy": "LastPrice",
                    "tpslMode": "Partial",
                    "slOrderType": "Market",
                    "positionIdx": 0,
                }
            else:
                data = {
                    "category": "linear",
                    "symbol": symbol,
                    "stopLoss": str(stop_loss),
                    "slTriggerBy": "LastPrice",
                    "tpslMode": "Full",
                    "slOrderType": "Market",
                    "positionIdx": 0,
                }
            return self._request("POST", "/v5/position/trading-stop", data)
        except Exception as e:
            logger.error(f"Failed to set SL-only for {symbol}: {e}")
            raise

    def place_reduce_only_limit(self, symbol: str, side: str, qty: float, price: float,
                                 post_only: bool = True, reduce_only: bool = True) -> Dict[str, Any]:
        """Place a reduce-only limit order (optionally PostOnly) for TP purposes.

        side: "Buy" or "Sell" relative to order direction. For long TP use "Sell"; for short TP use "Buy".
        """
        tif = "PostOnly" if post_only else "GTC"
        data = {
            "category": "linear",
            "symbol": symbol,
            "side": side.capitalize(),
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(price),
            "timeInForce": tif,
            "reduceOnly": reduce_only,
            "positionIdx": 0,
        }
        return self._request("POST", "/v5/order/create", data)
    
    def get_positions(self) -> list:
        """Get open positions"""
        try:
            resp = self._request("GET", "/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
            return resp["result"]["list"]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_open_orders(self, symbol: str = None) -> list:
        """Get open orders (reduce-only TP etc.) for a symbol or all symbols."""
        try:
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol
            resp = self._request("GET", "/v5/order/realtime", params)
            if resp and resp.get("result"):
                return resp["result"].get("list", []) or []
            return []
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def place_conditional_stop(self, symbol: str, side: str, trigger_price: float, qty: float,
                               reduce_only: bool = True) -> Dict[str, Any]:
        """Place a conditional stop-market reduce-only order as SL fallback.

        side is the order direction to close the position (Sell for long, Buy for short).
        """
        data = {
            "category": "linear",
            "symbol": symbol,
            "side": side.capitalize(),
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "GTC",
            "reduceOnly": reduce_only,
            "triggerPrice": str(trigger_price),
            "triggerBy": "LastPrice",
            "positionIdx": 0,
        }
        return self._request("POST", "/v5/order/create", data)
    
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
