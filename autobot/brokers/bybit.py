from __future__ import annotations
# Enhanced Bybit v5 REST wrapper with proper error handling (ASYNC)
import time, hmac, hashlib, json, os, asyncio
import aiohttp
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP

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
        self.session = None
        self.precisions_cache = {}  # {symbol: (tick_size_str, qty_step_str)}
        self.leverage_cache = {}    # {symbol: int_max_leverage}
        
    async def close(self):
        """Close the underlying aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def configure_active_symbols(self, target_symbols: list = None) -> dict:
        """Fetch AND SET max leverage for specific symbols one-by-one.
        
        Args:
           target_symbols: List of symbols to fetch. If None, does nothing.
        """
        try:
            if not target_symbols:
                logger.warning("[BYBIT] No symbols provided for leverage configuration")
                return {}

            logger.info(f"[BYBIT] 🔧 CONFIGURING LEVERAGE for {len(target_symbols)} symbols (Get & Set)...")
            count = 0
            
            # Fetch & Set individually
            for sym in target_symbols:
                try:
                    # set_leverage internally calls get_max_leverage (cache/fetch) 
                    # and then sends the POST request to set it
                    await self.set_leverage(sym)
                    count += 1
                    
                    # 0.1s delay = ~10 requests/sec max (GET+POST for each symbol)
                    await asyncio.sleep(0.1) 
                    
                    if count % 20 == 0:
                        logger.info(f"[BYBIT] Configured {count}/{len(target_symbols)} symbols...")
                        
                except Exception as e:
                    logger.warning(f"[BYBIT] Failed to configure {sym}: {e}")
                    continue
                        
            logger.info(f"[BYBIT] ✅ Completed configuration for {count} symbols")
            return self.leverage_cache
        except Exception as e:
            logger.error(f"[BYBIT] Failed to configure symbols: {e}")
            return {}

    # ... (rest of methods) ...

    async def get_max_leverage(self, symbol: str) -> int:
        """Get maximum allowed leverage for a symbol, preferring cache."""
        # 1. Check cache first
        if symbol in self.leverage_cache:
            return self.leverage_cache[symbol]
            
        # 2. Fallback to API call if not in cache (rare)
        try:
            instruments = await self.get_instruments_info(symbol=symbol)
            if instruments:
                for inst in instruments:
                    if inst.get('symbol') == symbol:
                        leverage_filter = inst.get('leverageFilter', {})
                        max_lev = leverage_filter.get('maxLeverage', '10')  # Default safely to 10x
                        result = int(float(max_lev))
                        # Cache it for next time
                        self.leverage_cache[symbol] = result
                        return result
            
            # Fallback if not found
            logger.warning(f"Could not get max leverage for {symbol}, using safe 10x")
            return 10
        except Exception as e:
            logger.warning(f"Error getting max leverage for {symbol}: {e}, using safe 10x")
            return 10
            
    def _ts(self) -> str:
        return str(int(time.time() * 1000))

    def _sign(self, ts:str, recv_window:str, body:str) -> str:
        # v5 signature: timestamp + api_key + recv_window + body
        msg = ts + self.cfg.api_key + recv_window + body
        return hmac.new(self.cfg.api_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

    async def _request(self, method:str, path:str, params:dict=None) -> Dict[str, Any]:
        """Async request wrapper"""
        import json # Ensure json is available for all paths
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            
        recv_window = "20000" # Increased to handle time drift
        ts = self._ts()
        
        if method == "POST":
            body = json.dumps(params or {}, separators=(",", ":"))
            query_string = ""
        else:
            body = ""
            # For GET requests, build query string properly
            if params:
                # 1. Sort by key (required by Bybit)
                # 2. Encode values (cursor params have commas, etc)
                import urllib.parse
                
                # Sort params by key
                sorted_items = sorted(params.items())
                
                # Construct query string manually to ensure consistency between SIGNATURE and REQUEST
                # We use urllib.parse.quote for values to handle special chars like commas in cursors
                query_string = "&".join([f"{k}={urllib.parse.quote(str(v))}" for k, v in sorted_items])
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
            # Compose URL for this base - append query_string manually for GET requests
            # This ensures signature matches actual request (aiohttp params= lowercases keys)
            url = base + path
            if method == "GET" and query_string:
                url += "?" + query_string
            
            # Simple retry with exponential backoff for transient failures
            max_retries = 3
            backoff = 0.5
            for attempt in range(max_retries):
                try:
                    # Use manual URL with query_string for GET, data for POST
                    async with self.session.request(method, url, headers=headers, data=body if method=="POST" else None, timeout=15) as r:
                         # Read raw bytes first to avoid encoding issues causing "truncation"
                        raw_data = await r.read()
                        try:
                             resp_text = raw_data.decode('utf-8')
                        except UnicodeDecodeError:
                             # Fallback if utf-8 fails (rare)
                             resp_text = raw_data.decode('utf-8', errors='replace')

                        try:
                            # Try to parse JSON even if status != 200 to get error msg
                            j = json.loads(resp_text)
                        except json.JSONDecodeError as e:
                            r.raise_for_status() # If not JSON, check HTTP status
                            # Log concise error with details
                            logger.error(f"JSON Decode Error: {e.msg} (Pos: {e.pos})")
                            logger.error(f"Raw Bytes: {len(raw_data)} | Text Len: {len(resp_text)}")
                            logger.error(f"Snippet: {resp_text[:200]}")
                            raise RuntimeError(f"Bybit invalid JSON response (len={len(resp_text)}): {resp_text[:100]}")
                            
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

                except aiohttp.ClientError as e:
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
                    await asyncio.sleep(max(0.2, min(5.0, sleep_s)))

            # If we exhausted retries on primary and have an alt, fall through to alt
            if base_idx == 0 and len(bases) > 1:
                logger.warning("Bybit primary base_url unreachable — attempting ALT base_url")

        # All bases failed
        logger.error(f"Bybit request failed across all endpoints: {last_err}")
        if isinstance(last_err, Exception):
            raise last_err
        raise RuntimeError("Bybit request failed")

    async def get_instruments_info(self, category:str="linear", symbol:Optional[str]=None) -> list:
        """Get instrument info (tick size, lot size, max leverage, etc.)"""
        try:
            params = {"category": category}
            
            # If specific symbol, NO limit/pagination needed - just direct fetch
            if symbol:
                params["symbol"] = symbol
                # [FIX] Force limit=1 to prevent large buffer/truncation issues
                params["limit"] = 1
                resp = await self._request("GET", "/v5/market/instruments-info", params)
                if resp and resp.get("result"):
                    return resp["result"].get("list", [])
                return []
            
            # Only use limit/pagination for BULK fetch
            params["limit"] = 200 
            
            # Handle pagination if fetching all symbols
            all_items = []
            cursor = ""
            
            while True:
                if cursor:
                    params["cursor"] = cursor
                
                resp = await self._request("GET", "/v5/market/instruments-info", params)
                
                if resp and resp.get("result"):
                    items = resp["result"].get("list", [])
                    all_items.extend(items)
                    cursor = resp["result"].get("nextPageCursor", "")
                    if not cursor:
                        break
                else:
                    break
                    
            return all_items
        except Exception as e:
            logger.error(f"Failed to get instruments info: {e}")
            return []

    async def get_api_key_info(self) -> dict:
        """Get API key information including expiry date.
        
        Returns dict with:
        - expiredAt: Expiry datetime string
        - deadlineDay: Days remaining (if no IP bound)
        - days_left: Calculated days until expiry
        """
        try:
            resp = await self._request("GET", "/v5/user/query-api", {})
            logger.info(f"API key info response: {resp}")  # Debug logging
            
            if not resp:
                logger.error("get_api_key_info: Response is None")
                return {"days_left": None}
            
            if not resp.get("result"):
                logger.error(f"get_api_key_info: No 'result' in response. Keys: {resp.keys()}")
                return {"days_left": None}
            
            result = resp["result"]
            logger.info(f"API key result: {result}")  # Debug logging
            
            # Bybit might return result as a list or dict
            if isinstance(result, list):
                if len(result) > 0:
                    result = result[0]  # Take first API key
                else:
                    logger.error("get_api_key_info: result list is empty")
                    return {"days_left": None}
            
            expired_at = result.get("expiredAt", "")
            deadline_day = result.get("deadlineDay", 0)
            
            # Calculate days left from expiredAt
            days_left = None
            if expired_at:
                try:
                    from datetime import datetime
                    # Parse Bybit datetime format (timestamp in milliseconds)
                    if isinstance(expired_at, (int, float)):
                        # Unix timestamp in milliseconds
                        expiry = datetime.fromtimestamp(int(expired_at) / 1000)
                    elif 'T' in str(expired_at):
                        expiry = datetime.fromisoformat(str(expired_at).replace('Z', '+00:00'))
                    else:
                        expiry = datetime.strptime(str(expired_at), "%Y-%m-%d %H:%M:%S")
                    
                    days_left = (expiry - datetime.now()).days
                    logger.info(f"Calculated days_left: {days_left} from expiredAt: {expired_at}")
                except Exception as parse_err:
                    logger.error(f"Failed to parse expiredAt '{expired_at}': {parse_err}")
                    days_left = deadline_day if deadline_day else None
            elif deadline_day:
                days_left = deadline_day
                logger.info(f"Using deadlineDay: {days_left}")
            
            return {
                "expiredAt": expired_at,
                "deadlineDay": deadline_day,
                "days_left": days_left
            }
        except Exception as e:
            logger.error(f"Failed to get API key info: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"days_left": None}

    async def get_balance(self) -> Optional[float]:
        """Get USDT balance (robust): try UNIFIED → CONTRACT → SPOT; prefer equity."""
        try:
            account_types = ["UNIFIED", "CONTRACT", "SPOT"]
            for acct in account_types:
                try:
                    resp = await self._request("GET", "/v5/account/wallet-balance", {"accountType": acct})
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

    async def get_max_leverage(self, symbol: str) -> int:
        """Get maximum allowed leverage for a symbol, preferring cache."""
        # 1. Check cache first
        if symbol in self.leverage_cache:
            return self.leverage_cache[symbol]
            
        # 2. Fallback to API call if not in cache (rare)
        try:
            instruments = await self.get_instruments_info(symbol=symbol)
            if instruments:
                for inst in instruments:
                    if inst.get('symbol') == symbol:
                        leverage_filter = inst.get('leverageFilter', {})
                        max_lev = leverage_filter.get('maxLeverage', '10')  # Default safely to 10x
                        result = int(float(max_lev))
                        # Cache it for next time
                        self.leverage_cache[symbol] = result
                        return result
            
            # Fallback if not found
            logger.warning(f"Could not get max leverage for {symbol}, using safe 10x")
            return 10
        except Exception as e:
            logger.warning(f"Error getting max leverage for {symbol}: {e}, using safe 10x")
            return 10

    async def set_leverage(self, symbol: str, leverage: int = None) -> Dict[str, Any] | None:
        """Set leverage for a symbol. If leverage is None, uses max allowed. Handles errors gracefully."""
        try:
            # If no leverage specified, use maximum allowed
            if leverage is None:
                leverage = await self.get_max_leverage(symbol)
                logger.info(f"🔧 Using MAX leverage for {symbol}: {leverage}x")
            
            data = {
                "category": "linear",
                "symbol": symbol,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage)
            }
            
            # Helper to try setting leverage
            async def try_set(lev):
                d = data.copy()
                d['buyLeverage'] = str(lev)
                d['sellLeverage'] = str(lev)
                return await self._request("POST", "/v5/position/set-leverage", d)

            try:
                # Primary Attempt
                return await try_set(leverage)
                
            except RuntimeError as e:
                err_msg = str(e).lower()
                
                # Case 1: Already set (Success)
                if "leverage not modified" in err_msg:
                    return {"retCode": 0, "result": "already_set"}
                
                # Case 2: Invalid Leverage (Too High)
                if "leverage invalid" in err_msg:
                    # If we tried something > 10x, fallback to 10x
                    if leverage > 10:
                        logger.warning(f"Leverage {leverage}x rejected for {symbol}, trying 10x...")
                        try:
                            return await try_set(10)
                        except RuntimeError as e2:
                            # If 10x fails, fallback to 1x
                            if "leverage invalid" in str(e2).lower():
                                logger.warning(f"Leverage 10x also rejected for {symbol}, trying 1x...")
                                return await try_set(1)
                            raise e2
                            
                    # If we started > 1x but <= 10x, fallback to 1x
                    elif leverage > 1:
                        logger.warning(f"Leverage {leverage}x rejected for {symbol}, trying 1x...")
                        return await try_set(1)
                        
                # Re-raise other errors
                raise e

        except Exception as e:
            logger.warning(f"Failed to set leverage for {symbol}: {e}")
            return None
        return None

    async def get_ticker(self, symbol: str) -> dict:
        """Get current ticker data."""
        try:
            resp = await self._request("GET", "/v5/market/tickers", {"category": "linear", "symbol": symbol})
            if resp and resp.get("result"):
                items = resp["result"].get("list", [])
                if items:
                    return items[0]
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
        return {}
    
    async def place_market(self, symbol:str, side:str, qty:float, reduce_only:bool=False,
                      take_profit:float=None, stop_loss:float=None) -> Dict[str, Any]:
        """Place market order with optional TP/SL (bracket order).
        
        When TP/SL are provided, they are set atomically with the order.
        This means the position is protected from the instant it opens.
        """
        # Convert 'long'/'short' to Bybit's 'Buy'/'Sell'
        bybit_side = "Buy" if side.lower() in ["long", "buy"] else "Sell"
        
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
        
        if take_profit or stop_loss is not None:
            tick_size, _ = await self._get_precisions(symbol)
            
            # Add TP/SL as bracket order for instant protection
            if take_profit is not None:
                data["takeProfit"] = self._round_price(take_profit, tick_size)
                data["tpTriggerBy"] = "LastPrice"
            
            if stop_loss is not None:
                data["stopLoss"] = self._round_price(stop_loss, tick_size)
                data["slTriggerBy"] = "LastPrice"
        
        bracket_status = ""
        if take_profit or stop_loss:
            bracket_status = f" [BRACKET: TP={take_profit} SL={stop_loss}]"
        
        logger.info(f"📤 Market order: {symbol} {side} qty={qty}{bracket_status}")
        return await self._request("POST", "/v5/order/create", data)
    
    async def place_limit(self, symbol: str, side: str, qty: float, price: float, 
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
            take_profit: Optional TP price (set at order creation for instant protection)
            stop_loss: Optional SL price (set at order creation for instant protection)
            post_only: If True, use PostOnly. If False, use GTC (recommended).
            
        Returns:
            API response dict with orderId if successful
        """
        bybit_side = "Buy" if side.lower() == "long" else "Sell"
        time_in_force = "PostOnly" if post_only else "GTC"
        
        # Round everything first
        tick_size, _ = await self._get_precisions(symbol)
        final_price = self._round_price(price, tick_size)
        
        data = {
            "category": "linear",
            "symbol": symbol,
            "side": bybit_side,
            "orderType": "Limit",
            "qty": str(qty),
            "price": final_price,
            "timeInForce": time_in_force,
            "reduceOnly": False,  # Entry order, not reduce-only
            "positionIdx": 0  # One-way mode
        }
        
        # Add TP/SL as bracket order for instant protection
        if take_profit is not None:
            data["takeProfit"] = self._round_price(take_profit, tick_size)
            data["tpTriggerBy"] = "LastPrice"
        
        if stop_loss is not None:
            data["stopLoss"] = self._round_price(stop_loss, tick_size)
            data["slTriggerBy"] = "LastPrice"
        
        bracket_status = ""
        if take_profit or stop_loss:
            bracket_status = f" [BRACKET: TP={take_profit} SL={stop_loss}]"
        
        logger.info(f"📤 Placing {time_in_force} limit order: {symbol} {side} qty={qty} price={price}{bracket_status}")
        response = await self._request("POST", "/v5/order/create", data)
        
        # Verify order status after placement
        if response and response.get('retCode') == 0:
            order_id = response.get('result', {}).get('orderId')
            if order_id:
                asyncio.sleep(0.3)  # Brief delay for Bybit to process
                status = await self.get_order_status(symbol, order_id)
                if status:
                    order_status = status.get('orderStatus', 'Unknown')
                    logger.info(f"📋 Order {order_id[:16]} status immediately after placement: {order_status}")
                    if order_status in ['Cancelled', 'Rejected', 'Deactivated']:
                        logger.warning(f"⚠️ Order was immediately {order_status}!")
                        response['_immediately_cancelled'] = True
                        response['_cancel_reason'] = order_status
        
        return response
    
    async def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific order.
        
        Returns order info including:
        - orderStatus: 'New', 'PartiallyFilled', 'Filled', 'Cancelled', 'Rejected'
        - cumExecQty: Filled quantity
        - avgPrice: Average fill price
        
        Returns None if order not found.
        """
        try:
            resp = await self._request("GET", "/v5/order/realtime", {
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id
            })
            
            if resp and resp.get("result"):
                orders = resp["result"].get("list", [])
                if orders:
                    return orders[0]
            
            # Order might be in history if already filled/cancelled
            resp = await self._request("GET", "/v5/order/history", {
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
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a specific order.
        
        Returns True if cancelled successfully or already cancelled.
        """
        try:
            data = {
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id
            }
            resp = await self._request("POST", "/v5/order/cancel", data)
            
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

    async def get_executions(self, symbol: str, limit: int = 50, start_time: Optional[int] = None) -> list:
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
            resp = await self._request("GET", "/v5/execution/list", params)
            if resp and resp.get("result"):
                return resp["result"].get("list", []) or []
            return []
        except Exception as e:
            logger.debug(f"Failed to get executions for {symbol}: {e}")
            return []
    
    async def get_closed_pnl(self, symbol: str, limit: int = 10) -> list:
        """Get closed PnL records for a symbol.
        
        This returns the ACTUAL realized PnL from Bybit, not a guess.
        Use this to definitively determine if a trade was a win or loss.
        
        Returns list of closed trades with closedPnl, avgEntryPrice, avgExitPrice etc.
        """
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "limit": str(min(max(1, limit), 200)),
            }
            resp = await self._request("GET", "/v5/position/closed-pnl", params)
            if resp and resp.get("result"):
                return resp["result"].get("list", []) or []
            return []
        except Exception as e:
            logger.debug(f"Failed to get closed pnl for {symbol}: {e}")
            return []
    
    async def get_all_closed_pnl(self, limit: int = 100, start_time: int = None) -> list:
        """Get ALL closed PnL records across all symbols.
        
        This returns the ACTUAL realized PnL from Bybit for all symbols.
        Used for the /pnl command to show exchange-verified P&L.
        
        Args:
            limit: Max number of records (up to 200)
            start_time: Optional start time in milliseconds
            
        Returns:
            List of closed trade records with closedPnl, symbol, side, etc.
        """
        try:
            params = {
                "category": "linear",
                "limit": str(min(max(1, limit), 200)),
            }
            if start_time:
                params["startTime"] = str(start_time)
            resp = await self._request("GET", "/v5/position/closed-pnl", params)
            if resp and resp.get("result"):
                return resp["result"].get("list", []) or []
            return []
        except Exception as e:
            logger.debug(f"Failed to get all closed pnl: {e}")
            return []
    
    async def get_position(self, symbol:str) -> Optional[Dict[str, Any]]:
        """Get current position for a symbol"""
        try:
            resp = await self._request("GET", "/v5/position/list", {
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

    async def set_tpsl(self, symbol:str, take_profit:float, stop_loss:float, qty:float=None) -> Dict[str, Any]:
        """Set position TP/SL - Use Partial mode with Limit TP for better fills"""
        
        # If qty not provided, try to get current position size
        position_qty = qty
        if not position_qty:
            try:
                # Add small delay to ensure position is registered
                await asyncio.sleep(0.5)
                
                positions = await self.get_positions()
                for pos in positions:
                    if pos.get("symbol") == symbol and float(pos.get("size", 0)) > 0:
                        position_qty = pos.get("size")
                        logger.info(f"Found position size for {symbol}: {position_qty}")
                        break
            except Exception as e:
                logger.warning(f"Could not get position size: {e}")
        
        # Precision rounding
        tick_size, _ = await self._get_precisions(symbol)
        tp_str = self._round_price(take_profit, tick_size)
        sl_str = self._round_price(stop_loss, tick_size)
        
        # Use Partial mode with sizes if we have them
        if position_qty:
            data = {
                "category": "linear",
                "symbol": symbol,
                "takeProfit": tp_str,
                "stopLoss": sl_str,
                "tpSize": str(position_qty),       # Required for Partial mode
                "slSize": str(position_qty),       # Required for Partial mode
                "tpLimitPrice": tp_str,            # Use proper rounded price
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
                "takeProfit": tp_str,
                "stopLoss": sl_str,
                "tpLimitPrice": tp_str,  # ensure Limit TP even in Full mode
                "tpTriggerBy": "LastPrice",
                "slTriggerBy": "LastPrice",
                "tpslMode": "Full",                # Full mode doesn't need sizes
                "tpOrderType": "Limit",            # Limit for TP (requested)
                "slOrderType": "Market",           # Market for SL
                "positionIdx": 0
            }
        
        return self._request("POST", "/v5/position/trading-stop", data)

    async def _get_precisions(self, symbol: str, force_fresh: bool = False) -> tuple[str, str]:
        """Get (tick_size, qty_step) as strings for a symbol (cached).
        
        CRITICAL: This is used for SL/TP/price rounding. 
        If cache miss, fetches from API. Always logs to help debug.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTCUSDT")
            force_fresh: If True, bypass cache and fetch fresh data from API
        """
        # Check cache first (unless force_fresh)
        if not force_fresh and symbol in self.precisions_cache:
            return self.precisions_cache[symbol]
        
        # Cache miss or force_fresh - fetch from API
        if force_fresh:
            logger.info(f"📐 Force-fetching fresh precision for {symbol}...")
        else:
            logger.warning(f"⚠️ Precision cache miss for {symbol} - fetching from API...")
        
        try:
            instruments = await self.get_instruments_info(symbol=symbol)
            if instruments:
                for i in instruments:
                    if i['symbol'] == symbol:
                        ts = i.get('priceFilter', {}).get('tickSize', '0.0001')
                        qs = i.get('lotSizeFilter', {}).get('qtyStep', '0.001')
                        self.precisions_cache[symbol] = (ts, qs)
                        logger.info(f"✅ Cached precisions for {symbol}: tickSize={ts}, qtyStep={qs}")
                        return (ts, qs)
        except Exception as e:
            logger.error(f"❌ Failed to fetch precisions for {symbol}: {e}")
        
        # Fallback - use conservative defaults and log loudly
        logger.error(f"❌ USING FALLBACK PRECISION for {symbol}: tickSize=0.0001, qtyStep=0.001")
        self.precisions_cache[symbol] = ("0.0001", "0.001")  # Cache fallback too
        return ("0.0001", "0.001")

    async def preload_all_precisions(self) -> int:
        """Preload precision info for ALL USDT perpetual symbols.
        
        Call this at bot startup to avoid per-trade API calls.
        Returns number of symbols cached.
        """
        try:
            logger.info("📥 Preloading precision data for all USDT perpetuals...")
            
            # Fetch ALL instruments at once (single API call)
            resp = await self._request("GET", "/v5/market/instruments-info", {"category": "linear", "limit": 1000})
            if not resp or resp.get("retCode") != 0:
                logger.error(f"Failed to fetch instruments: {resp}")
                return 0
            
            instruments = resp.get("result", {}).get("list", [])
            count = 0
            
            for inst in instruments:
                symbol = inst.get("symbol", "")
                if symbol.endswith("USDT"):
                    ts = inst.get('priceFilter', {}).get('tickSize', '0.0001')
                    qs = inst.get('lotSizeFilter', {}).get('qtyStep', '0.001')
                    self.precisions_cache[symbol] = (ts, qs)
                    count += 1
            
            logger.info(f"✅ Preloaded precisions for {count} USDT perpetual symbols")
            return count
            
        except Exception as e:
            logger.error(f"❌ Failed to preload precisions: {e}")
            return 0

    def _round_price(self, price: float, tick_size: str) -> str:
        """Round price to tick size string."""
        d_price = Decimal(str(price))
        d_tick = Decimal(tick_size)
        # Round to nearest tick
        rounded = (d_price / d_tick).quantize(Decimal('1')) * d_tick
        return str(rounded)  # Use str() to preserve quantized precision (e.g. 0.7540)

    async def set_sl_only(self, symbol: str, stop_loss: float, qty: float = None) -> Dict[str, Any]:
        """Set only Stop Loss for a position using trading-stop.

        Uses Partial mode when qty provided, otherwise Full mode.
        Returns response dict with retCode = 0 on success.
        """
        try:
            # ============================================
            # VALIDATION: Ensure SL is reasonable
            # ============================================
            if stop_loss <= 0:
                raise ValueError(f"Invalid SL: {stop_loss} <= 0")
            
            # === AUTO-ROUNDING FIX ===
            # ALWAYS fetch fresh precision before setting SL (avoid stale cache issues)
            tick_size, _ = await self._get_precisions(symbol, force_fresh=True)
            final_sl_str = self._round_price(stop_loss, tick_size)
            logger.info(f"📐 Rounded SL for {symbol}: {stop_loss} -> {final_sl_str} (tick: {tick_size})")
            
            # === SANITY CHECK: Reject obviously wrong SL values ===
            # Values like 2128000 are clearly wrong (should be ~0.02128)
            if stop_loss > 100000:
                raise ValueError(f"SL value {stop_loss} is absurdly large - likely a calculation error!")
            
            # Get current price to validate SL is reasonable
            try:
                ticker = await self.get_ticker(symbol)
                if ticker:
                    current_price = float(ticker.get('lastPrice', 0))
                    if current_price > 0:
                        # Check if SL is within reasonable range (100x current price at max)
                        if stop_loss > current_price * 100 or stop_loss < current_price / 100:
                            raise ValueError(f"SL {stop_loss:.8f} is way off from price {current_price:.8f} - rejecting!")
                        distance_pct = abs(stop_loss - current_price) / current_price * 100
                        if distance_pct > 20:
                            raise ValueError(f"SL {stop_loss:.4f} is {distance_pct:.1f}% from price {current_price:.4f} - too far!")
            except ValueError:
                raise  # Re-raise validation errors
            except Exception as e:
                logger.debug(f"Could not validate SL distance: {e}")
            
            if qty:
                data = {
                    "category": "linear",
                    "symbol": symbol,
                    "stopLoss": final_sl_str,
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
                    "stopLoss": final_sl_str,
                    "slTriggerBy": "LastPrice",
                    "tpslMode": "Full",
                    "slOrderType": "Market",
                    "positionIdx": 0,
                }
            
            # === DEBUG: Log exact request body ===
            import json as json_module
            debug_body = json_module.dumps(data, separators=(",", ":"))
            logger.info(f"🔍 DEBUG SL REQUEST for {symbol}: {debug_body}")
            
            resp = await self._request("POST", "/v5/position/trading-stop", data)
            
            # Validate response
            if resp and resp.get("retCode") == 0:
                logger.info(f"✅ SL SET: {symbol} @ {stop_loss} (qty={qty})")
                return resp
            else:
                ret_code = resp.get("retCode") if resp else "no response"
                ret_msg = resp.get("retMsg") if resp else "no message"
                logger.error(f"❌ SL SET FAILED: {symbol} - retCode={ret_code}, msg={ret_msg}")
                raise RuntimeError(f"SL set failed: {ret_msg}")
                
        except Exception as e:
            logger.error(f"Failed to set SL-only for {symbol}: {e}")
            raise
    
    async def set_trailing_sl(self, symbol: str, initial_sl: float, trail_distance: float, 
                        activation_price: float = None, side: str = None) -> Dict[str, Any]:
        """Set stop loss with Bybit's NATIVE trailing stop mechanism.
        
        This uses Bybit's `trailingStop` parameter which lets the exchange
        handle trailing logic, avoiding constant SL update API calls.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            initial_sl: Initial stop loss price
            trail_distance: Trail distance in PRICE units (not %)
            activation_price: Optional price at which trailing activates
            side: "long" or "short" for validation
            
        Returns:
            API response dict
        """
        try:
            # Fetch precision for formatting
            tick_size, _ = await self._get_precisions(symbol, force_fresh=True)
            
            # Format all prices correctly
            sl_str = self._round_price(initial_sl, tick_size)
            trail_str = self._round_price(trail_distance, tick_size)
            
            logger.info(f"📐 Setting native trailing for {symbol}:")
            logger.info(f"   Initial SL: {initial_sl} -> {sl_str}")
            logger.info(f"   Trail dist: {trail_distance} -> {trail_str}")
            
            # Validate values
            if initial_sl <= 0 or trail_distance <= 0:
                raise ValueError(f"Invalid values: SL={initial_sl}, trail={trail_distance}")
            
            # Get current price to validate
            ticker = await self.get_ticker(symbol)
            if ticker:
                current_price = float(ticker.get('lastPrice', 0))
                if current_price > 0:
                    # Validate SL is reasonable (within 20% of price)
                    sl_distance_pct = abs(initial_sl - current_price) / current_price * 100
                    if sl_distance_pct > 20:
                        raise ValueError(f"SL {sl_str} is {sl_distance_pct:.1f}% from price - too far!")
            
            # Build request data
            data = {
                "category": "linear",
                "symbol": symbol,
                "stopLoss": sl_str,
                "trailingStop": trail_str,  # Bybit handles trailing!
                "slTriggerBy": "LastPrice",
                "tpslMode": "Full",
                "positionIdx": 0,
            }
            
            # Add activation price if provided
            if activation_price:
                act_str = self._round_price(activation_price, tick_size)
                data["activePrice"] = act_str
                logger.info(f"   Activation: {activation_price} -> {act_str}")
            
            logger.info(f"🚀 Sending native trailing stop request for {symbol}")
            resp = await self._request("POST", "/v5/position/trading-stop", data)
            
            if resp and resp.get("retCode") == 0:
                logger.info(f"✅ NATIVE TRAILING SET: {symbol} SL={sl_str} Trail={trail_str}")
                return resp
            else:
                ret_code = resp.get("retCode") if resp else "no response"
                ret_msg = resp.get("retMsg") if resp else "no message"
                logger.error(f"❌ TRAILING SET FAILED: {symbol} - retCode={ret_code}, msg={ret_msg}")
                raise RuntimeError(f"Trailing stop set failed: {ret_msg}")
                
        except Exception as e:
            logger.error(f"Failed to set trailing SL for {symbol}: {e}")
            raise
    
    def verify_position_sl(self, symbol: str) -> tuple:
        """Verify the current SL set on Bybit for a position.
        
        Returns (sl_price, tp_price) or (None, None) if no position.
        """
        try:
            pos = self.get_position(symbol)
            if pos and float(pos.get('size', 0)) > 0:
                sl = float(pos.get('stopLoss', 0))
                tp = float(pos.get('takeProfit', 0))
                return (sl, tp)
            return (None, None)
        except Exception as e:
            logger.error(f"Failed to verify SL for {symbol}: {e}")
            return (None, None)

    async def place_reduce_only_limit(self, symbol: str, side: str, qty: float, price: float,
                                 post_only: bool = True, reduce_only: bool = True) -> Dict[str, Any]:
        """Place a reduce-only limit order (optionally PostOnly) for TP purposes.

        side: "Buy" or "Sell" relative to order direction. For long TP use "Sell"; for short TP use "Buy".
        """
        tif = "PostOnly" if post_only else "GTC"
        # Round price
        tick_size, _ = await self._get_precisions(symbol)
        final_price = self._round_price(price, tick_size)
        
        data = {
            "category": "linear",
            "symbol": symbol,
            "side": side.capitalize(),
            "orderType": "Limit",
            "qty": str(qty),
            "price": final_price,
            "timeInForce": tif,
            "reduceOnly": reduce_only,
            "positionIdx": 0,
        }
        return await self._request("POST", "/v5/order/create", data)
    
    async def get_positions(self) -> list:
        """Get ALL open positions with pagination. Tries with settleCoin first, then falls back if empty."""
        try:
            # Helper to fetch with specific params
            async def fetch_with_params(params):
                all_pos = []
                cursor = None
                page_num = 1
                while True:
                    current_params = params.copy()
                    if cursor:
                        current_params["cursor"] = cursor
                    
                    logger.info(f"[POSITIONS] Fetching page {page_num} with params: {current_params}")
                    resp = await self._request("GET", "/v5/position/list", current_params)
                    
                    if not resp or resp.get("retCode") != 0:
                        logger.error(f"[POSITIONS] API error or empty response: {resp}")
                        break
                        
                    result = resp.get("result", {})
                    positions = result.get("list", [])
                    all_pos.extend(positions)
                    
                    cursor = result.get("nextPageCursor")
                    if not cursor:
                        break
                    page_num += 1
                return all_pos

            # Attempt 1: Standard params (category=linear, settleCoin=USDT)
            params1 = {
                "category": "linear",
                "settleCoin": "USDT",
                "limit": 200
            }
            positions = await fetch_with_params(params1)
            
            # Check if we got any "open" positions
            open_count = sum(1 for p in positions if float(p.get('size', 0)) > 0)
            
            # Attempt 2: If we got 0 open positions, try without settleCoin (Unified Account sometimes prefers this)
            if open_count == 0:
                logger.info("[POSITIONS] Attempt 1 returned 0 open positions. Retrying without settleCoin filter...")
                params2 = {
                    "category": "linear",
                    "limit": 200
                }
                positions_retry = await fetch_with_params(params2)
                retry_open_count = sum(1 for p in positions_retry if float(p.get('size', 0)) > 0)
                
                if retry_open_count > 0:
                    logger.info(f"[POSITIONS] Attempt 2 success! Found {retry_open_count} positions.")
                    positions = positions_retry
                else:
                    logger.info("[POSITIONS] Attempt 2 also returned 0 positions.")

            # Final count
            final_open = [p for p in positions if float(p.get('size', 0)) > 0]
            logger.info(f"[POSITIONS] Final Total: {len(positions)} positions, {len(final_open)} with size > 0")
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    async def get_open_orders(self, symbol: str = None) -> list:
        """Get open orders (reduce-only TP etc.) for a symbol or all symbols."""
        try:
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol
            resp = await self._request("GET", "/v5/order/realtime", params)
            if resp and resp.get("result"):
                return resp["result"].get("list", []) or []
            return []
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    async def place_conditional_stop(self, symbol: str, side: str, trigger_price: float, qty: float,
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
        return await self._request("POST", "/v5/order/create", data)
    
    async def cancel_all_orders(self, symbol:str=None) -> bool:
        """Cancel all orders for a symbol or all symbols"""
        try:
            data = {
                "category": "linear",
                "settleCoin": "USDT"  # Required for cancel-all
            }
            if symbol:
                data["symbol"] = symbol
            resp = await self._request("POST", "/v5/order/cancel-all", data)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            # Not critical if no orders to cancel
            return True
    
    async def get_klines(self, symbol:str, interval:str, limit:int=200, start:Optional[int]=None, end:Optional[int]=None) -> list:
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
            resp = await self._request("GET", "/v5/market/kline", params)
            
            if resp and resp.get("result"):
                klines = resp["result"]["list"]
                # Bybit returns newest first, reverse for chronological order
                return list(reversed(klines))
            return []
            
        except Exception as e:
            logger.error(f"Failed to get klines for {symbol}: {e}")
            return []

    async def get_api_key_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current API key, including expiration."""
        try:
            resp = await self._request("GET", "/v5/user/query-api")
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
