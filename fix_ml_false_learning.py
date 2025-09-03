#!/usr/bin/env python3
"""
Fix for ML Learning from False Data Bug

The bug: Bot is marking open positions as "closed" and recording fake outcomes
This causes ML to learn from false data which ruins the model

Root cause: check_closed_positions() is incorrectly detecting positions as closed
when they're actually still open on the exchange
"""

# The problematic code in live_bot.py (around line 207-289):
"""
async def check_closed_positions(self, book: Book, meta: dict = None, ml_scorer=None, reset_symbol_state=None):
    # Get current positions from exchange
    current_positions = self.bybit.get_positions()
    current_symbols = {p['symbol'] for p in current_positions if float(p.get('size', 0)) > 0}
    
    # BUG: This might not include all open positions if API call fails or returns partial data
    for symbol, pos in list(book.positions.items()):
        if symbol not in current_symbols:  # <-- FALSE TRIGGER
            # Position marked as closed but might still be open!
"""

# FIXED VERSION:
async def check_closed_positions(self, book: Book, meta: dict = None, ml_scorer=None, reset_symbol_state=None):
    """Check for positions that have been closed and record them"""
    try:
        # Get current positions from exchange with retry
        max_retries = 3
        current_positions = None
        
        for attempt in range(max_retries):
            try:
                current_positions = self.bybit.get_positions()
                if current_positions is not None:
                    break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to get positions: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        
        if current_positions is None:
            logger.warning("Could not get positions from exchange, skipping closed position check")
            return
        
        # Build set of symbols that definitely have open positions
        current_symbols = set()
        for p in current_positions:
            # More robust checking - ensure position is truly open
            symbol = p.get('symbol')
            size = float(p.get('size', 0))
            # Check both size and position value to be sure
            position_value = float(p.get('positionValue', 0))
            
            if symbol and (size > 0 or position_value > 0):
                current_symbols.add(symbol)
                logger.debug(f"{symbol} has open position: size={size}, value={position_value}")
        
        # Track closed positions more carefully
        closed_positions = []
        for symbol, pos in list(book.positions.items()):
            if symbol not in current_symbols:
                # Double-check this position is really closed
                # Get specific position info to confirm
                try:
                    specific_pos = self.bybit.get_position(symbol)
                    if specific_pos and float(specific_pos.get('size', 0)) > 0:
                        logger.warning(f"{symbol} still has open position, not marking as closed")
                        continue
                except:
                    pass
                
                # Only mark as closed if we're certain
                logger.info(f"Confirmed {symbol} position is closed")
                closed_positions.append((symbol, pos))
        
        # Only record truly closed positions
        for symbol, pos in closed_positions:
            # Get the ACTUAL exit price from order history if possible
            try:
                # Try to get recent closed orders for this symbol
                orders = self.bybit.get_closed_orders(symbol, limit=10)
                exit_price = None
                exit_reason = "unknown"
                
                # Find the closing order
                for order in orders:
                    if order.get('reduceOnly', False):  # This was a closing order
                        exit_price = float(order.get('avgPrice', 0))
                        
                        # Determine if it was TP or SL
                        if 'takeProfit' in order.get('stopOrderType', '').lower():
                            exit_reason = "tp"
                        elif 'stopLoss' in order.get('stopOrderType', '').lower():
                            exit_reason = "sl"
                        else:
                            exit_reason = "manual"
                        break
                
                # Fallback to last price if we couldn't find the order
                if exit_price is None or exit_price == 0:
                    if symbol in self.frames and len(self.frames[symbol]) > 0:
                        exit_price = float(self.frames[symbol]['close'].iloc[-1])
                        logger.warning(f"Using last candle price for {symbol} as exit price")
                
                if exit_price and exit_price > 0:
                    # Calculate actual PnL
                    if pos.side == "long":
                        pnl = exit_price - pos.entry
                        # Determine reason based on actual exit vs targets
                        if exit_price >= pos.tp * 0.98:  # Within 2% of TP
                            exit_reason = "tp"
                        elif exit_price <= pos.sl * 1.02:  # Within 2% of SL
                            exit_reason = "sl"
                    else:  # short
                        pnl = pos.entry - exit_price
                        if exit_price <= pos.tp * 1.02:
                            exit_reason = "tp"
                        elif exit_price >= pos.sl * 0.98:
                            exit_reason = "sl"
                    
                    # Calculate R-multiple
                    risk = abs(pos.entry - pos.sl)
                    pnl_r = pnl / risk if risk > 0 else 0
                    
                    # Only update ML with REAL outcomes
                    if exit_reason in ["tp", "sl"]:
                        # Clear win or loss
                        outcome = "win" if exit_reason == "tp" else "loss"
                        
                        # Update ML scorer
                        if ml_scorer is not None and hasattr(pos, 'entry_time'):
                            ml_scorer.update_signal_outcome(symbol, pos.entry_time, outcome, pnl_r)
                            logger.info(f"[{symbol}] ML updated with REAL outcome: {outcome} ({pnl_r:.2f}R)")
                    else:
                        # Manual close or uncertain - don't train ML on this
                        logger.info(f"[{symbol}] Closed manually or unclear, not updating ML")
                    
                    # Record trade for statistics
                    self.record_closed_trade(symbol, pos, exit_price, exit_reason, 
                                            meta.get(symbol, {}).get("max_leverage", 1.0) if meta else 1.0)
                
            except Exception as e:
                logger.error(f"Error processing closed position {symbol}: {e}")
            
            # Remove from book and reset state
            book.positions.pop(symbol)
            if reset_symbol_state:
                reset_symbol_state(symbol)
                logger.info(f"[{symbol}] Position tracking cleared, state reset")
                
    except Exception as e:
        logger.error(f"Error in check_closed_positions: {e}")


# Additional helper method to add to Bybit class:
def get_position(self, symbol: str) -> dict:
    """Get specific position for a symbol"""
    try:
        resp = self._request("GET", "/v5/position/list", {
            "category": "linear",
            "symbol": symbol
        })
        positions = resp.get("result", {}).get("list", [])
        return positions[0] if positions else None
    except Exception as e:
        logger.error(f"Failed to get position for {symbol}: {e}")
        return None

def get_closed_orders(self, symbol: str, limit: int = 50) -> list:
    """Get recently closed orders for a symbol"""
    try:
        resp = self._request("GET", "/v5/order/history", {
            "category": "linear",
            "symbol": symbol,
            "limit": limit
        })
        return resp.get("result", {}).get("list", [])
    except Exception as e:
        logger.error(f"Failed to get closed orders for {symbol}: {e}")
        return []


print("""
FIX SUMMARY:
===========
1. More robust position checking (with retries)
2. Double-check before marking position as closed
3. Get ACTUAL exit price from order history
4. Only train ML on clear TP/SL hits
5. Ignore manual/unclear closes for ML training

TO APPLY THIS FIX:
==================
1. Update the check_closed_positions() method in live_bot.py
2. Add helper methods to Bybit class
3. Reset ML data to clear false learnings
4. Restart bot

This ensures ML only learns from REAL, CONFIRMED trade outcomes!
""")