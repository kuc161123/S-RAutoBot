#!/usr/bin/env python3
"""
Fixed version of check_closed_positions that only records REAL closed trades
Replace the method in live_bot.py with this version
"""

# Replace the check_closed_positions method in live_bot.py (around line 207) with this:

async def check_closed_positions(self, book: Book, meta: dict = None, ml_scorer=None, reset_symbol_state=None):
    """Check for positions that have been closed and record them"""
    try:
        # Get current positions from exchange
        current_positions = self.bybit.get_positions()
        if current_positions is None:
            logger.warning("Could not get positions from exchange, skipping check")
            return
        
        # Build set of symbols with CONFIRMED open positions
        current_symbols = set()
        for p in current_positions:
            symbol = p.get('symbol')
            size = float(p.get('size', 0))
            
            # Only add if we're SURE there's an open position
            if symbol and size > 0:
                current_symbols.add(symbol)
                # Log for debugging
                logger.debug(f"{symbol} confirmed open: size={size}")
        
        # Find positions that are TRULY closed
        closed_positions = []
        for symbol, pos in list(book.positions.items()):
            if symbol not in current_symbols:
                # IMPORTANT: Only mark as closed if we can verify the exit
                # Check if we have recent order history
                try:
                    # Try to get the last order for this symbol
                    resp = self.bybit._request("GET", "/v5/order/history", {
                        "category": "linear",
                        "symbol": symbol,
                        "limit": 20
                    })
                    orders = resp.get("result", {}).get("list", [])
                    
                    # Look for a FILLED reduce-only order (closing order)
                    found_close = False
                    exit_price = 0
                    exit_reason = "unknown"
                    
                    for order in orders:
                        # Check if this is a closing order
                        if (order.get("reduceOnly") == True and 
                            order.get("orderStatus") == "Filled"):
                            
                            found_close = True
                            exit_price = float(order.get("avgPrice", 0))
                            
                            # Check if TP or SL
                            trigger_price = float(order.get("triggerPrice", 0))
                            if trigger_price > 0:
                                if pos.side == "long":
                                    if trigger_price >= pos.tp * 0.98:
                                        exit_reason = "tp"
                                    elif trigger_price <= pos.sl * 1.02:
                                        exit_reason = "sl"
                                else:  # short
                                    if trigger_price <= pos.tp * 1.02:
                                        exit_reason = "tp"
                                    elif trigger_price >= pos.sl * 0.98:
                                        exit_reason = "sl"
                            break
                    
                    # ONLY record if we found a confirmed close
                    if found_close and exit_price > 0:
                        logger.info(f"[{symbol}] CONFIRMED closed at {exit_price:.4f} ({exit_reason})")
                        closed_positions.append((symbol, pos, exit_price, exit_reason))
                    else:
                        # Can't confirm close - might be API lag
                        logger.debug(f"[{symbol}] Not in positions but can't confirm close - keeping in book")
                        
                except Exception as e:
                    logger.warning(f"Could not verify {symbol} close status: {e}")
                    # Don't mark as closed if we can't verify
                    continue
        
        # Process ONLY confirmed closed positions
        for symbol, pos, exit_price, exit_reason in closed_positions:
            try:
                # Calculate actual PnL
                if pos.side == "long":
                    pnl = exit_price - pos.entry
                else:
                    pnl = pos.entry - exit_price
                
                # Calculate R-multiple
                risk = abs(pos.entry - pos.sl)
                pnl_r = pnl / risk if risk > 0 else 0
                
                # Determine outcome for ML
                if exit_reason == "tp":
                    outcome = "win"
                elif exit_reason == "sl":
                    outcome = "loss"
                else:
                    outcome = None  # Don't train on manual/unknown closes
                
                # Update ML ONLY for clear TP/SL hits
                if ml_scorer is not None and outcome is not None and hasattr(pos, 'entry_time'):
                    ml_scorer.update_signal_outcome(symbol, pos.entry_time, outcome, pnl_r)
                    logger.info(f"[{symbol}] ML updated: {outcome} ({pnl_r:.2f}R) - VERIFIED")
                
                # Record trade for statistics
                leverage = meta.get(symbol, {}).get("max_leverage", 1.0) if meta else 1.0
                self.record_closed_trade(symbol, pos, exit_price, exit_reason, leverage)
                
                # Remove from book
                book.positions.pop(symbol)
                
                # Reset strategy state
                if reset_symbol_state:
                    reset_symbol_state(symbol)
                    logger.info(f"[{symbol}] State reset - ready for new signals")
                    
            except Exception as e:
                logger.error(f"Error processing closed position {symbol}: {e}")
                
    except Exception as e:
        logger.error(f"Error in check_closed_positions: {e}")


print("""
========================================
FIX READY TO APPLY
========================================

This fixed version:
✅ Only marks positions as closed when CONFIRMED by order history
✅ Gets REAL exit prices from filled orders
✅ Only trains ML on verified TP/SL hits
✅ Ignores uncertain closes (no false data)
✅ Keeps positions in book if can't verify

TO APPLY:
1. Stop your bot
2. Replace check_closed_positions() in live_bot.py with this version
3. Restart bot

The ML will now only learn from REAL, VERIFIED trade outcomes!

NO MORE FALSE LEARNING!
""")