import asyncio
import json
import yaml
import os
import time
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import websockets
from dotenv import load_dotenv

# Import strategy based on configuration
from strategy_pullback import Settings  # Settings are the same for both strategies
from live_bot_selector import get_strategy_module
from position_mgr import RiskConfig, Book, Position
from sizer import Sizer
from broker_bybit import Bybit, BybitConfig
from telegram_bot import TGBot
from candle_storage_postgres import CandleStorage
# Use enhanced PostgreSQL trade tracker for persistence
try:
    from trade_tracker_postgres import TradeTrackerPostgres as TradeTracker, Trade
    USING_POSTGRES_TRACKER = True
except ImportError:
    from trade_tracker import TradeTracker, Trade
    USING_POSTGRES_TRACKER = False
from multi_websocket_handler import MultiWebSocketHandler
from ml_scorer_mean_reversion import get_mean_reversion_scorer

# Import ML scorers for parallel strategy system
try:
    from ml_signal_scorer_immediate import get_immediate_scorer
    from phantom_trade_tracker import get_phantom_tracker
    from enhanced_mr_scorer import get_enhanced_mr_scorer
    from mr_phantom_tracker import get_mr_phantom_tracker
    from enhanced_market_regime import get_enhanced_market_regime, get_regime_summary
    logger = logging.getLogger(__name__)
    logger.info("Using Enhanced Parallel ML System (Pullback + Mean Reversion)")
    ML_AVAILABLE = True
    ENHANCED_ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    ENHANCED_ML_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Enhanced ML not available: {e}")
    # Fallback to original ML if available
    try:
        from ml_signal_scorer_immediate import get_immediate_scorer
        from phantom_trade_tracker import get_phantom_tracker
        ML_AVAILABLE = True
        logger.info("Using Original ML Scorer only")
    except ImportError as e2:
        logger.warning(f"No ML available: {e2}")

# Import symbol data collector for future ML
try:
    from symbol_data_collector import get_symbol_collector
    SYMBOL_COLLECTOR_AVAILABLE = True
    logger.info("Symbol data collector initialized for future ML")
except ImportError as e:
    SYMBOL_COLLECTOR_AVAILABLE = False
    logger.warning(f"Symbol data collector not available: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log which trade tracker we're using
if USING_POSTGRES_TRACKER:
    logger.info("Using PostgreSQL trade tracker for persistence")
else:
    logger.info("Using JSON trade tracker")

# Load environment variables
load_dotenv()

def new_frame():
    return pd.DataFrame(columns=["open","high","low","close","volume"])

def meta_for(symbol:str, cfg_meta:dict):
    if symbol in cfg_meta: 
        return cfg_meta[symbol]
    return cfg_meta.get("default", {"qty_step":0.001,"min_qty":0.001,"tick_size":0.1})

def replace_env_vars(config:dict) -> dict:
    """Replace ${VAR} placeholders with environment variables"""
    def replace_in_value(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            env_value = os.getenv(env_var)
            if env_value is None:
                logger.warning(f"Environment variable {env_var} not found")
                return value
            # Try to parse numbers
            if env_value.isdigit():
                return int(env_value)
            try:
                return float(env_value)
            except:
                return env_value
        elif isinstance(value, dict):
            return {k: replace_in_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [replace_in_value(item) for item in value]
        return value
    
    return {k: replace_in_value(v) for k, v in config.items()}

class TradingBot:
    def __init__(self):
        self.running = False
        self.ws = None
        self.frames: Dict[str, pd.DataFrame] = {}
        self.tg: Optional[TGBot] = None
        self.bybit = None
        self.storage = CandleStorage()  # Will use DATABASE_URL from environment
        self.last_save_time = datetime.now()
        self.trade_tracker = TradeTracker()  # Initialize trade tracker
        
    async def load_or_fetch_initial_data(self, symbols:list[str], timeframe:str):
        """Load candles from database or fetch from API if not available"""
        logger.info("Loading historical data from database...")
        
        # First, try to load from database
        stored_frames = self.storage.load_all_frames(symbols)
        
        # Add rate limiting for large symbol lists
        api_delay = 0.1 if len(symbols) > 100 else 0  # 100ms delay for >100 symbols
        
        for idx, symbol in enumerate(symbols):
            if symbol in stored_frames and len(stored_frames[symbol]) >= 200:  # Need 200 for reliable analysis
                # Use stored data if we have enough candles
                self.frames[symbol] = stored_frames[symbol]
                logger.info(f"[{symbol}] Loaded {len(stored_frames[symbol])} candles from database")
            else:
                # Fetch from API if not in database or insufficient data
                try:
                    # Rate limit API calls
                    if api_delay > 0 and idx > 0:
                        await asyncio.sleep(api_delay)
                    
                    logger.info(f"[{symbol}] Fetching from API (not enough in database)")
                    klines = self.bybit.get_klines(symbol, timeframe, limit=200)
                    
                    # Retry once if no data
                    if not klines:
                        logger.info(f"[{symbol}] Retrying API fetch...")
                        await asyncio.sleep(1)
                        klines = self.bybit.get_klines(symbol, timeframe, limit=200)
                
                    if klines:
                        # Convert to DataFrame
                        data = []
                        for k in klines:
                            # k = [timestamp, open, high, low, close, volume, turnover]
                            data.append({
                                'open': float(k[1]),
                                'high': float(k[2]),
                                'low': float(k[3]),
                                'close': float(k[4]),
                                'volume': float(k[5])
                            })
                        
                        df = pd.DataFrame(data)
                        # Set index to timestamp
                        df.index = pd.to_datetime([int(k[0]) for k in klines], unit='ms', utc=True)
                        df.sort_index(inplace=True)
                        
                        self.frames[symbol] = df
                        logger.info(f"[{symbol}] Fetched {len(df)} candles from API")
                        
                        # Save to database for next time
                        self.storage.save_candles(symbol, df)
                    else:
                        self.frames[symbol] = new_frame()
                        logger.warning(f"[{symbol}] No data available from API")
                        
                except Exception as e:
                    logger.error(f"[{symbol}] Failed to fetch data: {e}")
                    self.frames[symbol] = new_frame()
        
        # Save all fetched data immediately
        if self.frames:
            logger.info("Saving initial data to database...")
            self.storage.save_all_frames(self.frames)
        
        # Show database stats
        stats = self.storage.get_stats()
        logger.info(f"Database: {stats.get('total_candles', 0)} candles, {stats.get('symbols', 0)} symbols, {stats.get('db_size_mb', 0):.2f} MB")
    
    async def save_all_candles(self):
        """Save all candles to database"""
        try:
            if self.frames:
                self.storage.save_all_frames(self.frames)
                logger.info("Auto-saved all candles to database")
        except Exception as e:
            logger.error(f"Failed to auto-save candles: {e}")
    
    def record_closed_trade(self, symbol: str, pos: Position, exit_price: float, exit_reason: str, leverage: float = 1.0):
        """Record a closed trade to history"""
        try:
            # Calculate PnL
            pnl_usd, pnl_percent = self.trade_tracker.calculate_pnl(
                symbol, pos.side, pos.entry, exit_price, pos.qty, leverage
            )
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                side=pos.side,
                entry_price=pos.entry,
                exit_price=exit_price,
                quantity=pos.qty,
                entry_time=pos.entry_time,  # Use entry time from position
                exit_time=datetime.now(),
                pnl_usd=pnl_usd,
                pnl_percent=pnl_percent,
                exit_reason=exit_reason,
                leverage=leverage,
                strategy_name=pos.strategy_name # Pass strategy name
            )
            
            # Add to tracker
            self.trade_tracker.add_trade(trade)
            logger.info(f"Trade recorded: {symbol} {exit_reason} PnL: ${pnl_usd:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
    
    async def check_closed_positions(self, book: Book, meta: dict = None, ml_scorer=None, reset_symbol_state=None, symbol_collector=None, ml_evolution=None):
        """Check for positions that have been closed and record them"""
        try:
            # Get current positions from exchange
            current_positions = self.bybit.get_positions()
            if current_positions is None:
                logger.warning("Could not get positions from exchange, skipping closed position check")
                return
            
            # Build set of symbols with CONFIRMED open positions
            current_symbols = set()
            for p in current_positions:
                symbol = p.get('symbol')
                size = float(p.get('size', 0))
                # Only add if we're SURE there's an open position
                if symbol and size > 0:
                    current_symbols.add(symbol)
            
            # Find positions that might be closed
            potentially_closed = []
            for symbol, pos in list(book.positions.items()):
                if symbol not in current_symbols:
                    potentially_closed.append((symbol, pos))
            
            # Verify each potentially closed position
            confirmed_closed = []
            for symbol, pos in potentially_closed:
                try:
                    # Try to get recent order history to confirm close
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
                            # Handle empty strings and None values
                            avg_price_str = order.get("avgPrice", 0)
                            exit_price = float(avg_price_str) if avg_price_str and avg_price_str != "" else 0
                            
                            # Log order details for debugging
                            logger.debug(f"[{symbol}] Found filled reduceOnly order: avgPrice={avg_price_str}, triggerPrice={order.get('triggerPrice')}, orderType={order.get('orderType')}")
                            
                            # Determine if it was TP or SL based on trigger price
                            trigger_price_str = order.get("triggerPrice", 0)
                            trigger_price = float(trigger_price_str) if trigger_price_str and trigger_price_str != "" else 0
                            
                            # Also check orderType for better detection
                            order_type = order.get("orderType", "")
                            
                            if trigger_price > 0:
                                if pos.side == "long":
                                    if trigger_price >= pos.tp * 0.98:  # Within 2% of TP
                                        exit_reason = "tp"
                                    elif trigger_price <= pos.sl * 1.02:  # Within 2% of SL
                                        exit_reason = "sl"
                                else:  # short
                                    if trigger_price <= pos.tp * 1.02:
                                        exit_reason = "tp"
                                    elif trigger_price >= pos.sl * 0.98:
                                        exit_reason = "sl"
                            
                            # Check orderType as fallback
                            if exit_reason == "unknown":
                                if "TakeProfit" in order_type:
                                    exit_reason = "tp"
                                elif "StopLoss" in order_type:
                                    exit_reason = "sl"
                            
                            # If no trigger price, check exit price vs targets
                            if exit_reason == "unknown" and exit_price > 0:
                                # Log position targets for debugging
                                logger.debug(f"[{symbol}] Checking exit price {exit_price:.4f} vs targets - TP: {pos.tp:.4f}, SL: {pos.sl:.4f}, Side: {pos.side}")
                                
                                if pos.side == "long":
                                    if exit_price >= pos.tp * 0.98:
                                        exit_reason = "tp"
                                        logger.debug(f"[{symbol}] Long TP hit: exit {exit_price:.4f} >= TP {pos.tp:.4f} * 0.98")
                                    elif exit_price <= pos.sl * 1.02:
                                        exit_reason = "sl"
                                        logger.debug(f"[{symbol}] Long SL hit: exit {exit_price:.4f} <= SL {pos.sl:.4f} * 1.02")
                                    else:
                                        exit_reason = "manual"
                                        logger.debug(f"[{symbol}] Long manual close: exit {exit_price:.4f} between SL {pos.sl:.4f} and TP {pos.tp:.4f}")
                                else:  # short
                                    if exit_price <= pos.tp * 1.02:
                                        exit_reason = "tp"
                                        logger.debug(f"[{symbol}] Short TP hit: exit {exit_price:.4f} <= TP {pos.tp:.4f} * 1.02")
                                    elif exit_price >= pos.sl * 0.98:
                                        exit_reason = "sl"
                                        logger.debug(f"[{symbol}] Short SL hit: exit {exit_price:.4f} >= SL {pos.sl:.4f} * 0.98")
                                    else:
                                        exit_reason = "manual"
                                        logger.debug(f"[{symbol}] Short manual close: exit {exit_price:.4f} between TP {pos.tp:.4f} and SL {pos.sl:.4f}")
                            break
                    
                    # Only add to confirmed closed if we found evidence
                    if found_close and exit_price > 0:
                        logger.info(f"[{symbol}] Position CONFIRMED closed at {exit_price:.4f} ({exit_reason})")
                        confirmed_closed.append((symbol, pos, exit_price, exit_reason))
                    else:
                        # Can't confirm it's closed - might be API lag
                        logger.debug(f"[{symbol}] Not in positions but can't confirm close - keeping in book")
                        
                except Exception as e:
                    logger.warning(f"Could not verify {symbol} close status: {e}")
                    # Don't process if we can't verify
            
            # Process only CONFIRMED closed positions
            for symbol, pos, exit_price, exit_reason in confirmed_closed:
                try:
                    # Get leverage
                    leverage = meta.get(symbol, {}).get("max_leverage", 1.0) if meta else 1.0
                    
                    # Record the trade
                    self.record_closed_trade(symbol, pos, exit_price, exit_reason, leverage)
                    
                    # Update symbol data collector with session performance
                    if symbol_collector:
                        try:
                            # Calculate hold time
                            hold_minutes = (datetime.now() - pos.entry_time).total_seconds() / 60
                            
                            # Calculate P&L for this position
                            if pos.side == "long":
                                pnl = (exit_price - pos.entry) * pos.qty
                            else:
                                pnl = (pos.entry - exit_price) * pos.qty
                            
                            # Determine session
                            session = symbol_collector.get_trading_session(datetime.now().hour)
                            
                            # Update session performance
                            symbol_collector.update_session_performance(
                                symbol=symbol,
                                session=session,
                                won=pnl > 0,
                                pnl=pnl,
                                hold_minutes=int(hold_minutes)
                            )
                            
                            logger.debug(f"[{symbol}] Updated session stats: {session}, PnL={pnl:.2f}, Hold={hold_minutes:.0f}min")
                            
                        except Exception as e:
                            logger.debug(f"Failed to update session performance: {e}")
                    
                    # Update ML scorer for all closed positions
                    if ml_scorer is not None:
                        # Always record outcome for ML learning (not just TP/SL)
                        # This ensures we learn from ALL trades including manual closes
                        try:
                            # Calculate P&L percentage
                            if pos.side == "long":
                                pnl_pct = ((exit_price - pos.entry) / pos.entry) * 100
                                # Debug logging for incorrect P&L
                                if exit_price > pos.entry and pnl_pct < 0:
                                    logger.error(f"[{symbol}] CALCULATION ERROR: Long position exit {exit_price:.4f} > entry {pos.entry:.4f} but P&L is {pnl_pct:.2f}%")
                            else:
                                pnl_pct = ((pos.entry - exit_price) / pos.entry) * 100
                                # Debug logging for incorrect P&L
                                if exit_price < pos.entry and pnl_pct < 0:
                                    logger.error(f"[{symbol}] CALCULATION ERROR: Short position exit {exit_price:.4f} < entry {pos.entry:.4f} but P&L is {pnl_pct:.2f}%")
                            
                            # CRITICAL FIX: Use actual P&L to determine win/loss, not exit_reason
                            # Negative P&L is ALWAYS a loss, positive is ALWAYS a win
                            outcome = "win" if pnl_pct > 0 else "loss"
                            
                            # Log warning if exit_reason doesn't match P&L for TP/SL
                            if exit_reason == "tp":
                                if pnl_pct < 0:
                                    logger.warning(f"[{symbol}] TP hit but negative P&L! Side: {pos.side}, Exit: {exit_price:.4f}, Entry: {pos.entry:.4f}, P&L: {pnl_pct:.2f}%")
                                    exit_reason = "sl"  # Correct the exit reason based on actual P&L
                            elif exit_reason == "sl":
                                if pnl_pct > 0:
                                    logger.warning(f"[{symbol}] SL hit but positive P&L! Exit: {exit_price:.4f}, Entry: {pos.entry:.4f}, P&L: {pnl_pct:.2f}%")
                                    exit_reason = "tp"  # Correct the exit reason based on actual P&L
                            
                            # For manual closes, try to determine if it was more like TP or SL based on P&L
                            if exit_reason == "manual":
                                # Check if the position's SL/TP values are valid
                                if pos.sl > 0 and pos.tp > 0:
                                    # For manual, still check proximity to targets
                                    if pos.side == "long":
                                        # If closer to TP than SL and profitable, likely a partial TP
                                        tp_distance = abs(exit_price - pos.tp)
                                        sl_distance = abs(exit_price - pos.sl)
                                        if tp_distance < sl_distance and pnl_pct > 0:
                                            logger.info(f"[{symbol}] Manual close near TP with profit - treating as TP for ML")
                                            exit_reason = "tp"
                                        elif sl_distance < tp_distance and pnl_pct < 0:
                                            logger.info(f"[{symbol}] Manual close near SL with loss - treating as SL for ML")
                                            exit_reason = "sl"
                                    else:  # short
                                        tp_distance = abs(exit_price - pos.tp)
                                        sl_distance = abs(exit_price - pos.sl)
                                        if tp_distance < sl_distance and pnl_pct > 0:
                                            logger.info(f"[{symbol}] Manual close near TP with profit - treating as TP for ML")
                                            exit_reason = "tp"
                                        elif sl_distance < tp_distance and pnl_pct < 0:
                                            logger.info(f"[{symbol}] Manual close near SL with loss - treating as SL for ML")
                                            exit_reason = "sl"
                            
                            # Create signal data for ML recording
                            signal_data = {
                                'symbol': symbol,
                                'features': {},  # Features were stored during signal detection
                                'score': 0  # Will be filled from phantom tracker if available
                            }
                            
                            # Debugging: Log strategy name and routing info for all closed trades
                            logger.info(f"[{symbol}] ML ROUTING DEBUG: strategy='{pos.strategy_name}', use_enhanced={shared.get('use_enhanced_parallel', False) if 'shared' in locals() else False}")

                            # Record outcome in appropriate ML scorer based on strategy (NO DUPLICATION)
                            # Get shared data components for ML scorers
                            shared_enhanced_mr = shared.get('enhanced_mr_scorer') if 'shared' in locals() else None
                            shared_mr_scorer = shared.get('mean_reversion_scorer') if 'shared' in locals() else None
                            use_enhanced = shared.get('use_enhanced_parallel', False) if 'shared' in locals() else False

                            logger.info(f"[{symbol}] ML COMPONENTS: enhanced_mr={shared_enhanced_mr is not None}, mr_scorer={shared_mr_scorer is not None}, use_enhanced={use_enhanced}")
                            logger.info(f"[{symbol}] ML ROUTING DECISION: Enhanced system active: {use_enhanced and shared_enhanced_mr}, Strategy: '{pos.strategy_name}'")

                            if use_enhanced and shared_enhanced_mr:
                                # Enhanced parallel system - route to correct ML scorer
                                if pos.strategy_name == "enhanced_mr":
                                    shared_enhanced_mr.record_outcome(signal_data, outcome, pnl_pct)
                                    logger.info(f"[{symbol}] Enhanced MR ML updated with outcome.")
                                else:
                                    # Pullback strategy
                                    if ml_scorer is not None:
                                        ml_scorer.record_outcome(signal_data, outcome, pnl_pct)
                                        logger.info(f"[{symbol}] Pullback ML updated with outcome.")
                            else:
                                # Original system - record in appropriate ML scorer
                                if pos.strategy_name == "mean_reversion" and shared_mr_scorer:
                                    shared_mr_scorer.record_outcome(signal_data, outcome, pnl_pct)
                                    logger.info(f"[{symbol}] Mean Reversion ML updated with outcome.")
                                elif ml_scorer is not None:
                                    ml_scorer.record_outcome(signal_data, outcome, pnl_pct)
                                    logger.info(f"[{symbol}] Pullback ML updated with outcome.")

                            # Also record in ML evolution if available
                            if ml_evolution is not None:
                                ml_evolution.record_outcome(symbol, outcome == "win", pnl_pct)                            
                            # Log with clear outcome based on actual P&L and corrected exit reason
                            actual_result = "WIN" if pnl_pct > 0 else "LOSS"
                            logger.info(f"[{symbol}] ML updated: {actual_result} ({pnl_pct:.2f}%) - Exit trigger: {exit_reason.upper()}")
                            
                            # Check if ML needs retraining after this trade completes
                            self._check_ml_retrain(ml_scorer)

                            # Also check mean reversion ML retraining (both enhanced and original systems)
                            if pos.strategy_name in ["enhanced_mr", "mean_reversion", "MeanReversion"]:
                                # Get the appropriate MR scorer references
                                enhanced_mr_scorer_ref = shared_enhanced_mr if use_enhanced and shared_enhanced_mr else None
                                original_mr_scorer_ref = shared_mr_scorer if not use_enhanced else None
                                
                                # Call the appropriate retrain check
                                self._check_mr_ml_retrain(original_mr_scorer_ref, enhanced_mr_scorer_ref)
                            
                        except Exception as e:
                            logger.error(f"Failed to update ML outcome: {e}")
                    
                    # Log position details for debugging
                    logger.debug(f"[{symbol}] Closed position details: side={pos.side}, entry={pos.entry:.4f}, exit={exit_price:.4f}, TP={pos.tp:.4f}, SL={pos.sl:.4f}, exit_reason={exit_reason}")
                    
                    # Remove from book
                    book.positions.pop(symbol)
                    logger.info(f"[{symbol}] Position removed from tracking")
                    
                    # Reset the strategy state
                    if reset_symbol_state:
                        reset_symbol_state(symbol)
                        logger.info(f"[{symbol}] Strategy state reset - ready for new signals")
                        
                except Exception as e:
                    logger.error(f"Error processing confirmed closed position {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in check_closed_positions: {e}")
    
    def _check_ml_retrain(self, ml_scorer):
        """Check if ML needs retraining after a trade completes"""
        if not ml_scorer:
            return
            
        try:
            # Get retrain info
            retrain_info = ml_scorer.get_retrain_info()
            
            # Check if ready to retrain
            if retrain_info['can_train'] and retrain_info['trades_until_next_retrain'] == 0:
                logger.info(f"🔄 ML retrain triggered after real trade completion - "
                           f"{retrain_info['total_combined']} total trades available")
                
                # Trigger retrain
                retrain_result = ml_scorer.startup_retrain()
                if retrain_result:
                    logger.info("✅ ML models successfully retrained after trade completion")
                else:
                    logger.warning("⚠️ ML retrain attempt failed")
            else:
                logger.debug(f"ML retrain check: {retrain_info['trades_until_next_retrain']} trades until next retrain")
                
        except Exception as e:
            logger.error(f"Error checking ML retrain: {e}")

    def _check_mr_ml_retrain(self, mean_reversion_scorer=None, enhanced_mr_scorer=None):
        """Check if Mean Reversion ML needs retraining after a trade completes"""
        # Use the appropriate MR scorer based on system configuration
        mr_scorer = enhanced_mr_scorer if enhanced_mr_scorer else mean_reversion_scorer
        
        if not mr_scorer:
            return

        try:
            # Get retrain info from the active MR scorer
            retrain_info = mr_scorer.get_retrain_info()
            
            scorer_type = "Enhanced MR" if enhanced_mr_scorer else "Original MR"

            # Check if ready to retrain
            if retrain_info['can_train'] and retrain_info['trades_until_next_retrain'] == 0:
                total_trades = retrain_info.get('total_combined', retrain_info.get('total_trades', 0))
                logger.info(f"🔄 {scorer_type} ML retrain triggered after trade completion - "
                           f"{total_trades} total trades available")

                # Trigger retrain
                retrain_result = mr_scorer.startup_retrain()
                if retrain_result:
                    logger.info(f"✅ {scorer_type} ML models successfully retrained after trade completion")
                else:
                    logger.warning(f"⚠️ {scorer_type} ML retrain attempt failed")
            else:
                logger.debug(f"{scorer_type} ML retrain check: {retrain_info['trades_until_next_retrain']} trades until next retrain")

        except Exception as e:
            logger.error(f"Error checking {scorer_type} ML retrain: {e}")

    async def recover_positions(self, book:Book, sizer:Sizer):
        """Recover existing positions from exchange - preserves all existing orders"""
        logger.info("Checking for existing positions to recover...")
        
        try:
            positions = self.bybit.get_positions()
            
            if positions:
                recovered = 0
                for pos in positions:
                    # Skip if no position size
                    size_str = pos.get('size', '0')
                    if not size_str or size_str == '' or float(size_str) == 0:
                        continue
                        
                    symbol = pos['symbol']
                    side = "long" if pos['side'] == "Buy" else "short"
                    
                    # Safe conversion with empty string handling
                    qty = float(size_str)
                    entry = float(pos.get('avgPrice') or 0)
                    
                    # Get current TP/SL if set - PRESERVE THESE
                    # Handle empty strings by converting to 0
                    tp_str = pos.get('takeProfit', '0')
                    sl_str = pos.get('stopLoss', '0')
                    tp = float(tp_str) if tp_str and tp_str != '' else 0
                    sl = float(sl_str) if sl_str and sl_str != '' else 0
                    
                    # For recovered positions, validate TP/SL make sense
                    if side == "long":
                        # For long: TP should be > entry, SL should be < entry
                        if tp > 0 and sl > 0 and tp < sl:
                            logger.warning(f"[{symbol}] TP/SL appear swapped for long position! TP={tp:.4f} SL={sl:.4f} Entry={entry:.4f}")
                            # Swap them
                            tp, sl = sl, tp
                    else:  # short
                        # For short: TP should be < entry, SL should be > entry
                        if tp > 0 and sl > 0 and tp > sl:
                            logger.warning(f"[{symbol}] TP/SL appear swapped for short position! TP={tp:.4f} SL={sl:.4f} Entry={entry:.4f}")
                            # Swap them
                            tp, sl = sl, tp
                    
                    # Add to book
                    from position_mgr import Position
                    book.positions[symbol] = Position(
                        side=side,
                        qty=qty,
                        entry=entry,
                        sl=sl if sl > 0 else (entry * 0.95 if side == "long" else entry * 1.05),
                        tp=tp if tp > 0 else (entry * 1.1 if side == "long" else entry * 0.9),
                        entry_time=datetime.now() - pd.Timedelta(hours=1)  # Approximate for recovered positions
                    )
                    
                    recovered += 1
                    logger.info(f"Recovered {side} position: {symbol} qty={qty} entry={entry:.4f} TP={tp:.4f} SL={sl:.4f}")
                
                if recovered > 0:
                    logger.info(f"Successfully recovered {recovered} position(s) - WILL NOT MODIFY THEM")
                    logger.info("Existing positions and their TP/SL orders will run their course without interference")
                    
                    # Send Telegram notification
                    if self.tg:
                        msg = f"📊 *Recovered {recovered} existing position(s)*\n"
                        msg += "⚠️ *These positions will NOT be modified*\n"
                        msg += "✅ *TP/SL orders preserved as-is*\n\n"
                        for sym, pos in book.positions.items():
                            emoji = "🟢" if pos.side == "long" else "🔴"
                            msg += f"{emoji} {sym}: {pos.side} qty={pos.qty:.4f}\n"
                        await self.tg.send_message(msg)
            else:
                logger.info("No existing positions to recover")
                
        except Exception as e:
            logger.error(f"Failed to recover positions: {e}")
    
    async def kline_stream(self, ws_url:str, topics:list[str]):
        """Stream klines from Bybit WebSocket"""
        sub = {"op":"subscribe","args":[f"kline.{t}" for t in topics]}
        
        while self.running:
            try:
                logger.info(f"Connecting to WebSocket: {ws_url}")
                async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                    self.ws = ws
                    await ws.send(json.dumps(sub))
                    logger.info(f"Subscribed to topics: {topics}")
                    
                    while self.running:
                        try:
                            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                            
                            if msg.get("success") == False:
                                logger.error(f"Subscription failed: {msg}")
                                continue
                                
                            topic = msg.get("topic","")
                            if topic.startswith("kline."):
                                sym = topic.split(".")[-1]
                                for k in msg.get("data", []):
                                    yield sym, k
                                    
                        except asyncio.TimeoutError:
                            logger.debug("WebSocket timeout, sending ping")
                            await ws.ping()
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("WebSocket connection closed, reconnecting...")
                            break
                            
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.running:
                    await asyncio.sleep(5)  # Wait before reconnecting

    async def auto_generate_enhanced_clusters(self):
        """Auto-generate or update enhanced clusters if needed"""
        try:
            # First try to load existing enhanced clusters
            from cluster_feature_enhancer import load_cluster_data
            simple_clusters, enhanced_clusters = load_cluster_data()
            
            # Check if we need to generate or update
            needs_generation = False
            if not enhanced_clusters:
                logger.info("🆕 No enhanced clusters found, auto-generating...")
                needs_generation = True
            else:
                # Check for broken clustering (too many borderline symbols)
                try:
                    import json
                    from datetime import datetime
                    with open('symbol_clusters_enhanced.json', 'r') as f:
                        cluster_data = json.load(f)
                        
                        # Count borderline symbols
                        enhanced_data = cluster_data.get('enhanced_clusters', {})
                        total_symbols = len(enhanced_data)
                        borderline_count = sum(1 for data in enhanced_data.values() 
                                             if data.get('is_borderline', False))
                        
                        # If more than 50% are borderline, clustering is broken
                        if total_symbols > 0 and borderline_count / total_symbols > 0.5:
                            logger.warning(f"🚨 Broken clustering detected: {borderline_count}/{total_symbols} symbols are borderline!")
                            logger.info("Deleting and regenerating clusters...")
                            import os
                            os.remove('symbol_clusters_enhanced.json')
                            needs_generation = True
                        # Also check for obviously wrong assignments
                        elif enhanced_data.get('BTCUSDT', {}).get('primary_cluster') != 1:
                            logger.warning("🚨 BTCUSDT not in Blue Chip cluster - clustering is broken!")
                            os.remove('symbol_clusters_enhanced.json')
                            needs_generation = True
                        else:
                            # Check age
                            if 'generated_at' in cluster_data:
                                gen_time = datetime.fromisoformat(cluster_data['generated_at'])
                                days_old = (datetime.now() - gen_time).days
                                if days_old > 7:  # Update weekly
                                    logger.info(f"🔄 Enhanced clusters are {days_old} days old, updating...")
                                    needs_generation = True
                                else:
                                    logger.info(f"✅ Enhanced clusters are {days_old} days old, still fresh")
                except Exception as e:
                    logger.warning(f"Error checking clusters: {e}")
                    needs_generation = True
            
            # Generate if needed
            if needs_generation:
                logger.info("🎯 Generating enhanced clusters from historical data...")
                try:
                    from symbol_clustering import SymbolClusterer
                    from datetime import datetime
                    import json
                    import numpy as np
                    
                    # Use loaded frames data
                    if self.frames and len(self.frames) > 0:
                        # Only use symbols with enough data
                        valid_frames = {sym: df for sym, df in self.frames.items() 
                                      if len(df) >= 500}
                        
                        if len(valid_frames) >= 20:  # Need at least 20 symbols
                            clusterer = SymbolClusterer(valid_frames)
                            metrics = clusterer.calculate_metrics(min_candles=500)
                            clusters = clusterer.cluster_symbols()
                            
                            # Convert simple clusters to enhanced format for compatibility
                            enhanced_data = {}
                            for symbol, cluster_id in clusters.items():
                                enhanced_data[symbol] = {
                                    "primary_cluster": cluster_id,
                                    "confidence": 0.95,  # High confidence for rule-based clustering
                                    "is_borderline": False,  # No borderline in simple clustering
                                    "secondary_cluster": None,
                                    "secondary_confidence": 0.0
                                }
                            
                            # Create enhanced format output
                            output = {
                                "generated_at": datetime.now().isoformat(),
                                "cluster_descriptions": clusterer.get_cluster_descriptions(),
                                "symbol_clusters": clusters,  # Backward compatible key name
                                "enhanced_clusters": enhanced_data,
                                "metrics_summary": {}
                            }
                            
                            # Add metrics summary per cluster
                            for cluster_id in range(1, 6):
                                cluster_symbols = [s for s, c in clusters.items() if c == cluster_id]
                                if cluster_symbols:
                                    cluster_metrics = [metrics[s] for s in cluster_symbols if s in metrics]
                                    
                                    output["metrics_summary"][cluster_id] = {
                                        "count": len(cluster_symbols),
                                        "symbols": cluster_symbols[:5],  # Show first 5 as examples
                                        "avg_volatility": np.mean([m.avg_volatility for m in cluster_metrics]) if cluster_metrics else 0,
                                        "avg_btc_correlation": np.mean([m.btc_correlation for m in cluster_metrics]) if cluster_metrics else 0
                                    }
                            
                            # Save to enhanced clusters file for compatibility
                            with open('symbol_clusters_enhanced.json', 'w') as f:
                                json.dump(output, f, indent=2)
                            
                            logger.info(f"✅ Generated enhanced clusters for {len(clusters)} symbols")
                            
                            # Notify via Telegram if available
                            if hasattr(self, 'tg') and self.tg:
                                await self.tg.send_message(
                                    f"✅ *Auto-generated enhanced clusters*\n"
                                    f"Analyzed {len(clusters)} symbols\n"
                                    f"Use /clusters to view status"
                                )
                        else:
                            logger.warning(f"Only {len(valid_frames)} symbols have enough data, skipping generation")
                    else:
                        logger.warning("No frame data available for cluster generation")
                        
                except Exception as e:
                    logger.error(f"Failed to auto-generate clusters: {e}")
                    # Don't fail the bot startup
            
        except Exception as e:
            logger.error(f"Error in auto cluster generation: {e}")
            # Don't fail the bot startup

    async def run(self):
        """Main bot loop"""
        # Load config
        with open("config.yaml","r") as f:
            cfg = yaml.safe_load(f)
        
        # Replace environment variables
        cfg = replace_env_vars(cfg)
        
        # Extract configuration
        symbols = [s.upper() for s in cfg["trade"]["symbols"]]
        tf = cfg["trade"]["timeframe"]
        topics = [f"{tf}.{s}" for s in symbols]
        
        logger.info(f"Trading symbols: {symbols}")
        logger.info(f"Timeframe: {tf} minutes")
        logger.info("📌 Bot Policy: Existing positions and orders will NOT be modified - they will run their course")
        
        # Get the appropriate strategy functions
        use_pullback = cfg["trade"].get("use_pullback_strategy", True)
        get_pullback_signals, reset_symbol_state = get_strategy_module(use_pullback)
        
        # Import strategies for parallel system
        from strategy_mean_reversion import detect_signal as detect_signal_mean_reversion
        use_enhanced_parallel = cfg["trade"].get("use_enhanced_parallel", True) and ENHANCED_ML_AVAILABLE
        use_regime_switching = cfg["trade"].get("use_regime_switching", False)

        if use_enhanced_parallel:
            strategy_type = "Enhanced Parallel (Pullback + Mean Reversion with ML)"
            logger.info(f"📊 Strategy: {strategy_type}")
            logger.info("🧠 Using Enhanced Parallel ML System with regime-based strategy routing")
        else:
            strategy_type = "Pullback (HL/LH + Confirmation)" if use_pullback else "Immediate Breakout"
            logger.info(f"📊 Strategy: {strategy_type}")
        
        # Initialize strategy settings
        settings = Settings(
            atr_len=cfg["trade"]["atr_len"],
            sl_buf_atr=cfg["trade"]["sl_buf_atr"],
            rr=cfg["trade"]["rr"],
            use_ema=cfg["trade"]["use_ema"],
            ema_len=cfg["trade"]["ema_len"],
            use_vol=cfg["trade"]["use_vol"],
            vol_len=cfg["trade"]["vol_len"],
            vol_mult=cfg["trade"]["vol_mult"],
            both_hit_rule=cfg["trade"]["both_hit_rule"],
            confirmation_candles=cfg["trade"].get("confirmation_candles", 2)
        )
        
        # Initialize components
        risk = RiskConfig(
            risk_usd=cfg["trade"]["risk_usd"],
            risk_percent=cfg["trade"]["risk_percent"],
            use_percent_risk=cfg["trade"]["use_percent_risk"],
            use_ml_dynamic_risk=False,  # Disabled - using fixed 1% risk until ML models are consistent
            ml_risk_min_score=70.0,
            ml_risk_max_score=100.0,
            ml_risk_min_percent=1.0,
            ml_risk_max_percent=5.0
        )
        sizer = Sizer(risk)
        book = Book()
        panic_list:list[str] = []
        
        # Initialize ALL ML components (including enhanced ones)
        ml_scorer = None
        phantom_tracker = None
        mean_reversion_scorer = None # Initialize Mean Reversion Scorer
        enhanced_mr_scorer = None  # Initialize Enhanced MR Scorer
        mr_phantom_tracker = None  # Initialize MR Phantom Tracker
        use_ml = cfg["trade"].get("use_ml_scoring", True)  # Default to True for immediate learning
        
        # Initialize ML Evolution System (optional)
        ml_evolution = None
        evolution_trainer = None
        enable_ml_evolution = cfg["trade"].get("enable_ml_evolution", False)
        
        # Initialize symbol data collector
        symbol_collector = None
        if SYMBOL_COLLECTOR_AVAILABLE:
            try:
                symbol_collector = get_symbol_collector()
                logger.info("📊 Symbol data collector active - tracking for future ML")
            except Exception as e:
                logger.warning(f"Could not initialize symbol collector: {e}")
        
        if ML_AVAILABLE and use_ml:
            try:
                if use_enhanced_parallel and ENHANCED_ML_AVAILABLE:
                    # Initialize Enhanced Parallel ML System
                    ml_scorer = get_immediate_scorer()  # Pullback ML
                    phantom_tracker = get_phantom_tracker()  # Pullback phantom tracker
                    enhanced_mr_scorer = get_enhanced_mr_scorer()  # Enhanced MR ML
                    mr_phantom_tracker = get_mr_phantom_tracker()  # MR phantom tracker
                    mean_reversion_scorer = None  # Not used in enhanced system

                    # Get and log stats for both systems
                    pullback_stats = ml_scorer.get_stats()
                    mr_stats = enhanced_mr_scorer.get_enhanced_stats()

                    logger.info(f"✅ Enhanced Parallel ML System initialized")
                    logger.info(f"   Pullback ML: {pullback_stats['status']} (threshold: {pullback_stats['current_threshold']:.0f}, trades: {pullback_stats['completed_trades']})")
                    logger.info(f"   Mean Reversion ML: {mr_stats['status']} (threshold: {mr_stats['current_threshold']:.0f}, trades: {mr_stats['completed_trades']})")

                    if pullback_stats['recent_win_rate'] > 0:
                        logger.info(f"   Pullback recent WR: {pullback_stats['recent_win_rate']:.1f}%")
                    if mr_stats['recent_win_rate'] > 0:
                        logger.info(f"   MR recent WR: {mr_stats['recent_win_rate']:.1f}%")

                    # Perform Enhanced MR startup retrain with phantom data
                    logger.info("🔄 Checking for Enhanced MR startup retrain...")
                    enhanced_mr_startup_result = enhanced_mr_scorer.startup_retrain()
                    if enhanced_mr_startup_result:
                        # Get updated stats after retrain
                        mr_stats = enhanced_mr_scorer.get_enhanced_stats()
                        logger.info(f"✅ Enhanced MR models retrained on startup")
                        logger.info(f"   Status: {mr_stats['status']}")
                        logger.info(f"   Threshold: {mr_stats['current_threshold']:.0f}")
                        if mr_stats.get('models_active'):
                            logger.info(f"   Active models: {', '.join(mr_stats['models_active'])}")
                    elif enhanced_mr_scorer.is_ml_ready:
                        logger.info("✅ Pre-trained Enhanced MR models loaded successfully.")
                    else:
                        logger.info("⚠️ No pre-trained Enhanced MR models found. Starting in online learning mode.")

                else:
                    # Initialize original ML system
                    ml_scorer = get_immediate_scorer()
                    phantom_tracker = get_phantom_tracker()
                    enhanced_mr_scorer = None
                    mr_phantom_tracker = None
                    mean_reversion_scorer = get_mean_reversion_scorer() # Original MR scorer

                    # Get and log ML stats
                    ml_stats = ml_scorer.get_stats()
                    logger.info(f"✅ Original ML Scorer initialized")
                    logger.info(f"   Status: {ml_stats['status']}")
                    logger.info(f"   Threshold: {ml_stats['current_threshold']:.0f}")
                    logger.info(f"   Completed trades: {ml_stats['completed_trades']}")
                    if ml_stats['recent_win_rate'] > 0:
                        logger.info(f"   Recent win rate: {ml_stats['recent_win_rate']:.1f}%")
                    if ml_stats['models_active']:
                        logger.info(f"   Active models: {', '.join(ml_stats['models_active'])}")
                    
                # Phantom trades now expire naturally on TP/SL - no timeout needed
                
                # Perform startup retrain with all available data
                logger.info("🔄 Checking for ML startup retrain...")
                startup_result = ml_scorer.startup_retrain()
                if startup_result:
                    # Get updated stats after retrain
                    ml_stats = ml_scorer.get_stats()
                    logger.info(f"✅ ML models retrained on startup")
                    logger.info(f"   Status: {ml_stats['status']}")
                    logger.info(f"   Threshold: {ml_stats['current_threshold']:.0f}")
                    if ml_stats.get('models_active'):
                        logger.info(f"   Active models: {', '.join(ml_stats['models_active'])}")
                elif ml_scorer.is_ml_ready:
                    logger.info("✅ Pre-trained Pullback ML Model loaded successfully.")
                else:
                    logger.warning("⚠️ No pre-trained Pullback model found. Starting in online learning mode.")
                
                # Initialize ML Evolution (always initialize for shadow learning)
                try:
                    from ml_evolution_system import get_evolution_system
                    from symbol_ml_trainer import get_symbol_trainer
                    
                    # Always create evolution system (enabled flag controls if it affects decisions)
                    ml_evolution = get_evolution_system(enabled=enable_ml_evolution)
                    evolution_trainer = get_symbol_trainer()
                    
                    if enable_ml_evolution:
                        logger.info("🚀 ML Evolution System ACTIVE - Symbol-specific models enabled")
                    else:
                        logger.info("👁️ ML Evolution System in SHADOW MODE - Learning but not affecting trades")
                    
                    # Always start background training for data collection
                    asyncio.create_task(evolution_trainer.start_background_training())
                    logger.info("📊 Background ML training started - Building symbol knowledge base")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize ML Evolution: {e}")
                    ml_evolution = None
                
            except Exception as e:
                logger.warning(f"Failed to initialize ML/Phantom system: {e}. Running without ML.")
                ml_scorer = None
                phantom_tracker = None
        elif use_ml:
            logger.warning("ML scoring requested but ML module not available")
        
        # Initialize basic symbol clustering (enhanced will be done after data load)
        symbol_clusters = {}
        try:
            from symbol_clustering import load_symbol_clusters
            symbol_clusters = load_symbol_clusters()
            logger.info(f"Loaded basic symbol clusters for {len(symbol_clusters)} symbols")
        except Exception as e:
            logger.warning(f"Could not load symbol clusters: {e}. Features will use defaults.")
            symbol_clusters = {}
        
        # Initialize Bybit client
        self.bybit = bybit = Bybit(BybitConfig(
            cfg["bybit"]["base_url"], 
            cfg["bybit"]["api_key"], 
            cfg["bybit"]["api_secret"]
        ))
        
        # Test connection
        balance = bybit.get_balance()
        if balance:
            logger.info(f"Connected to Bybit. Balance: ${balance:.2f} USDT")
        else:
            logger.warning("Could not fetch balance, continuing anyway...")
        
        # Fetch historical data for all symbols
        await self.load_or_fetch_initial_data(symbols, tf)
        
        # Initialize HTF support/resistance levels from all available data
        try:
            from multi_timeframe_sr import initialize_all_sr_levels, mtf_sr
            
            # Configure MTF update interval from config
            mtf_update_interval = cfg["trade"].get("mtf_update_interval", 100)
            mtf_sr.update_interval = mtf_update_interval
            logger.info(f"MTF S/R update interval set to {mtf_update_interval} candles")
            
            # Initialize all S/R levels
            sr_results = initialize_all_sr_levels(self.frames)
            
            # Send summary to Telegram if available
            if hasattr(self, 'tg') and self.tg and sr_results:
                symbols_with_levels = [sym for sym, count in sr_results.items() if count > 0]
                await self.tg.send_message(
                    f"📊 *HTF S/R Analysis Complete*\n"
                    f"Analyzed: {len(sr_results)} symbols\n" 
                    f"Found levels: {len(symbols_with_levels)} symbols\n"
                    f"Total levels: {sum(sr_results.values())}"
                )
        except Exception as e:
            logger.error(f"Failed to initialize HTF S/R levels: {e}")
        
        # Auto-generate enhanced clusters after data is loaded
        await self.auto_generate_enhanced_clusters()
        
        # Recover existing positions - PRESERVE THEIR ORDERS
        await self.recover_positions(book, sizer)
        
        # DISABLED: Not cancelling ANY orders to prevent accidental TP/SL removal
        # Orders will naturally cancel when positions close
        logger.info("Order cancellation DISABLED - all orders will be preserved")
        logger.info("TP/SL orders will close naturally with their positions")
        
        # Track analysis times
        last_analysis = {}
        
        # Setup shared data for Telegram - all ML components are now in scope
        shared = {
            "risk": risk,
            "book": book,
            "panic": panic_list,
            "meta": cfg.get("symbol_meta",{}),
            "broker": bybit,
            "frames": self.frames,
            "last_analysis": last_analysis,
            "trade_tracker": self.trade_tracker,
            "ml_scorer": ml_scorer,
            "bot_instance": self,
            # Enhanced ML system components
            "enhanced_mr_scorer": enhanced_mr_scorer,
            "mr_phantom_tracker": mr_phantom_tracker,
            "mean_reversion_scorer": mean_reversion_scorer,
            "phantom_tracker": phantom_tracker,
            "use_enhanced_parallel": use_enhanced_parallel
        }
        
        # Initialize Telegram bot with retry on conflict
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.tg = TGBot(cfg["telegram"]["token"], int(cfg["telegram"]["chat_id"]), shared)
                await self.tg.start_polling()
                # Send shorter startup message for 20 symbols
                # Format risk display
                if risk.use_percent_risk:
                    risk_display = f"{risk.risk_percent}%"
                else:
                    risk_display = f"${risk.risk_usd}"
                
                await self.tg.send_message(
                    "🚀 *Trading Bot Started*\n\n"
                    f"📊 Monitoring: {len(symbols)} symbols\n"
                    f"⏰ Timeframe: {tf} minutes\n"
                    f"💰 Risk per trade: {risk_display}\n"
                    f"📈 R:R Ratio: 1:{settings.rr}\n\n"
                    "_Use /risk to manage risk settings_\n"
                    "_Use /dashboard for full status_"
                )
                break  # Success
            except Exception as e:
                if "Conflict" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Telegram conflict detected, retrying in 5 seconds...")
                    await asyncio.sleep(5)
                    continue
                else:
                    logger.error(f"Telegram bot failed to start: {e}")
                    self.tg = None
                    break
        
        # Background training disabled - bot runs normally
        # Uncomment the lines below to enable background ML training:
        # try:
        #     from background_initial_trainer import get_background_trainer
        #     background_trainer = get_background_trainer(self.tg)
        #     training_started = await background_trainer.start_if_needed()
        #     if training_started:
        #         logger.info("🎯 Background ML training started - will run while bot trades")
        # except Exception as e:
        #     logger.error(f"Failed to start background trainer: {e}")
        logger.info("🚀 Bot starting in normal mode - background training disabled")
        
        # Signal tracking
        last_signal_time = {}
        signal_cooldown = 60  # Seconds between signals per symbol
        last_position_check = datetime.now()
        position_check_interval = 30  # Check for closed positions every 30 seconds
        
        # Add periodic logging summary to reduce log spam
        last_summary_log = datetime.now()
        summary_log_interval = 300  # Log summary every 5 minutes
        candles_processed = 0
        signals_detected = 0
        
        # Initialize ML breakout states dictionary
        ml_breakout_states = {}
        
        # Schedule weekly cluster updates
        last_cluster_update = datetime.now()
        cluster_update_interval = 7 * 24 * 60 * 60  # 7 days in seconds
        
        # Create background task for cluster updates
        async def weekly_cluster_updater():
            """Background task to update clusters weekly"""
            while self.running:
                try:
                    # Wait for next update time
                    await asyncio.sleep(cluster_update_interval)
                    
                    if self.running:  # Check if still running
                        logger.info("🔄 Running scheduled weekly cluster update...")
                        await self.auto_generate_enhanced_clusters()
                        
                except Exception as e:
                    logger.error(f"Error in weekly cluster updater: {e}")
                    # Continue running even if update fails
        
        # Start the weekly updater task
        cluster_updater_task = asyncio.create_task(weekly_cluster_updater())
        logger.info("📅 Started weekly cluster update scheduler")
        
        try:
            # Use multi-websocket handler if >190 topics
            if len(topics) > 190:
                logger.info(f"Using multi-websocket handler for {len(topics)} topics")
                ws_handler = MultiWebSocketHandler(cfg["bybit"]["ws_public"], self)
                stream = ws_handler.multi_kline_stream(topics)
            else:
                logger.info(f"Using single websocket for {len(topics)} topics")
                stream = self.kline_stream(cfg["bybit"]["ws_public"], topics)
            
            # Start streaming
            async for sym, k in stream:
                if not self.running:
                    break
                
                # Skip symbols not in our configured list
                if sym not in symbols:
                    continue
                
                try:
                    # Parse kline
                    ts = int(k["start"])
                    row = pd.DataFrame(
                        [[float(k["open"]), float(k["high"]), float(k["low"]), float(k["close"]), float(k["volume"])]],
                        index=[pd.to_datetime(ts, unit="ms", utc=True)],
                        columns=["open","high","low","close","volume"]
                    )
                    
                    # Get existing frame or create new if not exists
                    if sym in self.frames:
                        df = self.frames[sym]
                    else:
                        df = new_frame()
                    
                    # Ensure both dataframes have consistent timezone handling
                    if df.index.tz is None and row.index.tz is not None:
                        # Convert existing df to UTC if it's timezone-naive
                        df.index = df.index.tz_localize('UTC')
                    elif df.index.tz is not None and row.index.tz is None:
                        # Convert new row to UTC if it's timezone-naive
                        row.index = row.index.tz_localize('UTC')
                        
                    df.loc[row.index[0]] = row.iloc[0]
                    df.sort_index(inplace=True)
                    df = df.tail(10000)  # Keep last 10000 candles for maximum historical analysis
                    self.frames[sym] = df
                    
                    # Update phantom trades with current price
                    if phantom_tracker is not None:
                        current_price = df['close'].iloc[-1]
                        # Get BTC price for context
                        btc_price = None
                        if 'BTCUSDT' in self.frames and not self.frames['BTCUSDT'].empty:
                            btc_price = self.frames['BTCUSDT']['close'].iloc[-1]
                        # Update phantom prices for both systems
                        if use_enhanced_parallel and ENHANCED_ML_AVAILABLE:
                            # Update both phantom trackers in parallel system
                            phantom_tracker.update_phantom_prices(
                                sym, current_price, df=df, btc_price=btc_price, symbol_collector=symbol_collector
                            )
                            mr_phantom_tracker.update_mr_phantom_prices(sym, current_price, df=df)
                        else:
                            # Original system
                            phantom_tracker.update_phantom_prices(
                                sym, current_price, df=df, btc_price=btc_price, symbol_collector=symbol_collector
                            )
                
                    # Auto-save to database every 15 minutes
                    if (datetime.now() - self.last_save_time).total_seconds() > 900:
                        await self.save_all_candles()
                        self.last_save_time = datetime.now()
                
                    # Check for closed positions periodically
                    if (datetime.now() - last_position_check).total_seconds() > position_check_interval:
                        await self.check_closed_positions(book, shared.get("meta"), ml_scorer, reset_symbol_state, symbol_collector, ml_evolution)
                        last_position_check = datetime.now()
                
                    # Handle panic close requests
                    if sym in panic_list and sym in book.positions:
                        logger.warning(f"Executing panic close for {sym}")
                        pos = book.positions.pop(sym)
                        side = "Sell" if pos.side == "long" else "Buy"
                        try:
                            bybit.place_market(sym, side, pos.qty, reduce_only=True)
                            if self.tg:
                                await self.tg.send_message(f"✅ Panic closed {sym}")
                        except Exception as e:
                            logger.error(f"Panic close error: {e}")
                            if self.tg:
                                await self.tg.send_message(f"❌ Failed to panic close {sym}: {e}")
                        panic_list.remove(sym)
                
                    # Only act on bar close
                    if not k.get("confirm", False):
                        continue
                
                    # Increment candle counter for summary
                    candles_processed += 1
                    
                    # Track analysis time
                    last_analysis[sym] = datetime.now()
                
                    # Log summary periodically instead of every candle
                    if (datetime.now() - last_summary_log).total_seconds() > summary_log_interval:
                        logger.info(f"📊 5-min Summary: {candles_processed} candles processed, {signals_detected} signals, {len(book.positions)} positions open")
                        # Add ML stats to summary
                        if use_enhanced_parallel and ENHANCED_ML_AVAILABLE:
                            # Enhanced parallel system stats
                            if ml_scorer:
                                pullback_stats = ml_scorer.get_stats()
                                pullback_retrain = ml_scorer.get_retrain_info()
                                logger.info(f"🧠 Pullback ML: {pullback_stats.get('completed_trades', 0)} trades, "
                                           f"{pullback_retrain.get('trades_until_next_retrain', 'N/A')} to retrain")

                            if enhanced_mr_scorer:
                                mr_stats = enhanced_mr_scorer.get_enhanced_stats()
                                logger.info(f"🧠 Enhanced MR ML: {mr_stats.get('completed_trades', 0)} trades, "
                                           f"threshold: {mr_stats.get('current_threshold', 'N/A')}, "
                                           f"{mr_stats.get('trades_until_retrain', 'N/A')} to retrain")
                        else:
                            # Original system stats
                            if ml_scorer:
                                ml_stats = ml_scorer.get_stats()
                                retrain_info = ml_scorer.get_retrain_info()
                                logger.info(f"🧠 Pullback ML: {ml_stats.get('completed_trades', 0)} trades, {retrain_info.get('trades_until_next_retrain', 'N/A')} to retrain")
                            # Get mean reversion scorer from shared data for logging
                            shared_mr_scorer_log = shared.get('mean_reversion_scorer') if 'shared' in locals() else None
                            if shared_mr_scorer_log:
                                mr_ml_stats = shared_mr_scorer_log.get_stats()
                                logger.info(f"🧠 Mean Reversion ML: {mr_ml_stats.get('completed_trades', 0)} trades")
                        last_summary_log = datetime.now()
                        candles_processed = 0
                        signals_detected = 0
                    
                    # Check signal cooldown
                    now = time.time()
                    if sym in last_signal_time:
                        if now - last_signal_time[sym] < signal_cooldown:
                            # Skip silently - no need to log every cooldown
                            continue
                
                    # --- ENHANCED PARALLEL STRATEGY ROUTING ---
                    sig = None
                    selected_strategy = "pullback"  # Default
                    selected_ml_scorer = ml_scorer
                    selected_phantom_tracker = phantom_tracker

                    if use_enhanced_parallel and ENHANCED_ML_AVAILABLE:
                        # Use enhanced regime detection for strategy routing
                        regime_analysis = get_enhanced_market_regime(df, sym)

                        # Enhanced regime analysis logging
                        logger.info(f"🔍 [{sym}] MARKET ANALYSIS:")
                        logger.info(f"   📊 Regime: {regime_analysis.primary_regime.upper()} (confidence: {regime_analysis.regime_confidence:.1%})")
                        logger.info(f"   📈 Trend Strength: {regime_analysis.trend_strength:.1f} | Volatility: {regime_analysis.volatility_level}")
                        if regime_analysis.primary_regime == "ranging":
                            logger.info(f"   📦 Range Quality: {regime_analysis.range_quality} | Persistence: {regime_analysis.regime_persistence:.1%}")
                        logger.info(f"   🎯 Recommended Strategy: {regime_analysis.recommended_strategy.upper().replace('_', ' ')}")

                        if regime_analysis.recommended_strategy == "enhanced_mr":
                            # Use Enhanced Mean Reversion System
                            logger.info(f"🟢 [{sym}] ENHANCED MEAN REVERSION ANALYSIS:")
                            sig = detect_signal_mean_reversion(df.copy(), settings, sym)
                            selected_strategy = "enhanced_mr"
                            selected_ml_scorer = enhanced_mr_scorer
                            selected_phantom_tracker = mr_phantom_tracker

                            if sig:
                                logger.info(f"   ✅ Range Signal Detected: {sig.side.upper()} at {sig.entry:.4f}")
                                logger.info(f"   🎯 SL: {sig.sl:.4f} | TP: {sig.tp:.4f} | R:R: {((sig.tp-sig.entry)/(sig.entry-sig.sl) if sig.side=='long' else (sig.entry-sig.tp)/(sig.sl-sig.entry)):.2f}")
                                logger.info(f"   📝 Reason: {sig.reason}")
                            else:
                                logger.info(f"   ❌ No Mean Reversion Signal: Range conditions not met")
                                logger.info(f"   💡 Range quality: {regime_analysis.range_quality}, confidence: {regime_analysis.regime_confidence:.1%}")

                        elif regime_analysis.recommended_strategy == "pullback":
                            # Use Pullback System
                            logger.info(f"🔵 [{sym}] PULLBACK STRATEGY ANALYSIS:")
                            sig = get_pullback_signals(df.copy(), settings, sym)
                            selected_strategy = "pullback"
                            selected_ml_scorer = ml_scorer
                            selected_phantom_tracker = phantom_tracker

                            if sig:
                                logger.info(f"   ✅ Pullback Signal Detected: {sig.side.upper()} at {sig.entry:.4f}")
                                logger.info(f"   🎯 SL: {sig.sl:.4f} | TP: {sig.tp:.4f} | R:R: {((sig.tp-sig.entry)/(sig.entry-sig.sl) if sig.side=='long' else (sig.entry-sig.tp)/(sig.sl-sig.entry)):.2f}")
                                logger.info(f"   📝 Reason: {sig.reason}")
                            else:
                                logger.info(f"   ❌ No Pullback Signal: Trend structure insufficient")
                                logger.info(f"   💡 Trend strength: {regime_analysis.trend_strength:.1f}, volatility: {regime_analysis.volatility_level}")

                        else:
                            # Skip this symbol for now (volatile or poor conditions)
                            logger.info(f"⏭️ [{sym}] STRATEGY SELECTION:")
                            logger.info(f"   ❌ SKIPPING - {regime_analysis.primary_regime.upper()} regime not suitable")
                            logger.info(f"   💡 Volatility: {regime_analysis.volatility_level}, confidence: {regime_analysis.regime_confidence:.1%}")
                            logger.info(f"   📊 Market needs: trending (>25 strength) OR ranging (>medium quality)")
                            continue

                    elif use_regime_switching:
                        # Original regime switching logic (fallback)
                        from market_regime import get_market_regime
                        current_regime = get_market_regime(df)
                        logger.debug(f"[{sym}] Basic regime: {current_regime}")

                        if current_regime == "Ranging":
                            sig = detect_signal_mean_reversion(df.copy(), settings, sym)
                            selected_strategy = "mean_reversion"
                        else:
                            sig = get_pullback_signals(df.copy(), settings, sym)
                            selected_strategy = "pullback"

                    else:
                        # Default pullback strategy
                        sig = get_pullback_signals(df.copy(), settings, sym)
                        selected_strategy = "pullback"
                    # --- END PARALLEL STRATEGY ROUTING ---

                    if sig is None:
                        # Don't log every non-signal to reduce log spam
                        continue
                
                    logger.info(f"[{sym}] Signal detected: {sig.side} @ {sig.entry:.4f}")
                    signals_detected += 1
                
                    # Apply ML scoring and phantom tracking using selected system
                    ml_score = 0
                    ml_reason = "No ML Scoring"
                    should_take_trade = True

                    if selected_ml_scorer is not None and selected_phantom_tracker is not None:
                        try:
                            # Different feature extraction based on strategy
                            if selected_strategy == "enhanced_mr":
                                # Use enhanced MR features
                                from enhanced_mr_features import calculate_enhanced_mr_features

                                logger.info(f"🧠 [{sym}] ENHANCED MR ML ANALYSIS:")
                                enhanced_features = calculate_enhanced_mr_features(df, sig.__dict__, sym)
                                logger.info(f"   📊 Features: {len(enhanced_features)} range-specific features calculated")

                                # Score using Enhanced MR ML system
                                ml_score, ml_reason = selected_ml_scorer.score_signal(sig.__dict__, enhanced_features, df)

                                # Detailed ML decision logging
                                threshold = selected_ml_scorer.min_score
                                should_take_trade = ml_score >= threshold

                                logger.info(f"   🎯 ML Score: {ml_score:.1f} / {threshold:.0f} threshold")
                                logger.info(f"   🔍 Analysis: {ml_reason}")
                                logger.info(f"   📈 Key Factors:")

                                # Log top contributing factors if available
                                try:
                                    top_features = sorted(enhanced_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                                    for i, (feature, value) in enumerate(top_features):
                                        logger.info(f"      {i+1}. {feature}: {value:.3f}")
                                except:
                                    logger.info(f"      Range quality, oscillator signals, market microstructure")

                                if should_take_trade:
                                    logger.info(f"   ✅ DECISION: EXECUTE TRADE - ML confidence above {threshold}")
                                else:
                                    logger.info(f"   ❌ DECISION: REJECT TRADE - ML score {ml_score:.1f} below threshold {threshold}")
                                    logger.info(f"   💡 Rejection reason: {ml_reason}")
                                    continue  # Skip trade execution when ML score below threshold

                                # Record in MR phantom tracker (initially as ML approval, will update with actual execution)
                                selected_phantom_tracker.record_mr_signal(
                                    sym, sig.__dict__, ml_score, False, {}, enhanced_features  # Always False initially - will update after execution
                                )

                            else:
                                # Use pullback features (original system)
                                from strategy_pullback_ml_learning import calculate_ml_features, BreakoutState

                                logger.info(f"🧠 [{sym}] PULLBACK ML ANALYSIS:")

                                # Get or create state for this symbol
                                if sym not in ml_breakout_states:
                                    ml_breakout_states[sym] = BreakoutState()
                                state = ml_breakout_states[sym]

                                # Calculate retracement from entry price
                                if sig.side == "long":
                                    retracement = sig.entry  # Use entry as proxy for retracement level
                                else:
                                    retracement = sig.entry

                                # Calculate basic features (22 features for original ML)
                                basic_features = calculate_ml_features(df, state, sig.side, retracement)
                                logger.info(f"   📊 Core Features: {len(basic_features)} pullback features calculated")

                                # Add entry price for MTF feature calculation
                                basic_features['entry_price'] = sig.entry

                                # Add symbol cluster features with enhanced confidence scores
                                try:
                                    from cluster_feature_enhancer import enhance_ml_features
                                    basic_features = enhance_ml_features(basic_features, sym)
                                    logger.info(f"   🎯 Enhanced with cluster features for symbol group analysis")
                                except Exception as e:
                                    logger.debug(f"[{sym}] Enhanced clustering not available: {e}")
                                    # Fallback to simple clustering
                                    from symbol_clustering import load_symbol_clusters
                                    symbol_clusters = load_symbol_clusters()
                                    cluster_id = symbol_clusters.get(sym, 3)
                                    basic_features['symbol_cluster'] = cluster_id
                                    logger.info(f"   🏷️ Basic cluster ID: {cluster_id}")

                                # Score using pullback ML system
                                ml_score, ml_reason = selected_ml_scorer.score_signal(
                                    {'side': sig.side, 'entry': sig.entry, 'sl': sig.sl, 'tp': sig.tp},
                                    basic_features
                                )

                                # Detailed ML decision logging
                                threshold = selected_ml_scorer.min_score
                                should_take_trade = ml_score >= threshold

                                logger.info(f"   🎯 ML Score: {ml_score:.1f} / {threshold:.0f} threshold")
                                logger.info(f"   🔍 Analysis: {ml_reason}")
                                logger.info(f"   📈 Key Factors:")

                                # Log important technical factors
                                try:
                                    key_factors = []
                                    if 'trend_strength' in basic_features:
                                        key_factors.append(f"Trend: {basic_features['trend_strength']:.2f}")
                                    if 'atr_percentile' in basic_features:
                                        key_factors.append(f"Volatility: {basic_features['atr_percentile']:.2f}")
                                    if 'volume_ratio' in basic_features:
                                        key_factors.append(f"Volume: {basic_features['volume_ratio']:.2f}")
                                    if 'rsi' in basic_features:
                                        key_factors.append(f"RSI: {basic_features['rsi']:.1f}")
                                    if 'bb_position' in basic_features:
                                        key_factors.append(f"BB Pos: {basic_features['bb_position']:.2f}")

                                    for factor in key_factors[:4]:  # Top 4 factors
                                        logger.info(f"      • {factor}")
                                except:
                                    logger.info(f"      Trend strength, volume, volatility, momentum indicators")

                                if should_take_trade:
                                    logger.info(f"   ✅ DECISION: EXECUTE TRADE - ML confidence above {threshold}")
                                else:
                                    logger.info(f"   ❌ DECISION: REJECT TRADE - ML score {ml_score:.1f} below threshold {threshold}")
                                    logger.info(f"   💡 Rejection reason: {ml_reason}")
                                    continue  # Skip trade execution when ML score below threshold

                                # Record in pullback phantom tracker
                                selected_phantom_tracker.record_signal(
                                    symbol=sym,
                                    signal={'side': sig.side, 'entry': sig.entry, 'sl': sig.sl, 'tp': sig.tp},
                                    ml_score=ml_score,
                                    was_executed=should_take_trade,
                                    features=basic_features
                                )

                        except Exception as e:
                            logger.warning(f"🚨 [{sym}] ML SCORING ERROR: {e}")
                            logger.warning(f"   🛡️ FALLBACK: Allowing trade for safety (score: 75)")
                            # Fallback to default behavior
                            ml_score = 75.0
                            ml_reason = "ML Error - Using Default Safety Score"
                            should_take_trade = True

                    # Update strategy name for position tracking
                    strategy_name = selected_strategy

                    # Store ML data in signal for later use
                    if hasattr(sig, '__dict__'):
                        sig.__dict__['ml_score'] = ml_score
                        sig.__dict__['features'] = basic_features if 'basic_features' in locals() else {}
                        sig.__dict__['enhanced_features'] = enhanced_features if 'enhanced_features' in locals() else {}

                except Exception as e:
                    # Safety: Allow signal if ML fails but log the error
                    logger.warning(f"[{sym}] ML scoring error: {e}. Allowing signal for safety.")
                    should_take_trade = True

                # One position per symbol rule - wait for current position to close
                # Final trade execution decision logging
                logger.info(f"💯 [{sym}] FINAL TRADE DECISION:")

                # Check for existing positions
                if sym in book.positions:
                    logger.info(f"   ❌ POSITION CONFLICT: Existing position already open")
                    logger.info(f"   📊 Current positions: {list(book.positions.keys())}")
                    logger.info(f"   💡 One position per symbol rule prevents duplicate entries")
                    continue

                # Get symbol metadata
                m = meta_for(sym, shared["meta"])

                # Round TP/SL to symbol's tick size to prevent API errors
                from position_mgr import round_step
                tick_size = m.get("tick_size", 0.000001) # Default to a small tick size
                
                original_tp = sig.tp
                original_sl = sig.sl

                sig.tp = round_step(sig.tp, tick_size)
                sig.sl = round_step(sig.sl, tick_size)

                if original_tp != sig.tp or original_sl != sig.sl:
                    logger.info(f"[{sym}] Rounded TP/SL to tick size {tick_size}. TP: {original_tp:.6f} -> {sig.tp:.6f}, SL: {original_sl:.6f} -> {sig.sl:.6f}")
                
                # Check account balance and update risk calculation
                current_balance = bybit.get_balance()
                if current_balance:
                    balance = current_balance
                    # Update sizer with current balance for percentage-based risk
                    sizer.account_balance = balance
                
                    # Calculate actual risk amount for this trade
                    if risk.use_percent_risk:
                        risk_amount = balance * (risk.risk_percent / 100.0)
                    else:
                        risk_amount = risk.risk_usd
                
                    # Calculate required margin with max leverage
                    required_margin = (risk_amount * 100) / m.get("max_leverage", 50)  # Rough estimate
                
                    # Check if we have enough for margin + buffer
                    if balance < required_margin * 1.5:  # 1.5x for safety
                        logger.info(f"   ❌ INSUFFICIENT BALANCE:")
                        logger.info(f"      💰 Available: ${balance:.2f}")
                        logger.info(f"      📊 Required Margin: ≈${required_margin:.2f}")
                        logger.info(f"      ⚠️ Safety Buffer: {1.5}x margin = ${required_margin * 1.5:.2f}")
                        logger.info(f"      💡 Need ${(required_margin * 1.5) - balance:.2f} more to safely execute")
                        continue

                    logger.info(f"   ✅ BALANCE CHECK PASSED:")
                    logger.info(f"      💰 Available: ${balance:.2f}")
                    logger.info(f"      💸 Risk Amount: ${risk_amount:.2f}")
                    logger.info(f"      🛡️ Margin Required: ≈${required_margin:.2f}")

                # Calculate position size
                qty = sizer.qty_for(sig.entry, sig.sl, m.get("qty_step",0.001), m.get("min_qty",0.001), ml_score=ml_score)

                if qty <= 0:
                    logger.info(f"   ❌ POSITION SIZE ERROR:")
                    logger.info(f"      📊 Calculated Quantity: {qty}")
                    logger.info(f"      💡 Check: Risk amount, entry price, stop loss distance")
                    logger.info(f"      🔧 Symbol specs: min_qty={m.get('min_qty',0.001)}, qty_step={m.get('qty_step',0.001)}")
                    continue
                
                # Get current market price for stop loss validation
                current_price = df['close'].iloc[-1]
                
                # Validate stop loss is on correct side of market price
                logger.info(f"   🔍 STOP LOSS VALIDATION:")
                logger.info(f"      📍 Current Price: {current_price:.4f}")
                logger.info(f"      🛑 Stop Loss: {sig.sl:.4f}")
                logger.info(f"      📊 Entry: {sig.entry:.4f}")

                sl_valid = True
                if sig.side == "long":
                    if sig.sl >= current_price:
                        logger.info(f"      ❌ INVALID: Long SL ({sig.sl:.4f}) must be BELOW current price ({current_price:.4f})")
                        logger.info(f"      💡 Long stops protect against downward moves")
                        sl_valid = False
                    else:
                        logger.info(f"      ✅ VALID: Long SL ({sig.sl:.4f}) is below current price")
                else:  # short
                    if sig.sl <= current_price:
                        logger.info(f"      ❌ INVALID: Short SL ({sig.sl:.4f}) must be ABOVE current price ({current_price:.4f})")
                        logger.info(f"      💡 Short stops protect against upward moves")
                        sl_valid = False
                    else:
                        logger.info(f"      ✅ VALID: Short SL ({sig.sl:.4f}) is above current price")

                if not sl_valid:
                    continue
                
                # Final execution summary
                logger.info(f"   🚀 EXECUTING TRADE:")
                logger.info(f"      📊 Strategy: {selected_strategy.upper()}")
                logger.info(f"      🎯 Signal: {sig.side.upper()} @ {sig.entry:.4f}")
                logger.info(f"      🛑 Stop Loss: {sig.sl:.4f}")
                logger.info(f"      💰 Take Profit: {sig.tp:.4f}")
                logger.info(f"      📈 Risk:Reward: {((sig.tp-sig.entry)/(sig.entry-sig.sl) if sig.side=='long' else (sig.entry-sig.tp)/(sig.sl-sig.entry)):.2f}")
                logger.info(f"      🔢 Quantity: {qty}")
                logger.info(f"      🧠 ML Score: {ml_score:.1f}")
                logger.info(f"      💸 Risk Amount: ${risk_amount:.2f}")

                # IMPORTANT: Set leverage BEFORE opening position to prevent TP/SL cancellation
                max_lev = int(m.get("max_leverage", 10))
                logger.info(f"   ⚙️ Setting leverage to {max_lev}x (before position to preserve TP/SL)")
                bybit.set_leverage(sym, max_lev)
                
                # Place market order AFTER leverage is set
                side = "Buy" if sig.side == "long" else "Sell"
                try:
                    logger.info(f"[{sym}] Placing {side} order for {qty} units")
                    logger.debug(f"[{sym}] Order details: current_price={current_price:.4f}, sig.entry={sig.entry:.4f}, sig.sl={sig.sl:.4f}, sig.tp={sig.tp:.4f}")
                    order_result = bybit.place_market(sym, side, qty, reduce_only=False)
                    
                    # Get actual entry price from position
                    actual_entry = sig.entry  # Default to signal entry
                    try:
                        # Small delay to ensure position is updated
                        await asyncio.sleep(0.5)
                        
                        position = bybit.get_position(sym)
                        if position and position.get("avgPrice"):
                            actual_entry = float(position["avgPrice"])
                            logger.info(f"[{sym}] Actual entry price: {actual_entry:.4f} (signal was {sig.entry:.4f})")
                            
                            # Recalculate TP based on actual entry to maintain R:R ratio
                            if actual_entry != sig.entry:
                                risk_distance = abs(actual_entry - sig.sl)
                                # Apply same R:R ratio and fee adjustment
                                fee_adjustment = 1.00165  # Same as in strategy
                                if sig.side == "long":
                                    new_tp = actual_entry + (risk_distance * settings.rr * fee_adjustment)
                                else:
                                    new_tp = actual_entry - (risk_distance * settings.rr * fee_adjustment)

                                # Log the adjustment
                                tp_adjustment_pct = ((new_tp - sig.tp) / sig.tp) * 100
                                logger.info(f"[{sym}] Adjusting TP from {sig.tp:.4f} to {new_tp:.4f} ({tp_adjustment_pct:+.2f}%) to maintain {settings.rr}:1 R:R")
                                sig.tp = new_tp
                    except Exception as e:
                        logger.warning(f"[{sym}] Could not get actual entry price: {e}. Using signal entry.")
                    
                    # Set TP/SL - these will not be cancelled since leverage was set before position
                    logger.info(f"[{sym}] Setting TP={sig.tp:.4f} (Limit), SL={sig.sl:.4f} (Market)")
                    bybit.set_tpsl(sym, take_profit=sig.tp, stop_loss=sig.sl, qty=qty)
                
                    # Update book with actual entry price - use the selected strategy name
                    book.positions[sym] = Position(sig.side, qty, actual_entry, sig.sl, sig.tp, datetime.now(), strategy_name=selected_strategy)
                    last_signal_time[sym] = now
                    
                    # Debug log the position details
                    logger.debug(f"[{sym}] Stored position: side={sig.side}, entry={actual_entry:.4f}, TP={sig.tp:.4f}, SL={sig.sl:.4f}")
                    
                    # Record comprehensive trade context for future ML
                    if symbol_collector:
                        try:
                            # Get current BTC price for market context
                            btc_price = None
                            if 'BTCUSDT' in self.frames:
                                btc_price = self.frames['BTCUSDT']['close'].iloc[-1]
                            
                            # Record the context
                            trade_context = symbol_collector.record_trade_context(
                                symbol=sym,
                                df=df,
                                btc_price=btc_price
                            )
                            logger.debug(f"[{sym}] Recorded trade context: session={trade_context.session}, vol_ratio={trade_context.volume_vs_avg:.2f}")
                            
                            # Update symbol profile
                            symbol_collector.update_symbol_profile(sym)
                            
                        except Exception as e:
                            logger.debug(f"Failed to record trade context: {e}")
                
                    # Send notification
                    if self.tg:
                        emoji = "🟢" if sig.side == "long" else "🔴"
                        # Debug log to ensure correct side display
                        logger.debug(f"[{sym}] Notification: side='{sig.side}', emoji='{emoji}', order_side='{side}'")
                        # Calculate actual risk used
                        if risk.use_ml_dynamic_risk:
                            score_range = risk.ml_risk_max_score - risk.ml_risk_min_score
                            risk_range = risk.ml_risk_max_percent - risk.ml_risk_min_percent
                            clamped_score = max(risk.ml_risk_min_score, min(risk.ml_risk_max_score, ml_score))
                            if score_range > 0:
                                score_position = (clamped_score - risk.ml_risk_min_score) / score_range
                                actual_risk_pct = risk.ml_risk_min_percent + (score_position * risk_range)
                            else:
                                actual_risk_pct = risk.ml_risk_min_percent
                            risk_display = f"{actual_risk_pct:.2f}% (ML: {ml_score:.1f})"
                        else:
                            actual_risk_pct = risk.risk_percent
                            risk_display = f"{actual_risk_pct}%"
                        
                        # Add TP adjustment info if applicable
                        entry_info = f"Entry: {actual_entry:.4f}"
                        if actual_entry != sig.entry:
                            price_diff_pct = ((actual_entry - sig.entry) / sig.entry) * 100
                            entry_info += f" (signal: {sig.entry:.4f}, {price_diff_pct:+.2f}%)"
                        
                        msg = (
                            f"{emoji} *{sym} {sig.side.upper()}*\n\n"
                            f"{entry_info}\n"
                            f"Stop Loss: {sig.sl:.4f}\n"
                            f"Take Profit: {sig.tp:.4f}\n"
                            f"Quantity: {qty}\n"
                            f"Risk: {risk_display} (${risk_amount:.2f})\n"
                            f"Reason: {sig.reason}"
                        )
                        await self.tg.send_message(msg)
                
                    logger.info(f"[{sym}] {sig.side} position opened successfully")
                    
                    # Update phantom tracker with actual execution status for MR trades
                    if selected_strategy == "enhanced_mean_reversion" and ENHANCED_ML_AVAILABLE:
                        try:
                            # Update the phantom trade to mark it as actually executed
                            if sym in selected_phantom_tracker.active_mr_phantoms:
                                phantom = selected_phantom_tracker.active_mr_phantoms[sym]
                                phantom.was_executed = True
                                logger.debug(f"[{sym}] Updated MR phantom tracker: trade actually executed")
                                selected_phantom_tracker._save_to_redis()
                        except Exception as e:
                            logger.error(f"Error updating MR phantom execution status: {e}")
                    
                except Exception as e:
                    logger.error(f"Order error: {e}")
                    if self.tg:
                        await self.tg.send_message(f"❌ Failed to open {sym} {sig.side}: {str(e)[:100]}")
                
                except KeyError as e:
                    # KeyError likely means symbol not in config or metadata
                    if str(e).strip("'") == sym:
                        logger.debug(f"[{sym}] Not found in metadata/config - skipping")
                    else:
                        logger.error(f"[{sym}] KeyError accessing: {e}")
                    continue
                except Exception as e:
                    # Other errors - log with more detail
                    import traceback
                    logger.error(f"[{sym}] Processing error: {type(e).__name__}: {e}")
                    logger.debug(f"[{sym}] Traceback: {traceback.format_exc()}")
                    # Don't crash the whole bot for one symbol's error
                    continue
        
        except Exception as e:
            logger.error(f"Bot error: {e}")
            if self.tg:
                await self.tg.send_message(f"❌ Bot error: {str(e)[:100]}")
        
        finally:
            if self.tg:
                await self.tg.send_message("🛑 Trading bot stopped")
                await self.tg.stop()

    async def start(self):
        """Start the bot"""
        self.running = True
        logger.info("Starting trading bot...")
        await self.run()
    
    async def stop(self):
        """Stop the bot"""
        logger.info("Stopping trading bot...")
        self.running = False
        if self.ws:
            await self.ws.close()
        if self.tg:
            await self.tg.stop()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    asyncio.create_task(bot.stop())
    sys.exit(0)

# Global bot instance
bot = TradingBot()

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        # Save candles before shutdown
        if bot.frames:
            logger.info("Saving candles to database before shutdown...")
            bot.storage.save_all_frames(bot.frames)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Final cleanup
        if hasattr(bot, 'storage'):
            bot.storage.close()
        logger.info("Bot terminated")