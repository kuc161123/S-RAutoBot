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

import pandas as pd
import websockets
from dotenv import load_dotenv

from strategy import Settings, detect_signal
from position_mgr import RiskConfig, Book, Position
from sizer import Sizer
from broker_bybit import Bybit, BybitConfig
from telegram_bot import TGBot
from candle_storage_postgres import CandleStorage
from trade_tracker import TradeTracker, Trade

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        
        for symbol in symbols:
            if symbol in stored_frames and len(stored_frames[symbol]) >= 200:  # Need 200 for reliable analysis
                # Use stored data if we have enough candles
                self.frames[symbol] = stored_frames[symbol]
                logger.info(f"[{symbol}] Loaded {len(stored_frames[symbol])} candles from database")
            else:
                # Fetch from API if not in database or insufficient data
                try:
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
                entry_time=datetime.now() - pd.Timedelta(hours=1),  # Approximate
                exit_time=datetime.now(),
                pnl_usd=pnl_usd,
                pnl_percent=pnl_percent,
                exit_reason=exit_reason,
                leverage=leverage
            )
            
            # Add to tracker
            self.trade_tracker.add_trade(trade)
            logger.info(f"Trade recorded: {symbol} {exit_reason} PnL: ${pnl_usd:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
    
    async def check_closed_positions(self, book: Book, meta: dict = None):
        """Check for positions that have been closed and record them"""
        try:
            # Get current positions from exchange
            current_positions = self.bybit.get_positions()
            current_symbols = {p['symbol'] for p in current_positions if float(p.get('size', 0)) > 0}
            
            # Check each tracked position
            closed_positions = []
            for symbol, pos in list(book.positions.items()):
                if symbol not in current_symbols:
                    # Position has been closed
                    closed_positions.append((symbol, pos))
            
            # Record and remove closed positions
            for symbol, pos in closed_positions:
                # Get latest price for PnL calculation
                if symbol in self.frames and len(self.frames[symbol]) > 0:
                    exit_price = float(self.frames[symbol]['close'].iloc[-1])
                    
                    # Determine exit reason based on price
                    if pos.side == "long":
                        if exit_price >= pos.tp * 0.99:  # Near TP
                            exit_reason = "tp"
                        elif exit_price <= pos.sl * 1.01:  # Near SL
                            exit_reason = "sl"
                        else:
                            exit_reason = "manual"
                    else:  # short
                        if exit_price <= pos.tp * 1.01:  # Near TP
                            exit_reason = "tp"
                        elif exit_price >= pos.sl * 0.99:  # Near SL
                            exit_reason = "sl"
                        else:
                            exit_reason = "manual"
                    
                    # Get leverage from metadata
                    if meta:
                        leverage = meta.get(symbol, {}).get("max_leverage", 1.0)
                    else:
                        leverage = 1.0
                    
                    # Record the trade
                    self.record_closed_trade(symbol, pos, exit_price, exit_reason, leverage)
                
                # Remove from book
                book.positions.pop(symbol)
                logger.info(f"Position closed and removed from tracking: {symbol}")
                
        except Exception as e:
            logger.error(f"Error checking closed positions: {e}")
    
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
                    
                    # Add to book
                    from position_mgr import Position
                    book.positions[symbol] = Position(
                        side=side,
                        qty=qty,
                        entry=entry,
                        sl=sl if sl > 0 else entry * 0.95,  # Default 5% stop if not set
                        tp=tp if tp > 0 else entry * 1.1     # Default 10% target if not set
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
            both_hit_rule=cfg["trade"]["both_hit_rule"]
        )
        
        # Initialize components
        risk = RiskConfig(risk_usd=cfg["trade"]["risk_usd"])
        sizer = Sizer(risk)
        book = Book()
        panic_list:list[str] = []
        
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
        
        # Recover existing positions - PRESERVE THEIR ORDERS
        await self.recover_positions(book, sizer)
        
        # DO NOT cancel orders for symbols with existing positions
        # Only cancel orphaned orders (orders without positions)
        logger.info("Checking for orphaned orders to cancel...")
        try:
            # Get all open orders
            resp = bybit._request("GET", "/v5/order/realtime", {"category": "linear", "settleCoin": "USDT"})
            orders = resp.get("result", {}).get("list", [])
            
            for order in orders:
                symbol = order.get("symbol")
                # Only cancel if we don't have a position for this symbol
                if symbol not in book.positions:
                    logger.info(f"Cancelling orphaned order for {symbol}")
                    bybit.cancel_all_orders(symbol)
                else:
                    logger.info(f"Preserving orders for existing position: {symbol}")
        except Exception as e:
            logger.warning(f"Could not check orders: {e}")
        
        # Track analysis times
        last_analysis = {}
        
        # Setup shared data for Telegram
        shared = {
            "risk": risk, 
            "book": book, 
            "panic": panic_list, 
            "meta": cfg.get("symbol_meta",{}),
            "broker": bybit,
            "frames": self.frames,
            "last_analysis": last_analysis,
            "trade_tracker": self.trade_tracker
        }
        
        # Initialize Telegram bot with retry on conflict
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.tg = TGBot(cfg["telegram"]["token"], int(cfg["telegram"]["chat_id"]), shared)
                await self.tg.start_polling()
                # Send shorter startup message for 20 symbols
                await self.tg.send_message(
                    "🚀 *Trading Bot Started*\n\n"
                    f"📊 Monitoring: {len(symbols)} symbols\n"
                    f"⏰ Timeframe: {tf} minutes\n"
                    f"💰 Risk per trade: ${risk.risk_usd}\n"
                    f"📈 R:R Ratio: 1:{settings.rr}\n\n"
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
        
        try:
            # Start streaming
            async for sym, k in self.kline_stream(cfg["bybit"]["ws_public"], topics):
                if not self.running:
                    break
                    
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
                df = df.tail(500)  # Keep only last 500 candles
                self.frames[sym] = df
                
                # Auto-save to database every 2 minutes (more aggressive)
                if (datetime.now() - self.last_save_time).total_seconds() > 120:
                    await self.save_all_candles()
                    self.last_save_time = datetime.now()
                
                # Check for closed positions periodically
                if (datetime.now() - last_position_check).total_seconds() > position_check_interval:
                    await self.check_closed_positions(book, shared.get("meta"))
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
                    last_summary_log = datetime.now()
                    candles_processed = 0
                    signals_detected = 0
                
                # Check signal cooldown
                now = time.time()
                if sym in last_signal_time:
                    if now - last_signal_time[sym] < signal_cooldown:
                        # Skip silently - no need to log every cooldown
                        continue
                
                # Detect signal
                sig = detect_signal(df.copy(), settings, sym)
                if sig is None:
                    # Don't log every non-signal to reduce log spam
                    continue
                
                logger.info(f"[{sym}] Signal detected: {sig.side} @ {sig.entry:.4f}")
                signals_detected += 1
                
                # One position per symbol rule - wait for current position to close
                # This prevents overexposure to a single symbol and allows clean entry/exit
                if sym in book.positions:
                    logger.info(f"[{sym}] Already have position, waiting for it to close before taking new signal")
                    continue
                
                # Get symbol metadata
                m = meta_for(sym, shared["meta"])
                
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
                        logger.warning(f"[{sym}] Insufficient balance (${balance:.2f}, need ~${required_margin:.2f}), skipping signal")
                        continue
                    
                    logger.debug(f"[{sym}] Balance check passed: ${balance:.2f} available, risking ${risk_amount:.2f}")
                
                # Calculate position size
                qty = sizer.qty_for(sig.entry, sig.sl, m.get("qty_step",0.001), m.get("min_qty",0.001))
                
                if qty <= 0:
                    logger.warning(f"[{sym}] Quantity too small, skipping")
                    continue
                
                # Set maximum leverage for this symbol
                max_lev = int(m.get("max_leverage", 10))
                logger.info(f"[{sym}] Setting leverage to {max_lev}x (maximum available)")
                bybit.set_leverage(sym, max_lev)
                
                # Place market order
                side = "Buy" if sig.side == "long" else "Sell"
                try:
                    logger.info(f"[{sym}] Placing {side} order for {qty} units")
                    bybit.place_market(sym, side, qty, reduce_only=False)
                    
                    # Set TP/SL
                    logger.info(f"[{sym}] Setting TP={sig.tp:.4f} (Limit), SL={sig.sl:.4f} (Market)")
                    bybit.set_tpsl(sym, take_profit=sig.tp, stop_loss=sig.sl)
                    
                    # Update book
                    book.positions[sym] = Position(sig.side, qty, sig.entry, sig.sl, sig.tp)
                    last_signal_time[sym] = now
                    
                    # Send notification
                    if self.tg:
                        emoji = "🟢" if sig.side == "long" else "🔴"
                        msg = (
                            f"{emoji} *{sym} {sig.side.upper()}*\n\n"
                            f"Entry: {sig.entry:.4f}\n"
                            f"Stop Loss: {sig.sl:.4f}\n"
                            f"Take Profit: {sig.tp:.4f}\n"
                            f"Quantity: {qty}\n"
                            f"Risk: {risk.risk_percent}% (${risk_amount:.2f})\n"
                            f"Reason: {sig.reason}"
                        )
                        await self.tg.send_message(msg)
                    
                    logger.info(f"[{sym}] {sig.side} position opened successfully")
                    
                except Exception as e:
                    logger.error(f"Order error: {e}")
                    if self.tg:
                        await self.tg.send_message(f"❌ Failed to open {sym} {sig.side}: {str(e)[:100]}")
        
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