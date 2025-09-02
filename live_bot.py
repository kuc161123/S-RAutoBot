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
    
    async def recover_positions(self, book:Book, sizer:Sizer):
        """Recover existing positions from exchange"""
        logger.info("Checking for existing positions to recover...")
        
        try:
            positions = self.bybit.get_positions()
            
            if positions:
                recovered = 0
                for pos in positions:
                    # Skip if no position size
                    if float(pos.get('size', 0)) == 0:
                        continue
                        
                    symbol = pos['symbol']
                    side = "long" if pos['side'] == "Buy" else "short"
                    qty = float(pos['size'])
                    entry = float(pos['avgPrice'])
                    
                    # Get current TP/SL if set
                    tp = float(pos.get('takeProfit', 0))
                    sl = float(pos.get('stopLoss', 0))
                    
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
                    logger.info(f"Recovered {side} position: {symbol} qty={qty} entry={entry:.4f}")
                
                if recovered > 0:
                    logger.info(f"Successfully recovered {recovered} position(s)")
                    
                    # Send Telegram notification
                    if self.tg:
                        msg = f"📊 *Recovered {recovered} existing position(s)*\n\n"
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
        
        # Recover existing positions
        await self.recover_positions(book, sizer)
        
        # Cancel any existing orders (after recovery to not affect existing positions)
        logger.info("Cancelling any existing orders...")
        bybit.cancel_all_orders()
        
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
            "last_analysis": last_analysis
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
                
                # Log analysis activity
                logger.info(f"[{sym}] Analyzing completed candle - Close: {k['close']}, Volume: {k['volume']}")
                
                # Track analysis time
                last_analysis[sym] = datetime.now()
                
                # Check signal cooldown
                now = time.time()
                if sym in last_signal_time:
                    if now - last_signal_time[sym] < signal_cooldown:
                        logger.debug(f"[{sym}] In cooldown period, skipping analysis")
                        continue
                
                # Detect signal
                sig = detect_signal(df.copy(), settings, sym)
                if sig is None:
                    logger.debug(f"[{sym}] No signal detected - waiting for setup")
                    continue
                
                logger.info(f"[{sym}] Signal detected: {sig.side} @ {sig.entry:.4f}")
                
                # One position per symbol rule - wait for current position to close
                # This prevents overexposure to a single symbol and allows clean entry/exit
                if sym in book.positions:
                    logger.info(f"[{sym}] Already have position, waiting for it to close before taking new signal")
                    continue
                
                # Check account balance
                balance = bybit.get_balance()
                if balance:
                    # Calculate required margin with max leverage
                    required_margin = (cfg['trade']['risk_usd'] * 100) / m.get("max_leverage", 50)  # Rough estimate
                    
                    # Check if we have enough for margin + buffer
                    if balance < required_margin * 1.5:  # 1.5x for safety
                        logger.warning(f"[{sym}] Insufficient balance (${balance:.2f}, need ~${required_margin:.2f}), skipping signal")
                        continue
                    
                    logger.debug(f"[{sym}] Balance check passed: ${balance:.2f} available")
                
                # Calculate position size
                m = meta_for(sym, shared["meta"])
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
                            f"Risk: ${risk.risk_usd}\n"
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