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
        self.frames = {s:new_frame() for s in symbols}
        risk = RiskConfig(risk_usd=cfg["trade"]["risk_usd"])
        sizer = Sizer(risk)
        book = Book()
        panic_list:list[str] = []
        
        # Initialize Bybit client
        bybit = Bybit(BybitConfig(
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
        
        # Cancel any existing orders
        logger.info("Cancelling any existing orders...")
        bybit.cancel_all_orders()
        
        # Setup shared data for Telegram
        shared = {
            "risk": risk, 
            "book": book, 
            "panic": panic_list, 
            "meta": cfg.get("symbol_meta",{}),
            "broker": bybit
        }
        
        # Initialize Telegram bot with retry on conflict
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.tg = TGBot(cfg["telegram"]["token"], int(cfg["telegram"]["chat_id"]), shared)
                await self.tg.start_polling()
                await self.tg.send_message(
                    "🚀 *Trading Bot Started*\n\n"
                    f"Symbols: {', '.join(symbols)}\n"
                    f"Timeframe: {tf}m\n"
                    f"Risk per trade: ${risk.risk_usd}\n"
                    f"R:R Ratio: 1:{settings.rr}"
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
                    index=[pd.to_datetime(ts, unit="ms")],
                    columns=["open","high","low","close","volume"]
                )
                
                df = self.frames[sym]
                df.loc[row.index[0]] = row.iloc[0]
                df.sort_index(inplace=True)
                df = df.tail(500)  # Keep only last 500 candles
                self.frames[sym] = df
                
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
                
                # Check signal cooldown
                now = time.time()
                if sym in last_signal_time:
                    if now - last_signal_time[sym] < signal_cooldown:
                        continue
                
                # Detect signal
                sig = detect_signal(df.copy(), settings)
                if sig is None:
                    continue
                
                logger.info(f"[{sym}] Signal detected: {sig.side} @ {sig.entry:.4f}")
                
                # Check if we have a position
                if sym in book.positions:
                    cur = book.positions[sym]
                    if cur.side == sig.side:
                        logger.info(f"[{sym}] Already have {cur.side} position, skipping")
                        continue
                    
                    # Close opposite position first
                    logger.info(f"[{sym}] Closing opposite {cur.side} position")
                    try:
                        bybit.place_market(sym, "Sell" if cur.side=="long" else "Buy", cur.qty, reduce_only=True)
                        book.positions.pop(sym, None)
                        if self.tg:
                            await self.tg.send_message(f"🔄 Closed {sym} {cur.side} to flip position")
                    except Exception as e:
                        logger.error(f"Close error: {e}")
                        continue
                
                # Calculate position size
                m = meta_for(sym, shared["meta"])
                qty = sizer.qty_for(sig.entry, sig.sl, m.get("qty_step",0.001), m.get("min_qty",0.001))
                
                if qty <= 0:
                    logger.warning(f"[{sym}] Quantity too small, skipping")
                    continue
                
                # Place market order
                side = "Buy" if sig.side == "long" else "Sell"
                try:
                    logger.info(f"[{sym}] Placing {side} order for {qty} units")
                    bybit.place_market(sym, side, qty, reduce_only=False)
                    
                    # Set TP/SL
                    logger.info(f"[{sym}] Setting TP={sig.tp:.4f}, SL={sig.sl:.4f}")
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
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        logger.info("Bot terminated")