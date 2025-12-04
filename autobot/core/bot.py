import asyncio
import logging
import yaml
import os
import pandas as pd
import pandas_ta as ta
import aiohttp
import time
from datetime import datetime
from dataclasses import dataclass
from autobot.brokers.bybit import Bybit, BybitConfig
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vwap_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VWAPBot")

@dataclass
class PhantomTrade:
    symbol: str
    side: str
    entry: float
    tp: float
    sl: float
    combo: str
    start_time: float

class VWAPBot:
    def __init__(self):
        self.load_config()
        self.setup_broker()
        self.vwap_combos = {}
        self.active_positions = {} 
        
        # Risk Management (Dynamic)
        self.risk_config = {
            'type': 'percent', 
            'value': self.cfg.get('risk', {}).get('risk_percent', 0.5)
        }
        
        # Phantom Tracking
        self.phantom_trades = []
        self.phantom_stats = {'wins': 0, 'losses': 0, 'total': 0}
        self.phantom_history = []
        
    def load_config(self):
        # Load .env manually
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        k, v = line.strip().split('=', 1)
                        os.environ[k] = v
        except FileNotFoundError:
            pass

        with open('config.yaml', 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        # Env Var Replacement
        for k, v in self.cfg['bybit'].items():
            if isinstance(v, str) and v.startswith("${"):
                var = v[2:-1]
                val = os.getenv(var)
                if val: self.cfg['bybit'][k] = val
                
        for k, v in self.cfg['telegram'].items():
            if isinstance(v, str) and v.startswith("${"):
                var = v[2:-1]
                val = os.getenv(var)
                if val: self.cfg['telegram'][k] = val

    def setup_broker(self):
        self.broker = Bybit(BybitConfig(
            base_url=self.cfg['bybit']['base_url'],
            api_key=self.cfg['bybit']['api_key'],
            api_secret=self.cfg['bybit']['api_secret']
        ))
        
    def load_overrides(self):
        """Reload overrides to pick up new backtest findings"""
        try:
            with open('symbol_overrides_VWAP_Combo.yaml', 'r') as f:
                self.vwap_combos = yaml.safe_load(f) or {}
            # logger.info(f"Loaded {len(self.vwap_combos)} VWAP Combo overrides")
        except FileNotFoundError:
            self.vwap_combos = {}

    # --- Telegram Commands ---
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = (
            "🤖 **VWAP BOT COMMANDS**\n\n"
            "/help - Show this message\n"
            "/status - Show current bot status & stats\n"
            "/risk <value> <type> - Set risk (e.g. `/risk 1 %` or `/risk 10 $`)\n"
            "/phantoms - Show phantom trade stats"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        wins = self.phantom_stats['wins']
        total = self.phantom_stats['total']
        wr = (wins / total * 100) if total > 0 else 0.0
        
        msg = (
            "📊 **BOT STATUS**\n\n"
            f"**Risk Mode**: {self.risk_config['value']} {self.risk_config['type']}\n"
            f"**Active Phantoms**: {len(self.phantom_trades)}\n"
            f"**Phantom WR**: {wr:.1f}% ({wins}/{total})\n"
            f"**Combos Loaded**: {len(self.vwap_combos)} symbols"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            args = context.args
            if len(args) < 2:
                await update.message.reply_text("Usage: `/risk 1 %` or `/risk 10 $`", parse_mode='Markdown')
                return
            
            val = float(args[0])
            r_type = args[1].lower()
            
            if r_type in ['%', 'percent']:
                self.risk_config = {'type': 'percent', 'value': val}
                await update.message.reply_text(f"✅ Risk set to **{val}%** of balance", parse_mode='Markdown')
            elif r_type in ['$', 'usd', 'usdt']:
                self.risk_config = {'type': 'usd', 'value': val}
                await update.message.reply_text(f"✅ Risk set to **${val}** per trade", parse_mode='Markdown')
            else:
                await update.message.reply_text("Invalid type. Use `%` or `$`.")
                
        except ValueError:
            await update.message.reply_text("Invalid value.")

    async def send_telegram(self, msg):
        # Keep this for internal notifications (using the same token)
        # But now we also have the Application running.
        # We can use self.tg_app.bot.send_message if initialized, 
        # OR keep using aiohttp for simplicity/independence of the loop.
        # Let's stick to aiohttp for the "push" notifications to avoid context issues,
        # or better: use the bot instance from the app if available.
        if hasattr(self, 'tg_app') and self.tg_app:
            try:
                await self.tg_app.bot.send_message(chat_id=self.cfg['telegram']['chat_id'], text=msg, parse_mode='Markdown')
                return
            except Exception as e:
                logger.error(f"TG App Send Error: {e}")
        
        # Fallback to aiohttp
        token = self.cfg['telegram']['token']
        chat_id = self.cfg['telegram']['chat_id']
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        logger.error(f"Telegram failed: {await resp.text()}")
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    def calculate_indicators(self, df):
        if len(df) < 50: return df
        
        # ATR
        df['atr'] = df.ta.atr(length=14)
        
        # RSI
        df['rsi'] = df.ta.rsi(length=14)
        
        # MACD
        macd = df.ta.macd(close='close', fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        
        # VWAP
        try:
            vwap = df.ta.vwap(high='high', low='low', close='close', volume='volume')
            if isinstance(vwap, pd.DataFrame):
                df['vwap'] = vwap.iloc[:, 0]
            else:
                df['vwap'] = vwap
        except Exception:
            # Fallback VWAP
            tp = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (tp * df['volume']).rolling(1440//3).sum() / df['volume'].rolling(1440//3).sum()

        # Fib (Rolling 50 High/Low)
        df['roll_high'] = df['high'].rolling(50).max()
        df['roll_low'] = df['low'].rolling(50).min()
        
        return df.dropna()

    def get_combo(self, row):
        # RSI Bin
        rsi = row.rsi
        if rsi < 30: r_bin = '<30'
        elif rsi < 40: r_bin = '30-40'
        elif rsi < 60: r_bin = '40-60'
        elif rsi < 70: r_bin = '60-70'
        else: r_bin = '70+'
        
        # MACD Bin
        m_bin = 'bull' if row.macd > row.macd_signal else 'bear'
        
        # Fib Bin
        high = row.roll_high
        low = row.roll_low
        close = row.close
        if high == low: f_bin = '0-23'
        else:
            fib = (high - close) / (high - low) * 100
            if fib < 23.6: f_bin = '0-23'
            elif fib < 38.2: f_bin = '23-38'
            elif fib < 50.0: f_bin = '38-50'
            elif fib < 61.8: f_bin = '50-61'
            elif fib < 78.6: f_bin = '61-78'
            elif fib < 100: f_bin = '78-100'
            else: f_bin = '100+'
            
        return f"RSI:{r_bin} MACD:{m_bin} Fib:{f_bin}"

    async def process_symbol(self, sym):
        try:
            # 1. Check Overrides First (Optimization)
            if sym not in self.vwap_combos:
                # Still process for Phantom Tracking? 
                # If we want to track ALL phantoms, we should process.
                # But to save API calls, maybe only process if we are tracking phantoms generally?
                # User said "signals that get blocked get tracked".
                # Blocked means: Signal detected -> Checked against list -> Rejected.
                # So we MUST process every symbol to detect the signal first.
                pass 
                
            # 2. Fetch Data
            klines = self.broker.get_klines(sym, '3', limit=200)
            if not klines: return
            
            df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            # Convert start to datetime and set index
            df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
            df.set_index('start', inplace=True)
            df.sort_index(inplace=True)
            
            for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
            
            df = self.calculate_indicators(df)
            if df.empty: return
            
            row = df.iloc[-1] # Current candle (or last closed? Strategy uses CLOSE logic, so we check last CLOSED candle usually, but for live we check current if it just closed? 
            # Backtest uses: if row.low <= vwap and row.close > vwap. This implies the candle HAS closed.
            # So we should look at iloc[-2] if iloc[-1] is the currently forming candle?
            # Bybit get_klines returns the latest candle which is usually incomplete.
            # So we must use iloc[-2] (the last completed candle).
            
            last_candle = df.iloc[-2]
            
            # 3. Check Signal
            signal = None
            side = None
            
            # Long: Low touched VWAP, Close > VWAP
            if last_candle.low <= last_candle.vwap and last_candle.close > last_candle.vwap:
                side = 'long'
            # Short: High touched VWAP, Close < VWAP
            elif last_candle.high >= last_candle.vwap and last_candle.close < last_candle.vwap:
                side = 'short'
                
            if side:
                combo = self.get_combo(last_candle)
                
                # Risk Params (needed for Phantom too)
                atr = last_candle.atr
                entry = last_candle.close
                if side == 'long':
                    sl = entry - (2.0 * atr)
                    tp = entry + (4.0 * atr)
                else:
                    sl = entry + (2.0 * atr)
                    tp = entry - (4.0 * atr)
                
                # Check if allowed
                allowed = self.vwap_combos.get(sym, {}).get(side, [])
                
                if combo in allowed:
                    # VALID SIGNAL
                    logger.info(f"🚀 SIGNAL: {sym} {side} {combo}")
                    await self.execute_trade(sym, side, last_candle, combo)
                else:
                    # Phantom / Blocked
                    # Check if we already have a phantom for this symbol/side to avoid spam
                    existing = [p for p in self.phantom_trades if p.symbol == sym]
                    if not existing:
                        logger.info(f"👻 Blocked (Phantom Started): {sym} {side} {combo}")
                        
                        # Create Phantom Trade
                        pt = PhantomTrade(sym, side, entry, tp, sl, combo, time.time())
                        self.phantom_trades.append(pt)
                        self.phantom_stats['total'] += 1
                        
                        # Send Blocked Notification
                        msg = (
                            f"👻 **PHANTOM STARTED**\n"
                            f"Symbol: `{sym}`\n"
                            f"Side: {side.upper()}\n"
                            f"Combo: `{combo}`\n"
                            f"TP: `{tp:.4f}` | SL: `{sl:.4f}`\n"
                            f"Reason: WR < 45%"
                        )
                        await self.send_telegram(msg)
                    
        except Exception as e:
            logger.error(f"Error processing {sym}: {e}")

    async def update_phantoms(self):
        """Check status of running phantom trades"""
        for pt in self.phantom_trades[:]: # Copy to iterate
            try:
                # Fetch current price (lightweight)
                ticker = self.broker.get_ticker(pt.symbol)
                if not ticker: continue
                current_price = float(ticker['lastPrice'])
                
                outcome = None
                if pt.side == 'long':
                    if current_price >= pt.tp: outcome = 'win'
                    elif current_price <= pt.sl: outcome = 'loss'
                else:
                    if current_price <= pt.tp: outcome = 'win'
                    elif current_price >= pt.sl: outcome = 'loss'
                    
                if outcome:
                    self.phantom_trades.remove(pt)
                    self.phantom_stats[f"{outcome}s"] += 1
                    self.phantom_history.append({'symbol': pt.symbol, 'outcome': outcome, 'combo': pt.combo})
                    
                    # Notify Outcome
                    icon = "✅" if outcome == 'win' else "❌"
                    await self.send_telegram(f"{icon} **PHANTOM {outcome.upper()}**\n{pt.symbol} {pt.side} ({pt.combo})")
                    
            except Exception as e:
                logger.error(f"Phantom update error: {e}")

    def display_dashboard(self):
        """Print a cool terminal dashboard"""
        # Clear screen
        print("\033[H\033[J", end="")
        
        print("="*50)
        print(f"   🤖 VWAP BOT DASHBOARD   ")
        print("="*50)
        
        # Live Stats
        print(f"\n📊 LIVE TRADING")
        print(f"   Risk Mode: {self.risk_config['value']} {self.risk_config['type']}")
        print(f"   Active Positions: {len(self.active_positions)}")
        
        # Phantom Stats
        wins = self.phantom_stats['wins']
        losses = self.phantom_stats['losses']
        total = wins + losses
        wr = (wins / total * 100) if total > 0 else 0.0
        
        print(f"\n👻 PHANTOM TRACKER")
        print(f"   Active Phantoms: {len(self.phantom_trades)}")
        print(f"   History: {wins}W - {losses}L (WR: {wr:.1f}%)")
        
        if self.phantom_trades:
            print(f"\n   Running Phantoms:")
            for pt in self.phantom_trades[-5:]: # Show last 5
                print(f"   - {pt.symbol} {pt.side} ({pt.combo})")
                
        print("\n" + "="*50)

    async def execute_trade(self, sym, side, row, combo):
        # Check if already in position
        pos = self.broker.get_position(sym)
        if pos and float(pos['size']) > 0:
            logger.info(f"Skipping {sym}: Already in position")
            return

        # Dynamic Risk Calculation
        balance = self.broker.get_balance() or 0
        risk_val = self.risk_config['value']
        risk_type = self.risk_config['type']
        
        if risk_type == 'percent':
            risk_amt = balance * (risk_val / 100)
        else: # usd
            risk_amt = risk_val
            
        # R:R 1:2
        atr = row.atr
        entry = row.close
        
        if side == 'long':
            sl = entry - (2.0 * atr)
            tp = entry + (4.0 * atr)
            dist = entry - sl
        else:
            sl = entry + (2.0 * atr)
            tp = entry - (4.0 * atr)
            dist = sl - entry
            
        if dist == 0: return
        
        qty = risk_amt / dist
        # Round qty (simple logic, ideally use instrument info)
        # For now, just round to 0 decimals for simplicity or 3 sig figs?
        # Bybit needs specific precision. 
        # Let's try to be safe: use a simple rounding or fetch precision.
        # To be safe/fast: just try to execute. If fail, log it.
        # Better: use a helper to round.
        
        # Simple rounding based on price
        if entry > 1000: qty = round(qty, 3)
        elif entry > 1: qty = round(qty, 1)
        else: qty = round(qty, 0)
        
        if qty <= 0: return

        logger.info(f"Executing {side} on {sym}: Size {qty}, SL {sl}, TP {tp}")
        
        # Execute
        # 1. Market Order
        res = self.broker.place_market(sym, side, qty)
        if res and res.get('retCode') == 0:
            # 2. Set TP/SL
            self.broker.set_tpsl(sym, tp, sl, qty)
            
            # 3. Notify
            msg = (
                f"🚀 **VWAP SNIPER ENTRY**\n"
                f"Symbol: `{sym}`\n"
                f"Side: **{side.upper()}**\n"
                f"Combo: `{combo}`\n"
                f"Price: {entry}\n"
                f"Size: {qty}\n"
                f"Risk: ${risk_amt:.2f} ({risk_type})\n"
                f"TP: {tp:.4f} (4ATR)\n"
                f"SL: {sl:.4f} (2ATR)"
            )
            await self.send_telegram(msg)
        else:
            logger.error(f"Order failed: {res}")

    async def run(self):
        logger.info("🤖 VWAP Bot Started")
        
        # Initialize Telegram App
        token = self.cfg['telegram']['token']
        self.tg_app = ApplicationBuilder().token(token).build()
        
        # Add Handlers
        self.tg_app.add_handler(CommandHandler("help", self.cmd_help))
        self.tg_app.add_handler(CommandHandler("status", self.cmd_status))
        self.tg_app.add_handler(CommandHandler("risk", self.cmd_risk))
        self.tg_app.add_handler(CommandHandler("phantoms", self.cmd_status)) # Alias
        
        await self.tg_app.initialize()
        await self.tg_app.start()
        await self.tg_app.updater.start_polling()
        
        await self.send_telegram("🤖 **VWAP Bot Started**\nCommands: /help, /status, /risk")
        
        # Load symbols
        with open('symbols_400.yaml', 'r') as f:
            symbols = yaml.safe_load(f)['symbols']
            
        try:
            while True:
                self.load_overrides()
                
                # Process Symbols
                for sym in symbols:
                    await self.process_symbol(sym)
                    await asyncio.sleep(0.1) # Rate limit protection
                
                # Update Phantoms
                await self.update_phantoms()
                
                # Show Dashboard
                self.display_dashboard()
                    
                logger.info("Loop complete. Sleeping...")
                await asyncio.sleep(10) # 10s sleep between full loops
        except Exception as e:
            logger.error(f"Fatal Loop Error: {e}")
        finally:
            await self.tg_app.updater.stop()
            await self.tg_app.stop()
