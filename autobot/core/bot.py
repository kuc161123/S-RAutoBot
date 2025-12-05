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
        self.tg_app = None
        
        # Risk Management (Dynamic)
        self.risk_config = {
            'type': 'percent', 
            'value': self.cfg.get('risk', {}).get('risk_percent', 0.5)
        }
        
        # Phantom Tracking
        self.phantom_trades = []
        self.phantom_stats = {'wins': 0, 'losses': 0, 'total': 0}
        self.phantom_history = []
        self.last_phantom_notify = {}  # Cooldown tracker
        
        # Stats
        self.loop_count = 0
        self.signals_detected = 0
        self.trades_executed = 0
        
        # Position Tracking
        self.trade_history = []  # List of {symbol, side, entry, exit, pnl, time}
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        
        # Daily Summary
        self.last_daily_summary = time.time()
        self.start_time = time.time()
        
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
        except FileNotFoundError:
            self.vwap_combos = {}

    def save_state(self):
        """Save bot state to persist across restarts"""
        import json
        state = {
            'phantom_stats': self.phantom_stats,
            'phantom_history': self.phantom_history[-100:],  # Keep last 100
            'trade_history': self.trade_history[-100:],  # Keep last 100
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'wins': self.wins,
            'losses': self.losses,
            'signals_detected': self.signals_detected,
            'trades_executed': self.trades_executed,
            'last_daily_summary': self.last_daily_summary,
            'saved_at': time.time()
        }
        try:
            with open('bot_state.json', 'w') as f:
                json.dump(state, f, indent=2)
            logger.info("💾 State saved")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self):
        """Load bot state from previous session"""
        import json
        try:
            with open('bot_state.json', 'r') as f:
                state = json.load(f)
            
            self.phantom_stats = state.get('phantom_stats', {'wins': 0, 'losses': 0, 'total': 0})
            self.phantom_history = state.get('phantom_history', [])
            self.trade_history = state.get('trade_history', [])
            self.daily_pnl = state.get('daily_pnl', 0.0)
            self.total_pnl = state.get('total_pnl', 0.0)
            self.wins = state.get('wins', 0)
            self.losses = state.get('losses', 0)
            self.signals_detected = state.get('signals_detected', 0)
            self.trades_executed = state.get('trades_executed', 0)
            self.last_daily_summary = state.get('last_daily_summary', time.time())
            
            saved_at = state.get('saved_at', 0)
            age_hrs = (time.time() - saved_at) / 3600
            logger.info(f"📂 State loaded (saved {age_hrs:.1f}h ago)")
            logger.info(f"   Stats: {self.wins}W/{self.losses}L, PnL: ${self.total_pnl:.2f}")
        except FileNotFoundError:
            logger.info("📂 No previous state found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    # --- Telegram Commands ---
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = (
            "🤖 **VWAP BOT COMMANDS**\n\n"
            "/help - Show this message\n"
            "/status - Show bot status & stats\n"
            "/risk <value> <type> - Set risk\n"
            "  Example: `/risk 1 %` or `/risk 10 $`\n"
            "/phantoms - Show phantom trades"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Phantom stats
        p_wins = self.phantom_stats['wins']
        p_losses = self.phantom_stats['losses']
        p_total = p_wins + p_losses
        p_wr = (p_wins / p_total * 100) if p_total > 0 else 0.0
        
        # Trade stats
        t_total = self.wins + self.losses
        t_wr = (self.wins / t_total * 100) if t_total > 0 else 0.0
        
        # Uptime
        uptime_sec = int(time.time() - self.start_time)
        uptime_hr = uptime_sec // 3600
        uptime_min = (uptime_sec % 3600) // 60
        
        msg = (
            "📊 **BOT STATUS**\n\n"
            f"⏱️ Uptime: {uptime_hr}h {uptime_min}m\n"
            f"⚙️ Risk: {self.risk_config['value']} {self.risk_config['type']}\n"
            f"📈 Signals: {self.signals_detected}\n"
            f"🔄 Loops: {self.loop_count}\n\n"
            f"💰 **Trading**\n"
            f"Trades: {self.trades_executed}\n"
            f"WR: {t_wr:.1f}% ({self.wins}W/{self.losses}L)\n"
            f"Daily PnL: ${self.daily_pnl:.2f}\n"
            f"Total PnL: ${self.total_pnl:.2f}\n\n"
            f"👻 **Phantoms**\n"
            f"Active: {len(self.phantom_trades)}\n"
            f"WR: {p_wr:.1f}% ({p_wins}W/{p_losses}L)\n\n"
            f"📂 Combos: {len(self.vwap_combos)} symbols"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def cmd_phantoms(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.phantom_trades:
            await update.message.reply_text("👻 No active phantom trades.")
            return
            
        msg = "👻 **ACTIVE PHANTOMS**\n\n"
        for pt in self.phantom_trades[-10:]:
            elapsed = int((time.time() - pt.start_time) / 60)
            msg += f"• `{pt.symbol}` {pt.side.upper()} ({elapsed}m)\n"
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def cmd_dashboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show combo dashboard summary via Telegram"""
        from collections import Counter
        
        # Count combos
        total_symbols = len(self.vwap_combos)
        long_combos = 0
        short_combos = 0
        rsi_counter = Counter()
        macd_counter = Counter()
        
        for symbol, data in self.vwap_combos.items():
            longs = data.get('long', [])
            shorts = data.get('short', [])
            long_combos += len(longs)
            short_combos += len(shorts)
            
            # Parse combos for analytics
            for combo in longs + shorts:
                parts = combo.split()
                for part in parts:
                    if part.startswith('RSI:'):
                        rsi_counter[part.replace('RSI:', '')] += 1
                    elif part.startswith('MACD:'):
                        macd_counter[part.replace('MACD:', '')] += 1
        
        # Top RSI levels
        top_rsi = rsi_counter.most_common(3)
        rsi_str = ', '.join([f"{r[0]}({r[1]})" for r in top_rsi]) if top_rsi else "None"
        
        msg = (
            "📊 **COMBO DASHBOARD**\n\n"
            f"🎰 **Overview**\n"
            f"Active Symbols: {total_symbols}\n"
            f"Total Combos: {long_combos + short_combos}\n"
            f"🟢 Long: {long_combos}\n"
            f"🔴 Short: {short_combos}\n\n"
            f"📈 **MACD Trend**\n"
            f"Bullish: {macd_counter.get('bull', 0)}\n"
            f"Bearish: {macd_counter.get('bear', 0)}\n\n"
            f"🎯 **Top RSI Zones**\n"
            f"{rsi_str}\n\n"
            f"🌐 Web: http://localhost:8888"
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
        """Send Telegram notification"""
        try:
            if self.tg_app and self.tg_app.bot:
                await self.tg_app.bot.send_message(
                    chat_id=self.cfg['telegram']['chat_id'], 
                    text=msg, 
                    parse_mode='Markdown'
                )
                return
        except Exception as e:
            logger.warning(f"TG App error, using fallback: {e}")
        
        # Fallback to aiohttp
        try:
            token = self.cfg['telegram']['token']
            chat_id = self.cfg['telegram']['chat_id']
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            async with aiohttp.ClientSession() as session:
                payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
                async with session.post(url, json=payload, timeout=10) as resp:
                    if resp.status != 200:
                        logger.error(f"Telegram failed: {resp.status}")
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    async def send_daily_summary(self):
        """Send daily summary at midnight or every 24 hours"""
        # Check if 24 hours have passed
        if time.time() - self.last_daily_summary < 86400:  # 24 hours
            return
            
        self.last_daily_summary = time.time()
        
        # Phantom stats
        p_wins = self.phantom_stats['wins']
        p_losses = self.phantom_stats['losses']
        p_total = p_wins + p_losses
        p_wr = (p_wins / p_total * 100) if p_total > 0 else 0.0
        
        # Trade stats
        t_total = self.wins + self.losses
        t_wr = (self.wins / t_total * 100) if t_total > 0 else 0.0
        
        msg = (
            "📅 **DAILY SUMMARY**\n\n"
            f"💰 **Trading**\n"
            f"Trades: {self.trades_executed}\n"
            f"WR: {t_wr:.1f}% ({self.wins}W/{self.losses}L)\n"
            f"Daily PnL: ${self.daily_pnl:.2f}\n"
            f"Total PnL: ${self.total_pnl:.2f}\n\n"
            f"👻 **Phantoms**\n"
            f"WR: {p_wr:.1f}% ({p_wins}W/{p_losses}L)\n\n"
            f"📂 Active Combos: {len(self.vwap_combos)} symbols\n"
            f"📈 Signals Detected: {self.signals_detected}"
        )
        await self.send_telegram(msg)
        
        # Reset daily stats
        self.daily_pnl = 0.0

    def calculate_indicators(self, df):
        if len(df) < 50: return df
        
        df['atr'] = df.ta.atr(length=14)
        df['rsi'] = df.ta.rsi(length=14)
        
        macd = df.ta.macd(close='close', fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        df['prev_hist'] = df['macd_hist'].shift(1)
        df['prev_rsi'] = df['rsi'].shift(1)
        
        try:
            vwap = df.ta.vwap(high='high', low='low', close='close', volume='volume')
            df['vwap'] = vwap.iloc[:, 0] if isinstance(vwap, pd.DataFrame) else vwap
        except Exception:
            tp = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (tp * df['volume']).rolling(480).sum() / df['volume'].rolling(480).sum()

        df['roll_high'] = df['high'].rolling(50).max()
        df['roll_low'] = df['low'].rolling(50).min()
        
        return df.dropna()

    def get_combo(self, row):
        """Original combo: 70 combinations (5 RSI x 2 MACD x 7 Fib)"""
        # RSI: 5 levels
        rsi = row.rsi
        if rsi < 30: r_bin = '<30'
        elif rsi < 40: r_bin = '30-40'
        elif rsi < 60: r_bin = '40-60'
        elif rsi < 70: r_bin = '60-70'
        else: r_bin = '70+'
        
        # MACD: 2 levels
        m_bin = 'bull' if row.macd > row.macd_signal else 'bear'
        
        # Fib: 7 levels
        high, low, close = row.roll_high, row.roll_low, row.close
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
            klines = self.broker.get_klines(sym, '3', limit=200)
            if not klines: return
            
            df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
            df.set_index('start', inplace=True)
            df.sort_index(inplace=True)
            
            for c in ['open','high','low','close','volume']: 
                df[c] = df[c].astype(float)
            
            df = self.calculate_indicators(df)
            if df.empty or len(df) < 3: return
            
            last_candle = df.iloc[-2]  # Last CLOSED candle
            
            # Check Signal
            side = None
            if last_candle.low <= last_candle.vwap and last_candle.close > last_candle.vwap:
                side = 'long'
            elif last_candle.high >= last_candle.vwap and last_candle.close < last_candle.vwap:
                side = 'short'
                
            if side:
                self.signals_detected += 1
                combo = self.get_combo(last_candle)
                
                atr = last_candle.atr
                entry = last_candle.close
                if side == 'long':
                    sl, tp = entry - (2.0 * atr), entry + (4.0 * atr)
                else:
                    sl, tp = entry + (2.0 * atr), entry - (4.0 * atr)
                
                allowed = self.vwap_combos.get(sym, {}).get(side, [])
                
                if combo in allowed:
                    logger.info(f"🚀 VALID SIGNAL: {sym} {side} {combo}")
                    await self.execute_trade(sym, side, last_candle, combo)
                else:
                    # Phantom - with cooldown (1 notification per symbol per 30 min)
                    existing = [p for p in self.phantom_trades if p.symbol == sym]
                    cooldown_key = f"{sym}_{side}"
                    last_notify = self.last_phantom_notify.get(cooldown_key, 0)
                    
                    if not existing and (time.time() - last_notify) > 1800:  # 30 min cooldown
                        logger.info(f"👻 PHANTOM: {sym} {side} {combo}")
                        
                        pt = PhantomTrade(sym, side, entry, tp, sl, combo, time.time())
                        self.phantom_trades.append(pt)
                        self.phantom_stats['total'] += 1
                        self.last_phantom_notify[cooldown_key] = time.time()
                        
                        # Show allowed combos for this symbol/side
                        allowed_str = '\n'.join([f"  • `{c}`" for c in allowed[:3]]) if allowed else "  None"
                        
                        # Notification with allowed combos
                        await self.send_telegram(
                            f"👻 `{sym}` {side.upper()}\n"
                            f"❌ Combo: `{combo}`\n"
                            f"TP: {tp:.4f} | SL: {sl:.4f}\n\n"
                            f"✅ Allowed combos:\n{allowed_str}"
                        )
                    
        except Exception as e:
            logger.error(f"Error {sym}: {e}")

    async def update_phantoms(self):
        """Check phantom trade outcomes"""
        for pt in self.phantom_trades[:]:
            try:
                ticker = self.broker.get_ticker(pt.symbol)
                if not ticker: continue
                
                price = float(ticker.get('lastPrice', 0))
                if price == 0: continue
                
                outcome = None
                if pt.side == 'long':
                    if price >= pt.tp: outcome = 'win'
                    elif price <= pt.sl: outcome = 'loss'
                else:
                    if price <= pt.tp: outcome = 'win'
                    elif price >= pt.sl: outcome = 'loss'
                    
                if outcome:
                    self.phantom_trades.remove(pt)
                    self.phantom_stats[f"{outcome}s"] += 1
                    self.phantom_history.append({
                        'symbol': pt.symbol, 
                        'outcome': outcome, 
                        'combo': pt.combo
                    })
                    
                    icon = "✅" if outcome == 'win' else "❌"
                    await self.send_telegram(f"{icon} PHANTOM {outcome.upper()}: `{pt.symbol}` {pt.side}")
                    
            except Exception as e:
                logger.error(f"Phantom update error: {e}")

    async def execute_trade(self, sym, side, row, combo):
        try:
            pos = self.broker.get_position(sym)
            if pos and float(pos.get('size', 0)) > 0:
                logger.info(f"Skip {sym}: Already in position")
                return

            balance = self.broker.get_balance() or 0
            if balance <= 0:
                logger.error("Balance is 0")
                return
                
            risk_val = self.risk_config['value']
            risk_type = self.risk_config['type']
            
            risk_amt = balance * (risk_val / 100) if risk_type == 'percent' else risk_val
                
            atr = row.atr
            entry = row.close
            
            if side == 'long':
                sl, tp = entry - (2.0 * atr), entry + (4.0 * atr)
                dist = entry - sl
            else:
                sl, tp = entry + (2.0 * atr), entry - (4.0 * atr)
                dist = sl - entry
                
            if dist <= 0: return
            
            qty = risk_amt / dist
            
            # Round based on price magnitude
            if entry > 1000: qty = round(qty, 3)
            elif entry > 10: qty = round(qty, 2)
            elif entry > 1: qty = round(qty, 1)
            else: qty = round(qty, 0)
            
            if qty <= 0: return

            logger.info(f"EXECUTE: {sym} {side} qty={qty} entry={entry}")
            
            res = self.broker.place_market(sym, side, qty)
            if res and res.get('retCode') == 0:
                self.broker.set_tpsl(sym, tp, sl, qty)
                self.trades_executed += 1
                
                await self.send_telegram(
                    f"🚀 **ENTRY** `{sym}`\n"
                    f"Side: **{side.upper()}**\n"
                    f"Combo: `{combo}`\n"
                    f"Size: {qty} @ {entry:.4f}\n"
                    f"TP: {tp:.4f} | SL: {sl:.4f}\n"
                    f"Risk: ${risk_amt:.2f}"
                )
            else:
                logger.error(f"Order failed: {res}")
                
        except Exception as e:
            logger.error(f"Execute error: {e}")

    async def run(self):
        logger.info("🤖 VWAP Bot Starting...")
        
        # Send starting notification
        await self.send_telegram("⏳ **VWAP Bot Starting...**\nInitializing systems...")
        
        # Initialize Telegram
        try:
            token = self.cfg['telegram']['token']
            self.tg_app = ApplicationBuilder().token(token).build()
            
            self.tg_app.add_handler(CommandHandler("help", self.cmd_help))
            self.tg_app.add_handler(CommandHandler("status", self.cmd_status))
            self.tg_app.add_handler(CommandHandler("risk", self.cmd_risk))
            self.tg_app.add_handler(CommandHandler("phantoms", self.cmd_phantoms))
            self.tg_app.add_handler(CommandHandler("dashboard", self.cmd_dashboard))
            
            await self.tg_app.initialize()
            await self.tg_app.start()
            await self.tg_app.updater.start_polling()
            logger.info("Telegram bot initialized")
        except Exception as e:
            logger.error(f"Telegram init failed: {e}")
            self.tg_app = None
        
        # Load symbols from backtest results ONLY
        self.load_overrides()
        self.load_state()  # Restore previous session data
        symbols = list(self.vwap_combos.keys())
        
        if not symbols:
            await self.send_telegram("⚠️ **No symbols found!**\nAdd combos to symbol_overrides_VWAP_Combo.yaml")
            logger.error("No symbols in overrides file")
            return
        
        # Send success notification
        await self.send_telegram(
            f"✅ **VWAP Bot Online!**\n\n"
            f"📊 Scanning: **{len(symbols)}** symbols\n"
            f"⚙️ Risk: {self.risk_config['value']} {self.risk_config['type']}\n\n"
            f"Symbols: `{', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}`\n\n"
            f"Commands: /help /status /risk /phantoms"
        )
        
        logger.info(f"Scanning {len(symbols)} symbols from backtest: {symbols}")
            
        try:
            while True:
                self.load_overrides()  # Reload to pick up new combos
                symbols = list(self.vwap_combos.keys())  # Update symbol list
                self.loop_count += 1
                
                for sym in symbols:
                    await self.process_symbol(sym)
                    await asyncio.sleep(0.2)
                
                await self.update_phantoms()
                
                # Daily summary (every 24 hours)
                await self.send_daily_summary()
                
                # Log stats and save state every 10 loops
                if self.loop_count % 10 == 0:
                    logger.info(f"Stats: Loop={self.loop_count} Symbols={len(symbols)} Signals={self.signals_detected} Trades={self.trades_executed}")
                    self.save_state()  # Periodic state save
                    
                await asyncio.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.save_state()  # Save on shutdown
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            self.save_state()  # Save on error
            await self.send_telegram(f"❌ **Bot Error**: {e}")
        finally:
            self.save_state()  # Final save
            if self.tg_app:
                try:
                    await self.tg_app.updater.stop()
                    await self.tg_app.stop()
                except:
                    pass
