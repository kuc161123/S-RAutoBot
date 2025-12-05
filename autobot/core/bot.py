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
from autobot.core.combo_learner import ComboLearner
from autobot.core.smart_learner import SmartLearner
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

PHANTOM_TIMEOUT = 14400  # 4 hours same as learning

@dataclass
class PhantomTrade:
    symbol: str
    side: str
    entry: float
    tp: float
    sl: float
    combo: str
    start_time: float
    max_price: float = 0.0   # Highest price seen
    min_price: float = 0.0   # Lowest price seen

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
        
        # Combo Learning System (silent background tracker)
        self.combo_learner = ComboLearner()
        
        # Smart Learning System (adaptive R:R, volatility, BTC context)
        self.smart_learner = SmartLearner()
        
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
        from dataclasses import asdict
        
        # Convert phantom trades to serializable format
        phantom_trades_data = []
        for pt in self.phantom_trades:
            phantom_trades_data.append({
                'symbol': pt.symbol,
                'side': pt.side,
                'entry': pt.entry,
                'tp': pt.tp,
                'sl': pt.sl,
                'combo': pt.combo,
                'start_time': pt.start_time,
                'max_price': pt.max_price,
                'min_price': pt.min_price
            })
        
        state = {
            'phantom_trades': phantom_trades_data,  # Active phantoms!
            'phantom_stats': self.phantom_stats,
            'phantom_history': self.phantom_history[-100:],
            'trade_history': self.trade_history[-100:],
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'wins': self.wins,
            'losses': self.losses,
            'signals_detected': self.signals_detected,
            'trades_executed': self.trades_executed,
            'last_daily_summary': self.last_daily_summary,
            'last_phantom_notify': self.last_phantom_notify,
            'saved_at': time.time()
        }
        try:
            with open('bot_state.json', 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"💾 State saved ({len(phantom_trades_data)} active phantoms)")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self):
        """Load bot state from previous session"""
        import json
        try:
            with open('bot_state.json', 'r') as f:
                state = json.load(f)
            
            # Restore active phantom trades
            self.phantom_trades = []
            for pt_data in state.get('phantom_trades', []):
                try:
                    pt = PhantomTrade(
                        symbol=pt_data['symbol'],
                        side=pt_data['side'],
                        entry=pt_data['entry'],
                        tp=pt_data['tp'],
                        sl=pt_data['sl'],
                        combo=pt_data['combo'],
                        start_time=pt_data['start_time'],
                        max_price=pt_data.get('max_price', 0.0),
                        min_price=pt_data.get('min_price', 0.0)
                    )
                    self.phantom_trades.append(pt)
                except:
                    pass
            
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
            self.last_phantom_notify = state.get('last_phantom_notify', {})
            
            saved_at = state.get('saved_at', 0)
            age_hrs = (time.time() - saved_at) / 3600
            logger.info(f"📂 State loaded (saved {age_hrs:.1f}h ago)")
            logger.info(f"   Stats: {self.wins}W/{self.losses}L, Phantoms: {len(self.phantom_trades)} active")
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
            "/dashboard - Comprehensive overview\n"
            "/risk <value> <type> - Set risk\n"
            "/phantoms - Show phantom trades\n\n"
            "📚 **LEARNING SYSTEM**\n"
            "/learn - Learning report\n"
            "/smart - Smart learning (adaptive R:R)\n"
            "/promote - Promotion candidates\n"
            "/sessions - Session performance\n"
            "/blacklist - Blacklisted combos"
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
        """Show comprehensive bot dashboard"""
        try:
            from autobot.core.combo_learner import wilson_lower_bound
            
            # === SYSTEM STATUS ===
            uptime_hrs = (time.time() - self.combo_learner.started_at) / 3600
            
            # === TRADING SYMBOLS ===
            total_symbols = len(self.vwap_combos)
            learning_symbols = len(getattr(self, 'all_symbols', []))
            long_combos = sum(len(d.get('long', [])) for d in self.vwap_combos.values())
            short_combos = sum(len(d.get('short', [])) for d in self.vwap_combos.values())
            
            # === PHANTOM PERFORMANCE ===
            phantom_wins = self.phantom_stats.get('wins', 0)
            phantom_losses = self.phantom_stats.get('losses', 0)
            phantom_total = phantom_wins + phantom_losses
            phantom_wr = (phantom_wins / phantom_total * 100) if phantom_total > 0 else 0
            phantom_lb_wr = wilson_lower_bound(phantom_wins, phantom_total)
            active_phantoms = len(self.phantom_trades)
            
            # === LEARNING STATS ===
            learning = self.combo_learner
            total_signals = learning.total_signals_tracked
            total_wins = learning.total_wins
            total_losses = learning.total_losses
            learning_total = total_wins + total_losses
            learning_wr = (total_wins / learning_total * 100) if learning_total > 0 else 0
            lower_wr = wilson_lower_bound(total_wins, learning_total)
            
            unique_combos = len(learning.get_all_combos())
            promoted = len(learning.promoted)
            blacklisted = len(learning.blacklist)
            pending = len(learning.pending_signals)
            
            # Top performers
            top_combos = learning.get_top_combos(min_trades=3, min_lower_wr=40)[:3]
            
            # Recent phantom activity
            recent_phantoms = self.phantom_history[-3:] if self.phantom_history else []
            
            # === BUILD MESSAGE ===
            msg = (
                "📊 **VWAP BOT DASHBOARD**\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                
                f"⚙️ **SYSTEM**\n"
                f"├ Uptime: {uptime_hrs:.1f}h | Loops: {self.loop_count}\n"
                f"└ Risk: {self.risk_config['value']} {self.risk_config['type']}\n\n"
                
                f"🎯 **TRADING**\n"
                f"├ Symbols: {total_symbols} active\n"
                f"├ Combos: 🟢{long_combos} / 🔴{short_combos}\n"
                f"└ Signals: {self.signals_detected} detected\n\n"
                
                f"👻 **PHANTOMS**\n"
                f"├ Active: {active_phantoms} tracking\n"
                f"├ Done: {phantom_total} ({phantom_wins}W/{phantom_losses}L)\n"
                f"└ WR: {phantom_wr:.0f}% (LB: **{phantom_lb_wr:.0f}%**)\n\n"
                
                f"📚 **LEARNING** ({learning_symbols} symbols)\n"
                f"├ Signals: {total_signals} | Pending: {pending}\n"
                f"├ Resolved: {learning_total} ({total_wins}W/{total_losses}L)\n"
                f"├ WR: {learning_wr:.0f}% (LB: **{lower_wr:.0f}%**)\n"
                f"├ Combos: {unique_combos} learned\n"
                f"├ 🚀 Promoted: {promoted}\n"
                f"└ 🚫 Blacklisted: {blacklisted}\n"
            )
            
            # Add top performers if any
            if top_combos:
                msg += "\n🏆 **TOP PERFORMERS**\n"
                for c in top_combos:
                    msg += f"├ `{c['symbol']}` {c['side'][0].upper()}: {c['lower_wr']:.0f}%\n"
            
            # Add recent phantom activity
            if recent_phantoms:
                msg += "\n📈 **RECENT**\n"
                for p in reversed(recent_phantoms):
                    icon = "✅" if p['outcome'] == 'win' else "❌"
                    msg += f"├ {icon} `{p['symbol']}`\n"
            
            msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
            msg += "💡 /learn /sessions /phantoms"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Dashboard error: {e}")
            logger.error(f"Dashboard error: {e}")

    async def cmd_learn(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show learning system report"""
        try:
            report = self.combo_learner.generate_report()
            await update.message.reply_text(report, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"cmd_learn error: {e}")

    async def cmd_promote(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show combos that could be promoted to active trading"""
        candidates = self.combo_learner.get_promote_candidates()
        
        if not candidates:
            await update.message.reply_text(
                "📊 **NO PROMOTION CANDIDATES YET**\n\n"
                "Need combos with:\n"
                "• Lower Bound WR ≥ 40%\n"
                "• N ≥ 10 trades\n"
                "• Positive EV\n\n"
                "Keep running to collect more data!",
                parse_mode='Markdown'
            )
            return
        
        msg = "🚀 **PROMOTION CANDIDATES**\n\n"
        
        for c in candidates[:10]:
            msg += f"**{c['symbol']}** {c['side'].upper()}\n"
            msg += f"`{c['combo']}`\n"
            msg += f"LB_WR: {c['lower_wr']:.0f}% (N={c['total']}) EV: {c['ev']:.2f}R\n\n"
        
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def cmd_sessions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show session performance report"""
        try:
            report = self.combo_learner.get_session_report()
            await update.message.reply_text(report, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"cmd_sessions error: {e}")

    async def cmd_blacklist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show blacklisted combos"""
        try:
            blacklist = self.combo_learner.blacklist
            if not blacklist:
                await update.message.reply_text("🚫 No blacklisted combos yet.", parse_mode='Markdown')
                return
            
            msg = f"🚫 **BLACKLISTED COMBOS** ({len(blacklist)})\n\n"
            for item in list(blacklist)[:15]:
                parts = item.split(':')
                if len(parts) >= 3:
                    msg += f"• `{parts[0]}` {parts[1]}\n"
            
            if len(blacklist) > 15:
                msg += f"\n... and {len(blacklist) - 15} more"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"cmd_blacklist error: {e}")

    async def cmd_smart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show smart learning report with adaptive parameters"""
        try:
            smart = self.smart_learner
            
            # Basic stats
            uptime = (time.time() - smart.started_at) / 3600
            total = smart.total_wins + smart.total_losses
            wr = (smart.total_wins / total * 100) if total > 0 else 0
            
            from autobot.core.smart_learner import wilson_lower_bound
            lb_wr = wilson_lower_bound(smart.total_wins, total)
            
            btc_change = smart.get_btc_change_1h()
            btc_trend = smart.get_btc_trend(btc_change)
            
            msg = (
                "🧠 **SMART LEARNING**\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                f"⏱️ Running: {uptime:.1f}h\n"
                f"📊 Signals: {smart.total_signals}\n"
                f"📈 Resolved: {total} ({smart.total_wins}W/{smart.total_losses}L)\n"
                f"🎯 WR: {wr:.0f}% (LB: **{lb_wr:.0f}%**)\n"
                f"⏳ Pending: {len(smart.pending_signals)}\n\n"
                f"₿ **BTC Context**\n"
                f"├ Trend: {btc_trend}\n"
                f"└ 1h Change: {btc_change:+.2f}%\n\n"
            )
            
            # Recent auto-adjustments
            if smart.adjustments_made:
                msg += "🔧 **AUTO-ADJUSTMENTS**\n"
                for adj in smart.adjustments_made[-5:]:
                    msg += f"├ `{adj['symbol']}` {adj['side']}\n"
                    msg += f"│  {adj['type']}: {adj['old']}→{adj['new']}\n"
                msg += "\n"
            else:
                msg += "🔧 No adjustments yet (need more data)\n\n"
            
            msg += "━━━━━━━━━━━━━━━━━━━━\n"
            msg += "💡 Bot self-adjusts R:R & filters"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"cmd_smart error: {e}")

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
                
                # LEARNING: Record ALL signals (silently, for learning)
                self.combo_learner.record_signal(sym, side, combo, entry, tp, sl)
                
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
        """Check phantom trade outcomes with high/low accuracy"""
        for pt in self.phantom_trades[:]:
            try:
                # Check timeout first
                if time.time() - pt.start_time > PHANTOM_TIMEOUT:
                    self.phantom_trades.remove(pt)
                    logger.info(f"⏰ Phantom timeout: {pt.symbol}")
                    continue
                
                # Get recent candle for high/low
                klines = self.broker.get_klines(pt.symbol, '3', limit=1)
                if not klines:
                    continue
                
                candle = klines[0]
                high = float(candle[2])
                low = float(candle[3])
                current = float(candle[4])
                
                # Track max/min prices seen
                if pt.max_price == 0:
                    pt.max_price = high
                    pt.min_price = low
                else:
                    pt.max_price = max(pt.max_price, high)
                    pt.min_price = min(pt.min_price, low)
                
                outcome = None
                
                if pt.side == 'long':
                    # Check if SL was hit first (using low)
                    if low <= pt.sl:
                        outcome = 'loss'
                    # Check if TP was hit (using high)
                    elif high >= pt.tp:
                        outcome = 'win'
                else:  # short
                    # Check if SL was hit first (using high)
                    if high >= pt.sl:
                        outcome = 'loss'
                    # Check if TP was hit (using low)
                    elif low <= pt.tp:
                        outcome = 'win'
                
                if outcome:
                    self.phantom_trades.remove(pt)
                    self.phantom_stats[f"{outcome}s"] += 1
                    
                    # Calculate time and drawdown
                    time_mins = (time.time() - pt.start_time) / 60
                    if pt.side == 'long':
                        max_dd = (pt.entry - pt.min_price) / pt.entry * 100
                    else:
                        max_dd = (pt.max_price - pt.entry) / pt.entry * 100
                    
                    self.phantom_history.append({
                        'symbol': pt.symbol, 
                        'outcome': outcome, 
                        'combo': pt.combo,
                        'time_mins': round(time_mins, 1),
                        'max_drawdown': round(max_dd, 2)
                    })
                    
                    icon = "✅" if outcome == 'win' else "❌"
                    await self.send_telegram(
                        f"{icon} PHANTOM {outcome.upper()}: `{pt.symbol}` {pt.side}\n"
                        f"⏱️ {time_mins:.0f}m | DD: {max_dd:.1f}%"
                    )
                    
            except Exception as e:
                logger.debug(f"Phantom update error: {e}")

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
            self.tg_app.add_handler(CommandHandler("learn", self.cmd_learn))
            self.tg_app.add_handler(CommandHandler("promote", self.cmd_promote))
            self.tg_app.add_handler(CommandHandler("sessions", self.cmd_sessions))
            self.tg_app.add_handler(CommandHandler("blacklist", self.cmd_blacklist))
            self.tg_app.add_handler(CommandHandler("smart", self.cmd_smart))
            
            # Global error handler
            async def error_handler(update, context):
                logger.error(f"Telegram error: {context.error}")
                if update and update.message:
                    await update.message.reply_text(f"❌ Command error: {context.error}")
            self.tg_app.add_error_handler(error_handler)
            
            await self.tg_app.initialize()
            await self.tg_app.start()
            await self.tg_app.updater.start_polling(drop_pending_updates=True)
            logger.info("Telegram bot initialized")
        except Exception as e:
            logger.error(f"Telegram init failed: {e}")
            self.tg_app = None
        
        # Load symbols from backtest results (for TRADING)
        self.load_overrides()
        self.load_state()  # Restore previous session data
        trading_symbols = list(self.vwap_combos.keys())
        
        # Load ALL 400 symbols for LEARNING
        try:
            with open('symbols_400.yaml', 'r') as f:
                all_symbols_data = yaml.safe_load(f)
                self.all_symbols = all_symbols_data.get('symbols', trading_symbols)
            logger.info(f"📚 Learning mode: scanning {len(self.all_symbols)} symbols")
        except FileNotFoundError:
            self.all_symbols = trading_symbols
            logger.warning("symbols_400.yaml not found, using trading symbols only")
        
        if not trading_symbols:
            await self.send_telegram("⚠️ **No trading symbols!**\nLearning will still run on all 400 symbols.")
            logger.warning("No trading symbols, learning only mode")
        
        # Send success notification
        await self.send_telegram(
            f"✅ **VWAP Bot Online!**\n\n"
            f"📊 Trading: **{len(trading_symbols)}** symbols\n"
            f"📚 Learning: **{len(self.all_symbols)}** symbols\n"
            f"⚙️ Risk: {self.risk_config['value']} {self.risk_config['type']}\n\n"
            f"Commands: /help /status /risk /phantoms"
        )
        
        logger.info(f"Trading {len(trading_symbols)} symbols, Learning {len(self.all_symbols)} symbols")
            
        try:
            while True:
                self.load_overrides()  # Reload to pick up new combos
                trading_symbols = list(self.vwap_combos.keys())
                self.loop_count += 1
                
                # Scan ALL symbols for learning, but only trade allowed ones
                for sym in self.all_symbols:
                    await self.process_symbol(sym)
                    await asyncio.sleep(0.1)  # Faster for learning
                
                await self.update_phantoms()
                
                # Update combo learner with current prices for ALL symbols
                try:
                    prices = {}
                    for sym in self.all_symbols:
                        ticker = self.broker.get_ticker(sym)
                        if ticker:
                            prices[sym] = float(ticker.get('lastPrice', 0))
                    
                    # Update both learners
                    self.combo_learner.update_signals(prices)
                    self.smart_learner.update_signals(prices)
                    
                    # Update BTC price for smart learner
                    btc_price = prices.get('BTCUSDT', 0)
                    if btc_price > 0:
                        self.smart_learner.update_btc_price(btc_price)
                        
                except Exception as e:
                    logger.debug(f"Learner update error: {e}")
                
                # Daily summary (every 24 hours)
                await self.send_daily_summary()
                
                # Log stats and save state every 10 loops
                if self.loop_count % 10 == 0:
                    logger.info(f"Stats: Loop={self.loop_count} Trading={len(trading_symbols)} Learning={len(self.all_symbols)} Signals={self.signals_detected}")
                    self.save_state()  # Periodic state save
                    self.combo_learner.save()  # Save learning data
                    self.smart_learner.save()  # Save smart learning
                    
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
