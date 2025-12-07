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
from autobot.core.unified_learner import UnifiedLearner, wilson_lower_bound
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

# Phantom tracking now handled by UnifiedLearner for accuracy

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
        
        # Phantom tracking now in unified learner
        self.last_phantom_notify = {}  # Cooldown tracker for notifications
        
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
        
        # Track active trades for close monitoring
        # Format: {symbol: {side, combo, entry, order_id, open_time}}
        self.active_trades = {}
        
        # Unified Learning System (all learning features in one)
        self.learner = UnifiedLearner()
        
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

    def _get_data_dir(self):
        """Get persistent data directory"""
        if os.path.isdir('/data'):
            return '/data'
        if os.path.isdir('/app/data'):
            return '/app/data'
        return '.'

    def save_state(self):
        """Save bot state to persist across restarts"""
        import json
        
        # Phantoms now tracked by learner (learner.save() handles its own state)
        state = {
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
        
        data_dir = self._get_data_dir()
        file_path = os.path.join(data_dir, 'bot_state.json')
        
        try:
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            pending = len(self.learner.pending_signals)
            logger.info(f"💾 State saved to {file_path} (learner tracking {pending} signals)")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
        
        # Also save executed trades to Redis (survives container restarts)
        if self.learner.redis_client:
            try:
                trade_stats = {
                    'wins': self.wins,
                    'losses': self.losses,
                    'trades_executed': self.trades_executed,
                    'daily_pnl': self.daily_pnl,
                    'total_pnl': self.total_pnl,
                    'signals_detected': self.signals_detected
                }
                self.learner.redis_client.set('vwap_bot:executed_trades', json.dumps(trade_stats))
            except Exception as e:
                logger.debug(f"Failed to save trades to Redis: {e}")

    def load_state(self):
        """Load bot state from previous session"""
        import json
        
        data_dir = self._get_data_dir()
        file_path = os.path.join(data_dir, 'bot_state.json')
        
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Phantoms now loaded by learner.load()
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
            pending = len(self.learner.pending_signals)
            logger.info(f"📂 State loaded from {file_path} (saved {age_hrs:.1f}h ago)")
            logger.info(f"   Stats: {self.wins}W/{self.losses}L | Learner: {pending} pending")
        except FileNotFoundError:
            logger.info("📂 No previous state found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
        
        # Load executed trades from Redis (survives container restarts)
        if self.learner.redis_client:
            try:
                data = self.learner.redis_client.get('vwap_bot:executed_trades')
                if data:
                    trade_stats = json.loads(data)
                    self.wins = trade_stats.get('wins', self.wins)
                    self.losses = trade_stats.get('losses', self.losses)
                    self.trades_executed = trade_stats.get('trades_executed', self.trades_executed)
                    self.daily_pnl = trade_stats.get('daily_pnl', self.daily_pnl)
                    self.total_pnl = trade_stats.get('total_pnl', self.total_pnl)
                    self.signals_detected = trade_stats.get('signals_detected', self.signals_detected)
                    logger.info(f"📂 Loaded trade stats from Redis: {self.wins}W/{self.losses}L/{self.trades_executed} trades")
            except Exception as e:
                logger.debug(f"Failed to load trades from Redis: {e}")

    def _sync_promoted_to_yaml(self):
        """Ensure all promoted combos are in the YAML file.
        
        This handles the case where promoted set has combos but YAML was cleared.
        """
        if not self.learner.promoted:
            logger.info("📂 No promoted combos to sync")
            return
            
        try:
            # Load current YAML
            with open('symbol_overrides_VWAP_Combo.yaml', 'r') as f:
                current_yaml = yaml.safe_load(f) or {}
            
            synced_count = 0
            
            for key in self.learner.promoted:
                # Parse key: "SYMBOL:side:combo"
                parts = key.split(':', 2)
                if len(parts) != 3:
                    continue
                    
                sym, side, combo = parts
                
                # Ensure symbol structure exists
                if sym not in current_yaml:
                    current_yaml[sym] = {'long': [], 'short': []}
                elif not isinstance(current_yaml[sym], dict):
                    current_yaml[sym] = {'long': [], 'short': []}
                else:
                    if 'long' not in current_yaml[sym]:
                        current_yaml[sym]['long'] = []
                    if 'short' not in current_yaml[sym]:
                        current_yaml[sym]['short'] = []
                
                # Add combo if not already there
                if combo not in current_yaml[sym][side]:
                    current_yaml[sym][side].append(combo)
                    synced_count += 1
            
            if synced_count > 0:
                # Write back to file
                with open('symbol_overrides_VWAP_Combo.yaml', 'w') as f:
                    yaml.dump(current_yaml, f, default_flow_style=False)
                logger.info(f"📂 Synced {synced_count} promoted combos to YAML")
            else:
                logger.info("📂 All promoted combos already in YAML")
                
        except Exception as e:
            logger.error(f"Failed to sync promoted combos: {e}")

    # --- Telegram Commands ---
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = (
            "🤖 **VWAP BOT COMMANDS**\n\n"
            "/help - Show this message\n"
            "/status - System health & connections\n"
            "/dashboard - Live trading stats\n"
            "/analytics - Deep pattern analysis (30d)\n"
            "/learn - Learning system report\n"
            "/promote - Show promotion candidates\n"
            "/sessions - Session win rates\n"
            "/blacklist - Show blacklisted combos\n"
            "/smart - Smart filter status\n"
            "/risk - Current risk settings"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show system status"""
        uptime = (time.time() - self.start_time) / 3600
        
        # Check connections
        redis_ok = "🟢" if self.learner.redis_client else "🔴"
        pg_ok = "🟢" if self.learner.pg_conn else "🔴"
        
        msg = (
            f"🤖 **SYSTEM STATUS**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⏱️ Uptime: {uptime:.1f} hours\n"
            f"💾 Persistence: Redis {redis_ok} | DB {pg_ok}\n"
            f"🔄 Loops: {self.loop_count}\n"
            f"📡 Trading: {len(self.vwap_combos)} symbols\n"
            f"🧠 Learning: {len(self.all_symbols)} symbols\n"
            f"⚡ Risk: {self.risk_config['value']} {self.risk_config['type']}"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def cmd_phantoms(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show pending signals being tracked by the learner"""
        pending = self.learner.pending_signals
        if not pending:
            await update.message.reply_text("👻 No pending signals being tracked.")
            return
        
        msg = "👻 **PENDING SIGNALS** (Unified Learner)\n\n"
        for sig in pending[-10:]:
            elapsed = int((time.time() - sig.start_time) / 60)
            icon = "🟢" if sig.is_allowed_combo else "🔴"
            msg += f"{icon} `{sig.symbol}` {sig.side.upper()} ({elapsed}m)\n"
        
        total = len(pending)
        if total > 10:
            msg += f"\n... and {total - 10} more"
        
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def cmd_dashboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show comprehensive bot dashboard - UNIFIED stats (phantoms merged into learner)"""
        try:
            # === SYSTEM STATUS ===
            uptime_hrs = (time.time() - self.learner.started_at) / 3600
            
            # === TRADING SYMBOLS ===
            # Use max of YAML-loaded combos and promoted set (handles ephemeral file systems)
            yaml_symbols = len(self.vwap_combos)
            promoted_symbols = len(set(k.split(':')[0] for k in self.learner.promoted))
            total_symbols = max(yaml_symbols, promoted_symbols)
            
            learning_symbols = len(getattr(self, 'all_symbols', []))
            long_combos = sum(len(d.get('long', [])) for d in self.vwap_combos.values())
            short_combos = sum(len(d.get('short', [])) for d in self.vwap_combos.values())
            
            # If YAML is empty but promoted set has combos, count from promoted
            if long_combos == 0 and short_combos == 0 and self.learner.promoted:
                long_combos = sum(1 for k in self.learner.promoted if ':long:' in k)
                short_combos = sum(1 for k in self.learner.promoted if ':short:' in k)
            
            # === UNIFIED LEARNING STATS ===
            learning = self.learner
            total_signals = learning.total_signals
            total_wins = learning.total_wins
            total_losses = learning.total_losses
            learning_total = total_wins + total_losses
            learning_wr = (total_wins / learning_total * 100) if learning_total > 0 else 0
            lower_wr = wilson_lower_bound(total_wins, learning_total)
            
            # Calculate overall EV at 2:1 R:R
            if learning_total > 0:
                wr_decimal = total_wins / learning_total
                ev = (wr_decimal * 2.0) - ((1 - wr_decimal) * 1.0)
            else:
                ev = 0
            
            unique_combos = len(learning.get_all_combos())
            promoted = len(learning.promoted)
            blacklisted = len(learning.blacklist)
            pending = len(learning.pending_signals)
            
            # Count combos approaching thresholds
            all_combos = learning.get_all_combos()
            approaching_promote = len([c for c in all_combos if c['total'] >= 10 and c['lower_wr'] >= 40])
            approaching_blacklist = len([c for c in all_combos if c['total'] >= 5 and c['lower_wr'] <= 35])
            
            # === SESSION BREAKDOWN ===
            sessions = {'asian': {'w': 0, 'l': 0}, 'london': {'w': 0, 'l': 0}, 'newyork': {'w': 0, 'l': 0}}
            long_stats = {'w': 0, 'l': 0}
            short_stats = {'w': 0, 'l': 0}
            
            for symbol, sides in learning.combo_stats.items():
                for side, combos in sides.items():
                    for combo, stats in combos.items():
                        # Aggregate session stats
                        for session, data in stats.get('sessions', {}).items():
                            if session in sessions:
                                sessions[session]['w'] += data.get('w', 0)
                                sessions[session]['l'] += data.get('l', 0)
                        
                        # Aggregate side stats
                        if side == 'long':
                            long_stats['w'] += stats.get('wins', 0)
                            long_stats['l'] += stats.get('losses', 0)
                        else:
                            short_stats['w'] += stats.get('wins', 0)
                            short_stats['l'] += stats.get('losses', 0)
            
            # Calculate session WRs
            asian_total = sessions['asian']['w'] + sessions['asian']['l']
            london_total = sessions['london']['w'] + sessions['london']['l']
            ny_total = sessions['newyork']['w'] + sessions['newyork']['l']
            asian_wr = (sessions['asian']['w'] / asian_total * 100) if asian_total > 0 else 0
            london_wr = (sessions['london']['w'] / london_total * 100) if london_total > 0 else 0
            ny_wr = (sessions['newyork']['w'] / ny_total * 100) if ny_total > 0 else 0
            
            # Calculate side WRs
            long_total = long_stats['w'] + long_stats['l']
            short_total = short_stats['w'] + short_stats['l']
            long_wr = (long_stats['w'] / long_total * 100) if long_total > 0 else 0
            short_wr = (short_stats['w'] / short_total * 100) if short_total > 0 else 0
            
            # === R:R PERFORMANCE ===
            rr_stats = {1.5: {'w': 0, 'l': 0}, 2.0: {'w': 0, 'l': 0}, 2.5: {'w': 0, 'l': 0}, 3.0: {'w': 0, 'l': 0}}
            
            for symbol, sides in learning.combo_stats.items():
                for side, combos in sides.items():
                    for combo, stats in combos.items():
                        for rr, data in stats.get('by_rr', {}).items():
                            rr_float = float(rr)
                            if rr_float in rr_stats:
                                rr_stats[rr_float]['w'] += data.get('w', 0)
                                rr_stats[rr_float]['l'] += data.get('l', 0)
            
            # Build R:R message part
            rr_msg = ""
            for rr in [1.5, 2.0, 2.5, 3.0]:
                data = rr_stats.get(rr, {'w': 0, 'l': 0})
                total = data['w'] + data['l']
                if total > 0:
                    wr = (data['w'] / total * 100)
                    ev = (data['w']/total * rr) - (data['l']/total * 1.0)
                    rr_msg += f"├ {rr}:1 → {wr:.0f}% ({ev:+.2f}R)\n"
            
            # Top performers
            top_combos = learning.get_top_combos(min_trades=3, min_lower_wr=40)[:3]
            
            # Recent activity from learner
            recent = getattr(learning, 'last_resolved', [])[-3:] if hasattr(learning, 'last_resolved') else []
            
            # BTC context
            btc_trend = learning.get_btc_trend()
            btc_change = learning.get_btc_change_1h()
            
            # Day-of-week aggregation
            days = {'monday': {'w': 0, 'l': 0}, 'tuesday': {'w': 0, 'l': 0}, 
                    'wednesday': {'w': 0, 'l': 0}, 'thursday': {'w': 0, 'l': 0},
                    'friday': {'w': 0, 'l': 0}, 'saturday': {'w': 0, 'l': 0}, 'sunday': {'w': 0, 'l': 0}}
            
            for symbol, sides in learning.combo_stats.items():
                for side, combos in sides.items():
                    for combo, stats in combos.items():
                        for day, data in stats.get('by_day', {}).items():
                            if day in days:
                                days[day]['w'] += data.get('w', 0)
                                days[day]['l'] += data.get('l', 0)
            
            # Build day performance string (only weekdays with data)
            day_msg = ""
            day_icons = {'monday': 'Mon', 'tuesday': 'Tue', 'wednesday': 'Wed', 
                         'thursday': 'Thu', 'friday': 'Fri', 'saturday': 'Sat', 'sunday': 'Sun'}
            for d in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
                total = days[d]['w'] + days[d]['l']
                if total > 0:
                    wr = days[d]['w'] / total * 100
                    day_msg += f"├ {day_icons[d]}: {wr:.0f}% ({days[d]['w']}/{total})\n"
            
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
                f"├ 🚀 Auto-Promoted: {len(self.learner.promoted)}\n"
                f"└ Signals: {self.signals_detected} detected\n\n"
                
                f"📚 **UNIFIED TRACKER** ({learning_symbols} symbols)\n"
                f"├ Signals: {total_signals} | Pending: {pending}\n"
                f"├ Resolved: {learning_total} ({total_wins}W/{total_losses}L)\n"
                f"├ WR: {learning_wr:.0f}% (LB: **{lower_wr:.0f}%**)\n"
                f"├ EV: **{ev:+.2f}R** at 2:1 R:R\n"
                f"├ Combos: {unique_combos} learned\n"
                f"├ 📈 Near Promote: {approaching_promote}\n"
                f"├ 📉 Near Blacklist: {approaching_blacklist}\n"
                f"├ 🚀 Promoted: {promoted}\n"
                f"├ 🔽 Demoted: {getattr(self, 'demoted_count', 0)}\n"
                f"└ 🚫 Blacklisted: {blacklisted}\n\n"
                
                f"📊 **SESSION BREAKDOWN**\n"
                f"├ 🌏 Asian:  {asian_wr:.0f}% ({sessions['asian']['w']}/{asian_total})\n"
                f"├ 🌍 London: {london_wr:.0f}% ({sessions['london']['w']}/{london_total})\n"
                f"└ 🌎 NY:     {ny_wr:.0f}% ({sessions['newyork']['w']}/{ny_total})\n\n"
                
                f"📈 **SIDE PERFORMANCE**\n"
                f"├ 🟢 Long:  {long_wr:.0f}% ({long_stats['w']}/{long_total})\n"
                f"└ 🔴 Short: {short_wr:.0f}% ({short_stats['w']}/{short_total})\n\n"
                
                f"💰 **EXECUTED TRADES** (Real Positions)\n"
                f"├ 📊 Total: {self.trades_executed}\n"
                f"├ 🟢 Open: {len(self.active_trades)}\n"
                f"├ ✅ Won: {self.wins}\n"
                f"├ ❌ Lost: {self.losses}\n"
                f"└ 📈 WR: {(self.wins / (self.wins + self.losses) * 100) if (self.wins + self.losses) > 0 else 0:.0f}%\n\n"
                
                f"📅 **DAY BREAKDOWN**\n"
                f"{day_msg}\n"
                
                f"💹 **BEST R:R PERFORMANCE**\n"
                f"{rr_msg}\n"
                
                f"₿ **BTC**: {btc_trend} ({btc_change:+.1f}%)\n"
            )
            
            # Add top performers if any
            if top_combos:
                msg += "\n🏆 **TOP PERFORMERS**\n"
                for c in top_combos:
                    # Find best session
                    sessions = c.get('sessions', {})
                    best_session = 'N/A'
                    best_session_wr = 0
                    for s, data in sessions.items():
                        total = data.get('w', 0) + data.get('l', 0)
                        if total > 0:
                            wr = data['w'] / total * 100
                            if wr > best_session_wr:
                                best_session_wr = wr
                                best_session = {'asian': '🌏', 'london': '🌍', 'newyork': '🌎'}.get(s, s)
                    
                    side_icon = "🟢" if c['side'] == 'long' else "🔴"
                    ev_str = f"{c['ev']:+.2f}R" if c['ev'] != 0 else "0R"
                    combo_short = c['combo'][:15] + '..' if len(c['combo']) > 17 else c['combo']
                    msg += f"├ {side_icon} `{c['symbol']}` {combo_short}\n"
                    msg += f"│  WR:{c['lower_wr']:.0f}% | EV:{ev_str} | {c['optimal_rr']}:1 | {best_session} (N={c['total']})\n"
            
            # Add recent activity from learner
            if recent:
                msg += "\n📈 **RECENT**\n"
                for p in reversed(recent):
                    icon = "✅" if p.get('outcome') == 'win' else "❌"
                    msg += f"├ {icon} `{p.get('symbol', 'N/A')}`\n"
            
            msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
            msg += "💡 /learn /smart /promote"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Dashboard error: {e}")
            logger.error(f"Dashboard error: {e}")


    async def cmd_learn(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show learning system report"""
        try:
            report = self.learner.generate_report()
            await update.message.reply_text(report, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"cmd_learn error: {e}")

    async def cmd_analytics(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show deep analytics (Day/Hour/Patterns)"""
        try:
            # We can reuse the learner's DB connection to get analytics
            if not self.learner.pg_conn:
                await update.message.reply_text("❌ Analytics requires PostgreSQL connection.")
                return

            # Import analytics logic dynamically to avoid circular imports
            from analytics import fetch_trade_history, analyze_by_day, analyze_by_hour, find_winning_patterns
            import psycopg2
            import psycopg2.extras
            
            # Fetch last 30 days
            with self.learner.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM trade_history WHERE created_at > NOW() - INTERVAL '30 days'")
                trades = cur.fetchall()
            
            if not trades:
                await update.message.reply_text("📉 No trades recorded in history yet.")
                return
                
            # Generate Report
            total = len(trades)
            wins = sum(1 for t in trades if t['outcome'] == 'win')
            wr = (wins / total * 100)
            
            msg = (
                f"📊 **DEEP ANALYTICS** (30d)\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Total: {total} | WR: {wr:.1f}%\n\n"
            )
            
            # Best Days
            days = analyze_by_day(trades)[:3]
            msg += "📅 **BEST DAYS**\n"
            for d in days:
                msg += f"├ {d['key']}: {d['wr']:.0f}% ({d['wins']}/{d['total']})\n"
            msg += "\n"
            
            # Best Hours
            hours = [h for h in analyze_by_hour(trades) if h['total'] >= 3][:3]
            if hours:
                msg += "⏰ **BEST HOURS** (UTC)\n"
                for h in hours:
                    msg += f"├ {h['key']}: {h['wr']:.0f}% ({h['wins']}/{h['total']})\n"
                msg += "\n"
            
            # Top Patterns
            patterns = find_winning_patterns(trades, min_trades=3)[:3]
            if patterns:
                msg += "🏆 **TOP PATTERNS**\n"
                for p in patterns:
                    combo_short = p['combo'][:15] + '..' if len(p['combo']) > 17 else p['combo']
                    msg += f"├ {p['symbol']} {p['side'][0].upper()} {combo_short}\n"
                    msg += f"│  WR:{p['wr']:.0f}% (N={p['total']})\n"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Analytics error: {e}")
            logger.error(f"cmd_analytics error: {e}")

    async def cmd_promote(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show combos that could be promoted to active trading"""
        candidates = self.learner.get_promote_candidates()
        
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
            report = self.learner.get_session_report()
            await update.message.reply_text(report, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"cmd_sessions error: {e}")

    async def cmd_blacklist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show blacklisted combos"""
        try:
            blacklist = self.learner.blacklist
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
            report = self.learner.get_smart_report()
            await update.message.reply_text(report, parse_mode='Markdown')
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
                atr_percent = (atr / entry) * 100 if entry > 0 else 1.0
                
                # Get BTC price for context
                btc_price = 0
                try:
                    btc_ticker = self.broker.get_ticker('BTCUSDT')
                    if btc_ticker:
                        btc_price = float(btc_ticker.get('lastPrice', 0))
                except:
                    pass
                
                # Check if allowed to trade
                allowed = self.vwap_combos.get(sym, {}).get(side, [])
                
                # UNIFIED LEARNING: Record signal with full context
                # Returns optimized TP/SL based on learned R:R
                # The learner now handles ALL signal tracking (both allowed and phantom)
                is_allowed = combo in allowed
                
                # Rate limit phantom notifications (max 1 per symbol per 30 min)
                should_notify = True
                if not is_allowed:
                    cooldown_key = f"{sym}_{side}"
                    now = time.time()
                    if not hasattr(self, 'last_phantom_notify_times'):
                        self.last_phantom_notify_times = {}
                    
                    last_notify = self.last_phantom_notify_times.get(cooldown_key, 0)
                    if now - last_notify < 1800: # 30 min cooldown
                        should_notify = False
                    else:
                        self.last_phantom_notify_times[cooldown_key] = now

                smart_tp, smart_sl, smart_explanation = self.learner.record_signal(
                    sym, side, combo, entry, atr, btc_price, is_allowed=is_allowed, notify=should_notify
                )
                
                # Use smart R:R if available, otherwise default
                if smart_tp and smart_sl:
                    tp, sl = smart_tp, smart_sl
                else:
                    if side == 'long':
                        sl, tp = entry - (2.0 * atr), entry + (4.0 * atr)
                    else:
                        sl, tp = entry + (2.0 * atr), entry - (4.0 * atr)
                
                if combo in allowed:
                    # SMART FILTER: Check if smart learner recommends taking this
                    btc_change = self.learner.get_btc_change_1h()
                    should_take, smart_reason = self.learner.should_take_signal(
                        sym, side, combo, atr_percent, btc_change
                    )
                    
                    if should_take:
                        logger.info(f"🚀 SIGNAL: {sym} {side} | {smart_explanation}")
                        await self.execute_trade(sym, side, last_candle, combo)
                    else:
                        logger.info(f"🛑 SMART BLOCK: {sym} {side} | {smart_reason}")
                        # Learner already tracks this signal (recorded above)
                else:
                    # Phantom signal - learner already tracks it (recorded above)
                    # Telegram notifications DISABLED to prevent 429 errors
                    # Just log locally
                    logger.debug(f"👻 PHANTOM: {sym} {side} {combo}")
                    
        except Exception as e:
            logger.error(f"Error {sym}: {e}")

    # NOTE: update_phantoms() removed - phantom tracking now handled by learner.update_signals()


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
            
            # Get optimal R:R from smart learner
            optimal_rr, rr_reason = self.learner.get_optimal_rr(sym, side, combo)
            
            # Calculate TP/SL with optimal R:R
            # Using 1 ATR for SL (1R), RR*ATR for TP
            if side == 'long':
                sl = entry - (1.0 * atr)  # 1 ATR stop
                tp = entry + (optimal_rr * atr)  # R:R * ATR profit
                dist = entry - sl
            else:
                sl = entry + (1.0 * atr)  
                tp = entry - (optimal_rr * atr)
                dist = sl - entry
                
            if dist <= 0: return
            
            qty = risk_amt / dist
            
            # Round based on price magnitude
            if entry > 1000: qty = round(qty, 3)
            elif entry > 10: qty = round(qty, 2)
            elif entry > 1: qty = round(qty, 1)
            else: qty = round(qty, 0)
            
            if qty <= 0: return

            logger.info(f"EXECUTE: {sym} {side} qty={qty} R:R={optimal_rr}:1")
            
            res = self.broker.place_market(sym, side, qty)
            
            if res and res.get('retCode') == 0:
                # Extract order details from response
                result = res.get('result', {})
                order_id = result.get('orderId', 'N/A')
                
                # Set TP/SL and capture result
                tpsl_res = self.broker.set_tpsl(sym, tp, sl, qty)
                tpsl_ok = tpsl_res and tpsl_res.get('retCode') == 0 if tpsl_res else False
                tpsl_status = "✅ SET" if tpsl_ok else "⚠️ FAILED"
                
                self.trades_executed += 1
                
                # Calculate actual values for notification
                sl_pct = abs(entry - sl) / entry * 100
                tp_pct = abs(tp - entry) / entry * 100
                position_value = qty * entry
                
                # Determine if from backtest or auto-promote
                combo_key = f"{sym}:{side}:{combo}"
                is_auto_promoted = combo_key in self.learner.promoted
                source = "🚀 Auto-Promoted" if is_auto_promoted else "📊 Backtest"
                
                # Get current WR and N for this combo
                combo_stats = self.learner.get_combo_stats(sym, side, combo)
                if combo_stats:
                    wr_info = f"WR: {combo_stats['wr']:.0f}% (LB: {combo_stats['lower_wr']:.0f}%) | N={combo_stats['total']}"
                else:
                    wr_info = "WR: N/A (new combo)"
                
                await self.send_telegram(
                    f"🚀 **TRADE EXECUTED**\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 Symbol: `{sym}`\n"
                    f"📈 Side: **{side.upper()}**\n"
                    f"🎯 Combo: `{combo}`\n"
                    f"📁 Source: **{source}**\n"
                    f"📈 {wr_info}\n\n"
                    f"💰 **ORDER DETAILS**\n"
                    f"├ Order ID: `{order_id}`\n"
                    f"├ Quantity: {qty}\n"
                    f"├ Entry: ${entry:.4f}\n"
                    f"├ Position Value: ${position_value:.2f}\n"
                    f"└ Risk: ${risk_amt:.2f} ({self.risk_config['value']}{self.risk_config['type']})\n\n"
                    f"🎯 **RISK MANAGEMENT** {tpsl_status}\n"
                    f"├ Take Profit: ${tp:.4f} (+{tp_pct:.2f}%)\n"
                    f"├ Stop Loss: ${sl:.4f} (-{sl_pct:.2f}%)\n"
                    f"├ R:R Ratio: **{optimal_rr}:1**\n"
                    f"└ ATR: {atr:.4f}\n\n"
                    f"💵 Balance: ${balance:.2f}"
                )
                
                # Track this trade for close monitoring
                self.active_trades[sym] = {
                    'side': side,
                    'combo': combo,
                    'entry': entry,
                    'order_id': order_id,
                    'qty': qty,
                    'tp': tp,
                    'sl': sl,
                    'open_time': time.time(),
                    'is_auto_promoted': is_auto_promoted
                }
            else:
                # Order failed - notify with details
                error_msg = res.get('retMsg', 'Unknown error') if res else 'No response'
                error_code = res.get('retCode', 'N/A') if res else 'N/A'
                
                await self.send_telegram(
                    f"❌ **ORDER FAILED**\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 Symbol: `{sym}`\n"
                    f"📈 Side: **{side.upper()}**\n"
                    f"🎯 Combo: `{combo}`\n\n"
                    f"⚠️ Error Code: `{error_code}`\n"
                    f"📝 Message: {error_msg}\n\n"
                    f"Attempted: qty={qty} @ ${entry:.4f}"
                )
                logger.error(f"Order failed: {res}")
                
        except Exception as e:
            logger.error(f"Execute error: {e}")
            # Notify about execution error
            await self.send_telegram(
                f"❌ **EXECUTION ERROR**\n"
                f"Symbol: `{sym}` {side.upper()}\n"
                f"Error: `{str(e)[:100]}`"
            )

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
            self.tg_app.add_handler(CommandHandler("analytics", self.cmd_analytics))
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
        
        # Sync promoted combos to YAML (ensures YAML matches promoted set)
        self._sync_promoted_to_yaml()
        self.load_overrides()  # Reload after sync
        trading_symbols = list(self.vwap_combos.keys())
        
        if not trading_symbols:
            await self.send_telegram("⚠️ **No trading symbols!**\nLearning will still run on all 400 symbols.")
            logger.warning("No trading symbols, learning only mode")
        
        # Check connections
        redis_ok = "🟢" if self.learner.redis_client else "🔴"
        pg_ok = "🟢" if self.learner.pg_conn else "🔴"

        # Send success notification
        await self.send_telegram(
            f"✅ **VWAP Bot Online!**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Trading: **{len(trading_symbols)}** symbols\n"
            f"📚 Learning: **{len(self.all_symbols)}** symbols\n"
            f"⚙️ Risk: {self.risk_config['value']} {self.risk_config['type']}\n"
            f"🚀 Auto-Promote: **>40% LB WR**\n\n"
            f"💾 System Health:\n"
            f"• Redis: {redis_ok}\n"
            f"• Postgres: {pg_ok}\n\n"
            f"🖥️ **Dashboard**: `http://localhost:8888`\n"
            f"Commands: /help /status /analytics"
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
                
                # Phantom tracking now handled by learner.update_signals() below
                
                # Update learner with candle data (high/low) for accurate resolution
                try:
                    candle_data = {}
                    for sym in self.all_symbols:
                        klines = self.broker.get_klines(sym, '3', limit=1)
                        if klines and len(klines) > 0:
                            candle = klines[0]
                            candle_data[sym] = {
                                'high': float(candle[2]),
                                'low': float(candle[3]),
                                'close': float(candle[4])
                            }
                    
                    # Update unified learner with accurate high/low
                    self.learner.update_signals(candle_data)
                    
                    # Check for closed trades and send notifications
                    for sym in list(self.active_trades.keys()):
                        try:
                            pos = self.broker.get_position(sym)
                            has_position = pos and float(pos.get('size', 0)) > 0
                            
                            if not has_position:
                                # Trade closed - determine outcome
                                trade_info = self.active_trades.pop(sym)
                                
                                # Get current price to estimate P/L
                                current_price = candle_data.get(sym, {}).get('close', 0)
                                candle_high = candle_data.get(sym, {}).get('high', 0)
                                candle_low = candle_data.get(sym, {}).get('low', 0)
                                entry = trade_info['entry']
                                side = trade_info['side']
                                combo = trade_info['combo']
                                
                                if current_price and entry:
                                    if side == 'long':
                                        pnl_pct = ((current_price - entry) / entry) * 100
                                    else:
                                        pnl_pct = ((entry - current_price) / entry) * 100
                                    
                                    # Determine outcome based on P/L
                                    if pnl_pct > 0:
                                        outcome = "win"
                                        outcome_display = "✅ WIN"
                                        self.wins += 1
                                    else:
                                        outcome = "loss"
                                        outcome_display = "❌ LOSS"
                                        self.losses += 1
                                    
                                    # *** CRITICAL FIX: Update learner analytics ***
                                    # This ensures combo stats are updated for executed trades
                                    resolved = self.learner.resolve_executed_trade(
                                        sym, side, outcome, 
                                        exit_price=current_price,
                                        max_high=candle_high,
                                        min_low=candle_low
                                    )
                                    if not resolved:
                                        logger.warning(f"Could not resolve trade in learner: {sym} {side}")
                                    
                                    # Get updated WR/N from analytics (now includes this trade)
                                    updated_stats = self.learner.get_combo_stats(sym, side, combo)
                                    if updated_stats:
                                        wr_info = f"WR: {updated_stats['wr']:.0f}% (LB: {updated_stats['lower_wr']:.0f}%) | N={updated_stats['total']}"
                                    else:
                                        wr_info = "WR: Updating..."
                                    
                                    # Source
                                    source = "🚀 Auto-Promoted" if trade_info.get('is_auto_promoted') else "📊 Backtest"
                                    
                                    # Duration
                                    duration_mins = (time.time() - trade_info['open_time']) / 60
                                    
                                    await self.send_telegram(
                                        f"📝 **TRADE CLOSED**\n"
                                        f"━━━━━━━━━━━━━━━━━━━━\n"
                                        f"📊 Symbol: `{sym}`\n"
                                        f"📈 Side: **{side.upper()}**\n"
                                        f"🎯 Combo: `{combo}`\n"
                                        f"📁 Source: **{source}**\n\n"
                                        f"💰 **RESULT**: {outcome_display}\n"
                                        f"├ P/L: **{pnl_pct:+.2f}%**\n"
                                        f"├ Entry: ${entry:.4f}\n"
                                        f"├ Exit: ${current_price:.4f}\n"
                                        f"└ Duration: {duration_mins:.0f}m\n\n"
                                        f"📊 **UPDATED ANALYTICS**\n"
                                        f"└ {wr_info}"
                                    )
                        except Exception as e:
                            logger.debug(f"Trade close check error for {sym}: {e}")
                    
                    # Clear learner's last_resolved (we handle our own close notifications now)
                    if hasattr(self.learner, 'last_resolved'):
                        self.learner.last_resolved = []
                    
                    # Update BTC price for context tracking
                    btc_candle = candle_data.get('BTCUSDT', {})
                    btc_price = btc_candle.get('close', 0)
                    if btc_price > 0:
                        self.learner.update_btc_price(btc_price)
                        
                except Exception as e:
                    logger.debug(f"Learner update error: {e}")
                
                # Daily summary (every 24 hours)
                await self.send_daily_summary()
                
                # Log stats and save state every loop (for faster auto-promote)
                if self.loop_count % 1 == 0:
                    logger.info(f"Stats: Loop={self.loop_count} Trading={len(trading_symbols)} Learning={len(self.all_symbols)} Signals={self.signals_detected}")
                    self.save_state()
                    self.learner.save()
                    
                    # === AUTO-ACTIVATION ===
                    # Check for high-performing combos to auto-promote
                    logger.info("🔍 Checking for auto-promote candidates...")
                    candidates = self.learner.get_auto_activate_candidates(min_wr=40.0, min_trades=5)
                    logger.info(f"🔍 Found {len(candidates)} candidates")
                    
                    if candidates:
                        logger.info(f"🔍 Top candidate: {candidates[0]}")
                        new_promotions = []
                        for c in candidates:
                            if self.learner.activate_combo(c['symbol'], c['side'], c['combo']):
                                new_promotions.append(c)
                                
                                # Add to active trading config (YAML)
                                try:
                                    # Read current YAML
                                    with open('symbol_overrides_VWAP_Combo.yaml', 'r') as f:
                                        current_yaml = yaml.safe_load(f) or {}
                                    
                                    # Update structure
                                    sym = c['symbol']
                                    side = c['side']
                                    combo = c['combo']
                                    
                                    if sym not in current_yaml:
                                        current_yaml[sym] = {'long': [], 'short': []}
                                    # Ensure proper structure even if symbol exists
                                    elif not isinstance(current_yaml[sym], dict):
                                        current_yaml[sym] = {'long': [], 'short': []}
                                    else:
                                        if 'long' not in current_yaml[sym]:
                                            current_yaml[sym]['long'] = []
                                        if 'short' not in current_yaml[sym]:
                                            current_yaml[sym]['short'] = []
                                    
                                    if combo not in current_yaml[sym][side]:
                                        current_yaml[sym][side].append(combo)
                                        
                                        # Write back to file
                                        with open('symbol_overrides_VWAP_Combo.yaml', 'w') as f:
                                            yaml.dump(current_yaml, f, default_flow_style=False)
                                            
                                        logger.info(f"💾 Added {sym} {side} {combo} to overrides")
                                except Exception as e:
                                    logger.error(f"Failed to update YAML for {c['symbol']}: {e}")

                        # Notify user
                        if new_promotions:
                            msg = "🚀 **AUTO-PROMOTED COMBOS**\n(LB WR > 40%)\n\n"
                            for p in new_promotions:
                                msg += (
                                    f"✅ `{p['symbol']}` {p['side'].upper()}\n"
                                    f"Combo: `{p['combo']}`\n"
                                    f"WR: {p['wr']:.0f}% (LB: {p['lower_wr']:.0f}%) | N={p['total']}\n\n"
                                )
                            await self.send_telegram(msg)  # ENABLED - auto-promote notifications
                            logger.info(f"Auto-promoted {len(new_promotions)} combos")
                    
                    # === AUTO-DEMOTION: Remove poor performers from YAML ===
                    # Check every 5 loops (less frequent than promotion)
                    if self.loop_count % 5 == 0:
                        demoted = []
                        
                        try:
                            # Load current YAML
                            with open('symbol_overrides_VWAP_Combo.yaml', 'r') as f:
                                current_yaml = yaml.safe_load(f) or {}
                            
                            # Check each combo in YAML against analytics
                            combos_to_remove = []
                            
                            for sym, sides in list(current_yaml.items()):
                                if not isinstance(sides, dict):
                                    continue
                                    
                                for side in ['long', 'short']:
                                    combos = sides.get(side, [])
                                    if not isinstance(combos, list):
                                        continue
                                        
                                    for combo in list(combos):
                                        # Get performance from trade_history
                                        stats = self.learner.get_combo_stats(sym, side, combo)
                                        
                                        if stats and stats.get('total', 0) >= 5:
                                            lb_wr = stats.get('lower_wr', 100)
                                            
                                            if lb_wr < 40:  # Below threshold
                                                combos_to_remove.append({
                                                    'symbol': sym,
                                                    'side': side,
                                                    'combo': combo,
                                                    'lb_wr': lb_wr,
                                                    'total': stats['total']
                                                })
                            
                            # Remove poor performers from YAML
                            if combos_to_remove:
                                for item in combos_to_remove:
                                    sym = item['symbol']
                                    side = item['side']
                                    combo = item['combo']
                                    
                                    try:
                                        if sym in current_yaml and side in current_yaml[sym]:
                                            if combo in current_yaml[sym][side]:
                                                current_yaml[sym][side].remove(combo)
                                                demoted.append(item)
                                                # Track demoted count for dashboard
                                                if not hasattr(self, 'demoted_count'):
                                                    self.demoted_count = 0
                                                self.demoted_count += 1
                                                logger.info(f"🔽 Demoted {sym} {side} {combo} (LB WR: {item['lb_wr']:.0f}%)")
                                                
                                                # Clean up empty entries
                                                if not current_yaml[sym]['long'] and not current_yaml[sym]['short']:
                                                    del current_yaml[sym]
                                    except Exception as e:
                                        logger.error(f"Failed to demote {sym}: {e}")
                                
                                # Write updated YAML
                                with open('symbol_overrides_VWAP_Combo.yaml', 'w') as f:
                                    yaml.dump(current_yaml, f, default_flow_style=False)
                                
                                # Notify user
                                if demoted:
                                    msg = "🔽 **AUTO-DEMOTED COMBOS**\n(LB WR < 40%)\n\n"
                                    for d in demoted[:10]:  # Limit to 10
                                        msg += (
                                            f"❌ `{d['symbol']}` {d['side'].upper()}\n"
                                            f"Combo: `{d['combo']}`\n"
                                            f"LB WR: {d['lb_wr']:.0f}% | N={d['total']}\n\n"
                                        )
                                    if len(demoted) > 10:
                                        msg += f"... and {len(demoted) - 10} more"
                                    await self.send_telegram(msg)
                                    
                        except Exception as e:
                            logger.error(f"Auto-demotion error: {e}")
                
                await asyncio.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.save_state()
            self.learner.save()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            self.save_state()
            self.learner.save()
            await self.send_telegram(f"❌ **Bot Error**: {e}")
        finally:
            self.save_state()
            self.learner.save()
            if self.tg_app:
                try:
                    await self.tg_app.updater.stop()
                    await self.tg_app.stop()
                except:
                    pass
