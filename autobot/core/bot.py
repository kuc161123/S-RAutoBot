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
        
        # Track pending limit orders waiting to be filled
        # Format: {symbol: {order_id, side, combo, entry_price, tp, sl, qty, created_at}}
        self.pending_limit_orders = {}
        
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
        """Load combos - AUTO-PROMOTE/DEMOTE ONLY MODE
        
        Backtest golden combos are DISABLED.
        Only learner.promoted combos will be used for trading.
        """
        # DISABLED: Backtest golden combos
        # Previously loaded from backtest_golden_combos.yaml
        # Now using ONLY auto-promoted combos from live learning
        self.vwap_combos = {}  # Empty - no backtest combos
        logger.info("📂 AUTO-PROMOTE MODE: No backtest combos loaded (live learning only)")

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
            'pending_limit_orders': self.pending_limit_orders,
            'active_trades': self.active_trades,
            'demoted_count': getattr(self, 'demoted_count', 0),  # Persist demotion count
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
            
            # Load pending orders and active trades
            self.pending_limit_orders = state.get('pending_limit_orders', {})
            self.active_trades = state.get('active_trades', {})
            self.demoted_count = state.get('demoted_count', 0)
            
            saved_at = state.get('saved_at', 0)
            age_hrs = (time.time() - saved_at) / 3600
            pending = len(self.learner.pending_signals)
            pending_orders = len(self.pending_limit_orders)
            active = len(self.active_trades)
            logger.info(f"📂 State loaded from {file_path} (saved {age_hrs:.1f}h ago)")
            logger.info(f"   Stats: {self.wins}W/{self.losses}L | Learner: {pending} pending")
            logger.info(f"   Orders: {pending_orders} pending, {active} active trades")
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
        """DISABLED: Using static backtest golden combos now.
        
        Previously synced promoted combos to YAML, but we're now using
        backtest_golden_combos.yaml as the single source of truth.
        """
        logger.debug("📂 _sync_promoted_to_yaml DISABLED - using backtest golden combos")
        return
    
    def _check_feature_filters(self) -> tuple:
        """Check market feature filters before executing trade.
        
        DISABLED: Backtest validated combos at ALL hours, not filtered by session.
        To match backtest exactly, we allow trading at all hours.
        
        If you want to add session filtering later, uncomment the code below.
        """
        # # Session filter (DISABLED - backtest didn't use this)
        # from datetime import datetime
        # hour = datetime.utcnow().hour
        # if hour < 8:  # Asia session
        #     return False, "asia_session"
        
        # All hours allowed to match backtest behavior
        return True, "all_hours_allowed"

    # --- Telegram Commands ---
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = (
            "🤖 **VWAP BOT COMMANDS**\n\n"
            "/help - Show this message\n"
            "/status - System health & connections\n"
            "/dashboard - Live trading stats\n"
            "/top - Top performing combos\n"
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
            long_combos = sum(len(d.get('allowed_combos_long', [])) for d in self.vwap_combos.values())
            short_combos = sum(len(d.get('allowed_combos_short', [])) for d in self.vwap_combos.values())
            
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
            
            # Calculate overall EV at 1:1 R:R (net after fees)
            if learning_total > 0:
                wr_decimal = total_wins / learning_total
                ev = (wr_decimal * 1.0) - ((1 - wr_decimal) * 1.0)  # 1:1 R:R
            else:
                ev = 0
            
            unique_combos = len(learning.get_all_combos())
            promoted = len(learning.promoted)
            blacklisted = len(learning.blacklist)
            pending = len(learning.pending_signals)
            
            # Count combos approaching thresholds (use actual promotion criteria)
            all_combos = learning.get_all_combos()
            PROMOTE_TRADES = getattr(learning, 'PROMOTE_MIN_TRADES', 20)
            PROMOTE_WR = getattr(learning, 'PROMOTE_MIN_LOWER_WR', 45.0)
            
            # Near promotion: combos with at least 5 trades, WR >= 35%, progressing toward threshold
            # Filter out already promoted combos
            near_promote_candidates = []
            for c in all_combos:
                key = f"{c['symbol']}:{c['side']}:{c['combo']}"
                if key in learning.promoted:
                    continue  # Already promoted
                if c['total'] >= 5 and c['lower_wr'] >= 35:
                    # Calculate progress: N progress + WR progress
                    n_progress = min(100, (c['total'] / PROMOTE_TRADES) * 100)
                    wr_progress = min(100, (c['lower_wr'] / PROMOTE_WR) * 100) if PROMOTE_WR > 0 else 100
                    overall_progress = (n_progress + wr_progress) / 2
                    c['n_progress'] = n_progress
                    c['wr_progress'] = wr_progress
                    c['overall_progress'] = overall_progress
                    near_promote_candidates.append(c)
            
            # Sort by overall progress descending
            near_promote_candidates.sort(key=lambda x: x['overall_progress'], reverse=True)
            approaching_promote = len(near_promote_candidates)
            
            # Near blacklist: combos with at least 5 trades and WR <= 35%
            approaching_blacklist = len([c for c in all_combos if c['total'] >= 5 and c['lower_wr'] <= 30 
                                          and f"{c['symbol']}:{c['side']}:{c['combo']}" not in learning.blacklist])
            
            # === COMBO MATURITY DISTRIBUTION ===
            # New: N < 5, Growing: 5-15, Mature: 15-20, Ready: 20+
            maturity_new = len([c for c in all_combos if c['total'] < 5])
            maturity_growing = len([c for c in all_combos if 5 <= c['total'] < 15])
            maturity_mature = len([c for c in all_combos if 15 <= c['total'] < PROMOTE_TRADES])
            maturity_ready = len([c for c in all_combos if c['total'] >= PROMOTE_TRADES])
            
            # === PROMOTION FORECAST ===
            # Calculate uptime and signal rate
            uptime_hours = max(0.1, uptime_hrs)  # Avoid division by zero
            total_signals_rate = learning.total_signals / uptime_hours if uptime_hours > 0 else 0
            
            # Find top promotion candidates (need both N progress and good WR)
            # Only consider combos with reasonable WR (at least 30% LB WR)
            promotion_candidates = []
            for c in all_combos:
                key = f"{c['symbol']}:{c['side']}:{c['combo']}"
                if key in learning.promoted or key in learning.blacklist:
                    continue
                # ONLY consider combos with minimum 30% LB WR (realistic promotion path)
                if c['total'] >= 5 and c['lower_wr'] >= 30:
                    trades_needed = max(0, PROMOTE_TRADES - c['total'])
                    wr_margin = c['lower_wr'] - PROMOTE_WR  # Positive if above threshold
                    
                    # Probability estimation based on current WR
                    # If WR is already above threshold, high prob; if below, lower
                    if c['lower_wr'] >= PROMOTE_WR:
                        prob = 90 if trades_needed == 0 else min(85, 70 + (c['lower_wr'] - PROMOTE_WR))
                    elif c['lower_wr'] >= PROMOTE_WR - 10:
                        prob = max(30, 50 + wr_margin * 2)  # Close to threshold
                    else:
                        prob = max(15, 25 + wr_margin)  # Far from threshold
                    
                    # ETA calculation: assume ~2 signals per combo per hour on average
                    signals_per_combo_per_hour = max(0.5, total_signals_rate / max(1, len(all_combos)))
                    eta_hours = trades_needed / signals_per_combo_per_hour if signals_per_combo_per_hour > 0 else 999
                    
                    # Score: PRIORITIZE WR (70%) over N count (30%)
                    # This ensures high-WR combos rank above high-N/low-WR combos
                    wr_score = min(100, c['lower_wr'] * 1.8)  # Scale WR to 0-100 (55% -> 99)
                    n_score = min(100, (c['total'] / PROMOTE_TRADES) * 100)  # N progress 0-100
                    combined_score = wr_score * 0.7 + n_score * 0.3
                    
                    promotion_candidates.append({
                        'symbol': c['symbol'],
                        'side': c['side'],
                        'total': c['total'],
                        'lower_wr': c['lower_wr'],
                        'wins': c.get('wins', 0),
                        'trades_needed': trades_needed,
                        'prob': prob,
                        'eta_hours': eta_hours,
                        'score': combined_score
                    })
            
            # Sort by score descending (best candidates = high WR + high N)
            promotion_candidates.sort(key=lambda x: x['score'], reverse=True)
            top_candidates = promotion_candidates[:3]
            
            # Probability distribution
            prob_high = len([c for c in promotion_candidates if c['prob'] >= 70])
            prob_medium = len([c for c in promotion_candidates if 40 <= c['prob'] < 70])
            prob_low = len([c for c in promotion_candidates if c['prob'] < 40])
            
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
                f"├ EV: **{ev:+.2f}R** at 1:1 R:R\n"
                f"├ Combos: {unique_combos} learned\n"
                f"├ 📈 Near Promote: {approaching_promote}\n"
                f"├ 📉 Near Blacklist: {approaching_blacklist}\n"
                f"├ 🚀 Promoted: {promoted}\n"
                f"├ 🔽 Demoted: {getattr(self, 'demoted_count', 0)}\n"
                f"└ 🚫 Blacklisted: {blacklisted}\n\n"
                
                f"🌱 **COMBO MATURITY**\n"
                f"├ 🌱 New (N<5):      {maturity_new}\n"
                f"├ 🌿 Growing (5-15): {maturity_growing}\n"
                f"├ 🌳 Mature (15-20): {maturity_mature}\n"
                f"├ 🎯 Ready (N≥20):   {maturity_ready}\n"
                f"├ 🏆 Promoted:       {promoted}\n"
                f"└ 🚫 Blacklisted:    {blacklisted}\n\n"
                
                f"🔮 **PROMOTION FORECAST**\n"
                f"├ 📊 Signal Rate: {total_signals_rate:.0f}/hr\n"
                f"├ 🎲 High Prob (>70%): {prob_high}\n"
                f"├ 🎲 Med Prob (40-70%): {prob_medium}\n"
                f"└ 🎲 Low Prob (<40%): {prob_low}\n"
            )
            
            # Add top candidates if any
            if top_candidates:
                msg += "\n🎯 **TOP CANDIDATES**\n"
                for i, cand in enumerate(top_candidates, 1):
                    side_icon = "🟢" if cand['side'] == 'long' else "🔴"
                    sym = cand['symbol'][:8]
                    eta_str = f"{cand['eta_hours']:.1f}h" if cand['eta_hours'] < 100 else "∞"
                    msg += f"├ {i}. {side_icon} `{sym}` N:{cand['total']}/{PROMOTE_TRADES} LB:{cand['lower_wr']:.0f}%\n"
                    msg += f"│   ETA: {eta_str} | Prob: {cand['prob']:.0f}%\n"
            
            msg += "\n"
                
            msg += (
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
            
            # Add NEAR PROMOTION section - show top candidates approaching threshold
            if near_promote_candidates:
                msg += f"\n🚀 **NEAR PROMOTION** (top 5 of {len(near_promote_candidates)})\n"
                msg += f"├ _Requires: N≥{PROMOTE_TRADES}, LB WR≥{PROMOTE_WR:.0f}%_\n"
                for c in near_promote_candidates[:5]:
                    side_icon = "🟢" if c['side'] == 'long' else "🔴"
                    # Progress bar: ▓▓▓▓▓▒▒▒▒▒ (5 filled out of 10)
                    n_bars = int(c['n_progress'] / 10)
                    wr_bars = int(c['wr_progress'] / 10)
                    n_bar_str = "▓" * n_bars + "▒" * (10 - n_bars)
                    wr_bar_str = "▓" * wr_bars + "▒" * (10 - wr_bars)
                    msg += f"├ {side_icon} `{c['symbol'][:10]}`\n"
                    msg += f"│  N: {n_bar_str} {c['total']}/{PROMOTE_TRADES}\n"
                    msg += f"│  WR: {wr_bar_str} {c['lower_wr']:.0f}%/{PROMOTE_WR:.0f}%\n"
            
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
            
            # Get optional pattern count from args (default 3)
            pattern_count = 3
            if context.args and context.args[0].isdigit():
                pattern_count = min(int(context.args[0]), 20)  # Max 20

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
            
            # Top Patterns (now configurable)
            patterns = find_winning_patterns(trades, min_trades=3)[:pattern_count]
            if patterns:
                msg += f"🏆 **TOP PATTERNS** (Top {len(patterns)})\n"
                for p in patterns:
                    combo_short = p['combo'][:15] + '..' if len(p['combo']) > 17 else p['combo']
                    msg += f"├ {p['symbol']} {p['side'][0].upper()} {combo_short}\n"
                    msg += f"│  WR:{p['wr']:.0f}% (N={p['total']})\n"
                
                # Add usage hint if showing default count
                if pattern_count == 3:
                    msg += f"\n💡 Use `/analytics 10` for more patterns"
            
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

    async def cmd_top(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show top performing combos"""
        try:
            # Get optional limit from args (default 10)
            limit = 10
            if context.args and context.args[0].isdigit():
                limit = min(int(context.args[0]), 20)  # Max 20
            
            # Get top combos from learner
            top_combos = self.learner.get_top_combos(min_trades=3, min_lower_wr=35)[:limit]
            
            if not top_combos:
                await update.message.reply_text(
                    "🏆 **NO TOP PERFORMERS YET**\n\n"
                    "Need combos with:\n"
                    "• N ≥ 3 trades\n"
                    "• Lower Bound WR ≥ 35%\n\n"
                    "Keep running to collect more data!",
                    parse_mode='Markdown'
                )
                return
            
            msg = f"🏆 **TOP PERFORMERS** (Top {len(top_combos)})\n"
            msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
            
            for c in top_combos:
                # Side icon
                side_icon = "🟢" if c['side'] == 'long' else "🔴"
                
                # Find best session
                sessions = c.get('sessions', {})
                best_session = '🌐'
                best_session_wr = 0
                session_icons = {'asian': '🌏', 'london': '🌍', 'newyork': '🌎'}
                
                for s, data in sessions.items():
                    total = data.get('w', 0) + data.get('l', 0)
                    if total >= 2:
                        wr = data['w'] / total * 100
                        if wr > best_session_wr:
                            best_session_wr = wr
                            best_session = session_icons.get(s, '🌐')
                
                # Truncate combo for display
                combo_short = c['combo'][:20] + '..' if len(c['combo']) > 22 else c['combo']
                
                # EV string
                ev_str = f"{c['ev']:+.2f}R" if c['ev'] != 0 else "0R"
                
                msg += f"├ {side_icon} **{c['symbol']}**\n"
                msg += f"│  `{combo_short}`\n"
                msg += f"│  WR:{c['lower_wr']:.0f}% | EV:{ev_str} | {c['optimal_rr']}:1 | {best_session} (N={c['total']})\n"
                msg += f"│\n"
            
            # Summary
            total_trades = sum(c['total'] for c in top_combos)
            avg_wr = sum(c['lower_wr'] for c in top_combos) / len(top_combos) if top_combos else 0
            
            msg += f"━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"📊 Avg LB WR: {avg_wr:.0f}% | Total N: {total_trades}\n"
            msg += f"💡 Use `/top 20` for more results"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"cmd_top error: {e}")

    async def cmd_ladder(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show combo ladder/leaderboard - all combos ranked by progress"""
        try:
            # Get optional page number from args (default 1)
            page = 1
            if context.args and context.args[0].isdigit():
                page = max(1, int(context.args[0]))
            
            per_page = 10
            
            # Get promotion thresholds
            PROMOTE_TRADES = getattr(self.learner, 'PROMOTE_MIN_TRADES', 20)
            PROMOTE_WR = getattr(self.learner, 'PROMOTE_MIN_LOWER_WR', 45.0)
            
            # Get all combos
            all_combos = self.learner.get_all_combos()
            
            if not all_combos:
                await update.message.reply_text(
                    "📊 **NO COMBOS TRACKED YET**\n\n"
                    "Run the bot to detect signals and build stats!",
                    parse_mode='Markdown'
                )
                return
            
            # Calculate progress for each combo
            for c in all_combos:
                key = f"{c['symbol']}:{c['side']}:{c['combo']}"
                c['is_promoted'] = key in self.learner.promoted
                c['is_blacklisted'] = key in self.learner.blacklist
                
                # Progress scores
                n_progress = min(100, (c['total'] / PROMOTE_TRADES) * 100)
                wr_progress = min(100, (c['lower_wr'] / PROMOTE_WR) * 100) if PROMOTE_WR > 0 else 100
                c['n_progress'] = n_progress
                c['wr_progress'] = wr_progress
                c['overall_progress'] = (n_progress + wr_progress) / 2
            
            # Sort by: promoted first, then by overall progress
            all_combos.sort(key=lambda x: (x['is_promoted'], x['overall_progress']), reverse=True)
            
            # Pagination
            total_pages = (len(all_combos) + per_page - 1) // per_page
            page = min(page, total_pages)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            page_combos = all_combos[start_idx:end_idx]
            
            # Build message
            msg = f"📊 **COMBO LADDER** (Page {page}/{total_pages})\n"
            msg += f"━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"📏 Thresholds: N≥{PROMOTE_TRADES}, LB WR≥{PROMOTE_WR:.0f}%\n\n"
            
            for i, c in enumerate(page_combos, start=start_idx + 1):
                # Status icon
                if c['is_promoted']:
                    status = "🏆"  # Promoted
                elif c['is_blacklisted']:
                    status = "🚫"  # Blacklisted
                elif c['overall_progress'] >= 90:
                    status = "🔥"  # Almost there
                elif c['overall_progress'] >= 70:
                    status = "📈"  # Good progress
                elif c['overall_progress'] >= 50:
                    status = "📊"  # Moderate
                else:
                    status = "📉"  # Low
                
                side_icon = "🟢" if c['side'] == 'long' else "🔴"
                
                # Mini progress bar (5 chars)
                bars_filled = int(c['overall_progress'] / 20)
                bar = "█" * bars_filled + "░" * (5 - bars_filled)
                
                # Truncate symbol for display
                sym = c['symbol'][:8]
                
                # Calculate raw WR and EV
                raw_wr = (c.get('wins', 0) / c['total'] * 100) if c['total'] > 0 else 0
                wr_decimal = raw_wr / 100
                ev = (wr_decimal * 1.0) - ((1 - wr_decimal) * 1.0)  # EV at 1:1 R:R
                ev_str = f"{ev:+.2f}R"
                
                msg += f"{i}. {status} {side_icon} `{sym}`\n"
                msg += f"   [{bar}] {c['overall_progress']:.0f}% | N:{c['total']}/{PROMOTE_TRADES} | WR:{raw_wr:.0f}% (LB:{c['lower_wr']:.0f}%) | EV:{ev_str}\n"
            
            # Summary
            promoted_count = sum(1 for c in all_combos if c['is_promoted'])
            near_count = sum(1 for c in all_combos if not c['is_promoted'] and c['overall_progress'] >= 70)
            
            msg += f"\n━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"🏆 Promoted: {promoted_count} | 📈 Near: {near_count} | 📚 Total: {len(all_combos)}\n"
            
            if total_pages > 1:
                msg += f"💡 Use `/ladder {page + 1}` for next page"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
            logger.error(f"cmd_ladder error: {e}")

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
        """
        SIMPLIFIED combo: 18 combinations (3 RSI x 2 MACD x 3 Fib)
        Updated to match backtest_simplified_2to1.py results.
        
        Original 70-combo version backed up - can revert by restoring
        backtest_golden_combos_BACKUP_20251210.yaml
        """
        # RSI: 3 levels (simplified)
        rsi = row.rsi
        if rsi < 40: 
            r_bin = 'oversold'
        elif rsi > 60: 
            r_bin = 'overbought'
        else: 
            r_bin = 'neutral'
        
        # MACD: 2 levels
        m_bin = 'bull' if row.macd > row.macd_signal else 'bear'
        
        # Fib: 3 levels (simplified)
        high, low, close = row.roll_high, row.roll_low, row.close
        if high == low: 
            f_bin = 'low'
        else:
            fib = (high - close) / (high - low) * 100
            if fib < 38: 
                f_bin = 'low'
            elif fib < 62: 
                f_bin = 'mid'
            else: 
                f_bin = 'high'
        
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
                
                # Check for "Heartbeat" / Proof of Life logging
                # This logs ANY cross, even if not matched, so user knows bot is scanning
                yaml_key = f"allowed_combos_{side}"
                allowed = self.vwap_combos.get(sym, {}).get(yaml_key, [])
                is_allowed = combo in allowed

                if not is_allowed:
                     # Log mismatch at INFO level for "Proof of Life"
                     # Rate limit this slightly if needed, but for now user wants VISIBILITY
                     logger.info(f"👀 SCAN: {sym} {side} {combo} (Not in Golden Combos)")

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
                # YAML uses: allowed_combos_long / allowed_combos_short
                yaml_key = f"allowed_combos_{side}"
                allowed = self.vwap_combos.get(sym, {}).get(yaml_key, [])
                
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
                
                # Use smart R:R if available, otherwise default 1:1 R:R
                # TP has 5% buffer (1.05x) to account for:
                # - Bybit taker fees: 0.055% x 2 = 0.11% round-trip
                # - Slippage on market TP: ~0.05-0.1%
                # This ensures NET 1:1 R:R after all costs
                if smart_tp and smart_sl:
                    tp, sl = smart_tp, smart_sl
                else:
                    # 1:1 R:R with fee buffer: SL = 1 ATR, TP = 1.05 ATR (net ~1R after fees)
                    if side == 'long':
                        sl = entry - (1.0 * atr)
                        tp = entry + (1.05 * atr)  # 5% buffer for fees/slippage
                    else:
                        sl = entry + (1.0 * atr)
                        tp = entry - (1.05 * atr)  # 5% buffer for fees/slippage
                
                # Check if COMBO is in learner.promoted (auto-promoted from live stats)
                combo_key = f"{sym}:{side}:{combo}"
                is_auto_promoted = combo_key in self.learner.promoted
                
                if is_auto_promoted:
                    # AUTO-PROMOTED COMBO - Execute!
                    # This combo was promoted based on live learning stats (>40% LB WR)
                    logger.info(f"🚀 AUTO-PROMOTED: {sym} {side} {combo}")
                    await self.execute_trade(sym, side, last_candle, combo)
                else:
                    # Phantom signal - learner already tracks it (recorded above)
                    logger.info(f"👻 SIGNAL DETECTED: {sym} {side} {combo} (Phantom)")
                    
                    # NEAR-MISS NOTIFICATION: Only for symbols in Golden Combos file
                    # This helps user see when their tracked symbols are active but combo doesn't match
                    if sym in self.vwap_combos:
                        # Get allowed combos for this symbol
                        allowed_long = self.vwap_combos[sym].get('allowed_combos_long', [])
                        allowed_short = self.vwap_combos[sym].get('allowed_combos_short', [])
                        
                        # Format allowed combos for display
                        long_list = "\n".join([f"  • `{c}`" for c in allowed_long]) if allowed_long else "  • None"
                        short_list = "\n".join([f"  • `{c}`" for c in allowed_short]) if allowed_short else "  • None"
                        
                        await self.send_telegram(
                            f"🔍 **NEAR MISS** (Golden Combo Symbol)\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📊 Symbol: `{sym}`\n"
                            f"📈 Side: **{side.upper()}**\n"
                            f"🎯 Detected: `{combo}`\n\n"
                            f"✅ **Allowed Long Combos:**\n{long_list}\n\n"
                            f"✅ **Allowed Short Combos:**\n{short_list}\n\n"
                            f"⏳ Waiting for exact match..."
                        )
                    
        except Exception as e:
            logger.error(f"Error {sym}: {e}")

    async def _immediate_demote(self, symbol: str, side: str, combo: str, lb_wr: float, total: int):
        """Immediately demote a combo from YAML after a trade loss drops WR below threshold.
        
        This ensures poor-performing combos are removed right away, not waiting for periodic check.
        """
        try:
            import yaml
            yaml_file = 'symbol_overrides_VWAP_Combo.yaml'
            
            with open(yaml_file, 'r') as f:
                current_yaml = yaml.safe_load(f) or {}
            
            # Check if combo exists in YAML
            if symbol in current_yaml and isinstance(current_yaml[symbol], dict):
                if side in current_yaml[symbol] and isinstance(current_yaml[symbol][side], list):
                    if combo in current_yaml[symbol][side]:
                        # Remove the combo
                        current_yaml[symbol][side].remove(combo)
                        
                        # Track demotion
                        if not hasattr(self, 'demoted_count'):
                            self.demoted_count = 0
                        self.demoted_count += 1
                        
                        # Add to blacklist for this session
                        self.learner.blacklist.add(f"{symbol}:{side}:{combo}")
                        self.learner.save_blacklist()
                        
                        # Clean up empty entries
                        if not current_yaml[symbol]['long'] and not current_yaml[symbol]['short']:
                            del current_yaml[symbol]
                        elif not current_yaml[symbol][side]:
                            del current_yaml[symbol][side]
                        
                        # Save YAML
                        with open(yaml_file, 'w') as f:
                            yaml.dump(current_yaml, f, default_flow_style=False)
                        
                        logger.info(f"🔽 IMMEDIATE DEMOTE: {symbol} {side} {combo} (LB WR: {lb_wr:.0f}%, N={total})")
                        
                        await self.send_telegram(
                            f"🔽 **COMBO DEMOTED**\n"
                            f"Symbol: `{symbol}` {side.upper()}\n"
                            f"Combo: `{combo}`\n"
                            f"Reason: LB WR dropped to {lb_wr:.0f}% (below 40%)\n"
                            f"Trades: {total}"
                        )
        except FileNotFoundError:
            logger.debug("YAML file not found for demotion check")
        except Exception as e:
            logger.error(f"Immediate demotion error: {e}")

    async def monitor_pending_limit_orders(self, candle_data: dict):
        """
        Monitor pending limit orders for:
        1. Fills (then set TP/SL and move to active_trades)
        2. Invalidation (SL/TP breached before fill → cancel)
        3. Timeout (5 minutes → cancel)
        4. Partial fills (cancel remainder, protect filled portion)
        """
        if not self.pending_limit_orders:
            return
        
        TIMEOUT_SECONDS = 300  # 5 minutes
        
        for sym in list(self.pending_limit_orders.keys()):
            try:
                order_info = self.pending_limit_orders[sym]
                order_id = order_info['order_id']
                side = order_info['side']
                tp = order_info['tp']
                sl = order_info['sl']
                entry_price = order_info['entry_price']
                created_at = order_info['created_at']
                
                # Get current price from candle data
                current_price = candle_data.get(sym, {}).get('close', 0)
                if current_price <= 0:
                    continue
                
                # Get order status from Bybit
                status = self.broker.get_order_status(sym, order_id)
                
                if not status:
                    # Order not found - might have been cancelled/filled externally
                    logger.warning(f"Order {order_id[:16]} for {sym} not found on Bybit, removing from tracking")
                    
                    # Record to analytics
                    try:
                        self.learner.resolve_executed_trade(
                            sym, side, 'no_fill',
                            exit_price=current_price,
                            max_high=0,
                            min_low=0,
                            combo=order_info.get('combo', 'UNKNOWN')
                        )
                        logger.info(f"📊 Recorded order not found: {sym} {side} → no_fill")
                    except Exception as e:
                        logger.warning(f"Could not record to analytics: {e}")
                    
                    # Send notification
                    await self.send_telegram(
                        f"⚠️ **ORDER NOT FOUND ON BYBIT**\n"
                        f"Symbol: `{sym}` {side.upper()}\n"
                        f"Entry: ${entry_price:.4f}\n"
                        f"Order ID: `{order_id[:16]}...`\n\n"
                        f"Order may have expired or been cancelled.\n"
                        f"⏱️ **Recorded as**: NO_FILL"
                    )
                    
                    del self.pending_limit_orders[sym]
                    continue
                
                order_status = status.get('orderStatus', '')
                filled_qty = float(status.get('cumExecQty', 0) or 0)
                avg_price = float(status.get('avgPrice', entry_price) or entry_price)
                
                logger.debug(f"📊 {sym} order status: {order_status}, filled: {filled_qty}")
                
                # CASE 1: Order fully filled
                if order_status == 'Filled':
                    logger.info(f"✅ BRACKET ORDER FILLED: {sym} {side} @ {avg_price}")
                    
                    # TP/SL already set via bracket order - no need to call set_tpsl()
                    # Just log confirmation
                    logger.info(f"🛡️ TP/SL already active (bracket order): TP={tp:.6f} SL={sl:.6f}")
                    
                    # Move to active_trades
                    self.active_trades[sym] = {
                        'side': side,
                        'combo': order_info['combo'],
                        'entry': avg_price,
                        'order_id': order_id,
                        'qty': filled_qty,
                        'tp': tp,
                        'sl': sl,
                        'open_time': created_at,
                        'is_auto_promoted': order_info.get('is_auto_promoted', False)
                    }
                    
                    self.trades_executed += 1
                    del self.pending_limit_orders[sym]
                    
                    # Calculate values for notification
                    sl_pct = abs(avg_price - sl) / avg_price * 100
                    tp_pct = abs(tp - avg_price) / avg_price * 100
                    position_value = filled_qty * avg_price
                    
                    # Notify user with step-by-step status
                    source = "🚀 Auto-Promoted" if order_info.get('is_auto_promoted') else "📊 Backtest"
                    await self.send_telegram(
                        f"✅ **BRACKET ORDER FILLED**\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📊 Symbol: `{sym}`\n"
                        f"📈 Side: **{side.upper()}**\n"
                        f"🎯 Combo: `{order_info['combo']}`\n"
                        f"📁 Source: **{source}**\n\n"
                        f"📋 **COMPLETION STEPS**\n"
                        f"├ ✅ Order filled @ ${avg_price:.4f}\n"
                        f"├ ✅ TP/SL already active (bracket)\n"
                        f"└ ✅ Position tracking started\n\n"
                        f"💰 **POSITION DETAILS**\n"
                        f"├ Quantity: {filled_qty}\n"
                        f"├ Fill Price: ${avg_price:.4f}\n"
                        f"└ Position Value: ${position_value:.2f}\n\n"
                        f"🛡️ **TP/SL PROTECTION**\n"
                        f"├ Take Profit: ${tp:.4f} (+{tp_pct:.2f}%)\n"
                        f"├ Stop Loss: ${sl:.4f} (-{sl_pct:.2f}%)\n"
                        f"└ R:R: **{order_info['optimal_rr']}:1**"
                    )
                    continue
                
                # CASE 2: Order cancelled/rejected externally (by Bybit)
                if order_status in ['Cancelled', 'Rejected', 'Expired', 'Deactivated']:
                    logger.info(f"Order for {sym} was {order_status} by Bybit")
                    
                    # Record to analytics as 'no_fill' (we don't know why Bybit cancelled)
                    combo = order_info.get('combo', 'EXTERNAL_CANCEL')
                    try:
                        self.learner.resolve_executed_trade(
                            sym, side, 'no_fill',
                            exit_price=current_price,
                            max_high=0,
                            min_low=0,
                            combo=combo
                        )
                        logger.info(f"📊 Recorded externally cancelled order: {sym} {side} → no_fill")
                    except Exception as e:
                        logger.warning(f"Could not record cancelled order to analytics: {e}")
                    
                    # Send notification
                    await self.send_telegram(
                        f"⚠️ **ORDER CANCELLED BY BYBIT**\n"
                        f"Symbol: `{sym}` {side.upper()}\n"
                        f"Entry: ${order_info['entry_price']:.4f}\n"
                        f"Status: {order_status}\n\n"
                        f"⏱️ **Recorded as**: NO_FILL\n"
                        f"(Signal data saved for analytics)"
                    )
                    
                    del self.pending_limit_orders[sym]
                    continue
                
                # CASE 3: Check for invalidation (still pending or partially filled)
                should_cancel = False
                reason = ""
                
                if side == 'long':
                    if current_price <= sl:
                        should_cancel = True
                        reason = f"SL breached ({current_price:.4f} ≤ {sl:.4f})"
                    elif current_price >= tp:
                        should_cancel = True
                        reason = f"TP breached ({current_price:.4f} ≥ {tp:.4f}) - missed entry"
                else:  # short
                    if current_price >= sl:
                        should_cancel = True
                        reason = f"SL breached ({current_price:.4f} ≥ {sl:.4f})"
                    elif current_price <= tp:
                        should_cancel = True
                        reason = f"TP breached ({current_price:.4f} ≤ {tp:.4f}) - missed entry"
                
                # CASE 4: Timeout check
                age = time.time() - created_at
                if age > TIMEOUT_SECONDS:
                    should_cancel = True
                    reason = f"Timeout ({age/60:.1f} min)"
                
                if should_cancel:
                    logger.info(f"❌ Cancelling {sym} order: {reason}")
                    self.broker.cancel_order(sym, order_id)
                    
                    # Determine theoretical outcome for analytics
                    # SL breach = would have been a loss
                    # TP breach = would have been a win (missed opportunity)
                    # Timeout = uncertain, record as 'no_fill'
                    if 'SL breached' in reason:
                        theoretical_outcome = 'loss'
                    elif 'TP breached' in reason:
                        theoretical_outcome = 'win'  # Signal was correct, just didn't fill
                    else:
                        theoretical_outcome = 'no_fill'  # Timeout or other
                    
                    # Record to analytics (preserves signal data for learning)
                    combo = order_info.get('combo', 'LIMIT_ORDER_CANCELLED')
                    try:
                        self.learner.resolve_executed_trade(
                            sym, side, theoretical_outcome,
                            exit_price=current_price,
                            max_high=current_price if side == 'short' else 0,
                            min_low=current_price if side == 'long' else 0,
                            combo=combo
                        )
                        logger.info(f"📊 Recorded cancelled order outcome: {sym} {side} → {theoretical_outcome}")
                    except Exception as e:
                        logger.warning(f"Could not record cancelled order to analytics: {e}")
                    
                    # Handle partial fills - protect filled portion
                    if filled_qty > 0 and order_status == 'PartiallyFilled':
                        logger.info(f"Partial fill {filled_qty} on {sym}, setting TP/SL for filled portion")
                        self.broker.set_tpsl(sym, tp, sl, filled_qty)
                        
                        # Track as active position
                        self.active_trades[sym] = {
                            'side': side,
                            'combo': order_info['combo'],
                            'entry': avg_price,
                            'order_id': order_id,
                            'qty': filled_qty,
                            'tp': tp,
                            'sl': sl,
                            'open_time': created_at,
                            'is_auto_promoted': order_info.get('is_auto_promoted', False)
                        }
                        
                        self.trades_executed += 1
                        
                        await self.send_telegram(
                            f"⚠️ **PARTIAL FILL - REMAINDER CANCELLED**\n"
                            f"Symbol: `{sym}` {side.upper()}\n"
                            f"Filled: {filled_qty} @ ${avg_price:.4f}\n"
                            f"Reason: {reason}\n"
                            f"TP/SL set for filled portion"
                        )
                    else:
                        # Show outcome in notification
                        outcome_emoji = "📈" if theoretical_outcome == 'win' else "📉" if theoretical_outcome == 'loss' else "⏱️"
                        await self.send_telegram(
                            f"❌ **ORDER CANCELLED**\n"
                            f"Symbol: `{sym}` {side.upper()}\n"
                            f"Entry: ${entry_price:.4f}\n"
                            f"Reason: {reason}\n\n"
                            f"{outcome_emoji} **Recorded as**: {theoretical_outcome.upper()}\n"
                            f"(Signal data saved for analytics)"
                        )
                    
                    del self.pending_limit_orders[sym]
                    continue
                
                # Still pending - log status periodically
                age_mins = age / 60
                logger.debug(f"⏳ {sym} pending: {order_status}, price={current_price:.4f}, age={age_mins:.1f}m")
                
            except Exception as e:
                logger.error(f"Error monitoring {sym} order: {e}")
                import traceback
                logger.error(traceback.format_exc())

    # NOTE: update_phantoms() removed - phantom tracking now handled by learner.update_signals()


    async def execute_trade(self, sym, side, row, combo):
        """Execute trade using LIMIT ORDER (not market) for precise entry."""
        try:
            # Check if already in position or have pending order
            pos = self.broker.get_position(sym)
            if pos and float(pos.get('size', 0)) > 0:
                logger.info(f"Skip {sym}: Already in position")
                return
            
            if sym in self.pending_limit_orders:
                logger.info(f"Skip {sym}: Already have pending limit order")
                return
            
            if sym in self.active_trades:
                logger.info(f"Skip {sym}: Already tracking active trade")
                return

            balance = self.broker.get_balance() or 0
            if balance <= 0:
                logger.error("Balance is 0")
                return
            
            # Feature filter: Skip during unfavorable market conditions
            feature_ok, feature_reason = self._check_feature_filters()
            if not feature_ok:
                logger.debug(f"Skip {sym}: Feature filter blocked ({feature_reason})")
                return
                
            risk_val = self.risk_config['value']
            risk_type = self.risk_config['type']
            
            risk_amt = balance * (risk_val / 100) if risk_type == 'percent' else risk_val
                
            atr = row.atr
            entry = row.close
            
            # 1:1 R:R with fee/slippage buffer
            # TP has 5% buffer to account for:
            # - Bybit taker fees: 0.055% x 2 = 0.11% round-trip
            # - Slippage on market TP: ~0.05-0.1%
            # This ensures NET 1:1 R:R after all costs
            optimal_rr = 1.05  # Gross 1.05:1 = Net ~1:1 after fees
            
            # Calculate TP/SL with 1:1 R:R (+ fee buffer)
            # Using 1 ATR for SL (1R), 1.05*ATR for TP (net ~1R after fees)
            MIN_SL_PCT = 0.5  # Minimum 0.5% distance for SL
            MIN_TP_PCT = 0.5  # Minimum 0.5% distance for TP (same as SL for 1:1)
            
            # Calculate minimum distances based on percentage
            min_sl_dist = entry * (MIN_SL_PCT / 100)
            min_tp_dist = entry * (MIN_TP_PCT / 100)
            
            # Use the LARGER of ATR-based or minimum distance (force minimum)
            sl_dist = max(1.0 * atr, min_sl_dist)
            tp_dist = max(optimal_rr * atr, min_tp_dist)
            
            if side == 'long':
                sl = entry - sl_dist
                tp = entry + tp_dist
                dist = sl_dist
            else:
                sl = entry + sl_dist
                tp = entry - tp_dist
                dist = sl_dist
                
            if dist <= 0: return
            
            # Log if we used minimum distance instead of ATR
            atr_sl_pct = (atr / entry) * 100
            if atr_sl_pct < MIN_SL_PCT:
                logger.info(f"📐 {sym}: Using minimum SL distance (ATR={atr_sl_pct:.2f}% < {MIN_SL_PCT}%)")

            
            qty = risk_amt / dist
            
            # Round based on price magnitude
            if entry > 1000: qty = round(qty, 3)
            elif entry > 10: qty = round(qty, 2)
            elif entry > 1: qty = round(qty, 1)
            else: qty = round(qty, 0)
            
            if qty <= 0: return

            logger.info(f"EXECUTE: {sym} {side} qty={qty} R:R={optimal_rr}:1 (LIMIT ORDER)")
            
            # Set leverage to maximum allowed for this symbol (reduces margin requirement)
            max_lev = self.broker.get_max_leverage(sym)
            lev_res = self.broker.set_leverage(sym, max_lev)
            if lev_res:
                logger.info(f"✅ Leverage set to MAX ({max_lev}x) for {sym}")
            else:
                logger.warning(f"⚠️ Could not set leverage for {sym}, proceeding anyway")
                max_lev = 10  # Fallback display value
            
            # Log the order details we're placing
            logger.info(f"📍 BRACKET ORDER: {sym} Entry={entry:.6f} TP={tp:.6f} SL={sl:.6f} ATR={atr:.6f}")
            
            # Place BRACKET LIMIT order with TP/SL included
            # TP/SL are set atomically - position protected from instant it opens
            res = self.broker.place_limit(
                sym, side, qty, entry,
                take_profit=tp,
                stop_loss=sl,
                post_only=False
            )
            
            if res and res.get('retCode') == 0:
                # DEBUG: Log full response to diagnose missing orders
                logger.info(f"📋 LIMIT ORDER RESPONSE: {res}")
                
                # Check if order was immediately cancelled (PostOnly crossed spread)
                if res.get('_immediately_cancelled'):
                    cancel_reason = res.get('_cancel_reason', 'Unknown')
                    await self.send_telegram(
                        f"⚠️ **LIMIT ORDER INSTANTLY CANCELLED**\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📊 Symbol: `{sym}`\n"
                        f"📈 Side: **{side.upper()}**\n"
                        f"🎯 Combo: `{combo}`\n\n"
                        f"❌ **Reason**: {cancel_reason}\n"
                        f"💡 PostOnly order was rejected because price\n"
                        f"   already crossed the limit price level.\n\n"
                        f"Entry was: ${entry:.4f}"
                    )
                    logger.warning(f"Order immediately cancelled for {sym} - PostOnly crossed spread")
                    return
                
                # Extract order details from response
                result = res.get('result', {})
                order_id = result.get('orderId', 'N/A')
                
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
                
                # Calculate values for notification
                sl_pct = abs(entry - sl) / entry * 100
                tp_pct = abs(tp - entry) / entry * 100
                position_value = qty * entry
                
                # Track as PENDING limit order (not active trade yet)
                self.pending_limit_orders[sym] = {
                    'order_id': order_id,
                    'side': side,
                    'combo': combo,
                    'entry_price': entry,
                    'tp': tp,
                    'sl': sl,
                    'qty': qty,
                    'atr': atr,
                    'optimal_rr': optimal_rr,
                    'created_at': time.time(),
                    'is_auto_promoted': is_auto_promoted,
                    'balance': balance,
                    'risk_amt': risk_amt
                }
                
                # Build step status for notification
                lev_status = "✅" if lev_res else "⚠️"
                order_status = "✅"  # Already confirmed success at this point
                tpsl_status = "✅"   # Bracket order - TP/SL set with order
                track_status = "✅"  # Just added to tracking
                
                # Send notification with step-by-step status
                await self.send_telegram(
                    f"⏳ **BRACKET ORDER PLACED**\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 Symbol: `{sym}`\n"
                    f"📈 Side: **{side.upper()}**\n"
                    f"🎯 Combo: `{combo}`\n"
                    f"📁 Source: **{source}**\n"
                    f"📈 {wr_info}\n\n"
                    f"📋 **EXECUTION STEPS**\n"
                    f"├ {lev_status} Leverage set to {max_lev}x (MAX)\n"
                    f"├ {order_status} Limit order placed\n"
                    f"├ {tpsl_status} TP/SL set with order\n"
                    f"└ {track_status} Order tracking started\n\n"
                    f"💰 **ORDER DETAILS**\n"
                    f"├ Order ID: `{order_id[:16]}...`\n"
                    f"├ Quantity: {qty}\n"
                    f"├ Limit Price: ${entry:.4f}\n"
                    f"├ Position Value: ${position_value:.2f}\n"
                    f"└ Risk: ${risk_amt:.2f}\n\n"
                    f"🛡️ **TP/SL PROTECTION** (Active on fill)\n"
                    f"├ Take Profit: ${tp:.4f} (+{tp_pct:.2f}%)\n"
                    f"├ Stop Loss: ${sl:.4f} (-{sl_pct:.2f}%)\n"
                    f"└ R:R Ratio: **{optimal_rr}:1**\n\n"
                    f"⏳ Monitoring for fill... (5m timeout)"
                )
                
                logger.info(f"✅ Limit order placed: {sym} {side} @ {entry} (ID: {order_id[:16]})")
                
            else:
                # Order failed - notify with details
                error_msg = res.get('retMsg', 'Unknown error') if res else 'No response'
                error_code = res.get('retCode', 'N/A') if res else 'N/A'
                
                # Check if PostOnly rejection (price crossed)
                if 'post only' in str(error_msg).lower() or 'price worse' in str(error_msg).lower():
                    logger.warning(f"PostOnly rejected for {sym}: price already crossed entry")
                    await self.send_telegram(
                        f"⚠️ **LIMIT ORDER REJECTED**\n"
                        f"Symbol: `{sym}` {side.upper()}\n"
                        f"Reason: Price already crossed entry level\n"
                        f"Entry was: ${entry:.4f}"
                    )
                else:
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
            import traceback
            logger.error(traceback.format_exc())
            # Notify about execution error
            await self.send_telegram(
                f"❌ **EXECUTION ERROR**\n"
                f"Symbol: `{sym}` {side.upper()}\n"
                f"Error: `{str(e)[:100]}`"
            )

    async def _startup_promote_demote_scan(self):
        """Show auto-promote system status on startup.
        
        We're in AUTO-PROMOTE ONLY mode - no static backtest combos.
        Only learner.promoted combos will be used for trading.
        """
        logger.info("🚀 Auto-Promote system initialized")
        
        # Get stats for startup message
        all_combos = self.learner.get_all_combos()
        PROMOTE_TRADES = getattr(self.learner, 'PROMOTE_MIN_TRADES', 20)
        PROMOTE_WR = getattr(self.learner, 'PROMOTE_MIN_LOWER_WR', 45.0)
        
        promoted_count = len(self.learner.promoted)
        blacklisted_count = len(self.learner.blacklist)
        near_promote = len([c for c in all_combos if c['total'] >= 5 and c['lower_wr'] >= 35
                           and f"{c['symbol']}:{c['side']}:{c['combo']}" not in self.learner.promoted])
        total_combos_tracked = len(all_combos)
        
        await self.send_telegram(
            f"🚀 **AUTO-PROMOTE MODE**\n"
            f"├ No static backtest combos\n"
            f"├ Live learning only\n"
            f"└ Source: `{self.learner.OVERRIDE_FILE}`\n\n"
            f"📊 **Current Status**\n"
            f"├ 🟢 Promoted: **{promoted_count}** (trading)\n"
            f"├ 📈 Near Promotion: **{near_promote}**\n"
            f"├ 🚫 Blacklisted: **{blacklisted_count}**\n"
            f"└ 📚 Total Tracked: **{total_combos_tracked}** combos\n\n"
            f"📏 **Thresholds**: N≥{PROMOTE_TRADES}, WR≥{PROMOTE_WR:.0f}%"
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
            self.tg_app.add_handler(CommandHandler("top", self.cmd_top))
            self.tg_app.add_handler(CommandHandler("ladder", self.cmd_ladder))
            
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
        
        # Run immediate promote/demote scan on startup
        await self._startup_promote_demote_scan()
        
        self.load_overrides()  # Reload after sync and startup scan
        trading_symbols = list(self.vwap_combos.keys())
        
        if not trading_symbols:
            await self.send_telegram("⚠️ **No trading symbols!**\nLearning will still run on all 400 symbols.")
            logger.warning("No trading symbols, learning only mode")
        
        # Check connections
        redis_ok = "🟢" if self.learner.redis_client else "🔴"
        pg_ok = "🟢" if self.learner.pg_conn else "🔴"

        # Get near-promotion stats for startup message
        all_combos = self.learner.get_all_combos()
        PROMOTE_TRADES = getattr(self.learner, 'PROMOTE_MIN_TRADES', 20)
        PROMOTE_WR = getattr(self.learner, 'PROMOTE_MIN_LOWER_WR', 45.0)
        near_promote = len([c for c in all_combos if c['total'] >= 5 and c['lower_wr'] >= 35
                           and f"{c['symbol']}:{c['side']}:{c['combo']}" not in self.learner.promoted])

        # Send success notification
        await self.send_telegram(
            f"✅ **VWAP Bot Online!**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Trading: **{len(trading_symbols)}** symbols\n"
            f"📚 Learning: **{len(self.all_symbols)}** symbols\n"
            f"⚙️ Risk: {self.risk_config['value']} {self.risk_config['type']}\n\n"
            f"🚀 **Auto-Promote**: N≥{PROMOTE_TRADES}, WR≥{PROMOTE_WR:.0f}%\n"
            f"├ 🟢 Promoted: **{len(self.learner.promoted)}**\n"
            f"├ 📈 Near Promotion: **{near_promote}**\n"
            f"└ 🚫 Blacklisted: **{len(self.learner.blacklist)}**\n\n"
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
                    
                    # Monitor pending limit orders (check for fills, invalidation, timeout)
                    await self.monitor_pending_limit_orders(candle_data)
                    
                    # Check for closed trades and send notifications
                    for sym in list(self.active_trades.keys()):
                        try:
                            pos = self.broker.get_position(sym)
                            has_position = pos and float(pos.get('size', 0)) > 0
                            
                            if not has_position:
                                # Trade closed - determine outcome
                                trade_info = self.active_trades.pop(sym)
                                
                                # Get price data
                                current_price = candle_data.get(sym, {}).get('close', 0)
                                candle_high = candle_data.get(sym, {}).get('high', 0)
                                candle_low = candle_data.get(sym, {}).get('low', 0)
                                entry = trade_info['entry']
                                side = trade_info['side']
                                combo = trade_info['combo']
                                tp = trade_info.get('tp', 0)
                                sl = trade_info.get('sl', 0)
                                
                                # Determine outcome based on which level was hit (TP or SL)
                                # Not candle close - actual exit is at TP or SL level
                                if entry and tp and sl:
                                    if side == 'long':
                                        # Long: Won if high reached TP, Lost if low reached SL
                                        # Check which was hit first (use candle extremes)
                                        hit_tp = candle_high >= tp
                                        hit_sl = candle_low <= sl
                                        
                                        if hit_tp and not hit_sl:
                                            outcome = "win"
                                            exit_price = tp
                                        elif hit_sl and not hit_tp:
                                            outcome = "loss"
                                            exit_price = sl
                                        elif hit_tp and hit_sl:
                                            # Both hit - use current price to guess
                                            outcome = "win" if current_price >= entry else "loss"
                                            exit_price = tp if outcome == "win" else sl
                                        else:
                                            # Neither? Use current price comparison
                                            outcome = "win" if current_price >= entry else "loss"
                                            exit_price = current_price
                                    else:
                                        # Short: Won if low reached TP, Lost if high reached SL
                                        hit_tp = candle_low <= tp
                                        hit_sl = candle_high >= sl
                                        
                                        if hit_tp and not hit_sl:
                                            outcome = "win"
                                            exit_price = tp
                                        elif hit_sl and not hit_tp:
                                            outcome = "loss"
                                            exit_price = sl
                                        elif hit_tp and hit_sl:
                                            outcome = "win" if current_price <= entry else "loss"
                                            exit_price = tp if outcome == "win" else sl
                                        else:
                                            outcome = "win" if current_price <= entry else "loss"
                                            exit_price = current_price
                                    
                                    # Calculate P/L percentage
                                    if side == 'long':
                                        pnl_pct = ((exit_price - entry) / entry) * 100
                                    else:
                                        pnl_pct = ((entry - exit_price) / entry) * 100
                                    
                                    if outcome == "win":
                                        outcome_display = "✅ WIN"
                                        self.wins += 1
                                    else:
                                        outcome_display = "❌ LOSS"
                                        self.losses += 1
                                    
                                    # *** CRITICAL FIX: Update learner analytics ***
                                    # This ensures combo stats are updated for executed trades
                                    resolved = self.learner.resolve_executed_trade(
                                        sym, side, outcome, 
                                        exit_price=exit_price,
                                        max_high=candle_high,
                                        min_low=candle_low,
                                        combo=combo
                                    )
                                    if not resolved:
                                        logger.warning(f"Could not resolve trade in learner: {sym} {side}")
                                    
                                    # Get updated WR/N from analytics (now includes this trade)
                                    updated_stats = self.learner.get_combo_stats(sym, side, combo)
                                    if updated_stats:
                                        wr_info = f"WR: {updated_stats['wr']:.0f}% (LB: {updated_stats['lower_wr']:.0f}%) | N={updated_stats['total']}"
                                    else:
                                        wr_info = "WR: Updating..."
                                    
                                    # Source - Explicitly trust Backtest Golden Combos
                                    source = "🟢 Backtest Golden Combo"
                                    if trade_info.get('is_auto_promoted'):
                                        source = "🚀 Auto-Promoted"
                                    
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
                                        f"├ Exit: ${exit_price:.4f}\n"
                                        f"└ Duration: {duration_mins:.0f}m\n\n"
                                        f"📊 **UPDATED ANALYTICS**\n"
                                        f"└ {wr_info}"
                                    )
                                    
                                    # IMMEDIATE DEMOTION CHECK after a loss
                                    if outcome == 'loss' and updated_stats:
                                        lb_wr = updated_stats.get('lower_wr', 100)
                                        if lb_wr < 40 and updated_stats.get('total', 0) >= 5:
                                            # Demote immediately - remove from YAML
                                            await self._immediate_demote(sym, side, combo, lb_wr, updated_stats['total'])
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
                    
                    # === AUTO-ACTIVATION DISABLED ===
                    # Now using static backtest_golden_combos.yaml
                    # Auto-promote/demote has been replaced with walk-forward validated combos
                    logger.debug("Auto-promote/demote DISABLED - using backtest golden combos")
                
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
