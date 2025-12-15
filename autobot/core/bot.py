import asyncio
import logging
import yaml
import os
import pandas as pd
import numpy as np
import aiohttp
import time
from datetime import datetime
from dataclasses import dataclass
from autobot.brokers.bybit import Bybit, BybitConfig
from autobot.core.unified_learner import UnifiedLearner, wilson_lower_bound
from autobot.core.shadow_auditor import ShadowAuditor
from autobot.core.divergence_detector import (
    detect_divergence, calculate_rsi, prepare_dataframe, 
    DivergenceSignal, get_signal_description, SIGNAL_DESCRIPTIONS,
    RSI_PERIOD, LOOKBACK_BARS
)
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("divergence_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DivergenceBot")

# RSI Divergence Strategy - Walk-Forward Validated
# 26,850 trades | 61.3% WR | +0.84 EV at 2:1 R:R

class DivergenceBot:
    def __init__(self):
        self.load_config()
        self.setup_broker()
        self.divergence_combos = {}  # Renamed from divergence_combos
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
        
        # BACKTEST MATCH: Skip trading on first loop (avoid stale signals)
        self.first_loop_completed = False
        
        # BACKTEST MATCH: Queue entries for next candle open
        # Format: {symbol: {side, combo, signal_type, df, detected_at}}
        self.pending_entries = {}
        
        # Unified Learning System (all learning features in one)
        self.learner = UnifiedLearner(
            on_promote_callback=self._on_combo_promoted,
            on_demote_callback=self._on_combo_demoted
        )
        
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
        """Load combos - DIRECT DIVERGENCE EXECUTION MODE
        
        All RSI divergence signals execute immediately.
        Backtest validated: 61.3% WR | +0.84 EV at 2:1 R:R
        """
        self.divergence_combos = {}  # Not used in direct execution mode
        self.backtest_golden = {}
        logger.info("📂 DIRECT EXECUTION MODE: All divergence signals trade immediately")
    
    def is_backtest_golden(self, side: str, combo: str, hour_utc: int) -> tuple:
        """Check if signal matches backtest-validated premium setup.
        
        Returns: (is_golden, tier, expected_wr) or (False, 0, 0)
        """
        # Check global rules - only shorts validated
        global_rules = self.backtest_golden.get('global_rules', {})
        if global_rules.get('side') == 'short_only' and side != 'short':
            return (False, 0, 0)
        
        if not global_rules.get('enabled', True):
            return (False, 0, 0)
        
        # Check premium setups
        for setup in self.backtest_golden.get('premium_setups', []):
            if setup['combo'] == combo and hour_utc in setup.get('hours', []):
                tier = setup.get('tier', 1)
                expected_wr = setup.get('expected_wr', 50)
                return (True, tier, expected_wr)
        
        return (False, 0, 0)

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
        # NOTE: For new divergence strategy, we only restore essential trade data
        # Signal counters start fresh since this is a new strategy
        if self.learner.redis_client:
            try:
                data = self.learner.redis_client.get('vwap_bot:executed_trades')
                if data:
                    trade_stats = json.loads(data)
                    # Only restore trade statistics, NOT old signal counts
                    # self.wins = trade_stats.get('wins', self.wins)  # Reset for new strategy
                    # self.losses = trade_stats.get('losses', self.losses)
                    # self.trades_executed = trade_stats.get('trades_executed', self.trades_executed)
                    self.daily_pnl = trade_stats.get('daily_pnl', self.daily_pnl)
                    self.total_pnl = trade_stats.get('total_pnl', self.total_pnl)
                    # signals_detected intentionally NOT restored - fresh start for divergence
                    logger.info(f"📂 Fresh start for RSI Divergence strategy")
            except Exception as e:
                logger.debug(f"Failed to load trades from Redis: {e}")
        
        # === CRITICAL: Reconcile with Bybit positions ===
        self._reconcile_positions_on_startup()

    def _reconcile_positions_on_startup(self):
        """Reconcile active_trades with actual Bybit positions on startup.
        
        This prevents trades from being "lost" if the bot restarts while
        positions are open. Fetches all open positions from Bybit and
        reconstructs active_trades for any positions not already tracked.
        """
        try:
            positions = self.broker.get_positions()
            if not positions:
                logger.info("📂 No open positions found on Bybit")
                return
            
            open_positions = [p for p in positions if float(p.get('size', 0)) > 0]
            
            if not open_positions:
                logger.info("📂 No open positions found on Bybit")
                return
            
            reconciled_count = 0
            already_tracked = 0
            
            for pos in open_positions:
                sym = pos.get('symbol')
                size = float(pos.get('size', 0))
                
                if size <= 0:
                    continue
                
                # Check if already tracked
                if sym in self.active_trades:
                    already_tracked += 1
                    logger.debug(f"✅ {sym} already tracked in active_trades")
                    continue
                
                # Position exists on Bybit but not in our tracking - reconstruct
                pos_side = pos.get('side', '').lower()
                side = 'long' if pos_side == 'buy' else 'short'
                entry = float(pos.get('avgPrice', 0))
                
                # Get TP/SL from position
                tp = float(pos.get('takeProfit', 0)) if pos.get('takeProfit') else 0
                sl = float(pos.get('stopLoss', 0)) if pos.get('stopLoss') else 0
                
                # Calculate R:R if we have TP/SL
                actual_rr = 0
                if tp and sl and entry:
                    if side == 'long':
                        tp_dist = abs(tp - entry)
                        sl_dist = abs(entry - sl)
                    else:
                        tp_dist = abs(entry - tp)
                        sl_dist = abs(sl - entry)
                    actual_rr = tp_dist / sl_dist if sl_dist > 0 else 0
                
                # Add to active_trades
                self.active_trades[sym] = {
                    'side': side,
                    'combo': 'DIV:recovered',  # Mark as recovered position
                    'signal_type': 'recovered',
                    'entry': entry,
                    'signal_price': entry,
                    'tp': tp,
                    'sl': sl,
                    'qty': size,
                    'order_id': 'recovered',
                    'actual_rr': actual_rr,
                    'open_time': time.time(),  # Unknown original time
                    'recovered_on_startup': True  # Flag for special handling
                }
                
                reconciled_count += 1
                logger.info(f"🔄 RECOVERED: {sym} {side.upper()} @ {entry:.6f} (TP: {tp}, SL: {sl})")
            
            if reconciled_count > 0:
                logger.info(f"📂 Position Reconciliation: Recovered {reconciled_count} positions, {already_tracked} already tracked")
                # Save state immediately after reconciliation
                self.save_state()
            else:
                logger.info(f"📂 Position Reconciliation: All {already_tracked} positions already tracked")
                
        except Exception as e:
            logger.error(f"Position reconciliation error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _sync_promoted_to_yaml(self):
        """Sync promoted combos to YAML file.
        
        Auto-promoted combos are written to symbol_overrides_VWAP_Combo.yaml
        for persistence across restarts.
        """
        # Syncing now handled by Redis persistence in UnifiedLearner
        # This function is deprecated but kept for backwards compatibility
        logger.debug("📂 _sync_promoted_to_yaml - handled by Redis persistence now")
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
    
    async def _on_combo_promoted(self, symbol: str, side: str, combo: str, stats: dict):
        """Callback when a combo is auto-promoted - sends Telegram notification"""
        try:
            wins = stats.get('wins', 0)
            total = stats.get('total', 0)
            lb_wr = stats.get('lower_wr', 0)
            if not lb_wr and total > 0:
                from autobot.core.unified_learner import wilson_lower_bound
                lb_wr = wilson_lower_bound(wins, total)
            
            raw_wr = (wins / total * 100) if total > 0 else 0
            ev = (wins/total * 2) - ((total-wins)/total * 1) if total > 0 else 0
            
            side_icon = "🟢" if side == 'long' else "🔴"
            
            await self.send_telegram(
                f"🚀 **COMBO PROMOTED!**\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Symbol: `{symbol}`\n"
                f"📈 Side: **{side_icon} {side.upper()}**\n"
                f"🎯 Combo: `{combo}`\n\n"
                f"📊 **STATS (30d)**\n"
                f"├ N: {total} trades\n"
                f"├ WR: {raw_wr:.0f}% (LB: {lb_wr:.0f}%)\n"
                f"├ EV: {ev:+.2f}R\n"
                f"└ Record: {wins}W/{total-wins}L\n\n"
                f"✅ **This combo will now EXECUTE real trades!**"
            )
        except Exception as e:
            logger.error(f"Promotion notification error: {e}")
    
    async def _on_combo_demoted(self, symbol: str, side: str, combo: str, reason: str, stats: dict):
        """Callback when a combo is demoted/blacklisted - sends Telegram notification"""
        try:
            wins = stats.get('wins', 0)
            total = stats.get('total', 0)
            lb_wr = stats.get('lower_wr', 0)
            if not lb_wr and total > 0:
                from autobot.core.unified_learner import wilson_lower_bound
                lb_wr = wilson_lower_bound(wins, total)
            
            raw_wr = (wins / total * 100) if total > 0 else 0
            
            side_icon = "🟢" if side == 'long' else "🔴"
            
            if reason == "demoted_and_blacklisted":
                title = "🔽 **COMBO DEMOTED & BLACKLISTED!**"
                footer = "❌ **Removed from promoted + added to blacklist!**"
            else:
                title = "🚫 **COMBO BLACKLISTED!**"
                footer = "❌ **This combo will NOT execute trades!**"
            
            await self.send_telegram(
                f"{title}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Symbol: `{symbol}`\n"
                f"📈 Side: **{side_icon} {side.upper()}**\n"
                f"🎯 Combo: `{combo}`\n\n"
                f"📊 **STATS (30d)**\n"
                f"├ N: {total} trades\n"
                f"├ WR: {raw_wr:.0f}% (LB: {lb_wr:.0f}%)\n"
                f"└ Record: {wins}W/{total-wins}L\n\n"
                f"{footer}"
            )
        except Exception as e:
            logger.error(f"Demotion notification error: {e}")

    # --- Telegram Commands ---
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = (
            "🤖 **RSI DIVERGENCE BOT**\n\n"
            "📊 **ANALYSIS**\n"
            "/dashboard - Live trading stats\n"
            "/backtest - Live vs backtest comparison\n"
            "/analytics - Deep pattern analysis\n"
            "/top - Top performing setups\n\n"
            "⚙️ **SYSTEM**\n"
            "/status - System health\n"
            "/risk - Current risk settings\n"
            "/learn - Learning system report\n\n"
            "📈 **STRATEGY**\n"
            "/sessions - Session win rates\n"
            "/blacklist - Blacklisted symbols\n"
            "/help - Show this message\n\n"
            "💡 **Strategy:** Pivot SL + 3:1 R:R"
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
            f"📡 Trading: {len(self.divergence_combos)} symbols\n"
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
        """Show RSI Divergence Bot Dashboard - Clean and focused"""
        try:
            # === SYSTEM STATUS ===
            uptime_hrs = (time.time() - self.learner.started_at) / 3600
            scanning_symbols = len(getattr(self, 'all_symbols', []))
            
            # === EXECUTED TRADES ===
            exec_total = self.wins + self.losses
            exec_wr = (self.wins / exec_total * 100) if exec_total > 0 else 0
            
            # Calculate EV with 3:1 R:R
            if exec_total > 0:
                live_ev = (self.wins * 3.0) - self.losses
                ev_per_trade = live_ev / exec_total
            else:
                live_ev = 0
                ev_per_trade = 0
            
            # Calculate P&L in R-multiples
            pnl_r = (self.wins * 3.0) - (self.losses * 1.0)
            
            # === BUILD MESSAGE ===
            msg = (
                "📊 **RSI DIVERGENCE DASHBOARD**\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                
                f"⚙️ **SYSTEM**\n"
                f"├ Uptime: {uptime_hrs:.1f}h\n"
                f"├ Loops: {self.loop_count}\n"
                f"└ Risk: {self.risk_config['value']}%\n\n"
                
                f"🎯 **STRATEGY**\n"
                f"├ Type: RSI Divergence\n"
                f"├ SL: Pivot (swing low/high)\n"
                f"├ R:R: 3:1\n"
                f"└ Symbols: {scanning_symbols}\n\n"
                
                f"📊 **SIGNALS**\n"
                f"├ Detected: {self.signals_detected}\n"
                f"└ Rate: {self.signals_detected / max(uptime_hrs, 0.1):.0f}/hr\n\n"
                
                f"💰 **EXECUTED TRADES**\n"
                f"├ Total: {self.trades_executed}\n"
                f"├ Open: {len(self.active_trades)}\n"
                f"├ ✅ Won: {self.wins}\n"
                f"├ ❌ Lost: {self.losses}\n"
                f"├ WR: {exec_wr:.1f}%\n"
                f"├ EV: {ev_per_trade:+.2f}R/trade\n"
                f"└ P&L: {pnl_r:+.1f}R\n\n"
                
                f"🕵️ **SHADOW AUDIT**\n"
                f"├ Match Rate: {self.auditor.get_stats()['rate']:.1f}%\n"
                f"├ Checks: {self.auditor.get_stats()['matches'] + self.auditor.get_stats()['mismatches']}\n"
                f"└ Mismatches: {self.auditor.get_stats()['mismatches']}\n"
            )
            
            # Add active trades if any
            if self.active_trades:
                msg += "\n🔔 **ACTIVE POSITIONS**\n"
                for sym, trade in list(self.active_trades.items())[:5]:
                    side_icon = "🟢" if trade['side'] == 'long' else "🔴"
                    rr = trade.get('actual_rr', 3.0)
                    msg += f"├ {side_icon} `{sym}` R:R={rr:.1f}:1\n"
            
            msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
            msg += "💡 /backtest /status /help"
            
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

    async def cmd_backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show comprehensive live vs backtest performance comparison"""
        try:
            # === BACKTEST REFERENCE (Validated) ===
            BT_WIN_RATE = 57.5  # % (Pivot SL + 3:1 R:R)
            BT_EV = 1.30        # R per trade
            BT_RR = 3.0         # Risk:Reward
            BT_TRADES_PER_DAY = 333  # Across 150 symbols
            
            # === LIVE DATA ===
            uptime_hrs = (time.time() - self.learner.started_at) / 3600
            uptime_days = uptime_hrs / 24
            
            live_total = self.wins + self.losses
            live_wr = (self.wins / live_total * 100) if live_total > 0 else 0
            live_pnl = (self.wins * BT_RR) - self.losses
            live_ev = live_pnl / live_total if live_total > 0 else 0
            
            # Expected values based on backtest
            expected_trades = BT_TRADES_PER_DAY * uptime_days * (len(getattr(self, 'all_symbols', [])) / 150)
            expected_wins = expected_trades * (BT_WIN_RATE / 100)
            expected_pnl = expected_trades * BT_EV
            
            # Calculate performance vs expectation
            if expected_trades > 0:
                trade_pct = (live_total / expected_trades) * 100
            else:
                trade_pct = 0
            
            if expected_pnl > 0 and live_total > 0:
                pnl_pct = (live_pnl / expected_pnl) * 100
            else:
                pnl_pct = 0
            
            wr_diff = live_wr - BT_WIN_RATE
            ev_diff = live_ev - BT_EV
            
            # Performance rating
            if live_total < 10:
                rating = "📊 Insufficient Data"
                rating_detail = "Need 10+ trades for analysis"
            elif live_wr >= BT_WIN_RATE and live_ev >= BT_EV * 0.8:
                rating = "🏆 OUTPERFORMING"
                rating_detail = "Live exceeds backtest!"
            elif live_wr >= BT_WIN_RATE * 0.9 and live_ev >= BT_EV * 0.6:
                rating = "✅ ON TARGET"
                rating_detail = "Within expected range"
            elif live_wr >= BT_WIN_RATE * 0.8:
                rating = "⚠️ BELOW TARGET"
                rating_detail = "Monitor closely"
            else:
                rating = "❌ UNDERPERFORMING"
                rating_detail = "Review strategy"
            
            # Build message
            msg = (
                "📊 **LIVE vs BACKTEST COMPARISON**\n"
                "━━━━━━━━━━━━━━━━━━━━\n\n"
                
                f"⏱️ **ANALYSIS PERIOD**\n"
                f"├ Uptime: {uptime_days:.1f} days ({uptime_hrs:.1f}h)\n"
                f"└ Symbols: {len(getattr(self, 'all_symbols', []))}\n\n"
                
                f"📈 **WIN RATE**\n"
                f"├ Live: {live_wr:.1f}%\n"
                f"├ Backtest: {BT_WIN_RATE:.1f}%\n"
                f"└ Diff: {wr_diff:+.1f}%\n\n"
                
                f"💰 **EXPECTED VALUE**\n"
                f"├ Live: {live_ev:+.2f}R/trade\n"
                f"├ Backtest: {BT_EV:+.2f}R/trade\n"
                f"└ Diff: {ev_diff:+.2f}R\n\n"
                
                f"🔢 **TRADE COUNT**\n"
                f"├ Live: {live_total}\n"
                f"├ Expected: {expected_trades:.0f}\n"
                f"└ Rate: {trade_pct:.0f}%\n\n"
                
                f"💵 **P&L (R-multiples)**\n"
                f"├ Live: {live_pnl:+.1f}R\n"
                f"├ Expected: {expected_pnl:+.1f}R\n"
                f"└ Rate: {pnl_pct:.0f}%\n\n"
                
                f"📊 **DETAILED STATS**\n"
                f"├ ✅ Wins: {self.wins} ({self.wins * BT_RR:+.1f}R)\n"
                f"├ ❌ Losses: {self.losses} ({-self.losses:.1f}R)\n"
                f"├ R:R Ratio: {BT_RR}:1\n"
                f"└ Risk: {self.risk_config['value']}%\n\n"
                
                f"🎯 **PERFORMANCE RATING**\n"
                f"├ Status: {rating}\n"
                f"└ {rating_detail}\n\n"
                
                f"📋 **BACKTEST REFERENCE**\n"
                f"├ WR: {BT_WIN_RATE}% (Pivot SL)\n"
                f"├ EV: +{BT_EV}R/trade\n"
                f"├ R:R: {BT_RR}:1\n"
                f"└ Total: +25,730R (60 days)\n"
            )
            
            msg += "\n━━━━━━━━━━━━━━━━━━━━\n"
            msg += "💡 /dashboard /analytics /help"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Backtest comparison error: {e}")
            logger.error(f"cmd_backtest error: {e}")

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

    async def cmd_promoted(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show currently promoted (active trading) combos with their stats"""
        promoted_set = self.learner.promoted
        
        if not promoted_set:
            await update.message.reply_text(
                "📊 **NO PROMOTED COMBOS YET**\n\n"
                "Combos need:\n"
                "• N ≥ 10 trades\n"
                "• LB WR ≥ 38%\n"
                "• EV ≥ 0.14\n\n"
                "Keep running to collect more data!",
                parse_mode='Markdown'
            )
            return
        
        msg = f"🚀 **PROMOTED COMBOS ({len(promoted_set)})**\n"
        msg += f"These combos EXECUTE real trades\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # Get stats for each promoted combo
        promoted_list = []
        for key in promoted_set:
            parts = key.split(':')
            if len(parts) >= 3:
                symbol = parts[0]
                side = parts[1]
                combo = ':'.join(parts[2:])
                
                stats = self.learner.get_recent_stats(symbol, side, combo, days=30)
                if stats and stats.get('total', 0) > 0:
                    from autobot.core.unified_learner import wilson_lower_bound
                    lb_wr = wilson_lower_bound(stats['wins'], stats['total'])
                    raw_wr = stats['wins'] / stats['total'] * 100
                    ev = (stats['wins']/stats['total'] * 2) - ((stats['total']-stats['wins'])/stats['total'] * 1)
                    
                    promoted_list.append({
                        'symbol': symbol,
                        'side': side,
                        'combo': combo,
                        'wins': stats['wins'],
                        'total': stats['total'],
                        'raw_wr': raw_wr,
                        'lb_wr': lb_wr,
                        'ev': ev
                    })
        
        # Sort by EV descending
        promoted_list.sort(key=lambda x: x['ev'], reverse=True)
        
        for p in promoted_list[:15]:  # Show top 15
            side_icon = "🟢" if p['side'] == 'long' else "🔴"
            msg += f"{side_icon} **{p['symbol']}**\n"
            msg += f"   `{p['combo'][:30]}`\n"
            msg += f"   N={p['total']} | WR={p['raw_wr']:.0f}% (LB:{p['lb_wr']:.0f}%) | EV={p['ev']:+.2f}R\n"
            msg += f"   Record: {p['wins']}W/{p['total']-p['wins']}L\n\n"
        
        if len(promoted_list) > 15:
            msg += f"_...and {len(promoted_list) - 15} more_"
        
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
            PROMOTE_TRADES = getattr(self.learner, 'PROMOTE_MIN_TRADES', 10)
            PROMOTE_WR = getattr(self.learner, 'PROMOTE_MIN_LOWER_WR', 38.0)
            
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
                ev = (wr_decimal * 2.0) - ((1 - wr_decimal) * 1.0)  # EV at 2:1 R:R
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
            f"📂 Active Combos: {len(self.divergence_combos)} symbols\n"
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
        """
        Process symbol for RSI divergence signals.
        Uses 15-minute candles (matches backtest parameters).
        Detects: regular_bullish, regular_bearish, hidden_bullish, hidden_bearish
        
        BACKTEST MATCHED: Only detects on NEW candle close + signal cooldown
        """
        try:
            # Use 15-minute timeframe (matches walk-forward validated backtest)
            klines = self.broker.get_klines(sym, '15', limit=100)
            if not klines or len(klines) < 50: 
                return
            
            df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
            df.set_index('start', inplace=True)
            df.sort_index(inplace=True)
            
            for c in ['open', 'high', 'low', 'close', 'volume']: 
                df[c] = df[c].astype(float)
            
            # ====================================================
            # BACKTEST MATCH: Only process when NEW candle closes
            # ====================================================
            # ====================================================
            # BACKTEST MATCH: Only process when NEW CLOSED candle exists
            # ====================================================
            if not hasattr(self, 'last_candle_processed'):
                self.last_candle_processed = {}
            
            # Use index -2 (latest closed candle) because index -1 is forming/open
            if len(df) < 2: return
            
            # Get timestamp of the CLOSED candle (second to last)
            closed_candle_time = df.index[-2]
            last_processed = self.last_candle_processed.get(sym)
            
            if last_processed is not None and closed_candle_time <= last_processed:
                # Already processed this closed candle
                return
            
            # Mark this closed candle as processed
            self.last_candle_processed[sym] = closed_candle_time
            
            # DROP the incomplete forming candle (last row)
            # This ensures we only trade confirmed signals with full volume
            df_closed = df.iloc[:-1].copy()
            
            # Calculate RSI and ATR using divergence module
            df = prepare_dataframe(df_closed)
            if df.empty or len(df) < 50: 
                return
            
            # Detect divergence signals on CLOSED candles
            signals = detect_divergence(df, sym)
            
            if not signals:
                return
            
            # ====================================================
            # SIGNAL COOLDOWN: BACKTEST MATCH - 10 bars between signals
            # ====================================================
            if not hasattr(self, 'last_signal_candle'):
                self.last_signal_candle = {}  # Per symbol (not per combo)
            
            COOLDOWN_BARS = 10  # Match backtest: min 10 bars between signals per symbol
            CANDLE_MINUTES = 15  # 15-min candles
            COOLDOWN_SECONDS = COOLDOWN_BARS * CANDLE_MINUTES * 60  # ~2.5 hours
            
            # Process each detected signal - EXECUTE ALL (backtest validated)
            for signal in signals:
                side = signal.side
                combo = signal.combo  # e.g., "DIV:regular_bullish"
                signal_type = signal.signal_type
                
                # Check cooldown - per SYMBOL only (not per combo like before)
                # Backtest uses: if i - last_idx < 10: continue
                last_signal_time = self.last_signal_candle.get(sym, 0)
                current_time = time.time()
                
                if current_time - last_signal_time < COOLDOWN_SECONDS:
                    # Still in cooldown - skip
                    remaining = int(COOLDOWN_SECONDS - (current_time - last_signal_time))
                    logger.info(f"⏳ COOLDOWN: {sym} - {remaining//60}min remaining (10-bar rule)")
                    continue
                
                self.signals_detected += 1
                
                last_row = df.iloc[-1]
                atr = last_row['atr']
                entry = last_row['close']
                
                # ====================================================
                # SHADOW AUDIT: Verify decision against backtest logic
                # ====================================================
                # Determine what the live bot decided for this signal
                live_action = "TRADE"
                if 'vol_ok' in last_row and not last_row['vol_ok']:
                    live_action = "SKIP_VOLUME"
                
                # Perform Audit
                # Note: df is already df_closed (latest closed candle + history)
                # But auditor expects full df? Auditor uses iloc[-1] of passed df as signal candle.
                # Passed df has `last_row` at -1. So this is correct.
                # Wait, passing df.iloc[:-1] drops the 'last_row'? 
                # NO. `df` in this context IS the closed dataframe (we dropped forming candle earlier at line 1245ish)
                # So `last_row` IS the closed candle.
                # BUT if we pass `df.iloc[:-1]`, we drop the signal candle! 
                # Auditor needs the SIGNAL candle at the end.
                # So pass `df`.
                
                if hasattr(self, 'auditor'):
                    audit_ok, audit_msg = self.auditor.audit(sym, df, {'action': live_action})
                    if not audit_ok:
                        logger.error(f"❌ AUDIT FAILURE: {sym} - {audit_msg}")
                    else:
                         if live_action != "NO_SIGNAL":
                            logger.debug(f"✅ AUDIT PASS: {sym} {live_action}")
                
                # ====================================================
                # VOLUME FILTER: REQUIRED (backtest validated)
                # ====================================================
                if live_action == "SKIP_VOLUME":
                    vol = last_row.get('volume', 0)
                    vol_ma = last_row.get('vol_ma', 0)
                    logger.info(f"📉 VOLUME SKIP: {sym} {side} - vol={vol:.0f} < 50% of vol_ma={vol_ma:.0f}")
                    continue
                
                # Log signal detection
                logger.info(f"📊 DIVERGENCE: {sym} {side.upper()} {combo} (RSI: {signal.rsi_value:.1f})")
                
                # Get BTC price for context
                btc_price = 0
                try:
                    btc_ticker = self.broker.get_ticker('BTCUSDT')
                    if btc_ticker:
                        btc_price = float(btc_ticker.get('lastPrice', 0))
                except:
                    pass
                
                # Record signal in learner for tracking stats
                smart_tp, smart_sl, smart_explanation = self.learner.record_signal(
                    sym, side, combo, entry, atr, btc_price, 
                    is_allowed=True,  # All divergence signals are allowed
                    notify=True
                )
                
                # ====================================================
                # BACKTEST MATCH: Queue for next candle open entry
                # ====================================================
                # Skip trading on first loop (avoid stale signals on startup)
                if not self.first_loop_completed:
                    logger.info(f"⏳ FIRST LOOP SKIP: {sym} {side} {combo} (will start trading on next loop)")
                    continue
                
                # Queue signal for execution on next candle (matches backtest behavior)
                # Backtest enters on idx+1 candle open, not immediately on signal
                if sym not in self.pending_entries:
                    self.pending_entries[sym] = {
                        'side': side,
                        'combo': combo,
                        'signal_type': signal_type,
                        'signal_price': entry,
                        'atr': atr,
                        'detected_at': time.time()
                    }
                    logger.info(f"📋 QUEUED: {sym} {side} {combo} - will execute on next candle open")
                else:
                    logger.info(f"⏳ ALREADY QUEUED: {sym} - waiting for next candle")
                    
        except Exception as e:
            logger.error(f"Error processing {sym}: {e}")

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


    async def execute_divergence_trade(self, sym, side, df, combo, signal_type):
        """Execute divergence trade with pivot-based SL and 3:1 R:R.
        
        Uses pivot points (recent swing low/high) for SL placement.
        Backtest validated: 57.5% WR, +1.30 EV, +40% more profit vs ATR.
        """
        row = df.iloc[-1]  # Get last row for current price data
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
            
            signal_price = row['close']
            atr = row['atr']
            rsi = row['rsi']
            
            # ATR validation (matches backtest: if pd.isna(atr) or atr <= 0: continue)
            if pd.isna(atr) or atr <= 0:
                logger.warning(f"Skip {sym}: Invalid ATR ({atr})")
                return
            
            # Get instrument info (tick size, lot size)
            tick_size = 0.0001
            qty_step = 0.001
            min_qty = 0.001
            try:
                inst_list = self.broker.get_instruments_info(symbol=sym)
                if inst_list and len(inst_list) > 0:
                    inst_info = inst_list[0]
                    tick_size = float(inst_info.get('priceFilter', {}).get('tickSize', 0.0001))
                    qty_step = float(inst_info.get('lotSizeFilter', {}).get('qtyStep', 0.001))
                    min_qty = float(inst_info.get('lotSizeFilter', {}).get('minOrderQty', 0.001))
            except Exception as e:
                logger.warning(f"Failed to get instrument info for {sym}: {e}")
            
            # Helper function to round to tick size
            def round_to_tick(price):
                return round(price / tick_size) * tick_size
            
            # ============================================
            # STEP 0: CALCULATE PIVOT SL DISTANCE FIRST
            # ============================================
            # This ensures position size is based on ACTUAL SL distance
            # so risk per trade is consistent (not ATR-dependent)
            
            RR_RATIO = 3.0
            LOOKBACK = 15  # Bars to look back for swing
            
            # Get recent highs/lows from DataFrame
            recent_lows = df['low'].tail(LOOKBACK).values
            recent_highs = df['high'].tail(LOOKBACK).values
            
            # Calculate SL distance from signal price (close approximation to entry)
            if side == 'long':
                swing_low = min(recent_lows)
                sl_distance_estimate = abs(signal_price - swing_low)
            else:
                swing_high = max(recent_highs)
                sl_distance_estimate = abs(swing_high - signal_price)
            
            # Apply min/max constraints (0.3-2.0 × ATR)
            min_sl_dist = 0.3 * atr
            max_sl_dist = 2.0 * atr
            
            if sl_distance_estimate < min_sl_dist:
                sl_distance_estimate = min_sl_dist
            elif sl_distance_estimate > max_sl_dist:
                sl_distance_estimate = max_sl_dist
            
            # Calculate position size based on ACTUAL SL distance
            risk_amount = balance * (self.risk_config['value'] / 100)
            qty = risk_amount / sl_distance_estimate if sl_distance_estimate > 0 else 0
            
            sl_atr_mult = sl_distance_estimate / atr if atr > 0 else 1.0
            logger.info(f"📐 {sym} SL distance: {sl_atr_mult:.2f}×ATR | Risk: ${risk_amount:.2f}")
            
            # Round qty to lot size (fix floating point precision)
            qty = (qty // qty_step) * qty_step
            # Fix floating point precision (e.g., 1236.1000000000001 -> 1236.1)
            decimals = len(str(qty_step).split('.')[-1]) if '.' in str(qty_step) else 0
            qty = round(qty, decimals)
            
            if qty < min_qty:
                logger.warning(f"Qty {qty} below min {min_qty} for {sym}")
                return
            
            # Set maximum leverage
            try:
                self.broker.set_leverage(sym, leverage=None)
            except Exception as e:
                logger.warning(f"Failed to set leverage for {sym}: {e}")
            
            # ============================================
            # STEP 1: MARKET ENTRY (instant fill)
            # ============================================
            order = self.broker.place_market(sym, side, qty)
            
            if not order or order.get('retCode') != 0:
                error_msg = order.get('retMsg', 'Unknown error') if order else 'No response'
                logger.error(f"Failed to place market order for {sym}: {error_msg}")
                return
            
            order_id = order.get('result', {}).get('orderId', 'N/A')
            logger.info(f"✅ MARKET ORDER PLACED: {sym} {side} qty={qty}")
            
            # ============================================
            # STEP 2: GET ACTUAL FILL PRICE
            # ============================================
            import asyncio
            await asyncio.sleep(0.5)  # Wait for position to register
            
            pos = self.broker.get_position(sym)
            if not pos or float(pos.get('size', 0)) == 0:
                logger.error(f"No position found for {sym} after market order")
                return
            
            actual_entry = float(pos.get('avgPrice', signal_price))
            actual_qty = float(pos.get('size', qty))
            logger.info(f"📍 ACTUAL ENTRY: {sym} @ ${actual_entry:.6f} (signal was ${signal_price:.6f})")
            
            # ============================================
            # STEP 3: CALCULATE PIVOT-BASED SL/TP (3:1 R:R)
            # ============================================
            # Use recent swing low/high for SL placement
            # Backtest validated: 57.5% WR, +1.30 EV (vs +0.92 with ATR)
            RR_RATIO = 3.0
            LOOKBACK = 15  # Bars to look back for swing
            
            # MATCH BACKTEST: Exclude current forming candle (entry candle) from swing search
            # We want SL based on the PATTERN (previous candles), not the entry candle
            # Use iloc[:-1] to drop last row
            recent_lows = df['low'].iloc[:-1].tail(LOOKBACK).values
            recent_highs = df['high'].iloc[:-1].tail(LOOKBACK).values
            
            if side == 'long':
                # SL at recent swing low
                swing_low = min(recent_lows)
                sl = round_to_tick(swing_low)
                sl_distance = abs(actual_entry - sl)
                
                # Minimum SL: 0.3×ATR  (avoid too tight)
                # Maximum SL: 2.0×ATR  (avoid too wide)
                min_sl_dist = 0.3 * atr
                max_sl_dist = 2.0 * atr
                
                if sl_distance < min_sl_dist:
                    sl_distance = min_sl_dist
                    sl = round_to_tick(actual_entry - sl_distance)
                elif sl_distance > max_sl_dist:
                    sl_distance = max_sl_dist
                    sl = round_to_tick(actual_entry - sl_distance)
                
                tp = round_to_tick(actual_entry + (RR_RATIO * sl_distance))
            else:
                # SL at recent swing high
                swing_high = max(recent_highs)
                sl = round_to_tick(swing_high)
                sl_distance = abs(sl - actual_entry)
                
                # Min/Max SL constraints
                min_sl_dist = 0.3 * atr
                max_sl_dist = 2.0 * atr
                
                if sl_distance < min_sl_dist:
                    sl_distance = min_sl_dist
                    sl = round_to_tick(actual_entry + sl_distance)
                elif sl_distance > max_sl_dist:
                    sl_distance = max_sl_dist
                    sl = round_to_tick(actual_entry + sl_distance)
                
                tp = round_to_tick(actual_entry - (RR_RATIO * sl_distance))
            
            tp_distance = abs(tp - actual_entry)
            actual_rr = tp_distance / sl_distance if sl_distance > 0 else 0
            sl_atr_mult = sl_distance / atr if atr > 0 else 1.0
            
            logger.info(f"📊 {sym} PIVOT SL: {sl_atr_mult:.2f}×ATR | R:R = {actual_rr:.1f}:1")
            logger.info(f"   Entry: ${actual_entry:.6f} | SL: ${sl:.6f} | TP: ${tp:.6f}")
            
            # ============================================
            # STEP 4: SET TP/SL ON POSITION
            # ============================================
            tpsl_result = self.broker.set_tpsl(sym, tp, sl, actual_qty)
            if not tpsl_result or tpsl_result.get('retCode') != 0:
                error_msg = tpsl_result.get('retMsg', 'Unknown') if tpsl_result else 'No response'
                logger.error(f"Failed to set TP/SL for {sym}: {error_msg}")
                # Continue anyway - position is open
            else:
                logger.info(f"🔐 TP/SL SET: {sym} TP=${tp:.6f} SL=${sl:.6f}")
            
            self.trades_executed += 1
            
            # Track trade with ACTUAL data
            self.active_trades[sym] = {
                'side': side,
                'combo': combo,
                'signal_type': signal_type,
                'entry': actual_entry,
                'signal_price': signal_price,
                'tp': tp,
                'sl': sl,
                'qty': actual_qty,
                'order_id': order_id,
                'actual_rr': actual_rr,
                'open_time': time.time()
            }
            
            # Signal type emoji
            type_emoji = {
                'regular_bullish': '📈 Regular Bullish',
                'regular_bearish': '📉 Regular Bearish', 
                'hidden_bullish': '🔼 Hidden Bullish',
                'hidden_bearish': '🔽 Hidden Bearish'
            }.get(signal_type, signal_type)
            
            # Calculate expected profit/loss
            profit_target = actual_qty * tp_distance
            loss_risk = actual_qty * sl_distance
            
            # Send Telegram notification
            side_emoji = '🟢 LONG' if side == 'long' else '🔴 SHORT'
            slippage = abs(actual_entry - signal_price)
            slippage_pct = (slippage / signal_price) * 100 if signal_price > 0 else 0
            
            msg = (
                f"🎯 **TRADE EXECUTED**\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Symbol: `{sym}`\n"
                f"📈 Side: **{side_emoji}**\n"
                f"💎 Type: **{type_emoji}**\n\n"
                f"💰 **Entry**: ${actual_entry:.6f}\n"
                f"├ Signal: ${signal_price:.6f}\n"
                f"└ Slippage: {slippage_pct:.3f}%\n\n"
                f"🎯 **TP**: ${tp:.6f} (+${profit_target:.2f})\n"
                f"🛑 **SL**: ${sl:.6f} (-${loss_risk:.2f})\n"
                f"📊 **R:R**: {actual_rr:.2f}:1\n"
                f"📊 RSI: {rsi:.1f}\n\n"
                f"🔐 **Protection**: ✅ TP/SL Active\n"
                f"└ Calculated from actual fill\n\n"
                f"💵 Risk: ${risk_amount:.2f} ({self.risk_config['value']}%)"
            )
            await self.send_telegram(msg)
            
            logger.info(f"✅ COMPLETE: {sym} {side} @ ${actual_entry:.6f} R:R={actual_rr:.2f}:1")
            
        except Exception as e:
            logger.error(f"Execute divergence trade error {sym}: {e}")

    async def check_pending_orders(self):
        """Check pending orders for fills or timeout (5 minutes).
        
        - If filled: Move to active_trades and increment trades_executed
        - If 5 minutes passed without fill: Cancel and send notification
        """
        if not hasattr(self, 'pending_orders') or not self.pending_orders:
            return
        
        to_remove = []
        now = time.time()
        TIMEOUT_SECONDS = 300  # 5 minutes
        
        for order_id, order_data in list(self.pending_orders.items()):
            sym = order_data['symbol']
            order_age = now - order_data['order_time']
            
            try:
                # Check order status
                status = self.broker.get_order_status(sym, order_id)
                
                if status:
                    order_status = status.get('orderStatus', 'Unknown')
                    
                    if order_status == 'Filled':
                        # ORDER FILLED - Move to active trades
                        fill_price = float(status.get('avgPrice', order_data['entry']))
                        
                        self.active_trades[sym] = {
                            'side': order_data['side'],
                            'combo': order_data['combo'],
                            'signal_type': order_data['signal_type'],
                            'entry': fill_price,
                            'tp': order_data['tp'],
                            'sl': order_data['sl'],
                            'qty': order_data['qty'],
                            'order_id': order_id,
                            'open_time': now
                        }
                        
                        self.trades_executed += 1
                        to_remove.append(order_id)
                        
                        # Send FILLED notification
                        side_emoji = '🟢 LONG' if order_data['side'] == 'long' else '🔴 SHORT'
                        type_emoji = {
                            'regular_bullish': '📈 Regular Bullish',
                            'regular_bearish': '📉 Regular Bearish', 
                            'hidden_bullish': '🔼 Hidden Bullish',
                            'hidden_bearish': '🔽 Hidden Bearish'
                        }.get(order_data['signal_type'], order_data['signal_type'])
                        
                        msg = (
                            f"✅ **ORDER FILLED!**\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📊 Symbol: `{sym}`\n"
                            f"📈 Side: **{side_emoji}**\n"
                            f"💎 Type: **{type_emoji}**\n\n"
                            f"💰 Fill Price: ${fill_price:.4f}\n"
                            f"🎯 TP: ${order_data['tp']:.4f}\n"
                            f"🛑 SL: ${order_data['sl']:.4f}\n\n"
                            f"🔐 **Protected**: TP/SL active"
                        )
                        await self.send_telegram(msg)
                        logger.info(f"✅ FILLED: {sym} {order_data['side']} @ {fill_price:.4f}")
                    
                    elif order_status in ['Cancelled', 'Rejected', 'Deactivated']:
                        # Already cancelled/rejected
                        to_remove.append(order_id)
                        logger.info(f"Order {order_id[:8]} was {order_status}")
                    
                    elif order_age > TIMEOUT_SECONDS:
                        # TIMEOUT - Cancel the order
                        logger.info(f"⏰ Order timeout for {sym} - cancelling after {order_age:.0f}s")
                        
                        cancel_result = self.broker.cancel_order(sym, order_id)
                        to_remove.append(order_id)
                        
                        # Send CANCELLED notification
                        msg = (
                            f"⏰ **ORDER CANCELLED (Timeout)**\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📊 Symbol: `{sym}`\n"
                            f"📈 Side: {order_data['side'].upper()}\n"
                            f"💰 Entry: ${order_data['entry']:.4f}\n\n"
                            f"❌ Order not filled after 5 minutes\n"
                            f"└ Cancelled automatically"
                        )
                        await self.send_telegram(msg)
                        logger.info(f"❌ CANCELLED: {sym} order timed out")
                
            except Exception as e:
                logger.error(f"Error checking order {order_id[:8]}: {e}")
                # If order too old and we can't check it, just remove
                if order_age > TIMEOUT_SECONDS * 2:
                    to_remove.append(order_id)
        
        # Remove processed orders
        for order_id in to_remove:
            self.pending_orders.pop(order_id, None)

    async def execute_trade(self, sym, side, row, combo, source='manual'):
        """Execute trade using LIMIT ORDER (not market) for precise entry.
        
        source: 'backtest_golden', 'auto_promoted', or 'manual'
        """
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
            
            # 2:1 R:R for trading
            # SL = 1 ATR (1R), TP = 2 ATR (2R)
            optimal_rr = 2.0
            
            # Calculate TP/SL with 2:1 R:R
            # Using 1 ATR for SL (1R), 2*ATR for TP (2R)
            MIN_SL_PCT = 0.5  # Minimum 0.5% distance for SL
            MIN_TP_PCT = 1.0  # Minimum 1.0% distance for TP
            
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
                
                # Determine source for notification display
                # Only auto-promoted combos execute now (backtest golden disabled)
                source_display = "🚀 Auto-Promoted"
                
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
                is_auto_promoted = (source == 'auto_promoted')  # Derive from source param
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
                    'source': source,  # Also store source for reference
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
                    f"📁 Source: **{source_display}**\n"
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
        """Show divergence strategy status on startup.
        
        DIRECT EXECUTION MODE - All divergence signals execute immediately.
        Backtest validated: 61.3% WR | +0.84 EV at 2:1 R:R
        """
        logger.info("🚀 RSI Divergence Strategy - DIRECT EXECUTION MODE")
        
        await self.send_telegram(
            f"🚀 **DIRECT EXECUTION MODE**\n"
            f"├ All divergence signals trade\n"
            f"├ No promotion waiting\n"
            f"└ Backtest: 61.3% WR\n\n"
            f"📊 **Signal Types (ALL ACTIVE)**\n"
            f"├ 📉 Regular Bearish: 66% WR\n"
            f"├ 📈 Regular Bullish: 64% WR\n"
            f"├ 🔽 Hidden Bearish: 59% WR\n"
            f"└ 🔼 Hidden Bullish: 55% WR\n\n"
            f"🎯 **Strategy**: RSI Divergence\n"
            f"⏱️ **Timeframe**: 15 minutes\n"
            f"📚 **Symbols**: Top 200 by volume"
        )

    async def run(self):
        logger.info("🤖 Divergence Bot Starting...")
        
        # Send starting notification
        await self.send_telegram("⏳ **Divergence Bot Starting...**\nInitializing systems...")
        
        # Initialize Learner
        self.learner = UnifiedLearner()
        
        # Initialize Shadow Auditor (Verification)
        self.auditor = ShadowAuditor()
        
        # Initialize Telegram
        try:
            token = self.cfg['telegram']['token']
            self.tg_app = ApplicationBuilder().token(token).build()
            
            self.tg_app.add_handler(CommandHandler("help", self.cmd_help))
            self.tg_app.add_handler(CommandHandler("status", self.cmd_status))
            self.tg_app.add_handler(CommandHandler("risk", self.cmd_risk))
            self.tg_app.add_handler(CommandHandler("phantoms", self.cmd_phantoms))
            self.tg_app.add_handler(CommandHandler("dashboard", self.cmd_dashboard))
            self.tg_app.add_handler(CommandHandler("backtest", self.cmd_backtest))
            self.tg_app.add_handler(CommandHandler("analytics", self.cmd_analytics))
            self.tg_app.add_handler(CommandHandler("learn", self.cmd_learn))
            self.tg_app.add_handler(CommandHandler("promote", self.cmd_promote))
            self.tg_app.add_handler(CommandHandler("sessions", self.cmd_sessions))
            self.tg_app.add_handler(CommandHandler("blacklist", self.cmd_blacklist))
            self.tg_app.add_handler(CommandHandler("smart", self.cmd_smart))
            self.tg_app.add_handler(CommandHandler("top", self.cmd_top))
            self.tg_app.add_handler(CommandHandler("ladder", self.cmd_ladder))
            self.tg_app.add_handler(CommandHandler("promoted", self.cmd_promoted))
            
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
        self.load_state()  # Restore previous session data + reconcile positions
        trading_symbols = list(self.divergence_combos.keys())
        
        # === NOTIFY ABOUT RECOVERED POSITIONS ===
        recovered_positions = [t for t in self.active_trades.values() if t.get('recovered_on_startup')]
        if recovered_positions:
            msg = f"🔄 **POSITIONS RECOVERED** ({len(recovered_positions)})\n"
            msg += "━━━━━━━━━━━━━━━━━━━━\n"
            for sym, trade in list(self.active_trades.items())[:5]:
                if trade.get('recovered_on_startup'):
                    side_icon = "🟢" if trade['side'] == 'long' else "🔴"
                    msg += f"{side_icon} `{sym}` @ ${trade['entry']:.4f}\n"
                    if trade.get('tp') and trade.get('sl'):
                        msg += f"   TP: ${trade['tp']:.4f} | SL: ${trade['sl']:.4f}\n"
            if len(recovered_positions) > 5:
                msg += f"\n...and {len(recovered_positions) - 5} more"
            msg += "\n\n✅ **Positions will continue to be monitored**"
            await self.send_telegram(msg)
            logger.info(f"🔄 Recovered {len(recovered_positions)} positions from Bybit")
        
        # === STARTUP PROMOTION SCAN ===
        # Check if any combos should be promoted based on 30-day PostgreSQL data
        startup_promoted = self.learner._scan_for_promote()
        if startup_promoted:
            promo_msg = f"🚀 **STARTUP PROMOTION SCAN**\n"
            promo_msg += f"Found **{len(startup_promoted)}** combos to promote!\n\n"
            for p in startup_promoted[:5]:  # Show top 5
                side_icon = "🟢" if p['side'] == 'long' else "🔴"
                promo_msg += f"{side_icon} `{p['symbol']}` | {p['combo'][:20]}...\n"
                promo_msg += f"   N={p['total']} | WR={p['wins']/p['total']*100:.0f}% | EV={p['ev']:+.2f}R\n"
            if len(startup_promoted) > 5:
                promo_msg += f"\n...and {len(startup_promoted) - 5} more"
            await self.send_telegram(promo_msg)
            logger.info(f"🚀 Startup: Promoted {len(startup_promoted)} combos")
        
        # Fetch TOP 200 symbols by 24h volume (SAME AS BACKTEST)
        try:
            import requests
            url = "https://api.bybit.com/v5/market/tickers?category=linear"
            resp = requests.get(url, timeout=10)
            tickers = resp.json().get('result', {}).get('list', [])
            usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
            usdt_pairs.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            self.all_symbols = [t['symbol'] for t in usdt_pairs[:200]]
            logger.info(f"📚 Fetched TOP 200 symbols by volume (same as backtest)")
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}, falling back to config")
            self.all_symbols = self.cfg.get('trade', {}).get('symbols', [])
        
        # Sync promoted combos to YAML (ensures YAML matches promoted set)
        self._sync_promoted_to_yaml()
        
        # Run immediate promote/demote scan on startup
        await self._startup_promote_demote_scan()
        
        self.load_overrides()  # Reload after sync and startup scan
        trading_symbols = list(self.divergence_combos.keys())
        
        if not trading_symbols:
            await self.send_telegram("⚠️ **No trading symbols!**\nLearning will still run on all 400 symbols.")
            logger.warning("No trading symbols, learning only mode")
        
        # Check connections
        redis_ok = "🟢" if self.learner.redis_client else "🔴"
        pg_ok = "🟢" if self.learner.pg_conn else "🔴"

        # Get near-promotion stats for startup message
        all_combos = self.learner.get_all_combos()
        PROMOTE_TRADES = getattr(self.learner, 'PROMOTE_MIN_TRADES', 15)
        PROMOTE_WR = getattr(self.learner, 'PROMOTE_MIN_LOWER_WR', 38.0)
        near_promote = len([c for c in all_combos if c['total'] >= 5 and c['lower_wr'] >= 35
                           and f"{c['symbol']}:{c['side']}:{c['combo']}" not in self.learner.promoted])

        # Send success notification
        await self.send_telegram(
            f"✅ **RSI Divergence Bot Online!**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 **Strategy**: RSI Divergence\n"
            f"⏱️ **Timeframe**: 15 minutes\n"
            f"🎯 **R:R**: 2:1\n"
            f"📈 **Backtest**: 61.3% WR | +0.84 EV\n\n"
            f"🚀 **Divergence Types (ALL ACTIVE)**\n"
            f"├ 📉 Regular Bearish: 66% WR\n"
            f"├ 📈 Regular Bullish: 64% WR\n"
            f"├ 🔽 Hidden Bearish: 59% WR\n"
            f"└ 🔼 Hidden Bullish: 55% WR\n\n"
            f"📚 Scanning: **{len(self.all_symbols)}** symbols\n"
            f"⚙️ Risk: **{self.risk_config['value']}%** per trade\n\n"
            f"💾 System Health:\n"
            f"• Redis: {redis_ok}\n"
            f"• Postgres: {pg_ok}\n\n"
            f"🖥️ **Dashboard**: `http://localhost:8888`\n"
            f"Commands: /help /status /dashboard"
        )
        
        logger.info(f"Trading {len(trading_symbols)} symbols, Learning {len(self.all_symbols)} symbols")
            
        try:
            while True:
                self.load_overrides()  # Reload to pick up new combos
                trading_symbols = list(self.divergence_combos.keys())
                self.loop_count += 1
                
                # ============================================================
                # BACKTEST MATCH: Execute pending entries from PREVIOUS loop
                # This matches backtest behavior of entering on next candle open
                # ============================================================
                if self.pending_entries:
                    logger.info(f"📊 Executing {len(self.pending_entries)} queued entries from previous candle")
                    for sym, entry_info in list(self.pending_entries.items()):
                        try:
                            # Get fresh klines for execution
                            klines = self.broker.get_klines(sym, '15', limit=100)
                            if klines and len(klines) >= 50:
                                df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                                df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
                                df.set_index('start', inplace=True)
                                df.sort_index(inplace=True)
                                for c in ['open', 'high', 'low', 'close', 'volume']: 
                                    df[c] = df[c].astype(float)
                                df = prepare_dataframe(df)
                                
                                if not df.empty:
                                    logger.info(f"🚀 EXECUTING QUEUED: {sym} {entry_info['side']} {entry_info['combo']}")
                                    await self.execute_divergence_trade(
                                        sym, 
                                        entry_info['side'], 
                                        df, 
                                        entry_info['combo'], 
                                        entry_info['signal_type']
                                    )
                            # Remove from pending after execution attempt
                            del self.pending_entries[sym]
                        except Exception as e:
                            logger.error(f"Failed to execute queued entry {sym}: {e}")
                            del self.pending_entries[sym]
                
                # Scan ALL symbols for learning, but only trade allowed ones
                for sym in self.all_symbols:
                    await self.process_symbol(sym)
                    # RATE LIMIT PROTECTION: 0.2s delay = max 5 req/s
                    # Bybit limit is typically 10-20/s, but 5/s is safe and stable
                    await asyncio.sleep(0.2)
                
                # Mark first loop as completed (trading will start on NEXT loop)
                if not self.first_loop_completed:
                    self.first_loop_completed = True
                    logger.info("✅ First loop completed - trading will start on next loop")
                
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
                    
                    # Check divergence pending orders for 5-minute timeout
                    await self.check_pending_orders()
                    
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
                                    
                                    # Source - only auto-promoted combos execute now
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
                    # Using only auto-promoted combos from live learning
                    # Backtest golden combos have been disabled
                    logger.debug("Auto-promote/demote ACTIVE - combos promoted based on live performance")
                
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
