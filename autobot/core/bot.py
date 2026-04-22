"""
1H Multi-Divergence Trading Bot - Multi-Config (Both Sides)
====================================================================
323 Configs across 275 Symbols | Long + Short | Dynamic Slippage Validated

6-Month Validation (Sep 2025 - Mar 2026):
  +3,644R | 24.2% WR | PF 1.94 | Max DD 26.1R | 4,736 trades
  Long: +650R (94 configs) | Short: +2,994R (229 configs)
  48 dual-side symbols trade both directions

Divergence Types:
- REG_BULL: Regular Bullish (29 configs)
- REG_BEAR: Regular Bearish (90 configs)
- HID_BULL: Hidden Bullish (65 configs)
- HID_BEAR: Hidden Bearish (139 configs)

Validation: 365-day data, 75/25 train/test split, Monte Carlo (1000 iter),
            Profit Factor > 1.2, Max DD < 15R, Dynamic volume-based slippage

Main Features:
- 1H divergence detection with EMA 200 trend filter
- Break of Structure (BOS) confirmation (12 candles max)
- Multi-config: each symbol can have independent long + short configs
- Per-config divergence type, R:R ratio, and ATR multiplier
- Fixed TP/SL (no trailing, no partial)
- Dynamic risk per trade (% of account balance)
"""

import asyncio
import logging
import yaml
import os
import time
import aiohttp
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from autobot.brokers.bybit import Bybit, BybitConfig
from autobot.core.divergence_detector import (
    detect_divergences,
    prepare_dataframe,
    check_bos,
    is_trend_aligned,
    DivergenceSignal,
    get_signal_emoji,
    get_signal_description
)
from autobot.config.symbol_rr_mapping import SymbolRRConfig
from autobot.utils.telegram_handler import TelegramHandler
from autobot.core.storage import StorageHandler

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class PendingSignal:
    """Tracks a divergence waiting for BOS confirmation"""
    signal: DivergenceSignal
    detected_at: datetime
    candles_waited: int = 0
    max_wait_candles: int = 12  # 12 hours on 1H - increased for better BOS rate
    
    def is_expired(self) -> bool:
        return self.candles_waited >= self.max_wait_candles
    
    def increment_wait(self):
        self.candles_waited += 1


@dataclass
class ActiveTrade:
    """Represents an active trade position"""
    symbol: str
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    position_size: float
    entry_time: datetime
    order_id: str = ""
    risk_usd_at_entry: float = 0.0  # Store risk at trade time for accurate R calculation


class Bot4H:
    """1H Multi-Divergence Trading Bot - Multi-Config (Both Sides)"""
    
    def __init__(self):
        """Initialize bot"""
        logger.info("="*60)
        logger.info("1H MULTI-DIVERGENCE BOT INITIALIZING")
        logger.info("="*60)
        
        self.load_config()
        self.setup_broker()
        self.setup_telegram()
        self.symbol_config = SymbolRRConfig()
        
        # Trading state
        self.pending_signals: Dict[str, List[PendingSignal]] = {}  # {symbol: [signals]}
        self.active_trades: Dict[str, ActiveTrade] = {}  # {symbol_side: trade} e.g. "BTCUSDT_long"
        self.confirmed_entries: Dict[str, DivergenceSignal] = {}  # {symbol_side: signal} - BOS confirmed, enter on NEXT candle
        
        # [STARTUP PROTECTION] Track which symbols have been seen since startup
        # First candle for each symbol is used for initialization only (no trades)
        self.symbols_initialized: set = set()
        
        # Stats
        self.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_r': 0.0,
            'win_rate': 0.0,
            'avg_r': 0.0
        }
        self.stats_file = 'stats.json'
        
        # Load stats if exists
        self.load_stats()
        
        # Lifetime Stats (persistent across all restarts - stored in PostgreSQL via StorageHandler)
        # Note: self.storage is initialized later, so we just set defaults here
        self.lifetime_stats = {
            'start_date': None,  # ISO format string
            'starting_balance': 0.0,
            'total_r': 0.0,
            'total_pnl': 0.0,
            'total_trades': 0,
            'wins': 0,
            'best_day_r': 0.0,
            'best_day_date': None,
            'worst_day_r': 0.0,
            'worst_day_date': None,
            'daily_r': {},  # {date_str: r_value}
            # New fields for comprehensive tracking
            'best_trade_r': 0.0,
            'best_trade_symbol': '',
            'best_trade_date': None,
            'worst_trade_r': 0.0,
            'worst_trade_symbol': '',
            'worst_trade_date': None,
            'max_drawdown_r': 0.0,
            'peak_equity_r': 0.0,  # For drawdown calculation
            'current_streak': 0,  # Positive = wins, negative = losses
            'longest_win_streak': 0,
            'longest_loss_streak': 0
        }
        
        # Recent trades for rolling regime detection (100 for asymmetric windows)
        self.recent_trades: deque = deque(maxlen=100)

        # Manual regime override (from /setregime command)
        self.regime_override = None        # 'favorable'|'cautious'|'adverse'|'critical' or None
        self.regime_override_trades = 0    # Trades since override was set (auto-clears at 10)

        # Per-symbol stats
        self.symbol_stats: Dict[str, dict] = {}  # {symbol: {trades, wins, total_r}}
        
        # Bot control
        self.trading_enabled = True
        
        # Start Time for Uptime
        self.start_time = datetime.now()
        
        # Candle tracking
        self.last_candle_close: Dict[str, datetime] = {}
        
        # Scan state for dashboard visibility
        self.scan_state = {
            'last_scan_time': None,
            'symbols_scanned': 0,
            'fresh_divergences': 0,
            'last_scan_signals': []  # List of symbols with fresh divergences
        }
        
        # RSI cache for dashboard
        self.rsi_cache: Dict[str, float] = {}  # {symbol: last_rsi}
        
        # Radar cache for developing setups
        self.radar_items: Dict[str, dict] = {}
        self.extreme_zone_tracker: Dict[str, datetime] = {}

        # Track whether first scan cycle has completed
        self.startup_scan_done = False

        # New startup logic: Immediate trading allowed (24h staleness filter protects us)
        logger.info("[STARTUP] Precision Mode Active. Immediate signal processing enabled.")
        
        # Signal deduplication & Lifetime Stats - Persistence (DB or File)
        self.storage = StorageHandler()
        # Load lifetime stats from PostgreSQL (or fallback to local file)
        self.lifetime_stats = self.storage.load_lifetime_stats()
        # Restore recent trades for regime detection
        saved_trades = self.lifetime_stats.get('recent_trades', [])
        for t in saved_trades[-100:]:
            self.recent_trades.append({
                'r': t['r'],
                'win': t['win'],
                'time': datetime.fromisoformat(t['time']),
                'symbol': t.get('symbol', '')
            })
        logger.info(f"Restored {len(self.recent_trades)} recent trades for regime detection")
        # Restore manual regime override
        self.regime_override = self.lifetime_stats.get('regime_override', None)
        self.regime_override_trades = self.lifetime_stats.get('regime_override_trades', 0)
        if self.regime_override:
            logger.info(f"[REGIME] Restored manual override: {self.regime_override} ({self.regime_override_trades}/10 trades since set)")
        logger.info(f"Lifetime stats loaded: {self.lifetime_stats['total_r']:.1f}R, {self.lifetime_stats['total_trades']} trades since {self.lifetime_stats.get('start_date', 'unknown')}")
        
        # BOS Performance Tracking (for monitoring stale filter removal impact)
        self.bos_tracking = {
            'divergences_detected_today': 0,
            'bos_confirmed_today': 0,
            'divergences_detected_total': 0,
            'bos_confirmed_total': 0,
            'last_reset': datetime.now().date()
        }
        
        logger.info(f"Loaded {len(self.symbol_config.get_enabled_symbols())} enabled symbols")
        # Logger info for signals is now handled by StorageHandler

    def load_stats(self):
        """Load internal stats from JSON file"""
        if os.path.exists(self.stats_file):
            try:
                import json
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                    self.stats.update(data)
                logger.info(f"Loaded stats: {self.stats['total_trades']} trades, {self.stats['total_r']:.2f}R")
            except Exception as e:
                logger.error(f"Failed to load stats: {e}")

    def save_stats(self):
        """Save internal stats to JSON file"""
        try:
            import json
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
    
    def load_lifetime_stats(self):
        """Load lifetime stats from PostgreSQL (via StorageHandler)"""
        self.lifetime_stats = self.storage.load_lifetime_stats()
        logger.info(f"Loaded lifetime stats: {self.lifetime_stats['total_r']:.1f}R since {self.lifetime_stats.get('start_date', 'unknown')}")
    
    def save_lifetime_stats(self):
        """Save lifetime stats to PostgreSQL (via StorageHandler)"""
        self.storage.save_lifetime_stats(self.lifetime_stats)
    
    async def initialize_lifetime_stats(self):
        """Initialize lifetime stats if this is the first run"""
        if self.lifetime_stats.get('start_date') is None:
            self.lifetime_stats['start_date'] = datetime.now().strftime('%Y-%m-%d')
            try:
                balance = await self.broker.get_balance()
                self.lifetime_stats['starting_balance'] = balance or 0.0
            except:
                self.lifetime_stats['starting_balance'] = 0.0
            self.save_lifetime_stats()
            logger.info(f"Initialized lifetime stats: Start {self.lifetime_stats['start_date']}, Balance ${self.lifetime_stats['starting_balance']:.2f}")
    
    def update_lifetime_stats(self, r_value: float, pnl: float, is_win: bool, symbol: str = ''):
        """Update lifetime stats after a trade closes
        
        Tracks comprehensive metrics including:
        - Total R, PnL, wins/losses
        - Best/worst individual trades
        - Maximum drawdown (peak-to-trough)
        - Win/loss streaks
        - Best/worst trading days
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"[LIFETIME] Updating stats: R={r_value:+.2f}, PnL=${pnl:.2f}, Win={is_win}, Symbol={symbol}")

        # Track recent trades for regime detection
        prev_regime, prev_mult, _ = self.get_regime_status()
        self.recent_trades.append({'r': r_value, 'win': is_win, 'time': datetime.now(), 'symbol': symbol})

        # Track trades since manual override (auto-clear at 10)
        if self.regime_override is not None:
            self.regime_override_trades += 1
            if self.regime_override_trades >= 10:
                logger.info(f"[REGIME] Manual override auto-cleared after 10 new trades")
                self.regime_override = None
                self.regime_override_trades = 0
            self.lifetime_stats['regime_override'] = self.regime_override
            self.lifetime_stats['regime_override_trades'] = self.regime_override_trades

        # Update totals
        self.lifetime_stats['total_r'] += r_value
        self.lifetime_stats['total_pnl'] += pnl
        self.lifetime_stats['total_trades'] += 1
        if is_win:
            self.lifetime_stats['wins'] += 1

        # Track gross profit/loss for accurate profit factor
        if r_value > 0:
            self.lifetime_stats['gross_profit_r'] = self.lifetime_stats.get('gross_profit_r', 0.0) + r_value
        else:
            self.lifetime_stats['gross_loss_r'] = self.lifetime_stats.get('gross_loss_r', 0.0) + r_value
        
        # Update best/worst individual trade
        if r_value > self.lifetime_stats.get('best_trade_r', 0):
            self.lifetime_stats['best_trade_r'] = r_value
            self.lifetime_stats['best_trade_symbol'] = symbol
            self.lifetime_stats['best_trade_date'] = today
        if r_value < self.lifetime_stats.get('worst_trade_r', 0):
            self.lifetime_stats['worst_trade_r'] = r_value
            self.lifetime_stats['worst_trade_symbol'] = symbol
            self.lifetime_stats['worst_trade_date'] = today
        
        # Update max drawdown (peak-to-trough equity curve)
        current_equity = self.lifetime_stats['total_r']
        peak_equity = self.lifetime_stats.get('peak_equity_r', 0)
        
        if current_equity > peak_equity:
            self.lifetime_stats['peak_equity_r'] = current_equity
            peak_equity = current_equity
        
        current_drawdown = current_equity - peak_equity  # Will be negative or zero
        if current_drawdown < self.lifetime_stats.get('max_drawdown_r', 0):
            self.lifetime_stats['max_drawdown_r'] = current_drawdown
        
        # Update streaks
        current_streak = self.lifetime_stats.get('current_streak', 0)
        if is_win:
            if current_streak >= 0:
                current_streak += 1
            else:
                current_streak = 1  # Reset to 1 win
            # Update longest win streak
            if current_streak > self.lifetime_stats.get('longest_win_streak', 0):
                self.lifetime_stats['longest_win_streak'] = current_streak
        else:
            if current_streak <= 0:
                current_streak -= 1
            else:
                current_streak = -1  # Reset to 1 loss
            # Update longest loss streak (store as positive number)
            if abs(current_streak) > self.lifetime_stats.get('longest_loss_streak', 0):
                self.lifetime_stats['longest_loss_streak'] = abs(current_streak)
        
        self.lifetime_stats['current_streak'] = current_streak
        
        # Update daily R
        if today not in self.lifetime_stats['daily_r']:
            self.lifetime_stats['daily_r'][today] = 0.0
        self.lifetime_stats['daily_r'][today] += r_value
        
        daily_r = self.lifetime_stats['daily_r'][today]
        
        # Update best/worst day
        if daily_r > self.lifetime_stats.get('best_day_r', 0):
            self.lifetime_stats['best_day_r'] = daily_r
            self.lifetime_stats['best_day_date'] = today
        if daily_r < self.lifetime_stats.get('worst_day_r', 0):
            self.lifetime_stats['worst_day_r'] = daily_r
            self.lifetime_stats['worst_day_date'] = today
        
        logger.info(f"[LIFETIME] New totals: R={self.lifetime_stats['total_r']:+.2f}, Trades={self.lifetime_stats['total_trades']}, Streak={current_streak}, MaxDD={self.lifetime_stats['max_drawdown_r']:.1f}R")

        # Detect and log regime changes
        new_regime, new_mult, diag = self.get_regime_status()
        if new_regime != prev_regime and new_regime != 'unknown':
            regime_icons = {
                'favorable': 'Favorable 🟢',
                'cautious': 'Cautious 🟡',
                'adverse': 'Adverse 🟠',
                'critical': 'Critical 🔴',
                'halted': 'HALTED 🛑',
            }
            label = regime_icons.get(new_regime, new_regime)
            active_signals = [k for k, v in diag.items() if v['mult'] < 1.0]
            wr = diag['quality']['wr']
            avg_r = diag['quality']['avg_r']
            dd = diag['drawdown']['dd_from_peak']
            daily_r = diag['daily']['daily_r']
            loss_streak = diag['streak']['loss_streak']
            logger.info(f"[REGIME] Changed: {prev_regime} → {new_regime} (mult={new_mult:.2f}, signals={active_signals})")
            if hasattr(self, 'telegram') and self.telegram:
                trigger_lines = []
                if diag['quality']['mult'] < 1.0:
                    trigger_lines.append(f"├ Quality: {wr:.0%} WR, {avg_r:+.2f}R → {diag['quality']['mult']:.2f}x")
                if diag['drawdown']['mult'] < 1.0:
                    trigger_lines.append(f"├ DD Breaker: {dd:.1f}R from peak → {diag['drawdown']['mult']:.2f}x")
                if diag['daily']['mult'] < 1.0:
                    trigger_lines.append(f"├ Daily Cap: {daily_r:+.1f}R today → {diag['daily']['mult']:.2f}x")
                if diag['streak']['mult'] < 1.0:
                    trigger_lines.append(f"├ Streak: {loss_streak} losses → {diag['streak']['mult']:.2f}x")
                triggers = "\n".join(trigger_lines) if trigger_lines else "├ All clear"
                asyncio.create_task(self.telegram.send_message(
                    f"⚙️ **Regime Change:** {label}\n{triggers}\n└ Risk Multiplier: {new_mult:.0%}"
                ))

        # Serialize recent_trades into lifetime_stats for persistence
        self.lifetime_stats['recent_trades'] = [
            {'r': t['r'], 'win': t['win'], 'time': t['time'].isoformat(), 'symbol': t['symbol']}
            for t in self.recent_trades
        ]
        self.lifetime_stats['regime_override'] = self.regime_override
        self.lifetime_stats['regime_override_trades'] = self.regime_override_trades
        self.save_lifetime_stats()

    # Legacy load/save methods removed in favor of StorageHandler

    def get_regime_status(self):
        """4-Tier Graduated regime detection with monitoring signals.

        Regime V2 — validated via simulate_regime_v2.py on 18,984 + 4,736 trades.
        Strategy #3 (4-Tier Graduated) won both datasets:
          - 1yr: R/DD 173.7, 13.3x more $ than current bot (daily compounding)
          - 6mo: R/DD 58.2, best final balance across all strategies

        Primary (drives multiplier):
          Trade quality tiers (20t window):
            favorable=1.0x, cautious=0.5x, adverse=0.25x, critical=0.1x

        Monitoring (dashboard display only, does NOT affect multiplier):
          DD from peak, daily R, loss streak

        Returns:
            (label, multiplier, diagnostics) where:
            - label: 'favorable'|'cautious'|'adverse'|'critical'|'unknown'
            - multiplier: 0.1 to 1.0
            - diagnostics: dict with quality + monitoring signals
        """
        diagnostics = {
            'quality': {'mult': 1.0, 'wr': 0.0, 'avg_r': 0.0, 'n_trades': 0},
            'drawdown': {'mult': 1.0, 'dd_from_peak': 0.0},
            'daily': {'mult': 1.0, 'daily_r': 0.0},
            'streak': {'mult': 1.0, 'loss_streak': 0},
        }

        # === PRIMARY: Trade quality tiers (20-trade window) ===
        n_trades = len(self.recent_trades)
        diagnostics['quality']['n_trades'] = n_trades
        if n_trades >= 10:
            trades = list(self.recent_trades)[-20:]
            wr = sum(1 for t in trades if t['win']) / len(trades)
            avg_r = sum(t['r'] for t in trades) / len(trades)
            diagnostics['quality']['wr'] = wr
            diagnostics['quality']['avg_r'] = avg_r
            if wr >= 0.18 and avg_r >= 0.15:
                diagnostics['quality']['mult'] = 1.0   # favorable
            elif wr >= 0.18 or avg_r >= 0.1:
                diagnostics['quality']['mult'] = 0.5   # cautious
            elif wr >= 0.10 or avg_r >= -0.5:
                diagnostics['quality']['mult'] = 0.25  # adverse
            else:
                diagnostics['quality']['mult'] = 0.1   # critical

        # === MONITORING ONLY (displayed on dashboard, does NOT affect multiplier) ===

        # Drawdown from peak (informational)
        peak = self.lifetime_stats.get('peak_equity_r', 0.0)
        current_equity = self.lifetime_stats.get('total_r', 0.0)
        dd = peak - current_equity  # positive = drawdown depth
        diagnostics['drawdown']['dd_from_peak'] = dd

        # Daily R (informational)
        today = datetime.now().strftime('%Y-%m-%d')
        daily_r = self.lifetime_stats.get('daily_r', {}).get(today, 0.0)
        diagnostics['daily']['daily_r'] = daily_r

        # Loss streak (informational)
        current_streak = self.lifetime_stats.get('current_streak', 0)
        loss_streak = abs(current_streak) if current_streak < 0 else 0
        diagnostics['streak']['loss_streak'] = loss_streak

        # --- Multiplier comes ONLY from quality tiers ---
        multiplier = diagnostics['quality']['mult']

        # Determine label from quality multiplier
        if n_trades < 10:
            label = 'unknown'
        elif multiplier <= 0.1:
            label = 'critical'
        elif multiplier <= 0.25:
            label = 'adverse'
        elif multiplier <= 0.5:
            label = 'cautious'
        else:
            label = 'favorable'

        # Manual override (from /setregime command)
        if self.regime_override is not None:
            override_map = {'favorable': 1.0, 'cautious': 0.5, 'adverse': 0.25, 'critical': 0.1}
            label = self.regime_override
            multiplier = override_map[label]
            # Keep diagnostics as-is so dashboard still shows underlying trade stats

        return label, multiplier, diagnostics

    def set_regime_override(self, regime: Optional[str]):
        """Set or clear manual regime override. Auto-clears after 10 trades."""
        if regime is not None and regime not in ('favorable', 'cautious', 'adverse', 'critical'):
            raise ValueError(f"Invalid regime: {regime}")
        self.regime_override = regime
        self.regime_override_trades = 0
        self.lifetime_stats['regime_override'] = regime
        self.lifetime_stats['regime_override_trades'] = 0
        self.save_lifetime_stats()

    def get_adaptive_risk(self):
        """Return risk_per_trade adjusted by 4-tier quality regime.

        Regime V2 (4-Tier Graduated): favorable=1.0x, cautious=0.5x,
        adverse=0.25x, critical=0.1x. Validated as best returns + R/DD.
        """
        base_risk = self.risk_config.get('risk_per_trade', 0.0005)
        label, multiplier, diagnostics = self.get_regime_status()
        if multiplier < 1.0:
            q = diagnostics['quality']
            logger.info(f"[RISK] Regime={label} (mult={multiplier:.2f}, WR={q['wr']:.0%}, avgR={q['avg_r']:+.2f}) — risk {base_risk:.6f} → {base_risk * multiplier:.6f}")
        return base_risk * multiplier

    def _track_divergence_detected(self):
        """Track a divergence detection"""
        # Reset daily counters if new day
        today = datetime.now().date()
        if self.bos_tracking['last_reset'] != today:
            self.bos_tracking['divergences_detected_today'] = 0
            self.bos_tracking['bos_confirmed_today'] = 0
            self.bos_tracking['last_reset'] = today
        
        self.bos_tracking['divergences_detected_today'] += 1
        self.bos_tracking['divergences_detected_total'] += 1
    
    def _track_bos_confirmed(self):
        """Track a BOS confirmation"""
        # Reset daily counters if new day
        today = datetime.now().date()
        if self.bos_tracking['last_reset'] != today:
            self.bos_tracking['divergences_detected_today'] = 0
            self.bos_tracking['bos_confirmed_today'] = 0
            self.bos_tracking['last_reset'] = today
        
        self.bos_tracking['bos_confirmed_today'] += 1
        self.bos_tracking['bos_confirmed_total'] += 1
    
    def get_signal_id(self, signal) -> str:
        """Generate unique ID using Pivot Timestamp (Backtest-Aligned Deduplication).
        
        Uses symbol + side + divergence_code + PIVOT timestamp to ensure
        multiple detections of the same setup are treated as one.
        """
        # Use pivot_timestamp for robust deduplication matching backtest
        # Fallback to timestamp if pivot_timestamp not available (legacy safety)
        ts = getattr(signal, 'pivot_timestamp', signal.timestamp)
        return f"{signal.symbol}_{signal.side}_{signal.divergence_code}_{ts.strftime('%Y%m%d_%H%M')}"
    
    def load_config(self):
        """Load configuration from config.yaml"""
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.strategy_config = self.config.get('strategy', {})
        self.risk_config = self.config.get('risk', {})
        self.timeframe = self.strategy_config.get('timeframe', '240')

        entry_params = self.strategy_config.get('entry_params', {})
        self.max_wait_candles = entry_params.get('max_wait_candles', 12)

        signal_params = self.strategy_config.get('signal_params', {})
        self.lookback_bars = signal_params.get('lookback_bars', 50)

        logger.info(f"Loaded config: Timeframe={self.timeframe}, Risk={self.risk_config.get('risk_per_trade', 0.01)*100}%, MaxWait={self.max_wait_candles}, Lookback={self.lookback_bars}")
    
    def setup_broker(self):
        """Initialize Bybit broker connection"""
        bybit_config_dict = self.config.get('bybit', {})
        
        # Expand environment variables
        api_key = os.path.expandvars(bybit_config_dict.get('api_key', ''))
        api_secret = os.path.expandvars(bybit_config_dict.get('api_secret', ''))
        
        bybit_config = BybitConfig(
            api_key=api_key,
            api_secret=api_secret,
            base_url=bybit_config_dict.get('base_url', 'https://api.bybit.com')
        )
        
        self.broker = Bybit(bybit_config)
        logger.info("Broker connection initialized")
    
    async def _apply_max_leverage(self):
        """
        Set maximum allowed leverage for all enabled symbols.
        This ensures optimal margin usage.
        """
        logger.info("🔧 CONFIGURING MAX LEVERAGE for all symbols...")
        enabled_symbols = self.symbol_config.get_enabled_symbols()
        
        count = 0
        for symbol in enabled_symbols:
            try:
                # set_leverage(None) defaults to max leverage in broker
                await self.broker.set_leverage(symbol)
                count += 1
                await asyncio.sleep(0.1)  # Avoid rate limits
            except Exception as e:
                logger.warning(f"[{symbol}] Could not set leverage: {e}")
        
        logger.info(f"✅ Max leverage configured for {count} symbols")

    
    def set_risk_per_trade(self, risk_pct: float):
        """
        Dynamically update risk per trade (percentage)
        
        Args:
            risk_pct: Risk percentage (e.g., 0.005 for 0.5%)
        """
        if 0.001 <= risk_pct <= 0.05:
            self.risk_config['risk_per_trade'] = risk_pct
            # Clear USD risk so percentage takes effect
            self.risk_config['risk_amount_usd'] = None
            logger.info(f"Risk updated to {risk_pct*100:.2f}%")
            return True, f"Risk updated to {risk_pct*100:.2f}%"
        else:
            return False, "Risk must be between 0.1% and 5.0%"
    
    def set_risk_usd(self, amount_usd: float):
        """
        Dynamically update risk per trade (fixed USD)
        
        Args:
            amount_usd: Fixed USD amount to risk per trade
        """
        if 0.1 <= amount_usd <= 1000:
            self.risk_config['risk_amount_usd'] = amount_usd
            logger.info(f"Risk updated to ${amount_usd:.2f} per trade")
            return True, f"Risk updated to ${amount_usd:.2f} per trade"
        else:
            return False, "USD risk must be between $0.1 and $1000"
    
    async def sync_with_exchange(self):
        """
        Sync internal active_trades with actual exchange positions.
        Removes stale trades that no longer exist on Bybit.
        """
        try:
            # Get actual positions from Bybit
            positions = await self.broker.get_positions()
            logger.info(f"[SYNC] Fetched {len(positions) if positions else 0} positions from Bybit")
            
            # Log details of what we got
            if positions:
                open_count = sum(1 for pos in positions if float(pos.get('size', 0)) > 0)
                logger.info(f"[SYNC] {open_count} positions with size > 0")
                for pos in positions[:5]:  # Log first 5
                    if float(pos.get('size', 0)) > 0:
                        logger.info(f"[SYNC] Found: {pos.get('symbol')} size={pos.get('size')} side={pos.get('side')}")
            else:
                logger.warning("[SYNC] get_positions() returned None or empty list!")
            
            # Build set of trade keys with actual open positions
            actual_open_keys = set()
            for pos in positions:
                if float(pos.get('size', 0)) > 0:
                    sym = pos.get('symbol')
                    side = "long" if pos.get('side') == "Buy" else "short"
                    actual_open_keys.add(f"{sym}_{side}")

            # Find stale trades (in our tracking but not on exchange)
            stale_trades = []
            for trade_key in list(self.active_trades.keys()):
                if trade_key not in actual_open_keys:
                    stale_trades.append(trade_key)

            # Remove stale trades and update stats with REAL PnL from exchange
            for trade_key in stale_trades:
                trade = self.active_trades.pop(trade_key, None)
                symbol = trade.symbol if trade else trade_key.rsplit('_', 1)[0]
                if trade:
                    logger.warning(f"[{symbol}] Removed stale trade from tracking (closed externally)")
                    
                    # Process exit via shared handler (same logic as normal exits)
                    await self.handle_trade_exit(trade_key, trade)
            
            # ADOPT EXISTING POSITIONS (On startup or manual intervention)
            adopted_count = 0
            for pos in positions:
                symbol = pos.get('symbol')
                size = float(pos.get('size', 0))
                side = "long" if pos.get('side') == "Buy" else "short"
                trade_key = f"{symbol}_{side}"

                if size > 0 and trade_key not in self.active_trades:
                    # Found an active position on exchange that bot doesn't know about
                    entry_price = float(pos.get('avgPrice', 0))
                    stop_loss = float(pos.get('stopLoss', 0))
                    take_profit = float(pos.get('takeProfit', 0))
                    entry_time = datetime.now()  # We don't know exact start, assume now

                    # Estimate R:R
                    rr_ratio = 0.0
                    if stop_loss > 0 and abs(entry_price - stop_loss) > 0:
                         risk = abs(entry_price - stop_loss)
                         reward = abs(take_profit - entry_price) if take_profit > 0 else 0
                         rr_ratio = round(reward / risk, 1)

                    # Create trade object
                    new_trade = ActiveTrade(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        rr_ratio=rr_ratio,
                        position_size=size,
                        entry_time=entry_time
                    )

                    self.active_trades[trade_key] = new_trade
                    adopted_count += 1
                    logger.info(f"[{symbol}] Adopted existing position: {side} {size} @ {entry_price}")
            
            if stale_trades or adopted_count > 0:
                logger.info(f"Synced with exchange: removed {len(stale_trades)} stale, adopted {adopted_count} existing trades")
            
            return len(stale_trades) + adopted_count
        except Exception as e:
            logger.error(f"Failed to sync with exchange: {e}")
            return 0
    
    def setup_telegram(self):
        """Initialize Telegram handler with commands"""
        telegram_config = self.config.get('telegram', {})
        
        bot_token = os.path.expandvars(telegram_config.get('token', ''))
        chat_id = os.path.expandvars(telegram_config.get('chat_id', ''))
        
        if bot_token and chat_id:
            self.telegram = TelegramHandler(bot_token, chat_id, self)
            logger.info("Telegram handler enabled (with commands)")
        else:
            self.telegram = None
            logger.warning("Telegram not configured - notifications disabled")
    
    async def fetch_4h_data(self, symbol: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Fetch candle data from Bybit (uses configured timeframe)

        Args:
            symbol: Trading pair
            limit: Number of candles to fetch (1000 for proper EMA 200 warmup)

        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            # Bybit klines API
            end_time = int(datetime.now().timestamp() * 1000)
            
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': self.timeframe,
                'limit': limit,
                'end': end_time
            }
            
            response = await self.broker._request('GET', '/v5/market/kline', params)
            
            if not response or 'result' not in response:
                logger.warning(f"[{symbol}] No data returned from API")
                return None
            
            klines = response['result'].get('list', [])
            
            if not klines:
                return None
            
            # Parse into DataFrame
            df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df.set_index('start', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"[{symbol}] Error fetching data: {e}")
            return None
    
    async def check_new_candle_close(self, symbol: str) -> bool:
        """
        Check if a new 1H candle has closed for a symbol
        
        Returns:
            True if new candle closed, False otherwise
        """
        now = datetime.now()
        
        # 1H candle closes at the top of every hour (00:00, 01:00, 02:00, etc.)
        current_hour = now.hour
        current_candle_close = now.replace(minute=0, second=0, microsecond=0)
        
        last_close = self.last_candle_close.get(symbol)
        
        if last_close is None or current_candle_close > last_close:
            self.last_candle_close[symbol] = current_candle_close
            return True
        
        return False
    
    async def process_symbol(self, symbol: str) -> int:
        """
        Process a single symbol for signals and trade execution
        
        Args:
            symbol: Trading pair to process
            
        Returns:
            int: Number of new signals found, or -1 if no new candle
        """
        # Check if new candle closed
        if not await self.check_new_candle_close(symbol):
            return -1
            
        # [STALENESS CHECK] Only skip truly stale candles (>55 min old)
        # Bypassed on first scan after startup so restart doesn't block all symbols.
        now = datetime.now()
        current_candle_close = now.replace(minute=0, second=0, microsecond=0)
        minutes_since_close = (now - current_candle_close).total_seconds() / 60

        if minutes_since_close > 55 and self.startup_scan_done:
            logger.info(f"[{symbol}] Skipping stale candle (Closed {minutes_since_close:.0f}m ago)")
            return 0
        
        # [REMOVED] First-candle skip
        # We now trust the 24h stale filter to prevent processing historical signals.
        # Immediate trading is now allowed.
        self.symbols_initialized.add(symbol)
        
        # NOTE: Divergence detection continues even if there's an active trade.
        # The block happens at BOS confirmation/trade entry stage instead.
        
        logger.info(f"[{symbol}] New 1H candle closed - processing...")

        # Fetch data
        df = await self.fetch_4h_data(symbol)
        if df is None or len(df) < 100:
            logger.warning(f"[{symbol}] Insufficient data")
            return 0

        # Prepare indicators
        df = prepare_dataframe(df)

        # [BACKTEST ALIGNMENT] Execute any queued entries from previous candle's BOS confirmation.
        # Entry uses the OPEN of the current (new) candle, matching the backtest exactly.
        # Check both long and short queues for this symbol
        for side in ['long', 'short']:
            trade_key = f"{symbol}_{side}"
            if trade_key in self.confirmed_entries:
                queued_signal = self.confirmed_entries.pop(trade_key)
                if trade_key not in self.active_trades:
                    logger.info(f"[{symbol}] Executing queued {side} entry at candle open (backtest-aligned)...")
                    await self.execute_trade(symbol, queued_signal, df, use_candle_open=True)
                else:
                    logger.info(f"[{symbol}] Queued {side} entry cancelled - already in {side} trade")
        
        # Cache latest RSI
        if 'rsi' in df.columns and len(df) > 0:
            last_rsi = df['rsi'].iloc[-1]
            self.rsi_cache[symbol] = last_rsi
            
            # === RADAR DETECTION (Developing Patterns) ===
            # specialized logic to see if we are 'close' to a divergence
            try:
                # 1. Get recent price extremes (last 20 candles)
                last_20 = df.iloc[-20:]
                low_20 = last_20['low'].min()
                high_20 = last_20['high'].max()
                current_close = df['close'].iloc[-1]
                
                # 2. Get trend
                ema = df['daily_ema'].iloc[-1] if 'daily_ema' in df.columns else 0
                ema_distance_pct = ((current_close - ema) / ema * 100) if ema > 0 else 0
                
                # 3. Look for previous pivot RSI to calculate divergence strength
                from autobot.core.divergence_detector import find_pivots
                close_arr = df['close'].values
                rsi_arr = df['rsi'].values
                _, price_lows = find_pivots(close_arr, 3, 3)
                price_highs, _ = find_pivots(close_arr, 3, 3)
                
                prev_pivot_rsi = None
                pivot_distance = 0
                
                # 4. Check Bullish Setup Forming
                # Price is near recent low (within 1%) BUT RSI is rising/higher
                if current_close < low_20 * 1.01 and current_close > ema:
                    # Find last pivot low RSI
                    # Match the main divergence detector logic: look back from -4 (after PIVOT_RIGHT=3)
                    for i in range(len(df) - 4, max(0, len(df) - 50), -1):
                        if not pd.isna(price_lows[i]):
                            prev_pivot_rsi = rsi_arr[i]
                            pivot_distance = len(df) - 1 - i
                            break
                    
                    rsi_divergence = last_rsi - prev_pivot_rsi if prev_pivot_rsi else 0
                    
                    # CRITICAL FIX: Only show as bullish if RSI is HIGHER than previous pivot
                    # Negative divergence = NOT a bullish divergence, don't show
                    if rsi_divergence > 0:
                        candles_since_potential = 0
                        pivot_progress = min(6, candles_since_potential + 3)
                        
                        self.radar_items[symbol] = {
                            'type': 'bullish_setup',
                            'price': current_close,
                            'rsi': last_rsi,
                            'prev_pivot_rsi': prev_pivot_rsi or 0,
                            'rsi_div': rsi_divergence,
                            'ema_dist': ema_distance_pct,
                            'pivot_progress': pivot_progress,
                            'pivot_distance': pivot_distance
                        }
                    elif symbol in self.radar_items and self.radar_items[symbol].get('type') == 'bullish_setup':
                        # Remove if no longer valid
                        del self.radar_items[symbol]
                
                # 5. Check Bearish Setup Forming
                # Price is near recent high (within 1%) BUT RSI is falling/lower
                elif current_close > high_20 * 0.99 and current_close < ema:
                    # Find last pivot high RSI
                    for i in range(len(df) - 4, max(0, len(df) - 50), -1):
                        if not pd.isna(price_highs[i]):
                            prev_pivot_rsi = rsi_arr[i]
                            pivot_distance = len(df) - 1 - i
                            break
                    
                    rsi_divergence = prev_pivot_rsi - last_rsi if prev_pivot_rsi else 0
                    
                    # CRITICAL FIX: Only show as bearish if RSI is LOWER than previous pivot
                    # Negative divergence = NOT a bearish divergence, don't show
                    if rsi_divergence > 0:
                        candles_since_potential = 0
                        pivot_progress = min(6, candles_since_potential + 3)
                        
                        self.radar_items[symbol] = {
                            'type': 'bearish_setup',
                            'price': current_close,
                            'rsi': last_rsi,
                            'prev_pivot_rsi': prev_pivot_rsi or 0,
                            'rsi_div': rsi_divergence,
                            'ema_dist': ema_distance_pct,
                            'pivot_progress': pivot_progress,
                            'pivot_distance': pivot_distance
                        }
                    elif symbol in self.radar_items and self.radar_items[symbol].get('type') == 'bearish_setup':
                        # Remove if no longer valid
                        del self.radar_items[symbol]
                
                # 6. RSI Extremes (Hot) - Track time in zone
                elif last_rsi <= 25:
                    # Track when it entered extreme zone
                    if symbol not in self.extreme_zone_tracker:
                        self.extreme_zone_tracker[symbol] = datetime.now()
                    
                    hours_in_zone = (datetime.now() - self.extreme_zone_tracker[symbol]).total_seconds() / 3600
                    
                    self.radar_items[symbol] = {
                        'type': 'extreme_oversold',
                        'price': current_close,
                        'rsi': last_rsi,
                        'hours_in_zone': hours_in_zone,
                        'ema_dist': ema_distance_pct
                    }
                    
                elif last_rsi >= 75:
                    # Track when it entered extreme zone
                    if symbol not in self.extreme_zone_tracker:
                        self.extreme_zone_tracker[symbol] = datetime.now()
                    
                    hours_in_zone = (datetime.now() - self.extreme_zone_tracker[symbol]).total_seconds() / 3600
                    
                    self.radar_items[symbol] = {
                        'type': 'extreme_overbought',
                        'price': current_close,
                        'rsi': last_rsi,
                        'hours_in_zone': hours_in_zone,
                        'ema_dist': ema_distance_pct
                    }
                
                # Remove if normal and clear tracker
                else:
                    if symbol in self.radar_items:
                        del self.radar_items[symbol]
                    if symbol in self.extreme_zone_tracker:
                        del self.extreme_zone_tracker[symbol]
                    
            except Exception as e:
                logger.debug(f"[{symbol}] Radar detection error: {e}")

        # 1. Detect new divergences
        # [MULTI-CONFIG] Pass ALL allowed divergence types for this symbol.
        # Supports both single-config and multi-config (long + short) per symbol.
        allowed_types = self.symbol_config.get_allowed_divergence_types(symbol)
        if not allowed_types:
            allowed_types = None  # Fallback: detect all types
        new_signals = detect_divergences(df, symbol, allowed_types=allowed_types, lookback_bars=self.lookback_bars)
        valid_signals_count = 0
        duplicate_count = 0
        
        for signal in new_signals:
            # Only accept trend-aligned signals
            if not signal.daily_trend_aligned:
                logger.debug(f"[{symbol}] {signal.divergence_code} divergence detected but NOT trend-aligned - SKIP")
                continue

            # [CRITICAL FIX] IGNORE HISTORY ON RESTART
            # Filter out signals that are too old to be relevant
            # 24 hours (time needed for valid signal) + buffer
            hours_since_signal = (df.index[-1] - signal.timestamp).total_seconds() / 3600
            if hours_since_signal > 24:
                # Silently skip old signals to avoid log spam on restart
                continue
            
            # DIVERGENCE TYPE FILTER: Only accept if this divergence type is allowed for this symbol
            if not self.symbol_config.is_divergence_allowed(symbol, signal.divergence_code):
                logger.debug(f"[{symbol}] {signal.divergence_code} detected but not in allowed types - SKIP")
                continue
            
            # DEDUPLICATION: Check if we've already seen this signal
            # On startup scan, allow re-detection of recent signals (within BOS wait window)
            # so pending_signals gets repopulated after restart
            signal_id = self.get_signal_id(signal)
            if self.storage.is_seen(signal_id):
                if self.startup_scan_done or hours_since_signal > self.max_wait_candles:
                    duplicate_count += 1
                    continue  # Already processed
                logger.info(f"[{symbol}] Re-detecting {signal.divergence_code} on startup (signal {hours_since_signal:.0f}h old, within BOS window)")
            
            # Mark signal as seen and save
            self.storage.add_signal(signal_id)
            
            # Add to pending signals
            pending = PendingSignal(signal=signal, detected_at=datetime.now(), max_wait_candles=self.max_wait_candles)
            
            if symbol not in self.pending_signals:
                self.pending_signals[symbol] = []
            
            self.pending_signals[symbol].append(pending)
            valid_signals_count += 1
            
            # Track divergence detection
            self._track_divergence_detected()
            
            logger.info(f"[{symbol}] 🔔 NEW {signal.divergence_code} divergence detected! Waiting for BOS...")
            logger.info(f"[{symbol}]   Price: ${signal.price:.2f}, RSI: {signal.rsi_value:.1f}, Swing: ${signal.swing_level:.2f}")
            logger.info(f"[{symbol}]   Signal ID: {signal_id}")
            
            # Send Telegram notification for divergence detected
            if self.telegram:
                div_emoji = get_signal_emoji(signal.divergence_code)
                div_name = signal.get_display_name()
                side_text = 'LONG' if signal.side == 'long' else 'SHORT'
                div_msg = f"""
🔔 **DIVERGENCE DETECTED**
━━━━━━━━━━━━━━━━━━━━

{div_emoji} **{symbol}** | {div_name}

**SIGNAL DETAILS**
├ Type: `{signal.divergence_code}`
├ Side: {side_text}
├ Price: ${signal.price:,.4f}
├ RSI: {signal.rsi_value:.1f}
├ Swing Level: ${signal.swing_level:,.4f}
└ Status: Waiting for BOS (0/12)

**ETA**
⏳ 0-12 hours to potential entry

━━━━━━━━━━━━━━━━━━━━
📊 [Chart](https://www.tradingview.com/chart/?symbol=BYBIT:{symbol})
"""
                try:
                    await self.telegram.send_message(div_msg)
                except Exception as e:
                    logger.error(f"Failed to send divergence notification: {e}")
        
        if valid_signals_count == 0:
            logger.info(f"[{symbol}] Scan complete - No new divergences (skipped {duplicate_count} duplicates)")
            
        # 2. Check pending signals for BOS
        await self.check_pending_bos(symbol, df)
        
        # 3. Update active trades
        await self.monitor_active_trades(symbol)
        
        return valid_signals_count
    
    async def check_pending_bos(self, symbol: str, df: pd.DataFrame):
        """
        Check if any pending signals have BOS confirmation
        
        Args:
            symbol: Trading pair
            df: Latest OHLCV data
        """
        if symbol not in self.pending_signals:
            return
        
        current_idx = len(df) - 1
        signals_to_remove = []
        
        for pending in self.pending_signals[symbol]:
            # Check if expired
            if pending.is_expired():
                logger.info(f"[{symbol}] Signal expired after {pending.candles_waited} candles - removing")
                signals_to_remove.append(pending)
                continue
            
            # Check for BOS
            if check_bos(df, pending.signal, current_idx):
                # [EMA GATE] Check EMA alignment at BOS confirmation time
                # EMA filter moved here from divergence detection to allow
                # divergences to queue and confirm when price crosses EMA
                if not is_trend_aligned(df, pending.signal, current_idx):
                    logger.info(f"[{symbol}] BOS confirmed but EMA not aligned ({pending.signal.side}) - skipping")
                    signals_to_remove.append(pending)
                    continue

                # [TRADE BLOCK CHECK] Block entry if already in a trade for this symbol+side
                trade_key = f"{symbol}_{pending.signal.side}"
                if trade_key in self.active_trades:
                    logger.info(f"[{symbol}] BOS confirmed but already in {pending.signal.side} trade - skipping entry")
                    signals_to_remove.append(pending)
                    continue

                # [BACKTEST ALIGNMENT] Queue entry for NEXT candle open instead of executing now.
                # This matches the backtest which enters at next candle open after BOS.
                logger.info(f"[{symbol}] ✅ BOS CONFIRMED! Queuing entry for next candle open...")

                # Send BOS confirmed notification
                if self.telegram:
                    div_emoji = get_signal_emoji(pending.signal.divergence_code)
                    div_name = pending.signal.get_display_name()
                    side_text = 'LONG' if pending.signal.side == 'long' else 'SHORT'
                    bos_msg = f"""
✅ **BOS CONFIRMED!**
━━━━━━━━━━━━━━━━━━━━

{div_emoji} **{symbol}** | {div_name}

🔓 Break of Structure confirmed after {pending.candles_waited} candles
├ Type: `{pending.signal.divergence_code}`
├ Side: {side_text}
⏳ Entry queued for next candle open...

━━━━━━━━━━━━━━━━━━━━
"""
                    try:
                        await self.telegram.send_message(bos_msg)
                    except Exception as e:
                         logger.error(f"Failed to send BOS notification: {e}")

                # Track BOS confirmation
                self._track_bos_confirmed()

                # Queue for next candle execution instead of immediate entry
                self.confirmed_entries[trade_key] = pending.signal
                signals_to_remove.append(pending)
            else:
                # Increment wait counter
                pending.increment_wait()
                logger.debug(f"[{symbol}] Waiting for BOS ({pending.candles_waited}/{pending.max_wait_candles})")
        
        # Remove signals
        for sig in signals_to_remove:
            self.pending_signals[symbol].remove(sig)
    
    async def execute_trade(self, symbol: str, signal: DivergenceSignal, df: pd.DataFrame, use_candle_open: bool = False):
        """
        Execute trade entry with TP/SL orders

        Args:
            symbol: Trading pair
            signal: Divergence signal
            df: Latest OHLCV data
            use_candle_open: If True, use current candle's open price for SL/TP calculation
                             (backtest-aligned entry on next candle after BOS)
        """
        # [REGIME V2] Halt check — block trade if multiplier is 0
        regime_label, regime_mult, regime_diag = self.get_regime_status()
        if regime_mult == 0.0:
            active_signals = [k for k, v in regime_diag.items() if v['mult'] <= 0.0]
            logger.warning(f"[{symbol}] TRADE BLOCKED — regime={regime_label}, halt signals: {active_signals}")
            if hasattr(self, 'telegram') and self.telegram:
                asyncio.create_task(self.telegram.send_message(
                    f"🛑 **TRADE BLOCKED** — {symbol} {signal.side}\n├ Regime: {regime_label}\n└ Halt signals: {', '.join(active_signals)}"
                ))
            return

        # Skip if already in a trade for this symbol+side (Internal Check)
        trade_key = f"{symbol}_{signal.side}"
        if trade_key in self.active_trades:
            logger.warning(f"[{symbol}] Already in a {signal.side} trade (internal) - skipping")
            return

        # [CRITICAL] Skip if already in a trade on EXCHANGE (prevents pyramiding after restart)
        try:
            # We fetch all positions to be sure. 
            # Optimization: could cache this or check specific symbol if API supports it,
            # but get_positions() is robust now.
            start_check = time.time()
            existing_positions = await self.broker.get_positions()
            for p in existing_positions:
                if p.get('symbol') == symbol and float(p.get('size', 0)) > 0:
                    pos_side = "long" if p.get('side') == "Buy" else "short"
                    if pos_side == signal.side:
                        logger.warning(f"[{symbol}] {signal.side} position already exists on EXCHANGE! Skipping to prevent pyramiding.")
                        # Sync internal state
                        self.active_trades[trade_key] = None
                        return
            logger.info(f"[{symbol}] No existing position found (took {time.time()-start_check:.2f}s). Proceeding...")
        except Exception as e:
            logger.error(f"[{symbol}] Failed to check existing positions: {e}")
            return # Safety fail-close
        
        # Get R:R for this specific signal's divergence type (multi-config support)
        rr = self.symbol_config.get_rr_for_symbol(symbol, signal.divergence_code)
        if rr is None:
            logger.error(f"[{symbol}] No R:R configured for {signal.divergence_code} - skipping")
            return
        
        # [BACKTEST ALIGNMENT] Use candle open for SL/TP calculation when entry is queued.
        # ATR comes from the PREVIOUS candle (the BOS candle), which is df.iloc[-2] when
        # use_candle_open=True (since df now includes the new candle).
        if use_candle_open:
            # Entry at current candle's open (matches backtest: next candle open after BOS)
            entry_price = df.iloc[-1]['open']
            atr = df.iloc[-2]['atr'] if len(df) >= 2 else df.iloc[-1]['atr']
            logger.info(f"[{symbol}] Using candle open price (backtest-aligned): ${entry_price:.6f}, ATR from prev candle: {atr:.6f}")
        else:
            # Fallback: use ticker price (legacy behavior)
            try:
                ticker = await self.broker.get_ticker(symbol)
                entry_price = float(ticker.get('lastPrice', df.iloc[-1]['close']))
                logger.info(f"[{symbol}] Using current ticker price: ${entry_price:.6f}")
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to get ticker, using candle close: {e}")
                entry_price = df.iloc[-1]['close']
            atr = df.iloc[-1]['atr']
        
        # SL distance - use PER-CONFIG atr_mult (multi-config: each div type has its own ATR mult)
        signal_cfg = self.symbol_config.get_config_for_divergence(symbol, signal.divergence_code)
        if signal_cfg:
            sl_mult = signal_cfg.get('atr_mult', 1.0)
        else:
            sl_mult = self.strategy_config.get('exit_params', {}).get('sl_atr_mult', 1.0)
        sl_distance = atr * sl_mult
        logger.info(f"[{symbol}] {signal.divergence_code} ATR mult: {sl_mult}x → SL distance: {sl_distance:.6f}")
        
        # Calculate position size based on RISK (Safety First)
        # Goal: If SL hits, loss = risk_amount
        try:
            # 1. Determine Risk Amount
            risk_amount_usd = self.risk_config.get('risk_amount_usd', None)
            account_balance = await self.broker.get_balance()
            
            if risk_amount_usd:
                risk_amount = float(risk_amount_usd)
            else:
                margin_pct = self.get_adaptive_risk()
                risk_amount = account_balance * margin_pct
            
            # 2. Calculate Qty based on Risk
            # Qty = Risk / SL_Distance
            if sl_distance <= 0:
                logger.error(f"[{symbol}] Invalid SL distance: {sl_distance}")
                return

            raw_qty = risk_amount / sl_distance
            
            # 3. Apply Max Leverage to minimize Margin Usage
            leverage = await self.broker.get_max_leverage(symbol)
            lev_result = await self.broker.set_leverage(symbol, leverage)
            # Re-read from cache in case set_leverage discovered a lower risk-limit max
            leverage = await self.broker.get_max_leverage(symbol)
            
            # 4. Check Margin Requirements
            position_value = raw_qty * entry_price
            required_margin = position_value / leverage
            
            # Get available balance
            available_balance = account_balance
            try:
                positions = await self.broker.get_positions()
                if positions:
                    total_margin_used = sum(float(p.get('positionIM', 0)) for p in positions if float(p.get('size', 0)) > 0)
                    available_balance = account_balance - total_margin_used
            except:
                pass

            # Skip if insufficient margin
            if required_margin > available_balance:
                logger.warning(f"[{symbol}] Insufficient margin for trade. Need ${required_margin:.2f}, Have ${available_balance:.2f} (Risk-Based Sizing)")
                return

            # 5. Round to Precision
            try:
                _, qty_step = await self.broker._get_precisions(symbol)
                from decimal import Decimal, ROUND_DOWN
                qty_step_dec = Decimal(qty_step)
                raw_qty_dec = Decimal(str(raw_qty))
                position_size_qty = float(raw_qty_dec.quantize(qty_step_dec, rounding=ROUND_DOWN))
                logger.info(f"[{symbol}] Risk-Based Qty: {raw_qty:.6f} → {position_size_qty} (Risk: ${risk_amount:.2f}, SL Dist: {sl_distance:.4f})")
            except Exception as e:
                logger.warning(f"[{symbol}] Precision fetch failed, using int(): {e}")
                position_size_qty = int(raw_qty)
            
            # Sanity check
            if position_size_qty <= 0:
                logger.error(f"[{symbol}] Invalid calculated qty: {position_size_qty} (Risk: ${risk_amount:.2f} too small for ATR {sl_distance:.4f})")
                return
            
        except Exception as e:
            logger.error(f"[{symbol}] Error calculating position size: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return
        
        # Calculate TP/SL prices
        if signal.side == 'long':
            sl_price = entry_price - sl_distance
            tp_price = entry_price + (sl_distance * rr)
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - (sl_distance * rr)
        
        # === VALIDATE SL IS ON CORRECT SIDE OF ENTRY ===
        # This catches edge cases where rapid price movement makes SL invalid
        if signal.side == 'long' and sl_price >= entry_price:
            logger.error(f"[{symbol}] Invalid SL for LONG: SL ${sl_price:.6f} >= Entry ${entry_price:.6f} (ATR too small or price moved)")
            return
        if signal.side == 'short' and sl_price <= entry_price:
            logger.error(f"[{symbol}] Invalid SL for SHORT: SL ${sl_price:.6f} <= Entry ${entry_price:.6f} (ATR too small or price moved)")
            return
        
        # === ATOMIC BRACKET ORDER: Market Entry + TP/SL ===
        # TP/SL set atomically with entry - position NEVER unprotected
        try:
            response = await self.broker.place_market(
                symbol=symbol,
                side=signal.side,
                qty=position_size_qty,
                take_profit=tp_price,  # Atomic TP protection
                stop_loss=sl_price     # Atomic SL protection
            )
            
            ret_code = response.get('retCode', -1) if isinstance(response, dict) else -1
            ret_msg = response.get('retMsg', 'Unknown') if isinstance(response, dict) else str(response)
            result = response.get('result', {}) if isinstance(response, dict) else {}
            order_id = result.get('orderId')
            
            if ret_code != 0 or not order_id:
                logger.error(f"[{symbol}] Failed to place bracket order: {ret_msg}")
                if self.telegram:
                    await self.telegram.send_message(f"❌ **TRADE FAILED**\\n\\n{symbol} | {ret_msg}\\nQty: {position_size_qty}\\nTP: ${tp_price:.4f}\\nSL: ${sl_price:.4f}")
                return
            
            actual_entry = float(result.get('avgPrice') or entry_price)
            logger.info(f"[{symbol}] ✅ BRACKET ORDER FILLED: {order_id}")
            logger.info(f"[{symbol}]   Entry: ${actual_entry:.4f} | TP: ${tp_price:.4f} | SL: ${sl_price:.4f}")
            
        except Exception as e:
            logger.error(f"[{symbol}] Error placing bracket order: {e}")
            if self.telegram:
                await self.telegram.send_message(f"❌ **TRADE FAILED**\n\n{symbol} | Error: {e}\nQty: {position_size_qty}")
            return
        
        # Track active trade
        trade = ActiveTrade(
            symbol=symbol,
            side=signal.side,
            entry_price=actual_entry,
            stop_loss=sl_price,
            take_profit=tp_price,
            rr_ratio=rr,
            position_size=position_size_qty,
            entry_time=datetime.now(),
            order_id=order_id,
            risk_usd_at_entry=risk_amount  # Store for accurate R calculation at closure
        )
        
        self.active_trades[trade_key] = trade

        # [BOT-BACKTEST ALIGNMENT - CHANGE 2] Clear pending signals for this side when trade opens
        if symbol in self.pending_signals:
            self.pending_signals[symbol] = [
                p for p in self.pending_signals[symbol]
                if p.signal.side != signal.side
            ]
        
        # Update stats
        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = {'trades': 0, 'wins': 0, 'total_r': 0.0}
        
        # Send enhanced notification
        await self.send_entry_notification(trade, signal)
        
        logger.info(f"[{symbol}] Trade opened: Entry=${actual_entry:.4f}, SL=${sl_price:.4f}, TP=${tp_price:.4f} ({rr}:1 R:R)")

    
    async def monitor_active_trades(self, symbol: str):
        """
        Monitor active trades for exits (checks both long and short for this symbol)

        Args:
            symbol: Trading pair
        """
        # Check both sides for this symbol
        for side in ['long', 'short']:
            trade_key = f"{symbol}_{side}"
            if trade_key not in self.active_trades:
                continue

            trade = self.active_trades[trade_key]

            # Check if position still open on exchange
            try:
                bybit_side = 'Buy' if side == 'long' else 'Sell'
                position = await self.broker.get_position(symbol, side=bybit_side)

                if position is None or position.get('size', 0) == 0:
                    # Position closed
                    await self.handle_trade_exit(trade_key, trade)

            except Exception as e:
                logger.error(f"[{symbol}] Error checking {side} position: {e}")
    
    async def handle_trade_exit(self, trade_key: str, trade: ActiveTrade):
        """
        Handle trade exit and update stats

        Args:
            trade_key: Trade key (symbol_side, e.g. "BTCUSDT_long")
            trade: Active trade object
        """
        symbol = trade.symbol if trade else trade_key.rsplit('_', 1)[0]
        logger.info(f"[{symbol}] Trade closed - processing exit...")
        
        # Get actual exit details from exchange (with retry for API latency)
        try:
            closed_pnl = None
            for attempt in range(3):
                closed_pnl = await self.broker.get_closed_pnl(symbol, limit=5)
                if closed_pnl:
                    break
                if attempt < 2:
                    logger.info(f"[{symbol}] Closed PnL not yet available, retry {attempt+1}/3...")
                    await asyncio.sleep(1.0)

            if closed_pnl:
                # Match record by side and entry price to avoid cross-contamination
                matched = None
                for record in closed_pnl:
                    rec_side = record.get('side', '').lower()
                    rec_entry = float(record.get('avgEntryPrice', 0))
                    # Bybit side: "Buy" = long, "Sell" = short
                    expected_side = 'buy' if trade.side == 'long' else 'sell'
                    entry_diff = abs(rec_entry - trade.entry_price) / trade.entry_price if trade.entry_price > 0 else 1
                    if rec_side == expected_side and entry_diff < 0.01:
                        matched = record
                        break

                if not matched:
                    logger.warning(f"[{symbol}] No matching closed PnL record (side={trade.side}, entry=${trade.entry_price:.4f}), using most recent")
                    matched = closed_pnl[0]

                pnl_usd = float(matched.get('closedPnl', 0))
                exit_price = float(matched.get('avgExitPrice', 0))

                # Calculate R value using stored risk from entry time
                if trade.risk_usd_at_entry > 0:
                    r_value = pnl_usd / trade.risk_usd_at_entry
                else:
                    # Fallback: price-based R for trades without stored risk
                    sl_distance = abs(trade.entry_price - trade.stop_loss)
                    if sl_distance > 0:
                        if trade.side == 'long':
                            price_diff = exit_price - trade.entry_price
                        else:
                            price_diff = trade.entry_price - exit_price
                        r_value = price_diff / sl_distance
                    else:
                        r_value = 0

                # Determine result
                result = 'WIN' if r_value > 0 else 'LOSS'

            else:
                # Fallback: use actual PnL direction, not hardcoded side assumption
                logger.warning(f"[{symbol}] No closed P&L found after 3 retries - using SL/TP estimates")
                # Estimate: assume SL hit (most common) unless position moved far enough for TP
                # Use -1R as conservative default for both longs AND shorts
                exit_price = trade.stop_loss
                r_value = -1.0
                result = 'LOSS'
                pnl_usd = 0

        except Exception as e:
            logger.error(f"[{symbol}] Error getting exit details: {e}")
            exit_price = 0
            r_value = 0
            result = 'UNKNOWN'
            pnl_usd = 0
        
        # Update stats
        self.stats['total_trades'] += 1
        self.stats['total_r'] += r_value

        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = {'trades': 0, 'wins': 0, 'total_r': 0.0}

        if result == 'WIN':
            self.stats['wins'] += 1
            self.symbol_stats[symbol]['wins'] += 1
        else:
            self.stats['losses'] += 1
        
        self.symbol_stats[symbol]['trades'] += 1
        self.symbol_stats[symbol]['total_r'] += r_value
        
        # Calculate new stats
        total = self.stats['wins'] + self.stats['losses']
        if total > 0:
            self.stats['win_rate'] = (self.stats['wins'] / total) * 100
            self.stats['avg_r'] = self.stats['total_r'] / total
        
        # Save session stats
        self.save_stats()
        
        # Update lifetime stats (persistent across restarts)
        is_win = result == 'WIN'
        self.update_lifetime_stats(r_value, pnl_usd, is_win, symbol)
        
        # Remove from active (may already be removed by sync)
        self.active_trades.pop(trade_key, None)

        # Calculate time held
        time_held = datetime.now() - trade.entry_time
        hours_held = time_held.total_seconds() / 3600

        # Send exit notification
        await self.send_exit_notification(
            symbol=symbol,
            trade=trade,
            exit_price=exit_price,
            result=result,
            r_value=r_value,
            pnl_usd=pnl_usd,
            hours_held=hours_held
        )

        logger.info(f"[{symbol}] Trade processed: {result}, {r_value:+.2f}R, held {hours_held:.1f}h")
    
    async def send_entry_notification(self, trade: ActiveTrade, signal: DivergenceSignal):
        """Send enhanced Telegram notification for trade entry"""
        if self.telegram:
            div_emoji = get_signal_emoji(signal.divergence_code)
            div_name = signal.get_display_name()
            direction = '🟢 LONG' if trade.side == 'long' else '🔴 SHORT'
            entry_time = trade.entry_time.strftime('%H:%M')
            
            # Calculate actual risk amount (use adaptive risk, not base)
            risk_usd = self.risk_config.get('risk_amount_usd')
            if risk_usd:
                risk_display = f"${float(risk_usd):.2f}"
            else:
                balance = await self.broker.get_balance() or 1000
                risk_pct = self.get_adaptive_risk()
                risk_display = f"${balance * risk_pct:.2f} ({risk_pct*100:.2f}%)"
            
            # Current active count
            active_count = len(self.active_trades)
            
            msg = f"""
🔔 **NEW TRADE OPENED**
━━━━━━━━━━━━━━━━━━━━

{div_emoji} **{trade.symbol}** | {direction}

⏰ {entry_time} | {div_name}

**ENTRY DETAILS**
├ 💵 Price: ${trade.entry_price:,.4f}
├ 📏 Size: {trade.position_size:.4f}
└ 💰 Risk: {risk_display}

**EXIT LEVELS**
├ 🎯 TP: ${trade.take_profit:,.4f} (+{trade.rr_ratio}R)
└ ⛔ SL: ${trade.stop_loss:,.4f} (-1R)

📊 Active Positions: {active_count}

━━━━━━━━━━━━━━━━━━━━
"""
            await self.telegram.send_message(msg)
        else:
            logger.info(f"[{trade.symbol}] Entry: ${trade.entry_price:.4f}, SL: ${trade.stop_loss:.4f}, TP: ${trade.take_profit:.4f}")



    async def send_exit_notification(self, symbol: str, trade: ActiveTrade, exit_price: float, 
                                     result: str, r_value: float, pnl_usd: float, hours_held: float):
        """Send enhanced exit notification"""
        if self.telegram:
            emoji = "✅" if result == "WIN" else "❌"
            direction = '🟢 LONG' if trade.side == 'long' else '🔴 SHORT'
            
            # Lifetime stats
            lifetime = self.lifetime_stats
            lifetime_r = lifetime.get('total_r', 0)
            lifetime_trades = lifetime.get('total_trades', 0)
            lifetime_wins = lifetime.get('wins', 0)
            lifetime_wr = (lifetime_wins / lifetime_trades * 100) if lifetime_trades > 0 else 0
            
            # Remaining positions (trade already removed from active_trades)
            remaining = len(self.active_trades)
            
            msg = f"""
{emoji} **TRADE CLOSED - {result}**
━━━━━━━━━━━━━━━━━━━━

📊 **{symbol}** | {direction}
⏱️ Held: {hours_held:.1f}h

**RESULT**
├ 💵 Entry: ${trade.entry_price:,.4f}
├ 💵 Exit: ${exit_price:,.4f}
└ {'🟢' if r_value > 0 else '🔴'} P&L: **{r_value:+.2f}R** (${pnl_usd:+.2f})

**LIFETIME STATS**
├ Total: {lifetime_r:+.1f}R ({lifetime_trades} trades)
├ Win Rate: {lifetime_wr:.1f}%
└ This Trade: {r_value:+.2f}R

📊 Remaining Positions: {max(0, remaining)}

━━━━━━━━━━━━━━━━━━━━
"""
            await self.telegram.send_message(msg)

    async def sync_positions_at_startup(self):
        """Fetch current open positions and sync internal state"""
        try:
            logger.info("[SYNC] Checking for existing positions on exchange...")
            positions = await self.broker.get_positions()
            count = 0
            for p in positions:
                if float(p.get('size', 0)) > 0:
                    sym = p.get('symbol')
                    side = "long" if p.get('side') == "Buy" else "short"
                    trade_key = f"{sym}_{side}"
                    entry_price = float(p.get('avgPrice', 0))
                    sl_price = float(p.get('stopLoss', 0))
                    tp_price = float(p.get('takeProfit', 0))
                    size = float(p.get('size', 0))
                    # Calculate approximate R:R from existing SL/TP
                    sl_dist = abs(entry_price - sl_price) if sl_price > 0 and entry_price > 0 else 0
                    rr = abs(tp_price - entry_price) / sl_dist if sl_dist > 0 and tp_price > 0 else 0
                    synced_trade = ActiveTrade(
                        symbol=sym,
                        side=side,
                        entry_price=entry_price,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        rr_ratio=round(rr, 1),
                        position_size=size,
                        entry_time=datetime.now()
                    )
                    self.active_trades[trade_key] = synced_trade
                    count += 1
                    logger.info(f"[SYNC] Found existing position: {sym} ({side}) entry=${entry_price:.4f} SL=${sl_price:.4f} TP=${tp_price:.4f}")
            logger.info(f"[SYNC] Synced {count} active positions.")
        except Exception as e:
            logger.error(f"[SYNC] Failed to sync positions: {e}")

    
    async def run(self):
        """Main bot loop"""
        if not self.broker.session:
             self.broker.session = aiohttp.ClientSession()

        # [NEW] Configure leverage at startup (SERIAL FETCH & SET)
        # This iterates enabled symbols one-by-one, gets max leverage, and SETS it on exchange
        # Ensures account is fully prepped and "Invalid Leverage" errors are handled early
        try:
            enabled_symbols = self.symbol_config.get_enabled_symbols()
            await self.broker.configure_active_symbols(enabled_symbols)
        except Exception as e:
            logger.error(f"Failed to configure leverage: {e}")

        # Start Telegram bot
        if self.telegram:
            await self.telegram.start()
            
        logger.info("============================================================")
        logger.info(f"🤖 1H TREND-DIVERGENCE BOT STARTED")
        logger.info("============================================================")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Enabled Symbols: {len(self.symbol_config.get_enabled_symbols())}")
        
        # Log risk setting clearly
        if self.risk_config.get('risk_amount_usd'):
             logger.info(f"Risk per trade: ${self.risk_config.get('risk_amount_usd')} (Fixed USD)")
        else:
             logger.info(f"Risk per trade: {self.risk_config.get('risk_per_trade', 0.01)*100}%")
             
        logger.info("============================================================")
        
        enabled_symbols = self.symbol_config.get_enabled_symbols()
        
        # Initialize Telegram handler (start command polling)
        if self.telegram:
            try:
                await self.telegram.start()
                
                # Send startup notification
                # Include lifetime stats if available
                lifetime = self.lifetime_stats
                lifetime_r = lifetime.get('total_r', 0)
                start_date = lifetime.get('start_date', 'Today')
                
                # Calculate risk display (use adaptive risk, not base)
                risk_pct = self.get_adaptive_risk()
                regime_label, regime_mult, _ = self.get_regime_status()
                regime_display = f" [{regime_label.upper()} {regime_mult:.0%}]" if regime_label != 'unknown' else ""

                msg = f"""
🤖 **BOT STARTED**
━━━━━━━━━━━━━━━━━━━━

📊 **Strategy**: 1H Multi-Div (Both Sides)
📈 **Symbols**: {len(enabled_symbols)} ({self.symbol_config.get_total_configs()} configs)
💰 **Risk**: {risk_pct*100:.2f}% per trade{regime_display}
🔬 **Mode**: Multi-config (long + short per symbol)

**LIFETIME STATS**
├ Total R: {lifetime_r:+.1f}R
└ Since: {start_date}

━━━━━━━━━━━━━━━━━━━━
💡 /dashboard /help
"""
                await self.telegram.send_message(msg)
            except Exception as e:
                logger.error(f"Telegram initialization failed: {e}")

        # [NEW] Sync existing positions to prevent pyramiding
        await self.sync_positions_at_startup()
        
        # Initialize lifetime stats (first run only)
        await self.initialize_lifetime_stats()

        # Prune old seen signals so recent divergences can be re-detected on startup
        self.storage.prune_old_signals(max_age_hours=48)

        # Start main loop
        last_check = 0
        last_auto_dashboard = datetime.now()
        
        while True:
            try:
                # Track cycle stats
                symbols_processed = 0
                total_signals_found = 0
                
                # Process each enabled symbol
                for symbol in enabled_symbols:
                    try:
                        signals = await self.process_symbol(symbol)
                        if signals >= 0:
                            symbols_processed += 1
                            total_signals_found += signals
                    except Exception as e:
                        logger.error(f"[{symbol}] Error processing: {e}")
                
                # If we completed a scan cycle (processed symbols implies candle close)
                if symbols_processed > 0:
                    if not self.startup_scan_done:
                        self.startup_scan_done = True
                        logger.info(f"[STARTUP] First scan complete — startup mode disabled")

                    # Update scan state for dashboard
                    self.scan_state['last_scan_time'] = datetime.now()
                    self.scan_state['symbols_scanned'] = symbols_processed
                    self.scan_state['fresh_divergences'] = total_signals_found

                    logger.info(f"✅ Hourly Scan Complete. Processed: {symbols_processed}, New Signals: {total_signals_found}")

                    if self.telegram:
                        # Build regime snippet for scan message
                        regime_snippet = ""
                        try:
                            regime_label, regime_mult, regime_diag = self.get_regime_status()
                            n_trades = len(self.recent_trades)
                            regime_icons = {
                                'favorable': '🟢', 'cautious': '🟡', 'adverse': '🟠',
                                'critical': '🔴', 'halted': '🛑', 'unknown': '⏳',
                            }
                            icon = regime_icons.get(regime_label, '❓')
                            if regime_label == 'unknown':
                                regime_snippet = f"├ Regime: Building data... ({n_trades}/10 trades)"
                            else:
                                q = regime_diag['quality']
                                regime_snippet = f"├ Regime: {regime_label.title()} {icon} ({regime_mult:.0%} risk) | {q['wr']:.0%} WR, {q['avg_r']:+.2f}R"
                        except Exception as e:
                            logger.error(f"Regime snippet error: {e}")

                        pending_count = sum(len(sigs) for sigs in self.pending_signals.values())
                        active_count = len(self.active_trades)

                        if total_signals_found == 0:
                            msg = f"""
🕵️ **HOURLY SCAN COMPLETE**

📊 Checked: {symbols_processed} Symbols
🔍 Result: **No New Divergences Found**

**Current Status:**
{regime_snippet}
├ ⏳ Pending BOS: {pending_count}
└ 📈 Active Trades: {active_count}

Next scan in ~60 mins ⏳
"""
                        else:
                            msg = f"""
🕵️ **HOURLY SCAN COMPLETE**

📊 Checked: {symbols_processed} Symbols
🔍 Result: **{total_signals_found} New Signal(s) Found!**

**Current Status:**
{regime_snippet}
├ ⏳ Pending BOS: {pending_count}
└ 📈 Active Trades: {active_count}

Next scan in ~60 mins ⏳
"""
                        await self.telegram.send_message(msg)

                        # Auto-send full dashboard every ~1 hour
                        elapsed = (datetime.now() - last_auto_dashboard).total_seconds()
                        if elapsed >= 3300:  # ~55 min guard to prevent double-firing
                            try:
                                dashboard_msg = await self.telegram.build_dashboard_message()
                                await self.telegram.send_message(dashboard_msg)
                                last_auto_dashboard = datetime.now()
                                logger.info("📊 Auto-dashboard sent")
                            except Exception as e:
                                logger.error(f"Auto-dashboard error: {e}")
                
                # Sleep for 1 minute before next check
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(60)


async def main():
    """Entry point"""
    bot = Bot4H()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
