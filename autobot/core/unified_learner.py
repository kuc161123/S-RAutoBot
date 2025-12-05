#!/usr/bin/env python3
"""
Unified Learning System v4.0

A single comprehensive learning system that combines:
- Basic W/L tracking per symbol/side/combo
- Session breakdown (Asian/London/NY)
- Volatility regime filtering (high/medium/low)
- BTC trend correlation
- Adaptive R:R optimization per symbol
- Auto-promote to YAML when profitable
- Blacklist for consistently losing combos
- Self-explanatory decision logging

All data stored in one file: unified_learning.json
"""

import json
import time
import math
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    """
    Calculate lower bound of Wilson score confidence interval.
    More reliable than raw win rate for small sample sizes.
    z=1.96 gives 95% confidence.
    """
    if total == 0:
        return 0.0
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    lower = (centre - spread) / denominator
    return max(0, lower * 100)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LearningSignal:
    """Complete signal with all context for learning"""
    symbol: str
    side: str
    combo: str
    entry_price: float
    tp_price: float
    sl_price: float
    start_time: float
    
    # Is this a phantom (not in allowed combos) or active trade?
    is_phantom: bool = True
    is_allowed_combo: bool = False
    
    # Market context
    atr_percent: float = 0.0          # ATR as % of price
    volatility_regime: str = "medium"  # 'high', 'medium', 'low'
    btc_trend: str = "neutral"        # 'bullish', 'bearish', 'neutral'
    btc_change_1h: float = 0.0        # BTC 1-hour change %
    session: str = "london"           # 'asian', 'london', 'newyork'
    hour_utc: int = 0
    
    # R:R used
    rr_ratio: float = 2.0
    
    # Accurate tracking (like phantom system - uses candle high/low)
    max_high: float = 0.0             # Max high seen since entry
    min_low: float = 0.0              # Min low seen since entry
    max_favorable: float = 0.0        # Max move towards TP (%)
    max_adverse: float = 0.0          # Max drawdown (%)
    
    # Outcome
    outcome: Optional[str] = None
    end_time: Optional[float] = None
    time_to_result: float = 0.0


# ============================================================================
# UNIFIED LEARNING SYSTEM
# ============================================================================

class UnifiedLearner:
    """
    Single comprehensive learning system for all trading intelligence.
    
    Features:
    - Tracks all signals with full market context
    - Learns optimal R:R per symbol/combo
    - Identifies best volatility regimes
    - Correlates with BTC trend
    - Auto-promotes profitable combos
    - Blacklists losing combos
    - Provides session breakdown
    """
    
    SAVE_FILE = 'unified_learning.json'
    BLACKLIST_FILE = 'combo_blacklist.json'
    OVERRIDE_FILE = 'symbol_overrides_VWAP_Combo.yaml'
    
    # Thresholds
    SIGNAL_TIMEOUT = 14400  # 4 hours
    MAX_PENDING = 2000
    
    # Volatility regimes (ATR as % of price)
    VOL_HIGH = 2.0
    VOL_LOW = 0.5
    
    # R:R options
    RR_OPTIONS = [1.5, 2.0, 2.5, 3.0]
    
    # Auto-promote thresholds
    PROMOTE_MIN_TRADES = 20
    PROMOTE_MIN_LOWER_WR = 45.0
    PROMOTE_MIN_EV = 0.3
    
    # Blacklist thresholds
    BLACKLIST_MIN_TRADES = 10
    BLACKLIST_MAX_LOWER_WR = 30.0
    
    # Minimum data for adaptive decisions
    MIN_TRADES_FOR_RR = 15
    MIN_TRADES_FOR_REGIME = 10
    
    def __init__(self, on_resolve_callback=None):
        """
        Initialize learner.
        
        on_resolve_callback: Optional async function(signal, outcome, time_mins, max_dd) 
                            called when a signal resolves - use for Telegram notifications
        """
        self.on_resolve_callback = on_resolve_callback
        
        # Combo stats: symbol -> side -> combo -> stats
        self.combo_stats: Dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
            'total': 0,
            'wins': 0,
            'losses': 0,
            # Session breakdown
            'sessions': {
                'asian': {'w': 0, 'l': 0},
                'london': {'w': 0, 'l': 0},
                'newyork': {'w': 0, 'l': 0}
            },
            # Volatility breakdown
            'by_regime': {
                'high': {'w': 0, 'l': 0},
                'medium': {'w': 0, 'l': 0},
                'low': {'w': 0, 'l': 0}
            },
            # BTC trend breakdown
            'by_btc': {
                'bullish': {'w': 0, 'l': 0},
                'bearish': {'w': 0, 'l': 0},
                'neutral': {'w': 0, 'l': 0}
            },
            # R:R performance
            'by_rr': {
                1.5: {'w': 0, 'l': 0},
                2.0: {'w': 0, 'l': 0},
                2.5: {'w': 0, 'l': 0},
                3.0: {'w': 0, 'l': 0}
            },
            # Time tracking
            'total_time_wins': 0,
            'total_time_losses': 0,
            'total_drawdown': 0
        })))
        
        # Pending signals
        self.pending_signals: List[LearningSignal] = []
        
        # Blacklist and promoted sets
        self.blacklist: Set[str] = set()
        self.promoted: Set[str] = set()
        
        # BTC tracking
        self.btc_price_1h_ago: float = 0
        self.btc_current: float = 0
        self.btc_last_update: float = 0
        
        # Global counters
        self.total_signals = 0
        self.total_wins = 0
        self.total_losses = 0
        self.started_at = time.time()
        
        # Auto-adjustments log
        self.adjustments: List[Dict] = []
        
        # Decision log (last 50)
        self.decision_log: List[Dict] = []
        
        # Load saved data
        self.load()
        self.load_blacklist()
    
    # ========================================================================
    # CONTEXT HELPERS
    # ========================================================================
    
    def get_session(self, hour_utc: int = None) -> str:
        """Determine trading session from UTC hour"""
        if hour_utc is None:
            hour_utc = datetime.utcnow().hour
        
        if 0 <= hour_utc < 8:
            return 'asian'
        elif 8 <= hour_utc < 16:
            return 'london'
        else:
            return 'newyork'
    
    def get_volatility_regime(self, atr_percent: float) -> str:
        """Classify volatility regime"""
        if atr_percent >= self.VOL_HIGH:
            return 'high'
        elif atr_percent <= self.VOL_LOW:
            return 'low'
        return 'medium'
    
    def get_btc_trend(self, btc_change: float = None) -> str:
        """Classify BTC trend"""
        if btc_change is None:
            btc_change = self.get_btc_change_1h()
        
        if btc_change > 0.5:
            return 'bullish'
        elif btc_change < -0.5:
            return 'bearish'
        return 'neutral'
    
    def update_btc_price(self, current_price: float):
        """Update BTC price tracking"""
        now = time.time()
        
        if now - self.btc_last_update > 3600 or self.btc_price_1h_ago == 0:
            self.btc_price_1h_ago = self.btc_current if self.btc_current > 0 else current_price
            self.btc_last_update = now
        
        self.btc_current = current_price
    
    def get_btc_change_1h(self) -> float:
        """Get BTC 1-hour change percentage"""
        if self.btc_price_1h_ago == 0 or self.btc_current == 0:
            return 0.0
        return ((self.btc_current - self.btc_price_1h_ago) / self.btc_price_1h_ago) * 100
    
    def is_blacklisted(self, symbol: str, side: str, combo: str) -> bool:
        """Check if combo is blacklisted"""
        key = f"{symbol}:{side}:{combo}"
        return key in self.blacklist
    
    # ========================================================================
    # ADAPTIVE R:R
    # ========================================================================
    
    def get_optimal_rr(self, symbol: str, side: str, combo: str) -> Tuple[float, str]:
        """
        Get optimal R:R for this setup based on historical data.
        Returns (rr_ratio, explanation)
        """
        stats = self.combo_stats[symbol][side][combo]
        
        if stats['total'] < self.MIN_TRADES_FOR_RR:
            return 2.0, f"Default 2:1 (only {stats['total']}/{self.MIN_TRADES_FOR_RR} trades)"
        
        best_rr = 2.0
        best_ev = -999
        
        for rr in self.RR_OPTIONS:
            rr_key = rr if isinstance(rr, float) else float(rr)
            rr_stats = stats['by_rr'].get(rr_key, {'w': 0, 'l': 0})
            total = rr_stats['w'] + rr_stats['l']
            
            if total < 5:
                continue
            
            wr = rr_stats['w'] / total
            ev = (wr * rr) - ((1 - wr) * 1)
            
            if ev > best_ev:
                best_ev = ev
                best_rr = rr_key
        
        if best_ev > 0:
            return best_rr, f"Optimal {best_rr}:1 (EV={best_ev:.2f}R)"
        
        return 2.0, "Default 2:1 (no positive EV found)"
    
    # ========================================================================
    # SMART FILTERING
    # ========================================================================
    
    def get_regime_filter(self, symbol: str, side: str, combo: str) -> Tuple[List[str], str]:
        """Get which volatility regimes work for this setup"""
        stats = self.combo_stats[symbol][side][combo]
        
        if stats['total'] < self.MIN_TRADES_FOR_REGIME:
            return ['high', 'medium', 'low'], "All regimes (not enough data)"
        
        allowed = []
        parts = []
        
        for regime in ['high', 'medium', 'low']:
            regime_stats = stats['by_regime'].get(regime, {'w': 0, 'l': 0})
            total = regime_stats['w'] + regime_stats['l']
            
            if total < 3:
                allowed.append(regime)
                continue
            
            lb_wr = wilson_lower_bound(regime_stats['w'], total)
            
            if lb_wr >= 35:
                allowed.append(regime)
                parts.append(f"{regime}:{lb_wr:.0f}%")
            else:
                parts.append(f"❌{regime}:{lb_wr:.0f}%")
        
        if not allowed:
            allowed = ['medium']
        
        return allowed, ", ".join(parts) if parts else "All regimes"
    
    def get_btc_filter(self, symbol: str, side: str, combo: str) -> Tuple[List[str], str]:
        """Get which BTC trends work for this setup"""
        stats = self.combo_stats[symbol][side][combo]
        
        if stats['total'] < self.MIN_TRADES_FOR_REGIME:
            return ['bullish', 'bearish', 'neutral'], "All BTC (not enough data)"
        
        allowed = []
        
        for trend in ['bullish', 'bearish', 'neutral']:
            trend_stats = stats['by_btc'].get(trend, {'w': 0, 'l': 0})
            total = trend_stats['w'] + trend_stats['l']
            
            if total < 3:
                allowed.append(trend)
                continue
            
            lb_wr = wilson_lower_bound(trend_stats['w'], total)
            if lb_wr >= 30:
                allowed.append(trend)
        
        if not allowed:
            allowed = ['neutral']
        
        return allowed, f"BTC: {', '.join(allowed)}"
    
    def should_take_signal(self, symbol: str, side: str, combo: str,
                          atr_percent: float, btc_change: float = None) -> Tuple[bool, str]:
        """
        Smart decision: Should we take this signal?
        Returns (should_take, explanation)
        """
        # Check blacklist
        if self.is_blacklisted(symbol, side, combo):
            return False, "❌ BLACKLISTED"
        
        regime = self.get_volatility_regime(atr_percent)
        btc_trend = self.get_btc_trend(btc_change)
        
        # Check regime filter
        allowed_regimes, regime_exp = self.get_regime_filter(symbol, side, combo)
        if regime not in allowed_regimes:
            return False, f"❌ Regime {regime} blocked | {regime_exp}"
        
        # Check BTC filter
        allowed_btc, btc_exp = self.get_btc_filter(symbol, side, combo)
        if btc_trend not in allowed_btc:
            return False, f"❌ BTC {btc_trend} blocked | {btc_exp}"
        
        # Check overall performance
        stats = self.combo_stats[symbol][side][combo]
        if stats['total'] >= 10:
            lb_wr = wilson_lower_bound(stats['wins'], stats['total'])
            if lb_wr < 35:
                return False, f"❌ LB_WR {lb_wr:.0f}% < 35%"
        
        # Get R:R info
        optimal_rr, rr_exp = self.get_optimal_rr(symbol, side, combo)
        
        return True, f"✅ {regime} | BTC:{btc_trend} | {rr_exp}"
    
    # ========================================================================
    # SIGNAL RECORDING (Accuracy-focused)
    # ========================================================================
    
    def record_signal(self, symbol: str, side: str, combo: str,
                     entry: float, atr: float, btc_price: float = 0,
                     is_allowed: bool = False) -> Tuple[float, float, str]:
        """
        Record a signal with full context and return optimal TP/SL.
        
        ACCURACY MEASURES:
        1. Strict input validation
        2. Deduplication by symbol+side (different combos allowed)
        3. Proper TP/SL calculation with validation
        4. Full context capture at entry time
        
        Returns (tp_price, sl_price, explanation)
        """
        # === STRICT INPUT VALIDATION ===
        if not symbol or not side or not combo:
            return None, None, "Invalid input: missing symbol/side/combo"
        
        if entry <= 0:
            return None, None, f"Invalid entry price: {entry}"
        
        if atr <= 0:
            return None, None, f"Invalid ATR: {atr}"
        
        if side not in ['long', 'short']:
            return None, None, f"Invalid side: {side}"
        
        # Skip if blacklisted
        if self.is_blacklisted(symbol, side, combo):
            return None, None, "Blacklisted"
        
        # Update BTC
        if btc_price > 0:
            self.update_btc_price(btc_price)
        
        # === DEDUPLICATION: Only one signal per symbol+side at a time ===
        # (This prevents duplicate tracking of same trade)
        existing = any(s.symbol == symbol and s.side == side 
                      for s in self.pending_signals)
        if existing:
            return None, None, "Already tracking this symbol+side"
        
        # Cleanup old signals if too many
        if len(self.pending_signals) > self.MAX_PENDING:
            self._cleanup_old_signals()
        
        # === CAPTURE CONTEXT AT ENTRY TIME ===
        now = datetime.utcnow()
        entry_timestamp = time.time()
        hour_utc = now.hour
        session = self.get_session(hour_utc)
        atr_percent = (atr / entry) * 100
        regime = self.get_volatility_regime(atr_percent)
        btc_change = self.get_btc_change_1h()
        btc_trend = self.get_btc_trend(btc_change)
        
        # === GET OPTIMAL R:R ===
        optimal_rr, rr_explanation = self.get_optimal_rr(symbol, side, combo)
        
        # === CALCULATE TP/SL WITH VALIDATION ===
        # Using 1 ATR for SL, optimal_rr * ATR for TP
        if side == 'long':
            sl = entry - atr
            tp = entry + (optimal_rr * atr)
            # Validate: SL must be below entry, TP must be above entry
            if sl >= entry or tp <= entry:
                return None, None, f"Invalid TP/SL calculation for long"
        else:
            sl = entry + atr
            tp = entry - (optimal_rr * atr)
            # Validate: SL must be above entry, TP must be below entry
            if sl <= entry or tp >= entry:
                return None, None, f"Invalid TP/SL calculation for short"
        
        # === CREATE SIGNAL WITH FULL CONTEXT ===
        signal = LearningSignal(
            symbol=symbol,
            side=side,
            combo=combo,
            entry_price=entry,
            tp_price=tp,
            sl_price=sl,
            start_time=entry_timestamp,
            is_phantom=not is_allowed,
            is_allowed_combo=is_allowed,
            atr_percent=atr_percent,
            volatility_regime=regime,
            btc_trend=btc_trend,
            btc_change_1h=btc_change,
            session=session,
            hour_utc=hour_utc,
            rr_ratio=optimal_rr,
            max_high=entry,    # Initialize with entry
            min_low=entry      # Initialize with entry
        )
        
        self.pending_signals.append(signal)
        self.total_signals += 1
        
        # Log for debugging
        logger.debug(
            f"📝 RECORDED: {symbol} {side} | Entry:{entry:.4f} TP:{tp:.4f} SL:{sl:.4f} | "
            f"R:R={optimal_rr}:1 | {regime} | BTC:{btc_trend}"
        )
        
        explanation = f"R:R={optimal_rr}:1 | {regime} | BTC:{btc_trend}"
        return tp, sl, explanation
    
    def _cleanup_old_signals(self):
        """Remove timed-out signals"""
        now = time.time()
        self.pending_signals = [
            s for s in self.pending_signals 
            if now - s.start_time < self.SIGNAL_TIMEOUT
        ]
    
    # ========================================================================
    # SIGNAL RESOLUTION (Accurate like phantom system)
    # ========================================================================
    
    def update_signals(self, candle_data: Dict[str, Dict]):
        """
        Update pending signals with candle data (high/low) for accurate resolution.
        
        candle_data format: {symbol: {'high': float, 'low': float, 'close': float}}
        
        This uses the same accurate method as the phantom system:
        - Longs: SL hit if low <= SL, TP hit if high >= TP
        - Shorts: SL hit if high >= SL, TP hit if low <= TP
        
        CRITICAL: SL is checked BEFORE TP (if both hit in same candle, it's a loss)
        """
        now = time.time()
        resolved_count = 0
        timeout_count = 0
        
        for signal in self.pending_signals[:]:
            if signal.outcome is not None:
                continue
            
            candle = candle_data.get(signal.symbol)
            if not candle:
                continue
            
            high = candle.get('high', 0)
            low = candle.get('low', 0)
            
            # === STRICT VALIDATION ===
            if high <= 0 or low <= 0:
                continue
            if high < low:  # Invalid candle data
                logger.warning(f"Invalid candle for {signal.symbol}: high={high} < low={low}")
                continue
            
            # === UPDATE MAX/MIN TRACKING ===
            # Use entry price as initial reference if not set
            if signal.max_high == signal.entry_price and signal.min_low == signal.entry_price:
                signal.max_high = max(signal.entry_price, high)
                signal.min_low = min(signal.entry_price, low)
            else:
                signal.max_high = max(signal.max_high, high)
                signal.min_low = min(signal.min_low, low)
            
            # === TRACK MAX FAVORABLE/ADVERSE MOVES ===
            if signal.side == 'long':
                # Favorable = price went up (high above entry)
                favorable_pct = max(0, (signal.max_high - signal.entry_price) / signal.entry_price * 100)
                # Adverse = price went down (low below entry)
                adverse_pct = max(0, (signal.entry_price - signal.min_low) / signal.entry_price * 100)
            else:
                # Favorable = price went down (low below entry)
                favorable_pct = max(0, (signal.entry_price - signal.min_low) / signal.entry_price * 100)
                # Adverse = price went up (high above entry)
                adverse_pct = max(0, (signal.max_high - signal.entry_price) / signal.entry_price * 100)
            
            signal.max_favorable = max(signal.max_favorable, favorable_pct)
            signal.max_adverse = max(signal.max_adverse, adverse_pct)
            
            # === CHECK TIMEOUT (4 hours) ===
            age_hours = (now - signal.start_time) / 3600
            if age_hours > 4:
                timeout_count += 1
                self.pending_signals.remove(signal)
                logger.debug(f"⏰ TIMEOUT: {signal.symbol} {signal.side} after {age_hours:.1f}h")
                continue
            
            # === CHECK OUTCOME USING HIGH/LOW ===
            # CRITICAL: Check SL FIRST - if both SL and TP hit in same candle, it's a LOSS
            outcome = None
            
            if signal.side == 'long':
                # Long trade: SL is below entry, TP is above entry
                sl_hit = low <= signal.sl_price
                tp_hit = high >= signal.tp_price
                
                if sl_hit:
                    outcome = 'loss'  # SL hit (even if TP also hit)
                elif tp_hit:
                    outcome = 'win'
                    
            else:  # short
                # Short trade: SL is above entry, TP is below entry
                sl_hit = high >= signal.sl_price
                tp_hit = low <= signal.tp_price
                
                if sl_hit:
                    outcome = 'loss'  # SL hit (even if TP also hit)
                elif tp_hit:
                    outcome = 'win'
            
            if outcome:
                resolved_count += 1
                self._resolve_signal(signal, outcome)
                self.pending_signals.remove(signal)
        
        if resolved_count > 0 or timeout_count > 0:
            logger.info(f"📊 Update: Resolved={resolved_count} Timeout={timeout_count} Pending={len(self.pending_signals)}")
    
    def _resolve_signal(self, signal: LearningSignal, outcome: str):
        """Resolve signal and update all stats"""
        signal.outcome = outcome
        signal.end_time = time.time()
        signal.time_to_result = signal.end_time - signal.start_time
        
        # Update global counters
        if outcome == 'win':
            self.total_wins += 1
        else:
            self.total_losses += 1
        
        # Update combo stats
        stats = self.combo_stats[signal.symbol][signal.side][signal.combo]
        stats['total'] += 1
        
        if outcome == 'win':
            stats['wins'] += 1
            stats['total_time_wins'] += signal.time_to_result
        else:
            stats['losses'] += 1
            stats['total_time_losses'] += signal.time_to_result
        
        stats['total_drawdown'] += signal.max_adverse
        
        # Update session stats
        session = signal.session
        if session in stats['sessions']:
            stats['sessions'][session]['w' if outcome == 'win' else 'l'] += 1
        
        # Update regime stats
        regime = signal.volatility_regime
        if regime in stats['by_regime']:
            stats['by_regime'][regime]['w' if outcome == 'win' else 'l'] += 1
        
        # Update BTC stats
        btc = signal.btc_trend
        if btc in stats['by_btc']:
            stats['by_btc'][btc]['w' if outcome == 'win' else 'l'] += 1
        
        # Update R:R stats
        rr = signal.rr_ratio
        if rr in stats['by_rr']:
            stats['by_rr'][rr]['w' if outcome == 'win' else 'l'] += 1
        
        # Check for auto-promote and blacklist
        self._check_promote(signal.symbol, signal.side, signal.combo)
        self._check_blacklist(signal.symbol, signal.side, signal.combo)
        
        # Store resolved signal data for notification
        time_mins = signal.time_to_result / 60
        max_dd = signal.max_adverse
        
        # Add to last_resolved (caller can use for notifications)
        if not hasattr(self, 'last_resolved'):
            self.last_resolved = []
        self.last_resolved.append({
            'symbol': signal.symbol,
            'side': signal.side,
            'combo': signal.combo,
            'outcome': outcome,
            'time_mins': round(time_mins, 1),
            'max_dd': round(max_dd, 2),
            'is_phantom': signal.is_phantom
        })
        
        logger.info(
            f"📊 {signal.symbol} {signal.side} {outcome.upper()} | "
            f"{time_mins:.0f}m | DD:{max_dd:.1f}%"
        )
    
    # ========================================================================
    # AUTO-PROMOTE
    # ========================================================================
    
    def _check_promote(self, symbol: str, side: str, combo: str):
        """Check if combo should be auto-promoted"""
        key = f"{symbol}:{side}:{combo}"
        if key in self.promoted:
            return
        
        stats = self.combo_stats[symbol][side][combo]
        if stats['total'] < self.PROMOTE_MIN_TRADES:
            return
        
        lb_wr = wilson_lower_bound(stats['wins'], stats['total'])
        if lb_wr < self.PROMOTE_MIN_LOWER_WR:
            return
        
        # Calculate EV (assuming current best R:R)
        optimal_rr, _ = self.get_optimal_rr(symbol, side, combo)
        raw_wr = stats['wins'] / stats['total']
        ev = (raw_wr * optimal_rr) - ((1 - raw_wr) * 1)
        
        if ev < self.PROMOTE_MIN_EV:
            return
        
        # Auto-promote!
        self._promote_combo(symbol, side, combo)
        self.promoted.add(key)
        
        self.adjustments.append({
            'time': time.time(),
            'type': 'PROMOTE',
            'symbol': symbol,
            'side': side,
            'combo': combo,
            'lb_wr': lb_wr,
            'ev': ev,
            'trades': stats['total']
        })
        
        logger.info(f"🚀 AUTO-PROMOTE: {symbol} {side} | LB_WR={lb_wr:.0f}% EV={ev:.2f}R")
    
    def _promote_combo(self, symbol: str, side: str, combo: str):
        """Add combo to override YAML"""
        try:
            import yaml
            
            try:
                with open(self.OVERRIDE_FILE, 'r') as f:
                    overrides = yaml.safe_load(f) or {}
            except FileNotFoundError:
                overrides = {}
            
            if symbol not in overrides:
                overrides[symbol] = {'long': [], 'short': []}
            
            if side not in overrides[symbol]:
                overrides[symbol][side] = []
            
            if combo not in overrides[symbol][side]:
                overrides[symbol][side].append(combo)
                
                with open(self.OVERRIDE_FILE, 'w') as f:
                    yaml.dump(overrides, f, default_flow_style=False)
                
                # Git commit
                try:
                    subprocess.run(['git', 'add', self.OVERRIDE_FILE], 
                                  capture_output=True, timeout=10)
                    subprocess.run(['git', 'commit', '-m', f'Auto-promote: {symbol} {side}'],
                                  capture_output=True, timeout=10)
                    subprocess.run(['git', 'push'], capture_output=True, timeout=30)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Promote error: {e}")
    
    # ========================================================================
    # BLACKLIST
    # ========================================================================
    
    def _check_blacklist(self, symbol: str, side: str, combo: str):
        """Check if combo should be blacklisted"""
        key = f"{symbol}:{side}:{combo}"
        if key in self.blacklist:
            return
        
        stats = self.combo_stats[symbol][side][combo]
        if stats['total'] < self.BLACKLIST_MIN_TRADES:
            return
        
        lb_wr = wilson_lower_bound(stats['wins'], stats['total'])
        if lb_wr <= self.BLACKLIST_MAX_LOWER_WR:
            self.blacklist.add(key)
            self.save_blacklist()
            
            self.adjustments.append({
                'time': time.time(),
                'type': 'BLACKLIST',
                'symbol': symbol,
                'side': side,
                'combo': combo,
                'lb_wr': lb_wr,
                'trades': stats['total']
            })
            
            logger.info(f"🚫 BLACKLIST: {symbol} {side} | LB_WR={lb_wr:.0f}%")
    
    def save_blacklist(self):
        """Save blacklist to file"""
        try:
            with open(self.BLACKLIST_FILE, 'w') as f:
                json.dump(list(self.blacklist), f)
        except Exception as e:
            logger.error(f"Failed to save blacklist: {e}")
    
    def load_blacklist(self):
        """Load blacklist from file"""
        try:
            with open(self.BLACKLIST_FILE, 'r') as f:
                self.blacklist = set(json.load(f))
            logger.info(f"🚫 Loaded {len(self.blacklist)} blacklisted combos")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Failed to load blacklist: {e}")
    
    # ========================================================================
    # REPORTS
    # ========================================================================
    
    def get_all_combos(self) -> List[Dict]:
        """Get all tracked combos with full stats"""
        result = []
        
        for symbol, sides in self.combo_stats.items():
            for side, combos in sides.items():
                for combo, stats in combos.items():
                    if stats['total'] == 0:
                        continue
                    
                    raw_wr = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    lb_wr = wilson_lower_bound(stats['wins'], stats['total'])
                    
                    optimal_rr, _ = self.get_optimal_rr(symbol, side, combo)
                    ev = (stats['wins']/stats['total'] * optimal_rr) - ((stats['losses']/stats['total']) * 1) if stats['total'] > 0 else 0
                    
                    result.append({
                        'symbol': symbol,
                        'side': side,
                        'combo': combo,
                        'total': stats['total'],
                        'wins': stats['wins'],
                        'losses': stats['losses'],
                        'raw_wr': raw_wr,
                        'lower_wr': lb_wr,
                        'optimal_rr': optimal_rr,
                        'ev': ev,
                        'sessions': stats['sessions'],
                        'regimes': stats['by_regime'],
                        'btc': stats['by_btc']
                    })
        
        return sorted(result, key=lambda x: x['lower_wr'], reverse=True)
    
    def get_top_combos(self, min_trades: int = 5, min_lower_wr: float = 40) -> List[Dict]:
        """Get top performing combos"""
        all_combos = self.get_all_combos()
        return [c for c in all_combos if c['total'] >= min_trades and c['lower_wr'] >= min_lower_wr]
    
    def get_promote_candidates(self) -> List[Dict]:
        """Get combos ready for promotion"""
        all_combos = self.get_all_combos()
        return [
            c for c in all_combos 
            if c['total'] >= 10 and c['lower_wr'] >= 40 and c['ev'] > 0
            and f"{c['symbol']}:{c['side']}:{c['combo']}" not in self.promoted
        ]
    
    def generate_report(self) -> str:
        """Generate comprehensive Telegram report"""
        uptime = (time.time() - self.started_at) / 3600
        total = self.total_wins + self.total_losses
        wr = (self.total_wins / total * 100) if total > 0 else 0
        lb_wr = wilson_lower_bound(self.total_wins, total)
        
        all_combos = self.get_all_combos()
        top = self.get_top_combos(min_trades=5, min_lower_wr=45)[:5]
        
        report = (
            "📚 **UNIFIED LEARNING REPORT**\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            f"⏱️ Running: {uptime:.1f}h\n"
            f"📊 Signals: {self.total_signals}\n"
            f"📈 Resolved: {total} ({self.total_wins}W/{self.total_losses}L)\n"
            f"🎯 WR: {wr:.0f}% (LB: **{lb_wr:.0f}%**)\n"
            f"⏳ Pending: {len(self.pending_signals)}\n"
            f"🔢 Combos: {len(all_combos)}\n"
            f"🚀 Promoted: {len(self.promoted)}\n"
            f"🚫 Blacklisted: {len(self.blacklist)}\n\n"
        )
        
        if top:
            report += "🏆 **TOP PERFORMERS**\n"
            for c in top:
                report += f"├ `{c['symbol']}` {c['side'][0].upper()}: {c['lower_wr']:.0f}% (N={c['total']})\n"
            report += "\n"
        
        # BTC context
        btc_trend = self.get_btc_trend()
        report += f"₿ BTC: {btc_trend} ({self.get_btc_change_1h():+.1f}%)"
        
        return report
    
    def get_session_report(self) -> str:
        """Generate session performance report"""
        sessions = {'asian': {'w': 0, 'l': 0}, 'london': {'w': 0, 'l': 0}, 'newyork': {'w': 0, 'l': 0}}
        
        for symbol, sides in self.combo_stats.items():
            for side, combos in sides.items():
                for combo, stats in combos.items():
                    for session, data in stats.get('sessions', {}).items():
                        if session in sessions:
                            sessions[session]['w'] += data.get('w', 0)
                            sessions[session]['l'] += data.get('l', 0)
        
        report = "🌍 **SESSION PERFORMANCE**\n━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for session, data in sessions.items():
            total = data['w'] + data['l']
            wr = (data['w'] / total * 100) if total > 0 else 0
            lb_wr = wilson_lower_bound(data['w'], total)
            icon = {'asian': '🌏', 'london': '🌍', 'newyork': '🌎'}.get(session, '🌐')
            report += f"{icon} **{session.upper()}**\n"
            report += f"   {data['w']}W/{data['l']}L | WR: {wr:.0f}% (LB: {lb_wr:.0f}%)\n\n"
        
        return report
    
    def get_smart_report(self) -> str:
        """Generate smart learning report with adaptive info"""
        uptime = (time.time() - self.started_at) / 3600
        total = self.total_wins + self.total_losses
        wr = (self.total_wins / total * 100) if total > 0 else 0
        lb_wr = wilson_lower_bound(self.total_wins, total)
        
        btc_trend = self.get_btc_trend()
        btc_change = self.get_btc_change_1h()
        
        report = (
            "🧠 **SMART LEARNING**\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            f"⏱️ Running: {uptime:.1f}h\n"
            f"📊 Signals: {self.total_signals}\n"
            f"📈 WR: {wr:.0f}% (LB: **{lb_wr:.0f}%**)\n"
            f"⏳ Pending: {len(self.pending_signals)}\n\n"
            f"₿ **BTC Context**\n"
            f"├ Trend: {btc_trend}\n"
            f"└ 1h Change: {btc_change:+.2f}%\n\n"
        )
        
        # Recent adjustments
        if self.adjustments:
            report += "🔧 **RECENT ACTIVITY**\n"
            for adj in self.adjustments[-5:]:
                adj_type = adj.get('type', 'ADJUST')
                symbol = adj.get('symbol', '?')
                side = adj.get('side', '?')
                if adj_type == 'PROMOTE':
                    report += f"├ 🚀 `{symbol}` {side}\n"
                elif adj_type == 'BLACKLIST':
                    report += f"├ 🚫 `{symbol}` {side}\n"
                else:
                    report += f"├ 🔧 `{symbol}` {side}\n"
        else:
            report += "🔧 No adjustments yet\n"
        
        report += "\n━━━━━━━━━━━━━━━━━━━━\n"
        report += "💡 Self-adjusting R:R & filters"
        
        return report
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save(self):
        """Save all learning data including pending signals for restart survival"""
        try:
            # Convert combo_stats to serializable format
            stats_data = {}
            for symbol, sides in self.combo_stats.items():
                stats_data[symbol] = {}
                for side, combos in sides.items():
                    stats_data[symbol][side] = {}
                    for combo, data in combos.items():
                        # Convert float keys to strings for JSON
                        by_rr = {str(k): v for k, v in data.get('by_rr', {}).items()}
                        stats_data[symbol][side][combo] = {**data, 'by_rr': by_rr}
            
            # Convert pending signals to serializable format (CRITICAL for restart survival)
            pending_data = []
            for sig in self.pending_signals:
                pending_data.append({
                    'symbol': sig.symbol,
                    'side': sig.side,
                    'combo': sig.combo,
                    'entry_price': sig.entry_price,
                    'tp_price': sig.tp_price,
                    'sl_price': sig.sl_price,
                    'start_time': sig.start_time,
                    'is_phantom': sig.is_phantom,
                    'is_allowed_combo': sig.is_allowed_combo,
                    'atr_percent': sig.atr_percent,
                    'volatility_regime': sig.volatility_regime,
                    'btc_trend': sig.btc_trend,
                    'btc_change_1h': sig.btc_change_1h,
                    'session': sig.session,
                    'hour_utc': sig.hour_utc,
                    'rr_ratio': sig.rr_ratio,
                    'max_high': sig.max_high,
                    'min_low': sig.min_low,
                    'max_favorable': sig.max_favorable,
                    'max_adverse': sig.max_adverse
                })
            
            data = {
                'combo_stats': stats_data,
                'pending_signals': pending_data,  # CRITICAL: Save pending for restart
                'promoted': list(self.promoted),
                'blacklist': list(self.blacklist),  # Also save blacklist
                'total_signals': self.total_signals,
                'total_wins': self.total_wins,
                'total_losses': self.total_losses,
                'btc_price_1h_ago': self.btc_price_1h_ago,
                'btc_current': self.btc_current,
                'started_at': self.started_at,
                'adjustments': self.adjustments[-100:],
                'saved_at': time.time()
            }
            
            with open(self.SAVE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"💾 Learning saved: {len(pending_data)} pending, {self.total_signals} total signals")
            
        except Exception as e:
            logger.error(f"Failed to save learning: {e}")
    
    def load(self):
        """Load learning data including pending signals for restart survival"""
        try:
            with open(self.SAVE_FILE, 'r') as f:
                data = json.load(f)
            
            # Load combo stats
            for symbol, sides in data.get('combo_stats', {}).items():
                for side, combos in sides.items():
                    for combo, stats in combos.items():
                        # Convert string keys back to floats for by_rr
                        if 'by_rr' in stats:
                            stats['by_rr'] = {float(k): v for k, v in stats['by_rr'].items()}
                        self.combo_stats[symbol][side][combo] = stats
            
            # Load pending signals (CRITICAL for restart survival)
            self.pending_signals = []
            for sig_data in data.get('pending_signals', []):
                try:
                    # Skip signals older than 4 hours (SIGNAL_TIMEOUT)
                    if time.time() - sig_data.get('start_time', 0) > self.SIGNAL_TIMEOUT:
                        continue
                    
                    signal = LearningSignal(
                        symbol=sig_data['symbol'],
                        side=sig_data['side'],
                        combo=sig_data['combo'],
                        entry_price=sig_data['entry_price'],
                        tp_price=sig_data['tp_price'],
                        sl_price=sig_data['sl_price'],
                        start_time=sig_data['start_time'],
                        is_phantom=sig_data.get('is_phantom', True),
                        is_allowed_combo=sig_data.get('is_allowed_combo', False),
                        atr_percent=sig_data.get('atr_percent', 0.0),
                        volatility_regime=sig_data.get('volatility_regime', 'medium'),
                        btc_trend=sig_data.get('btc_trend', 'neutral'),
                        btc_change_1h=sig_data.get('btc_change_1h', 0.0),
                        session=sig_data.get('session', 'london'),
                        hour_utc=sig_data.get('hour_utc', 0),
                        rr_ratio=sig_data.get('rr_ratio', 2.0),
                        max_high=sig_data.get('max_high', sig_data['entry_price']),
                        min_low=sig_data.get('min_low', sig_data['entry_price']),
                        max_favorable=sig_data.get('max_favorable', 0.0),
                        max_adverse=sig_data.get('max_adverse', 0.0)
                    )
                    self.pending_signals.append(signal)
                except Exception as e:
                    logger.debug(f"Failed to restore signal: {e}")
                    continue
            
            self.promoted = set(data.get('promoted', []))
            self.blacklist = set(data.get('blacklist', []))  # Also load blacklist
            self.total_signals = data.get('total_signals', 0)
            self.total_wins = data.get('total_wins', 0)
            self.total_losses = data.get('total_losses', 0)
            self.btc_price_1h_ago = data.get('btc_price_1h_ago', 0)
            self.btc_current = data.get('btc_current', 0)
            self.started_at = data.get('started_at', time.time())
            self.adjustments = data.get('adjustments', [])
            
            saved_at = data.get('saved_at', 0)
            age_mins = (time.time() - saved_at) / 60 if saved_at else 0
            logger.info(f"📂 Learning loaded: {len(self.pending_signals)} pending, {self.total_signals} total (saved {age_mins:.0f}m ago)")
            
        except FileNotFoundError:
            logger.info("📂 Starting fresh learning database")
        except Exception as e:
            logger.error(f"Failed to load learning: {e}")

