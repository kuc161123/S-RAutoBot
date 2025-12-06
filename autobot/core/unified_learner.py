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
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from datetime import datetime

# Persistence imports
try:
    import redis
except ImportError:
    redis = None

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
except ImportError:
    psycopg2 = None

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
    
    # Data directory: Use /data if exists (Railway), otherwise current dir
    DATA_DIR = '/data' if os.path.isdir('/data') else (
        '/app/data' if os.path.isdir('/app/data') else '.'
    )
    
    # Create data dir if needed (for local dev)
    if DATA_DIR not in ['/', '.'] and not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
        except:
            DATA_DIR = '.'
    
    SAVE_FILE = f'{DATA_DIR}/unified_learning.json'
    BLACKLIST_FILE = f'{DATA_DIR}/combo_blacklist.json'
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
            # Day of week performance
            'by_day': {
                'monday': {'w': 0, 'l': 0},
                'tuesday': {'w': 0, 'l': 0},
                'wednesday': {'w': 0, 'l': 0},
                'thursday': {'w': 0, 'l': 0},
                'friday': {'w': 0, 'l': 0},
                'saturday': {'w': 0, 'l': 0},
                'sunday': {'w': 0, 'l': 0}
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
        
        # Log data directory for debugging
        logger.info(f"📁 Data directory: {self.DATA_DIR}")
        logger.info(f"📁 Save file: {self.SAVE_FILE}")
        
        # Initialize Dual Persistence (Redis + Postgres)
        self.redis_client = None
        self.pg_conn = None
        self._init_persistence()
        
        # Load saved data
        self.load()
        self.load_blacklist()
    
    def _init_persistence(self):
        """Initialize Redis and Postgres connections"""
        # Redis for pending signals (fast, ephemeral)
        redis_url = os.getenv('REDIS_URL')
        if not redis:
            logger.warning("⚠️ Redis module not found. Install 'redis' package.")
        elif not redis_url:
            logger.warning("⚠️ REDIS_URL env var not set. Redis persistence disabled.")
        else:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("✅ Redis connected for pending signals")
            except Exception as e:
                logger.error(f"❌ Redis connection failed: {e}")
        
        # Postgres for historical stats (robust, queryable)
        pg_url = os.getenv('DATABASE_URL')
        if not psycopg2:
            logger.warning("⚠️ psycopg2 module not found. Install 'psycopg2-binary'.")
        elif not pg_url:
            logger.warning("⚠️ DATABASE_URL env var not set. Postgres persistence disabled.")
        else:
            try:
                self.pg_conn = psycopg2.connect(pg_url)
                self.pg_conn.autocommit = True
                self._init_db_tables()
                logger.info("✅ PostgreSQL connected for historical stats")
            except Exception as e:
                logger.error(f"❌ PostgreSQL connection failed: {e}")

    def _init_db_tables(self):
        """Create necessary tables in Postgres"""
        if not self.pg_conn:
            return
            
        try:
            with self.pg_conn.cursor() as cur:
                # Combo Stats Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS combo_stats (
                        symbol VARCHAR(20),
                        side VARCHAR(10),
                        combo VARCHAR(50),
                        total INT DEFAULT 0,
                        wins INT DEFAULT 0,
                        losses INT DEFAULT 0,
                        data JSONB,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (symbol, side, combo)
                    );
                """)
                
                # Adjustments Log Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS adjustments (
                        id SERIAL PRIMARY KEY,
                        type VARCHAR(20),
                        symbol VARCHAR(20),
                        side VARCHAR(10),
                        combo VARCHAR(50),
                        details JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Trade History Table (for time-based relevance)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trade_history (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20),
                        side VARCHAR(10),
                        combo VARCHAR(50),
                        outcome VARCHAR(10),
                        time_to_result FLOAT,
                        max_r_reached FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
        except Exception as e:
            logger.error(f"Failed to init DB tables: {e}")
    
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
                     is_allowed: bool = False, notify: bool = True) -> Tuple[float, float, str]:
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
            return None, None, "Missing required fields"
        
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
            sl_price = entry - atr
            tp_price = entry + (optimal_rr * atr)
            # Validate: SL must be below entry, TP must be above entry
            if sl_price <= 0 or tp_price <= 0:
                return None, None, f"Invalid TP/SL calculation for long"
        else:
            sl_price = entry + atr
            tp_price = entry - (optimal_rr * atr)
            # Validate: SL must be above entry, TP must be below entry
            if sl_price <= 0 or tp_price <= 0:
                return None, None, f"Invalid TP/SL calculation for short"
        
        # === CREATE SIGNAL WITH FULL CONTEXT ===
        signal = LearningSignal(
            symbol=symbol,
            side=side,
            combo=combo,
            entry_price=entry,
            tp_price=tp_price,
            sl_price=sl_price,
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
        
        # === NOTIFICATION ===
        # Send Telegram notification (if allowed by rate limiter)
        if notify:
            self._notify_new_signal(signal)
        
        self.pending_signals.append(signal)
        self.total_signals += 1
        
        # Log for debugging
        logger.debug(
            f"📝 RECORDED: {symbol} {side} | Entry:{entry:.4f} TP:{tp_price:.4f} SL:{sl_price:.4f} | "
            f"R:R={optimal_rr}:1 | {regime} | BTC:{btc_trend}"
        )
        
        explanation = f"R:R={optimal_rr}:1 | {regime} | BTC:{btc_trend}"
        return tp_price, sl_price, explanation
    
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
        
        # Update day-of-week stats
        from datetime import datetime
        day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day = day_names[datetime.now().weekday()]
        if 'by_day' not in stats:
            stats['by_day'] = {d: {'w': 0, 'l': 0} for d in day_names}
        if day in stats['by_day']:
            stats['by_day'][day]['w' if outcome == 'win' else 'l'] += 1
        
        # Update R:R stats (Counterfactual Analysis)
        # We calculate "what if" for all R:R options based on max favorable excursion
        entry = signal.entry_price
        sl = signal.sl_price
        risk = abs(entry - sl)
        
        if risk > 0:
            # Calculate max profit distance reached before resolution
            if signal.side == 'long':
                max_profit = signal.max_high - entry
            else:
                max_profit = entry - signal.min_low
            
            max_r_reached = max_profit / risk
            
            for rr_opt in [1.5, 2.0, 2.5, 3.0]:
                # Initialize if missing
                if rr_opt not in stats['by_rr']:
                    stats['by_rr'][rr_opt] = {'w': 0, 'l': 0}
                
                # Logic:
                # 1. If we reached the target R, it's a WIN
                # 2. If we didn't reach it AND the trade is resolved (hit SL or lower TP), it's a LOSS
                #    (Note: If we won at 2R, we don't strictly know if 3R would win, but 
                #     we assume LOSS for conservative stats unless max_r_reached >= 3.0)
                
                if max_r_reached >= rr_opt:
                    stats['by_rr'][rr_opt]['w'] += 1
                else:
                    stats['by_rr'][rr_opt]['l'] += 1
        
        # Check for auto-promote and blacklist
        self._check_promote(signal.symbol, signal.side, signal.combo)
        self._check_blacklist(signal.symbol, signal.side, signal.combo)
        
        # Store resolved signal data for notification
        time_mins = signal.time_to_result / 60
        max_dd = signal.max_adverse
        
        # Save to Postgres History (for time-based relevance)
        if self.pg_conn:
            try:
                with self.pg_conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO trade_history 
                        (symbol, side, combo, outcome, time_to_result, max_r_reached)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        signal.symbol, signal.side, signal.combo, outcome,
                        signal.time_to_result, max_r_reached
                    ))
            except Exception as e:
                logger.error(f"Failed to save trade history: {e}")
        
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
        """Check if combo should be auto-promoted (using last 30 days)"""
        key = f"{symbol}:{side}:{combo}"
        if key in self.promoted:
            return
        
        # Use RECENT stats (last 30 days) for promotion
        stats = self.get_recent_stats(symbol, side, combo, days=30)
        
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
        logger.info(f"🚀 PROMOTED {key} based on 30d stats: {stats['wins']}W/{stats['losses']}L")
    
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

    def get_auto_activate_candidates(self, min_wr: float = 40.0, min_trades: int = 5, days: int = 30) -> List[Dict]:
        """Get combos that meet strict criteria for auto-activation.
        
        NOW QUERIES trade_history DIRECTLY (same source as /analytics).
        """
        candidates = []
        
        if not self.pg_conn:
            logger.warning("No Postgres connection for auto-activate")
            return candidates
        
        try:
            with self.pg_conn.cursor() as cur:
                # Query aggregated stats from trade_history (last N days)
                cur.execute("""
                    SELECT symbol, side, combo,
                           COUNT(*) as total,
                           SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins
                    FROM trade_history
                    WHERE created_at > NOW() - INTERVAL '%s days'
                    GROUP BY symbol, side, combo
                    HAVING COUNT(*) >= %s
                """, (days, min_trades))
                
                rows = cur.fetchall()
                
                for symbol, side, combo, total, wins in rows:
                    # Calculate Lower Bound WR (Wilson score)
                    lb_wr = wilson_lower_bound(wins, total)
                    raw_wr = (wins / total * 100) if total > 0 else 0
                    
                    # Check if meets threshold and not already promoted
                    key = f"{symbol}:{side}:{combo}"
                    if lb_wr >= min_wr and key not in self.promoted:
                        candidates.append({
                            'symbol': symbol,
                            'side': side,
                            'combo': combo,
                            'total': total,
                            'wins': wins,
                            'wr': raw_wr,
                            'lower_wr': lb_wr
                        })
                        
        except Exception as e:
            logger.error(f"Auto-activate query failed: {e}")
        
        # Sort by lower_wr descending
        return sorted(candidates, key=lambda x: x['lower_wr'], reverse=True)

    def activate_combo(self, symbol: str, side: str, combo: str):
        """Mark a combo as promoted/active"""
        key = f"{symbol}:{side}:{combo}"
        if key not in self.promoted:
            self.promoted.add(key)
            self.save()
            return True
        return False
    
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
                report += f"├ {side_icon} `{c['symbol']}` {combo_short}\n"
                report += f"│  WR:{c['lower_wr']:.0f}% | EV:{ev_str} | {c['optimal_rr']}:1 | {best_session} (N={c['total']})\n"
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
        """Save all learning data using dual persistence (Redis + Postgres) with JSON fallback"""
        try:
            # 1. Prepare Data
            stats_data = {}
            for symbol, sides in self.combo_stats.items():
                stats_data[symbol] = {}
                for side, combos in sides.items():
                    stats_data[symbol][side] = {}
                    for combo, data in combos.items():
                        by_rr = {str(k): v for k, v in data.get('by_rr', {}).items()}
                        stats_data[symbol][side][combo] = {**data, 'by_rr': by_rr}
            
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

            # 2. Try Redis (Pending Signals)
            if self.redis_client:
                try:
                    self._save_to_redis(pending_data)
                except Exception as e:
                    logger.error(f"Redis save failed: {e}")
            
            # 3. Try Postgres (Historical Stats)
            if self.pg_conn:
                try:
                    self._save_to_postgres(stats_data)
                except Exception as e:
                    logger.error(f"Postgres save failed: {e}")

            # 4. Always save to JSON as backup/fallback
            data = {
                'combo_stats': stats_data,
                'pending_signals': pending_data,
                'promoted': list(self.promoted),
                'blacklist': list(self.blacklist),
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

    def get_recent_stats(self, symbol: str, side: str, combo: str, days: int = 30) -> Dict:
        """Get stats for a specific combo over the last X days"""
        if not self.pg_conn:
            return {'wins': 0, 'losses': 0, 'total': 0}
            
        try:
            with self.pg_conn.cursor() as cur:
                cur.execute("""
                    SELECT outcome, count(*)
                    FROM trade_history
                    WHERE symbol = %s AND side = %s AND combo = %s
                    AND created_at > NOW() - INTERVAL '%s days'
                    GROUP BY outcome
                """, (symbol, side, combo, days))
                
                rows = cur.fetchall()
                stats = {'wins': 0, 'losses': 0}
                for outcome, count in rows:
                    if outcome == 'win':
                        stats['wins'] = count
                    else:
                        stats['losses'] = count
                
                stats['total'] = stats['wins'] + stats['losses']
                return stats
        except Exception as e:
            logger.error(f"Failed to get recent stats: {e}")
            return {'wins': 0, 'losses': 0, 'total': 0}

    def _save_to_redis(self, pending_data: List[Dict]):
        """Save pending signals to Redis"""
        self.redis_client.set('vwap_bot:pending_signals', json.dumps(pending_data))
        # Also save lightweight stats for quick dashboard access
        stats = {
            'total_signals': self.total_signals,
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'updated_at': time.time()
        }
        self.redis_client.set('vwap_bot:stats_summary', json.dumps(stats))

    def _save_to_postgres(self, stats_data: Dict):
        """Save combo stats to Postgres"""
        if not self.pg_conn:
            return
            
        with self.pg_conn.cursor() as cur:
            for symbol, sides in stats_data.items():
                for side, combos in sides.items():
                    for combo, data in combos.items():
                        cur.execute("""
                            INSERT INTO combo_stats (symbol, side, combo, total, wins, losses, data, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                            ON CONFLICT (symbol, side, combo) 
                            DO UPDATE SET 
                                total = EXCLUDED.total,
                                wins = EXCLUDED.wins,
                                losses = EXCLUDED.losses,
                                data = EXCLUDED.data,
                                updated_at = NOW();
                        """, (
                            symbol, side, combo,
                            data.get('total', 0),
                            data.get('wins', 0),
                            data.get('losses', 0),
                            Json(data)
                        ))

    def load(self):
        """Load learning data from Redis/Postgres with JSON fallback"""
        loaded_from_db = False
        
        # 1. Try Load from DBs
        if self.redis_client and self.pg_conn:
            try:
                self._load_from_redis()
                self._load_from_postgres()
                loaded_from_db = True
                logger.info("📂 Loaded data from Redis & Postgres")
            except Exception as e:
                logger.error(f"DB Load failed, falling back to JSON: {e}")
        
        # 2. Fallback to JSON if DB load failed or not configured
        if not loaded_from_db:
            try:
                with open(self.SAVE_FILE, 'r') as f:
                    data = json.load(f)
                
                # Load combo stats
                for symbol, sides in data.get('combo_stats', {}).items():
                    for side, combos in sides.items():
                        for combo, stats in combos.items():
                            if 'by_rr' in stats:
                                stats['by_rr'] = {float(k): v for k, v in stats['by_rr'].items()}
                            self.combo_stats[symbol][side][combo] = stats
                
                # Load pending signals
                self._restore_pending_signals(data.get('pending_signals', []))
                
                self.promoted = set(data.get('promoted', []))
                self.blacklist = set(data.get('blacklist', []))
                self.total_signals = data.get('total_signals', 0)
                self.total_wins = data.get('total_wins', 0)
                self.total_losses = data.get('total_losses', 0)
                self.btc_price_1h_ago = data.get('btc_price_1h_ago', 0)
                self.btc_current = data.get('btc_current', 0)
                self.started_at = data.get('started_at', time.time())
                self.adjustments = data.get('adjustments', [])
                
                logger.info(f"📂 Loaded data from JSON file")
                
            except FileNotFoundError:
                logger.info("📂 No JSON file found, starting fresh")
            except Exception as e:
                logger.error(f"Failed to load JSON: {e}")

    def _load_from_redis(self):
        """Load pending signals from Redis"""
        data = self.redis_client.get('vwap_bot:pending_signals')
        if data:
            pending_data = json.loads(data)
            self._restore_pending_signals(pending_data)
            
        # Load summary stats
        stats_data = self.redis_client.get('vwap_bot:stats_summary')
        if stats_data:
            stats = json.loads(stats_data)
            self.total_signals = stats.get('total_signals', 0)
            self.total_wins = stats.get('total_wins', 0)
            self.total_losses = stats.get('total_losses', 0)

    def _load_from_postgres(self):
        """Load combo stats from Postgres"""
        if not self.pg_conn:
            return
            
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM combo_stats")
            rows = cur.fetchall()
            
            for row in rows:
                symbol = row['symbol']
                side = row['side']
                combo = row['combo']
                data = row['data']
                
                # Restore float keys in by_rr
                if 'by_rr' in data:
                    data['by_rr'] = {float(k): v for k, v in data['by_rr'].items()}
                
                self.combo_stats[symbol][side][combo] = data

    def _restore_pending_signals(self, pending_data: List[Dict]):
        """Helper to restore pending signals from list of dicts"""
        self.pending_signals = []
        for sig_data in pending_data:
            try:
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

