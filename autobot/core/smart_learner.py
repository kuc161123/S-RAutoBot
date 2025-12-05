#!/usr/bin/env python3
"""
Smart Learning System v3.0 - Self-Adjusting Intelligence

This module provides intelligent, self-adjusting trading decisions based on:
- Adaptive R:R optimization per symbol
- Volatility regime detection (high/medium/low)
- BTC trend correlation
- Automatic threshold adjustment
- Self-explanatory decision logging

The bot learns and adapts without user intervention.
"""

import json
import time
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    """Calculate lower bound of Wilson score confidence interval."""
    if total == 0:
        return 0.0
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    lower = (centre - spread) / denominator
    return max(0, lower * 100)

@dataclass
class SmartSignal:
    """Enhanced signal with full market context"""
    symbol: str
    side: str
    combo: str
    entry_price: float
    tp_price: float
    sl_price: float
    start_time: float
    
    # Market context at signal time
    atr_percent: float = 0.0      # Volatility
    volatility_regime: str = ""    # 'high', 'medium', 'low'
    btc_trend: str = ""           # 'bullish', 'bearish', 'neutral'
    btc_change_1h: float = 0.0    # BTC 1-hour change %
    session: str = ""              # 'asian', 'london', 'newyork'
    
    # R:R used
    rr_ratio: float = 2.0         # TP/SL ratio used
    
    # Outcome
    outcome: Optional[str] = None
    end_time: Optional[float] = None
    time_to_result: float = 0.0
    max_favorable: float = 0.0    # Max move towards TP
    max_adverse: float = 0.0      # Max move towards SL (drawdown)


class SmartLearner:
    """
    Self-adjusting intelligent learning system.
    Makes decisions and explains why.
    """
    
    SAVE_FILE = 'smart_learning.json'
    
    # Volatility thresholds (ATR as % of price)
    VOL_HIGH = 2.0    # > 2% = high volatility
    VOL_LOW = 0.5     # < 0.5% = low volatility
    
    # Minimum data for decisions
    MIN_TRADES_FOR_RR = 15    # Need 15 trades to adjust R:R
    MIN_TRADES_FOR_REGIME = 10
    
    # R:R options to try
    RR_OPTIONS = [1.5, 2.0, 2.5, 3.0]
    
    def __init__(self):
        # Symbol-specific learned parameters
        self.symbol_params: Dict[str, Dict] = defaultdict(lambda: {
            'optimal_rr': 2.0,
            'rr_confidence': 0,
            'best_regime': 'all',
            'regime_confidence': 0,
            'btc_alignment': 'neutral',
            'last_updated': 0
        })
        
        # Combo stats with regime breakdown
        self.combo_stats: Dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
            'total': 0, 'wins': 0, 'losses': 0,
            'by_regime': {'high': {'w': 0, 'l': 0}, 'medium': {'w': 0, 'l': 0}, 'low': {'w': 0, 'l': 0}},
            'by_btc': {'bullish': {'w': 0, 'l': 0}, 'bearish': {'w': 0, 'l': 0}, 'neutral': {'w': 0, 'l': 0}},
            'by_rr': {1.5: {'w': 0, 'l': 0}, 2.0: {'w': 0, 'l': 0}, 2.5: {'w': 0, 'l': 0}, 3.0: {'w': 0, 'l': 0}},
            'avg_time_win': 0, 'avg_time_loss': 0,
            'avg_drawdown': 0
        })))
        
        # Pending signals
        self.pending_signals: List[SmartSignal] = []
        
        # BTC price cache
        self.btc_price_1h_ago: float = 0
        self.btc_current: float = 0
        self.btc_last_update: float = 0
        
        # Decision log
        self.decision_log: List[Dict] = []
        
        # Global stats
        self.total_signals = 0
        self.total_wins = 0
        self.total_losses = 0
        self.started_at = time.time()
        
        # Auto-adjustments made
        self.adjustments_made: List[Dict] = []
        
        self.load()
    
    def get_volatility_regime(self, atr_percent: float) -> str:
        """Classify volatility regime"""
        if atr_percent >= self.VOL_HIGH:
            return 'high'
        elif atr_percent <= self.VOL_LOW:
            return 'low'
        return 'medium'
    
    def get_btc_trend(self, btc_change: float) -> str:
        """Classify BTC trend"""
        if btc_change > 0.5:
            return 'bullish'
        elif btc_change < -0.5:
            return 'bearish'
        return 'neutral'
    
    def update_btc_price(self, current_price: float):
        """Update BTC price tracking"""
        now = time.time()
        
        # Update hourly reference every hour
        if now - self.btc_last_update > 3600 or self.btc_price_1h_ago == 0:
            self.btc_price_1h_ago = self.btc_current if self.btc_current > 0 else current_price
            self.btc_last_update = now
        
        self.btc_current = current_price
    
    def get_btc_change_1h(self) -> float:
        """Get BTC 1-hour change percentage"""
        if self.btc_price_1h_ago == 0 or self.btc_current == 0:
            return 0.0
        return ((self.btc_current - self.btc_price_1h_ago) / self.btc_price_1h_ago) * 100
    
    def get_optimal_rr(self, symbol: str, side: str, combo: str) -> Tuple[float, str]:
        """
        Get optimal R:R for this setup and explain why.
        Returns (rr_ratio, explanation)
        """
        stats = self.combo_stats[symbol][side][combo]
        
        # Not enough data - use default
        if stats['total'] < self.MIN_TRADES_FOR_RR:
            return 2.0, f"Default 2:1 (only {stats['total']} trades, need {self.MIN_TRADES_FOR_RR})"
        
        # Find best R:R by win rate
        best_rr = 2.0
        best_ev = -999
        
        for rr in self.RR_OPTIONS:
            rr_stats = stats['by_rr'].get(rr, {'w': 0, 'l': 0})
            total = rr_stats['w'] + rr_stats['l']
            if total < 5:
                continue
            
            wr = rr_stats['w'] / total
            # Expected value = (WR * RR) - ((1-WR) * 1)
            ev = (wr * rr) - ((1 - wr) * 1)
            
            if ev > best_ev:
                best_ev = ev
                best_rr = rr
        
        if best_ev > 0:
            explanation = f"Optimal {best_rr}:1 (EV={best_ev:.2f}R based on {stats['total']} trades)"
        else:
            explanation = f"Default 2:1 (no positive EV found)"
        
        return best_rr, explanation
    
    def get_regime_filter(self, symbol: str, side: str, combo: str) -> Tuple[List[str], str]:
        """
        Get which volatility regimes work for this setup.
        Returns (allowed_regimes, explanation)
        """
        stats = self.combo_stats[symbol][side][combo]
        
        if stats['total'] < self.MIN_TRADES_FOR_REGIME:
            return ['high', 'medium', 'low'], f"All regimes (only {stats['total']} trades)"
        
        allowed = []
        explanations = []
        
        for regime in ['high', 'medium', 'low']:
            regime_stats = stats['by_regime'].get(regime, {'w': 0, 'l': 0})
            total = regime_stats['w'] + regime_stats['l']
            if total < 3:
                allowed.append(regime)  # Not enough data, allow
                continue
            
            wr = regime_stats['w'] / total * 100
            lb_wr = wilson_lower_bound(regime_stats['w'], total)
            
            if lb_wr >= 40:  # Good enough
                allowed.append(regime)
                explanations.append(f"{regime}:{lb_wr:.0f}%")
            else:
                explanations.append(f"❌{regime}:{lb_wr:.0f}%")
        
        if not allowed:
            allowed = ['medium']  # Fallback to medium
        
        exp_str = ", ".join(explanations) if explanations else "All regimes"
        return allowed, exp_str
    
    def get_btc_filter(self, symbol: str, side: str, combo: str) -> Tuple[List[str], str]:
        """
        Get which BTC trends work for this setup.
        Returns (allowed_btc_trends, explanation)
        """
        stats = self.combo_stats[symbol][side][combo]
        
        if stats['total'] < self.MIN_TRADES_FOR_REGIME:
            return ['bullish', 'bearish', 'neutral'], "All BTC trends (not enough data)"
        
        allowed = []
        
        for trend in ['bullish', 'bearish', 'neutral']:
            trend_stats = stats['by_btc'].get(trend, {'w': 0, 'l': 0})
            total = trend_stats['w'] + trend_stats['l']
            if total < 3:
                allowed.append(trend)
                continue
            
            lb_wr = wilson_lower_bound(trend_stats['w'], total)
            if lb_wr >= 35:
                allowed.append(trend)
        
        if not allowed:
            allowed = ['neutral']
        
        return allowed, f"BTC: {', '.join(allowed)}"
    
    def should_take_signal(self, symbol: str, side: str, combo: str,
                          atr_percent: float, btc_change: float) -> Tuple[bool, str]:
        """
        Smart decision: Should we take this signal?
        Returns (should_take, detailed_explanation)
        """
        reasons = []
        
        # Get current context
        regime = self.get_volatility_regime(atr_percent)
        btc_trend = self.get_btc_trend(btc_change)
        
        # Check regime filter
        allowed_regimes, regime_exp = self.get_regime_filter(symbol, side, combo)
        if regime not in allowed_regimes:
            return False, f"❌ BLOCKED: {regime} volatility not profitable. {regime_exp}"
        reasons.append(f"✓ Regime {regime} OK")
        
        # Check BTC filter
        allowed_btc, btc_exp = self.get_btc_filter(symbol, side, combo)
        if btc_trend not in allowed_btc:
            return False, f"❌ BLOCKED: BTC {btc_trend} conflicts. {btc_exp}"
        reasons.append(f"✓ BTC {btc_trend} OK")
        
        # Get optimal R:R
        optimal_rr, rr_exp = self.get_optimal_rr(symbol, side, combo)
        reasons.append(f"✓ R:R {optimal_rr}:1")
        
        # Get stats
        stats = self.combo_stats[symbol][side][combo]
        if stats['total'] >= 10:
            lb_wr = wilson_lower_bound(stats['wins'], stats['total'])
            if lb_wr < 35:
                return False, f"❌ BLOCKED: LB_WR {lb_wr:.0f}% too low (need 35%)"
            reasons.append(f"✓ LB_WR {lb_wr:.0f}%")
        
        explanation = f"✅ TAKE: {' | '.join(reasons)}"
        return True, explanation
    
    def record_signal(self, symbol: str, side: str, combo: str,
                     entry: float, atr: float, btc_price: float = 0) -> Tuple[float, float, str]:
        """
        Record signal and return optimal TP/SL with explanation.
        Returns (tp_price, sl_price, explanation)
        """
        # Update BTC
        if btc_price > 0:
            self.update_btc_price(btc_price)
        
        # Get context
        atr_percent = (atr / entry) * 100 if entry > 0 else 1.0
        regime = self.get_volatility_regime(atr_percent)
        btc_change = self.get_btc_change_1h()
        btc_trend = self.get_btc_trend(btc_change)
        
        # Get optimal R:R
        optimal_rr, rr_explanation = self.get_optimal_rr(symbol, side, combo)
        
        # Calculate TP/SL with optimal R:R
        if side == 'long':
            sl = entry - (1.0 * atr)  # 1 ATR stop
            tp = entry + (optimal_rr * atr)  # R:R * ATR profit
        else:
            sl = entry + (1.0 * atr)
            tp = entry - (optimal_rr * atr)
        
        # Create signal
        signal = SmartSignal(
            symbol=symbol,
            side=side,
            combo=combo,
            entry_price=entry,
            tp_price=tp,
            sl_price=sl,
            start_time=time.time(),
            atr_percent=atr_percent,
            volatility_regime=regime,
            btc_trend=btc_trend,
            btc_change_1h=btc_change,
            rr_ratio=optimal_rr
        )
        
        self.pending_signals.append(signal)
        self.total_signals += 1
        
        explanation = f"R:R={optimal_rr}:1 | Regime={regime} | BTC={btc_trend} ({btc_change:+.1f}%)"
        
        # Log decision
        self.decision_log.append({
            'time': time.time(),
            'symbol': symbol,
            'side': side,
            'action': 'SIGNAL',
            'reason': explanation
        })
        
        return tp, sl, explanation
    
    def update_signals(self, prices: Dict[str, float]):
        """Update pending signals with current prices"""
        for signal in self.pending_signals[:]:
            if signal.outcome is not None:
                continue
            
            price = prices.get(signal.symbol)
            if not price:
                continue
            
            # Track max favorable/adverse excursion
            if signal.side == 'long':
                favorable = (price - signal.entry_price) / signal.entry_price * 100
                adverse = (signal.entry_price - price) / signal.entry_price * 100
            else:
                favorable = (signal.entry_price - price) / signal.entry_price * 100
                adverse = (price - signal.entry_price) / signal.entry_price * 100
            
            signal.max_favorable = max(signal.max_favorable, favorable)
            signal.max_adverse = max(signal.max_adverse, adverse)
            
            # Check timeout (4 hours)
            if time.time() - signal.start_time > 14400:
                signal.outcome = 'timeout'
                signal.end_time = time.time()
                self.pending_signals.remove(signal)
                continue
            
            # Check outcome
            outcome = None
            if signal.side == 'long':
                if price <= signal.sl_price:
                    outcome = 'loss'
                elif price >= signal.tp_price:
                    outcome = 'win'
            else:
                if price >= signal.sl_price:
                    outcome = 'loss'
                elif price <= signal.tp_price:
                    outcome = 'win'
            
            if outcome:
                self._resolve_signal(signal, outcome)
                self.pending_signals.remove(signal)
    
    def _resolve_signal(self, signal: SmartSignal, outcome: str):
        """Resolve signal and update all stats"""
        signal.outcome = outcome
        signal.end_time = time.time()
        signal.time_to_result = signal.end_time - signal.start_time
        
        # Update combo stats
        stats = self.combo_stats[signal.symbol][signal.side][signal.combo]
        stats['total'] += 1
        
        if outcome == 'win':
            stats['wins'] += 1
            self.total_wins += 1
        else:
            stats['losses'] += 1
            self.total_losses += 1
        
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
        
        # Check if we need to auto-adjust
        self._check_auto_adjust(signal.symbol, signal.side, signal.combo)
        
        # Log
        logger.info(f"📊 {signal.symbol} {signal.side} {outcome.upper()} | "
                   f"Regime={signal.volatility_regime} BTC={signal.btc_trend} R:R={signal.rr_ratio}")
    
    def _check_auto_adjust(self, symbol: str, side: str, combo: str):
        """Check if we should auto-adjust parameters"""
        stats = self.combo_stats[symbol][side][combo]
        
        if stats['total'] < self.MIN_TRADES_FOR_RR:
            return
        
        # Check for R:R adjustment
        current_rr, _ = self.get_optimal_rr(symbol, side, combo)
        params = self.symbol_params[f"{symbol}:{side}:{combo}"]
        
        if params['optimal_rr'] != current_rr:
            old_rr = params['optimal_rr']
            params['optimal_rr'] = current_rr
            params['last_updated'] = time.time()
            
            self.adjustments_made.append({
                'time': time.time(),
                'symbol': symbol,
                'side': side,
                'type': 'R:R',
                'old': old_rr,
                'new': current_rr,
                'reason': f"Better EV at {current_rr}:1 based on {stats['total']} trades"
            })
            
            logger.info(f"🔧 AUTO-ADJUST: {symbol} {side} R:R changed {old_rr} → {current_rr}")
    
    def get_smart_report(self) -> str:
        """Generate smart status report"""
        uptime = (time.time() - self.started_at) / 3600
        total = self.total_wins + self.total_losses
        wr = (self.total_wins / total * 100) if total > 0 else 0
        lb_wr = wilson_lower_bound(self.total_wins, total)
        
        report = (
            "🧠 **SMART LEARNING REPORT**\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            f"⏱️ Running: {uptime:.1f}h\n"
            f"📊 Signals: {self.total_signals}\n"
            f"📈 WR: {wr:.0f}% (LB: **{lb_wr:.0f}%**)\n"
            f"🎯 Pending: {len(self.pending_signals)}\n\n"
        )
        
        # Recent adjustments
        if self.adjustments_made:
            report += "🔧 **RECENT ADJUSTMENTS**\n"
            for adj in self.adjustments_made[-3:]:
                report += f"├ `{adj['symbol']}` {adj['type']}: {adj['old']}→{adj['new']}\n"
            report += "\n"
        
        # BTC context
        btc_trend = self.get_btc_trend(self.get_btc_change_1h())
        report += f"₿ **BTC**: {btc_trend} ({self.get_btc_change_1h():+.1f}%)\n"
        
        return report
    
    def save(self):
        """Save learning data"""
        try:
            data = {
                'symbol_params': dict(self.symbol_params),
                'combo_stats': {k1: {k2: dict(v2) for k2, v2 in v1.items()} 
                               for k1, v1 in self.combo_stats.items()},
                'total_signals': self.total_signals,
                'total_wins': self.total_wins,
                'total_losses': self.total_losses,
                'adjustments_made': self.adjustments_made[-50:],
                'btc_price_1h_ago': self.btc_price_1h_ago,
                'started_at': self.started_at,
                'saved_at': time.time()
            }
            with open(self.SAVE_FILE, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save smart learning: {e}")
    
    def load(self):
        """Load learning data"""
        try:
            with open(self.SAVE_FILE, 'r') as f:
                data = json.load(f)
            
            for k, v in data.get('symbol_params', {}).items():
                self.symbol_params[k] = v
            
            for sym, sides in data.get('combo_stats', {}).items():
                for side, combos in sides.items():
                    for combo, stats in combos.items():
                        self.combo_stats[sym][side][combo] = stats
            
            self.total_signals = data.get('total_signals', 0)
            self.total_wins = data.get('total_wins', 0)
            self.total_losses = data.get('total_losses', 0)
            self.adjustments_made = data.get('adjustments_made', [])
            self.btc_price_1h_ago = data.get('btc_price_1h_ago', 0)
            self.started_at = data.get('started_at', time.time())
            
            logger.info(f"🧠 Smart learning loaded: {self.total_signals} signals")
        except FileNotFoundError:
            logger.info("🧠 Starting fresh smart learning")
        except Exception as e:
            logger.error(f"Failed to load smart learning: {e}")
