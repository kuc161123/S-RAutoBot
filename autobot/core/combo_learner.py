#!/usr/bin/env python3
"""
Combo Learning System - Silent Performance Tracker

This module tracks ALL signals across ALL symbols to build a database
of combo performance over time. It runs silently in the background
and persists data across bot restarts.

Key Features:
- Tracks every VWAP signal (not just allowed ones)
- Records outcome (win/loss) for each combo
- Calculates rolling win rates
- Discovers profitable combos automatically
- Provides insights via Telegram commands
"""

import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class LearningSignal:
    """A signal being tracked for learning"""
    symbol: str
    side: str  # 'long' or 'short'
    combo: str
    entry_price: float
    tp_price: float
    sl_price: float
    start_time: float
    outcome: Optional[str] = None  # 'win', 'loss', or None if pending
    end_time: Optional[float] = None

class ComboLearner:
    """
    Silent combo learning system that tracks all signals
    and builds a performance database over time.
    """
    
    SAVE_FILE = 'combo_learning.json'
    MAX_PENDING_SIGNALS = 1000  # Increased for 400 symbols
    SIGNAL_TIMEOUT = 14400  # 4 hours for scalps to resolve
    
    def __init__(self):
        # Performance database: {symbol: {side: {combo: {wins, losses, total}}}}
        self.combo_stats: Dict[str, Dict[str, Dict[str, Dict]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'total': 0}))
        )
        
        # Pending signals waiting for outcome
        self.pending_signals: List[LearningSignal] = []
        
        # Global stats
        self.total_signals_tracked = 0
        self.total_wins = 0
        self.total_losses = 0
        self.last_save = time.time()
        self.started_at = time.time()
        
        # Load existing data
        self.load()
    
    def load(self):
        """Load learning data from disk"""
        try:
            with open(self.SAVE_FILE, 'r') as f:
                data = json.load(f)
            
            # Restore stats with proper defaultdict structure
            for symbol, sides in data.get('combo_stats', {}).items():
                for side, combos in sides.items():
                    for combo, stats in combos.items():
                        self.combo_stats[symbol][side][combo] = stats
            
            # Restore pending signals
            for sig_data in data.get('pending_signals', []):
                self.pending_signals.append(LearningSignal(**sig_data))
            
            self.total_signals_tracked = data.get('total_signals_tracked', 0)
            self.total_wins = data.get('total_wins', 0)
            self.total_losses = data.get('total_losses', 0)
            self.started_at = data.get('started_at', time.time())
            
            logger.info(f"📚 Combo Learner loaded: {self.total_signals_tracked} signals tracked, {len(self.get_all_combos())} combos learned")
        except FileNotFoundError:
            logger.info("📚 Combo Learner: Starting fresh learning database")
        except Exception as e:
            logger.error(f"Failed to load learning data: {e}")
    
    def save(self):
        """Save learning data to disk"""
        try:
            # Convert defaultdict to regular dict for JSON
            combo_stats_dict = {}
            for symbol, sides in self.combo_stats.items():
                combo_stats_dict[symbol] = {}
                for side, combos in sides.items():
                    combo_stats_dict[symbol][side] = dict(combos)
            
            data = {
                'combo_stats': combo_stats_dict,
                'pending_signals': [asdict(s) for s in self.pending_signals[-self.MAX_PENDING_SIGNALS:]],
                'total_signals_tracked': self.total_signals_tracked,
                'total_wins': self.total_wins,
                'total_losses': self.total_losses,
                'started_at': self.started_at,
                'saved_at': time.time()
            }
            
            with open(self.SAVE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.last_save = time.time()
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    def record_signal(self, symbol: str, side: str, combo: str, 
                     entry: float, tp: float, sl: float):
        """Record a new signal for learning (called on every VWAP touch)"""
        # Don't duplicate - check if we're already tracking this exact combo
        for sig in self.pending_signals:
            if sig.symbol == symbol and sig.side == side and sig.combo == combo and sig.outcome is None:
                return  # Already tracking this exact combo
        
        signal = LearningSignal(
            symbol=symbol,
            side=side,
            combo=combo,
            entry_price=entry,
            tp_price=tp,
            sl_price=sl,
            start_time=time.time()
        )
        
        self.pending_signals.append(signal)
        self.total_signals_tracked += 1
        
        # Trim old pending signals
        if len(self.pending_signals) > self.MAX_PENDING_SIGNALS:
            self.pending_signals = self.pending_signals[-self.MAX_PENDING_SIGNALS:]
    
    def update_signals(self, current_prices: Dict[str, float]):
        """Update pending signals with current prices to determine outcomes"""
        for signal in self.pending_signals:
            if signal.outcome is not None:
                continue  # Already resolved
            
            price = current_prices.get(signal.symbol)
            if price is None:
                continue
            
            # Check timeout
            if time.time() - signal.start_time > self.SIGNAL_TIMEOUT:
                signal.outcome = 'timeout'
                signal.end_time = time.time()
                continue
            
            # Check outcome
            if signal.side == 'long':
                if price <= signal.sl_price:
                    signal.outcome = 'loss'
                    signal.end_time = time.time()
                    self._record_outcome(signal, 'loss')
                elif price >= signal.tp_price:
                    signal.outcome = 'win'
                    signal.end_time = time.time()
                    self._record_outcome(signal, 'win')
            else:  # short
                if price >= signal.sl_price:
                    signal.outcome = 'loss'
                    signal.end_time = time.time()
                    self._record_outcome(signal, 'loss')
                elif price <= signal.tp_price:
                    signal.outcome = 'win'
                    signal.end_time = time.time()
                    self._record_outcome(signal, 'win')
        
        # Clean up resolved signals properly
        pending_only = [s for s in self.pending_signals if s.outcome is None]
        self.pending_signals = pending_only
        
        # Trim if too many
        if len(self.pending_signals) > self.MAX_PENDING_SIGNALS:
            self.pending_signals = self.pending_signals[-self.MAX_PENDING_SIGNALS:]
        
        # Auto-save every 5 minutes
        if time.time() - self.last_save > 300:
            self.save()
    
    def _record_outcome(self, signal: LearningSignal, outcome: str):
        """Record the outcome of a signal in the stats database"""
        stats = self.combo_stats[signal.symbol][signal.side][signal.combo]
        stats['total'] += 1
        
        if outcome == 'win':
            stats['wins'] += 1
            self.total_wins += 1
        else:
            stats['losses'] += 1
            self.total_losses += 1
    
    def get_combo_stats(self, symbol: str, side: str, combo: str) -> Dict:
        """Get stats for a specific combo"""
        stats = self.combo_stats[symbol][side][combo]
        wr = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
        return {**stats, 'win_rate': wr}
    
    def get_all_combos(self) -> List[Dict]:
        """Get all combos with their stats, sorted by performance"""
        combos = []
        for symbol, sides in self.combo_stats.items():
            for side, combo_data in sides.items():
                for combo, stats in combo_data.items():
                    if stats['total'] > 0:
                        wr = stats['wins'] / stats['total'] * 100
                        combos.append({
                            'symbol': symbol,
                            'side': side,
                            'combo': combo,
                            'wins': stats['wins'],
                            'losses': stats['losses'],
                            'total': stats['total'],
                            'win_rate': wr
                        })
        return combos
    
    def get_top_combos(self, min_trades: int = 5, min_wr: float = 50.0) -> List[Dict]:
        """Get top performing combos that meet criteria"""
        all_combos = self.get_all_combos()
        filtered = [c for c in all_combos if c['total'] >= min_trades and c['win_rate'] >= min_wr]
        return sorted(filtered, key=lambda x: (x['win_rate'], x['total']), reverse=True)
    
    def get_losing_combos(self, min_trades: int = 5, max_wr: float = 40.0) -> List[Dict]:
        """Get consistently losing combos to avoid"""
        all_combos = self.get_all_combos()
        filtered = [c for c in all_combos if c['total'] >= min_trades and c['win_rate'] <= max_wr]
        return sorted(filtered, key=lambda x: x['win_rate'])
    
    def get_symbol_summary(self) -> Dict:
        """Get summary stats per symbol"""
        summary = {}
        for symbol, sides in self.combo_stats.items():
            total_wins = 0
            total_losses = 0
            for side, combos in sides.items():
                for combo, stats in combos.items():
                    total_wins += stats['wins']
                    total_losses += stats['losses']
            total = total_wins + total_losses
            wr = (total_wins / total * 100) if total > 0 else 0
            summary[symbol] = {'wins': total_wins, 'losses': total_losses, 'total': total, 'win_rate': wr}
        return summary
    
    def generate_report(self) -> str:
        """Generate a Telegram-friendly report"""
        uptime_hrs = (time.time() - self.started_at) / 3600
        total = self.total_wins + self.total_losses
        overall_wr = (self.total_wins / total * 100) if total > 0 else 0
        
        top_combos = self.get_top_combos(min_trades=3, min_wr=50.0)[:5]
        losing_combos = self.get_losing_combos(min_trades=3, max_wr=35.0)[:3]
        
        report = (
            "📚 **COMBO LEARNING REPORT**\n\n"
            f"⏱️ Learning for: {uptime_hrs:.1f} hours\n"
            f"📊 Signals tracked: {self.total_signals_tracked}\n"
            f"📈 Overall WR: {overall_wr:.1f}% ({self.total_wins}W/{self.total_losses}L)\n"
            f"🔢 Unique combos: {len(self.get_all_combos())}\n\n"
        )
        
        if top_combos:
            report += "🏆 **TOP PERFORMERS** (WR>50%, N≥3)\n"
            for c in top_combos:
                report += f"• `{c['symbol']}` {c['side'].upper()}\n"
                report += f"  {c['combo']}\n"
                report += f"  WR: {c['win_rate']:.0f}% ({c['wins']}W/{c['losses']}L)\n"
            report += "\n"
        else:
            report += "🏆 No top performers yet (need more data)\n\n"
        
        if losing_combos:
            report += "⚠️ **AVOID THESE** (WR<35%, N≥3)\n"
            for c in losing_combos[:3]:
                report += f"• `{c['symbol']}` {c['side']}: {c['win_rate']:.0f}%\n"
        
        return report
    
    def get_promote_candidates(self) -> List[Dict]:
        """Get combos that are good enough to be promoted to active trading"""
        # Criteria: WR >= 45%, at least 10 trades, positive expectancy
        candidates = []
        for c in self.get_all_combos():
            if c['total'] >= 10 and c['win_rate'] >= 45:
                # Calculate expected value (assuming 1:2 R:R)
                ev = (c['win_rate']/100 * 2) - ((100-c['win_rate'])/100 * 1)
                if ev > 0:
                    c['expected_value'] = ev
                    candidates.append(c)
        return sorted(candidates, key=lambda x: x['expected_value'], reverse=True)
