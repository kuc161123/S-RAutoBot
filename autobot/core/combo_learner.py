#!/usr/bin/env python3
"""
Combo Learning System v2.0 - Enhanced Performance Tracker

This module tracks ALL signals across ALL symbols to build a database
of combo performance over time with advanced analytics.

Key Features:
- Lower Bound Win Rate (Wilson score confidence interval)
- Auto-promote profitable combos to active trading
- Blacklist consistently losing combos
- Time-of-day / session analysis
- Volume and volatility context
- BTC correlation tracking
"""

import json
import time
import math
import logging
import subprocess
import yaml
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    """
    Calculate lower bound of Wilson score confidence interval.
    This gives a conservative estimate of true win rate.
    z=1.96 for 95% confidence, z=1.645 for 90% confidence
    """
    if total == 0:
        return 0.0
    
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    
    lower = (centre - spread) / denominator
    return max(0, lower * 100)  # Return as percentage

@dataclass
class LearningSignal:
    """Enhanced signal with full context for learning"""
    symbol: str
    side: str
    combo: str
    entry_price: float
    tp_price: float
    sl_price: float
    start_time: float
    
    # Context at signal time
    session: str = ""        # 'asian', 'london', 'newyork'
    hour_utc: int = 0
    volume_spike: bool = False
    atr_percent: float = 0.0  # ATR as % of price
    btc_trend: str = ""       # 'up', 'down', 'flat'
    
    # Outcome tracking
    outcome: Optional[str] = None
    end_time: Optional[float] = None
    time_to_result: float = 0.0
    max_drawdown_pct: float = 0.0

class ComboLearner:
    """
    Enhanced combo learning system with auto-promote and blacklist.
    """
    
    SAVE_FILE = 'combo_learning.json'
    BLACKLIST_FILE = 'combo_blacklist.json'
    OVERRIDES_FILE = 'symbol_overrides_VWAP_Combo.yaml'
    
    MAX_PENDING_SIGNALS = 1500
    SIGNAL_TIMEOUT = 14400  # 4 hours
    
    # Auto-promote thresholds
    PROMOTE_MIN_TRADES = 20
    PROMOTE_MIN_LOWER_WR = 45.0
    PROMOTE_MIN_EV = 0.3
    
    # Blacklist thresholds
    BLACKLIST_MIN_TRADES = 10
    BLACKLIST_MAX_LOWER_WR = 30.0
    
    def __init__(self):
        # Performance database: {symbol: {side: {combo: {wins, losses, total, ...}}}}
        self.combo_stats: Dict[str, Dict[str, Dict[str, Dict]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: {
                'wins': 0, 'losses': 0, 'total': 0,
                'sessions': {'asian': {'w': 0, 'l': 0}, 'london': {'w': 0, 'l': 0}, 'newyork': {'w': 0, 'l': 0}},
                'total_time': 0.0, 'max_drawdowns': []
            }))
        )
        
        self.pending_signals: List[LearningSignal] = []
        self.blacklist: set = set()  # Set of "symbol:side:combo" strings
        self.promoted: set = set()   # Already promoted combos
        
        # Global stats
        self.total_signals_tracked = 0
        self.total_wins = 0
        self.total_losses = 0
        self.last_save = time.time()
        self.started_at = time.time()
        
        # Load existing data
        self.load()
        self.load_blacklist()
    
    def get_session(self, hour_utc: int) -> str:
        """Determine trading session from UTC hour"""
        if 0 <= hour_utc < 8:
            return 'asian'
        elif 8 <= hour_utc < 13:
            return 'london'
        else:
            return 'newyork'
    
    def load(self):
        """Load learning data from disk"""
        try:
            with open(self.SAVE_FILE, 'r') as f:
                data = json.load(f)
            
            for symbol, sides in data.get('combo_stats', {}).items():
                for side, combos in sides.items():
                    for combo, stats in combos.items():
                        self.combo_stats[symbol][side][combo] = stats
            
            for sig_data in data.get('pending_signals', []):
                try:
                    self.pending_signals.append(LearningSignal(**sig_data))
                except:
                    pass  # Skip invalid signals
            
            self.promoted = set(data.get('promoted', []))
            self.total_signals_tracked = data.get('total_signals_tracked', 0)
            self.total_wins = data.get('total_wins', 0)
            self.total_losses = data.get('total_losses', 0)
            self.started_at = data.get('started_at', time.time())
            
            logger.info(f"📚 Learning loaded: {self.total_signals_tracked} signals, {len(self.get_all_combos())} combos")
        except FileNotFoundError:
            logger.info("📚 Starting fresh learning database")
        except Exception as e:
            logger.error(f"Failed to load learning data: {e}")
    
    def load_blacklist(self):
        """Load blacklist from disk"""
        try:
            with open(self.BLACKLIST_FILE, 'r') as f:
                data = json.load(f)
                self.blacklist = set(data.get('blacklist', []))
            logger.info(f"🚫 Loaded {len(self.blacklist)} blacklisted combos")
        except FileNotFoundError:
            pass
    
    def save(self):
        """Save learning data to disk"""
        try:
            combo_stats_dict = {}
            for symbol, sides in self.combo_stats.items():
                combo_stats_dict[symbol] = {}
                for side, combos in sides.items():
                    combo_stats_dict[symbol][side] = dict(combos)
            
            data = {
                'combo_stats': combo_stats_dict,
                'pending_signals': [asdict(s) for s in self.pending_signals[-self.MAX_PENDING_SIGNALS:]],
                'promoted': list(self.promoted),
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
    
    def save_blacklist(self):
        """Save blacklist to disk"""
        try:
            with open(self.BLACKLIST_FILE, 'w') as f:
                json.dump({'blacklist': list(self.blacklist), 'updated': time.time()}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save blacklist: {e}")
    
    def is_blacklisted(self, symbol: str, side: str, combo: str) -> bool:
        """Check if combo is blacklisted"""
        key = f"{symbol}:{side}:{combo}"
        return key in self.blacklist
    
    def record_signal(self, symbol: str, side: str, combo: str, 
                     entry: float, tp: float, sl: float,
                     volume_spike: bool = False, atr_percent: float = 0.0,
                     btc_trend: str = ""):
        """Record a new signal for learning"""
        
        # Skip if blacklisted
        if self.is_blacklisted(symbol, side, combo):
            return
        
        # Don't duplicate
        for sig in self.pending_signals:
            if sig.symbol == symbol and sig.side == side and sig.combo == combo and sig.outcome is None:
                return
        
        now = datetime.utcnow()
        hour_utc = now.hour
        session = self.get_session(hour_utc)
        
        signal = LearningSignal(
            symbol=symbol,
            side=side,
            combo=combo,
            entry_price=entry,
            tp_price=tp,
            sl_price=sl,
            start_time=time.time(),
            session=session,
            hour_utc=hour_utc,
            volume_spike=volume_spike,
            atr_percent=atr_percent,
            btc_trend=btc_trend
        )
        
        self.pending_signals.append(signal)
        self.total_signals_tracked += 1
        
        if len(self.pending_signals) > self.MAX_PENDING_SIGNALS:
            self.pending_signals = self.pending_signals[-self.MAX_PENDING_SIGNALS:]
    
    def update_signals(self, current_prices: Dict[str, float]):
        """Update pending signals and check for outcomes"""
        for signal in self.pending_signals:
            if signal.outcome is not None:
                continue
            
            price = current_prices.get(signal.symbol)
            if price is None:
                continue
            
            # Track max drawdown
            if signal.side == 'long':
                drawdown = (signal.entry_price - price) / signal.entry_price * 100
            else:
                drawdown = (price - signal.entry_price) / signal.entry_price * 100
            signal.max_drawdown_pct = max(signal.max_drawdown_pct, drawdown)
            
            # Check timeout
            if time.time() - signal.start_time > self.SIGNAL_TIMEOUT:
                signal.outcome = 'timeout'
                signal.end_time = time.time()
                continue
            
            # Check outcome
            if signal.side == 'long':
                if price <= signal.sl_price:
                    self._resolve_signal(signal, 'loss')
                elif price >= signal.tp_price:
                    self._resolve_signal(signal, 'win')
            else:
                if price >= signal.sl_price:
                    self._resolve_signal(signal, 'loss')
                elif price <= signal.tp_price:
                    self._resolve_signal(signal, 'win')
        
        # Cleanup
        self.pending_signals = [s for s in self.pending_signals if s.outcome is None]
        
        # Auto-save every 5 minutes
        if time.time() - self.last_save > 300:
            self.save()
    
    def _resolve_signal(self, signal: LearningSignal, outcome: str):
        """Resolve a signal and update stats"""
        signal.outcome = outcome
        signal.end_time = time.time()
        signal.time_to_result = signal.end_time - signal.start_time
        
        stats = self.combo_stats[signal.symbol][signal.side][signal.combo]
        stats['total'] += 1
        stats['total_time'] += signal.time_to_result
        stats['max_drawdowns'].append(signal.max_drawdown_pct)
        
        # Keep only last 20 drawdowns
        if len(stats['max_drawdowns']) > 20:
            stats['max_drawdowns'] = stats['max_drawdowns'][-20:]
        
        # Session stats
        session_key = signal.session
        if session_key in stats['sessions']:
            if outcome == 'win':
                stats['sessions'][session_key]['w'] += 1
            else:
                stats['sessions'][session_key]['l'] += 1
        
        if outcome == 'win':
            stats['wins'] += 1
            self.total_wins += 1
        else:
            stats['losses'] += 1
            self.total_losses += 1
        
        # Check for auto-promote or blacklist
        self._check_promote(signal.symbol, signal.side, signal.combo, stats)
        self._check_blacklist(signal.symbol, signal.side, signal.combo, stats)
    
    def _check_promote(self, symbol: str, side: str, combo: str, stats: Dict):
        """Check if combo should be promoted to active trading"""
        key = f"{symbol}:{side}:{combo}"
        if key in self.promoted:
            return
        
        if stats['total'] < self.PROMOTE_MIN_TRADES:
            return
        
        lower_wr = wilson_lower_bound(stats['wins'], stats['total'])
        if lower_wr < self.PROMOTE_MIN_LOWER_WR:
            return
        
        # Calculate EV (assuming 1:2 R:R)
        raw_wr = stats['wins'] / stats['total']
        ev = (raw_wr * 2) - ((1 - raw_wr) * 1)
        if ev < self.PROMOTE_MIN_EV:
            return
        
        # Promote!
        self._promote_combo(symbol, side, combo, lower_wr, ev)
    
    def _promote_combo(self, symbol: str, side: str, combo: str, lower_wr: float, ev: float):
        """Add combo to active trading"""
        key = f"{symbol}:{side}:{combo}"
        self.promoted.add(key)
        
        try:
            # Load existing overrides
            try:
                with open(self.OVERRIDES_FILE, 'r') as f:
                    overrides = yaml.safe_load(f) or {}
            except FileNotFoundError:
                overrides = {}
            
            # Add new combo
            if symbol not in overrides:
                overrides[symbol] = {}
            if side not in overrides[symbol]:
                overrides[symbol][side] = []
            
            if combo not in overrides[symbol][side]:
                overrides[symbol][side].append(combo)
                
                with open(self.OVERRIDES_FILE, 'w') as f:
                    yaml.dump(overrides, f, default_flow_style=False)
                
                # Git push
                try:
                    subprocess.run(['git', 'add', self.OVERRIDES_FILE], capture_output=True)
                    subprocess.run(['git', 'commit', '-m', f'Auto-promote: {symbol} {side} (LB_WR={lower_wr:.0f}%)'], capture_output=True)
                    subprocess.run(['git', 'push'], capture_output=True)
                except:
                    pass
                
                logger.info(f"🚀 AUTO-PROMOTED: {symbol} {side} {combo} (LB_WR={lower_wr:.0f}%, EV={ev:.2f})")
        except Exception as e:
            logger.error(f"Failed to promote combo: {e}")
    
    def _check_blacklist(self, symbol: str, side: str, combo: str, stats: Dict):
        """Check if combo should be blacklisted"""
        if stats['total'] < self.BLACKLIST_MIN_TRADES:
            return
        
        lower_wr = wilson_lower_bound(stats['wins'], stats['total'])
        if lower_wr >= self.BLACKLIST_MAX_LOWER_WR:
            return
        
        # Blacklist!
        key = f"{symbol}:{side}:{combo}"
        if key not in self.blacklist:
            self.blacklist.add(key)
            self.save_blacklist()
            logger.info(f"🚫 BLACKLISTED: {symbol} {side} {combo} (LB_WR={lower_wr:.0f}%)")
    
    def get_lower_bound_wr(self, symbol: str, side: str, combo: str) -> float:
        """Get lower bound WR for a specific combo"""
        stats = self.combo_stats[symbol][side][combo]
        return wilson_lower_bound(stats['wins'], stats['total'])
    
    def get_all_combos(self) -> List[Dict]:
        """Get all combos with stats including lower bound WR"""
        combos = []
        for symbol, sides in self.combo_stats.items():
            for side, combo_data in sides.items():
                for combo, stats in combo_data.items():
                    if stats['total'] > 0:
                        raw_wr = stats['wins'] / stats['total'] * 100
                        lower_wr = wilson_lower_bound(stats['wins'], stats['total'])
                        avg_time = stats['total_time'] / stats['total'] if stats['total'] > 0 else 0
                        avg_dd = sum(stats['max_drawdowns']) / len(stats['max_drawdowns']) if stats['max_drawdowns'] else 0
                        
                        combos.append({
                            'symbol': symbol,
                            'side': side,
                            'combo': combo,
                            'wins': stats['wins'],
                            'losses': stats['losses'],
                            'total': stats['total'],
                            'raw_wr': raw_wr,
                            'lower_wr': lower_wr,
                            'avg_time_min': avg_time / 60,
                            'avg_drawdown': avg_dd,
                            'sessions': stats['sessions']
                        })
        return combos
    
    def get_top_combos(self, min_trades: int = 5, min_lower_wr: float = 40.0) -> List[Dict]:
        """Get top performing combos by lower bound WR"""
        all_combos = self.get_all_combos()
        filtered = [c for c in all_combos if c['total'] >= min_trades and c['lower_wr'] >= min_lower_wr]
        return sorted(filtered, key=lambda x: x['lower_wr'], reverse=True)
    
    def get_session_report(self) -> str:
        """Generate session performance report"""
        sessions = {'asian': {'w': 0, 'l': 0}, 'london': {'w': 0, 'l': 0}, 'newyork': {'w': 0, 'l': 0}}
        
        for symbol, sides in self.combo_stats.items():
            for side, combos in sides.items():
                for combo, stats in combos.items():
                    for sess in ['asian', 'london', 'newyork']:
                        if sess in stats.get('sessions', {}):
                            sessions[sess]['w'] += stats['sessions'][sess].get('w', 0)
                            sessions[sess]['l'] += stats['sessions'][sess].get('l', 0)
        
        report = "📊 **SESSION PERFORMANCE**\n\n"
        for sess, data in sessions.items():
            total = data['w'] + data['l']
            wr = (data['w'] / total * 100) if total > 0 else 0
            lower_wr = wilson_lower_bound(data['w'], total)
            emoji = "🌏" if sess == "asian" else "🇬🇧" if sess == "london" else "🇺🇸"
            report += f"{emoji} **{sess.upper()}**\n"
            report += f"   WR: {wr:.0f}% (LB: {lower_wr:.0f}%)\n"
            report += f"   {data['w']}W / {data['l']}L\n\n"
        
        return report
    
    def generate_report(self) -> str:
        """Generate enhanced Telegram report"""
        uptime_hrs = (time.time() - self.started_at) / 3600
        total = self.total_wins + self.total_losses
        overall_wr = (self.total_wins / total * 100) if total > 0 else 0
        overall_lower_wr = wilson_lower_bound(self.total_wins, total)
        
        top_combos = self.get_top_combos(min_trades=3, min_lower_wr=40.0)[:5]
        
        report = (
            "📚 **COMBO LEARNING REPORT**\n\n"
            f"⏱️ Learning: {uptime_hrs:.1f}h\n"
            f"📊 Signals: {self.total_signals_tracked}\n"
            f"📈 WR: {overall_wr:.0f}% (LB: {overall_lower_wr:.0f}%)\n"
            f"🔢 Combos: {len(self.get_all_combos())}\n"
            f"🚀 Promoted: {len(self.promoted)}\n"
            f"🚫 Blacklisted: {len(self.blacklist)}\n\n"
        )
        
        if top_combos:
            report += "🏆 **TOP PERFORMERS** (LB_WR>40%)\n"
            for c in top_combos[:5]:
                report += f"• `{c['symbol']}` {c['side'].upper()}\n"
                report += f"  LB_WR: {c['lower_wr']:.0f}% (N={c['total']})\n"
        else:
            report += "🏆 No top performers yet\n"
        
        return report
    
    def get_promote_candidates(self) -> List[Dict]:
        """Get combos ready for promotion"""
        candidates = []
        for c in self.get_all_combos():
            key = f"{c['symbol']}:{c['side']}:{c['combo']}"
            if key in self.promoted:
                continue
            if c['total'] >= 10 and c['lower_wr'] >= 40:
                raw_wr = c['wins'] / c['total']
                ev = (raw_wr * 2) - ((1 - raw_wr) * 1)
                if ev > 0:
                    c['ev'] = ev
                    candidates.append(c)
        return sorted(candidates, key=lambda x: x['lower_wr'], reverse=True)
