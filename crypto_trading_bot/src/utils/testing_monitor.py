"""
Testing Phase Monitor
Tracks ML learning progress and ensures safe testing
"""
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import structlog
import pandas as pd
from dataclasses import dataclass

logger = structlog.get_logger(__name__)

@dataclass
class TestingMetrics:
    """Metrics for testing phase"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_usd: float = 0
    total_pnl_percent: float = 0
    max_drawdown_usd: float = 0
    max_drawdown_percent: float = 0
    best_symbol: str = ""
    worst_symbol: str = ""
    ml_accuracy: float = 0
    patterns_learned: int = 0
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def avg_win_loss_ratio(self) -> float:
        if self.losing_trades == 0:
            return float('inf') if self.winning_trades > 0 else 0
        return self.winning_trades / self.losing_trades

class TestingMonitor:
    """
    Monitors testing phase to ensure safe learning
    Prevents excessive losses while gathering ML training data
    """
    
    def __init__(self, initial_balance: float = 100):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.metrics = TestingMetrics()
        self.trade_history = []
        self.daily_pnl = {}
        self.symbol_performance = {}
        
        # Safety thresholds
        self.max_loss_usd = 10  # Stop if we lose $10 (10% of $100)
        self.max_daily_loss_usd = 2  # Stop trading for the day if down $2
        self.min_balance = 90  # Stop if balance drops below $90
        
        # Learning thresholds
        self.min_trades_for_ml = 50
        self.min_win_rate_to_increase_risk = 40  # Need 40% win rate to increase risk
        
    def should_continue_trading(self) -> tuple[bool, str]:
        """Check if we should continue trading"""
        
        # Check total loss
        if self.current_balance < self.min_balance:
            return False, f"Balance too low: ${self.current_balance:.2f} < ${self.min_balance}"
        
        # Check daily loss
        today = datetime.now().date()
        if today in self.daily_pnl:
            if self.daily_pnl[today] <= -self.max_daily_loss_usd:
                return False, f"Daily loss limit reached: ${self.daily_pnl[today]:.2f}"
        
        # Check drawdown
        if self.metrics.max_drawdown_usd >= self.max_loss_usd:
            return False, f"Max drawdown reached: ${self.metrics.max_drawdown_usd:.2f}"
        
        return True, "OK to continue"
    
    def record_trade(self, trade: Dict):
        """Record a completed trade"""
        
        self.trade_history.append(trade)
        self.metrics.total_trades += 1
        
        pnl = trade.get('pnl', 0)
        symbol = trade.get('symbol', '')
        
        # Update balance
        self.current_balance += pnl
        
        # Update metrics
        if pnl > 0:
            self.metrics.winning_trades += 1
        else:
            self.metrics.losing_trades += 1
        
        self.metrics.total_pnl_usd += pnl
        self.metrics.total_pnl_percent = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Update daily PnL
        today = datetime.now().date()
        if today not in self.daily_pnl:
            self.daily_pnl[today] = 0
        self.daily_pnl[today] += pnl
        
        # Update symbol performance
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = {'trades': 0, 'pnl': 0, 'wins': 0}
        
        self.symbol_performance[symbol]['trades'] += 1
        self.symbol_performance[symbol]['pnl'] += pnl
        if pnl > 0:
            self.symbol_performance[symbol]['wins'] += 1
        
        # Calculate drawdown
        peak_balance = max(t.get('balance', self.initial_balance) for t in self.trade_history)
        current_drawdown = peak_balance - self.current_balance
        self.metrics.max_drawdown_usd = max(self.metrics.max_drawdown_usd, current_drawdown)
        self.metrics.max_drawdown_percent = (self.metrics.max_drawdown_usd / peak_balance) * 100
        
        # Update best/worst symbol
        if self.symbol_performance:
            best = max(self.symbol_performance.items(), key=lambda x: x[1]['pnl'])
            worst = min(self.symbol_performance.items(), key=lambda x: x[1]['pnl'])
            self.metrics.best_symbol = best[0]
            self.metrics.worst_symbol = worst[0]
        
        # Log progress
        logger.info(f"Trade #{self.metrics.total_trades}: {symbol} PnL=${pnl:.2f} | "
                   f"Balance=${self.current_balance:.2f} | "
                   f"Win Rate={self.metrics.win_rate:.1f}% | "
                   f"Total PnL={self.metrics.total_pnl_percent:.1f}%")
    
    def get_risk_adjustment(self) -> float:
        """Get risk adjustment based on performance"""
        
        # First 20 trades: minimal risk
        if self.metrics.total_trades < 20:
            return 0.5
        
        # Next 30 trades: slightly increased if performing well
        elif self.metrics.total_trades < 50:
            if self.metrics.win_rate >= 40:
                return 0.75
            return 0.5
        
        # After 50 trades: adaptive risk
        else:
            if self.metrics.win_rate >= 60:
                return 1.5  # Increase risk if doing very well
            elif self.metrics.win_rate >= 50:
                return 1.0  # Normal risk if doing OK
            elif self.metrics.win_rate >= 40:
                return 0.75  # Reduce risk if struggling
            else:
                return 0.5  # Minimal risk if performing poorly
    
    def should_use_ml(self) -> bool:
        """Check if we have enough data for ML"""
        return self.metrics.total_trades >= self.min_trades_for_ml
    
    def get_summary(self) -> Dict:
        """Get testing summary"""
        
        return {
            'phase': self._get_current_phase(),
            'balance': f"${self.current_balance:.2f}",
            'total_trades': self.metrics.total_trades,
            'win_rate': f"{self.metrics.win_rate:.1f}%",
            'total_pnl_usd': f"${self.metrics.total_pnl_usd:.2f}",
            'total_pnl_percent': f"{self.metrics.total_pnl_percent:.1f}%",
            'max_drawdown': f"${self.metrics.max_drawdown_usd:.2f} ({self.metrics.max_drawdown_percent:.1f}%)",
            'best_symbol': self.metrics.best_symbol,
            'worst_symbol': self.metrics.worst_symbol,
            'risk_level': self.get_risk_adjustment(),
            'ml_ready': self.should_use_ml(),
            'can_continue': self.should_continue_trading()[0]
        }
    
    def _get_current_phase(self) -> str:
        """Get current testing phase"""
        
        trades = self.metrics.total_trades
        
        if trades < 20:
            return "Initial Data Collection"
        elif trades < 50:
            return "Early Learning"
        elif trades < 100:
            return "ML Training"
        elif trades < 150:
            return "Validation"
        else:
            return "Optimization"
    
    async def generate_report(self) -> str:
        """Generate detailed testing report"""
        
        report = []
        report.append("=" * 50)
        report.append("TESTING PHASE REPORT")
        report.append("=" * 50)
        report.append(f"Current Phase: {self._get_current_phase()}")
        report.append(f"Account Balance: ${self.current_balance:.2f} ({self.metrics.total_pnl_percent:+.1f}%)")
        report.append("")
        
        report.append("PERFORMANCE METRICS:")
        report.append(f"  Total Trades: {self.metrics.total_trades}")
        report.append(f"  Win Rate: {self.metrics.win_rate:.1f}%")
        report.append(f"  Wins/Losses: {self.metrics.winning_trades}/{self.metrics.losing_trades}")
        report.append(f"  Max Drawdown: ${self.metrics.max_drawdown_usd:.2f} ({self.metrics.max_drawdown_percent:.1f}%)")
        report.append("")
        
        report.append("SYMBOL PERFORMANCE:")
        for symbol, perf in sorted(self.symbol_performance.items(), key=lambda x: x[1]['pnl'], reverse=True):
            win_rate = (perf['wins'] / perf['trades'] * 100) if perf['trades'] > 0 else 0
            report.append(f"  {symbol}: ${perf['pnl']:.2f} ({perf['trades']} trades, {win_rate:.0f}% win)")
        report.append("")
        
        report.append("ML LEARNING STATUS:")
        if self.should_use_ml():
            report.append("  ✅ ML Ready - Sufficient data collected")
            report.append(f"  Patterns Learned: {self.metrics.patterns_learned}")
        else:
            trades_needed = self.min_trades_for_ml - self.metrics.total_trades
            report.append(f"  ⏳ Need {trades_needed} more trades for ML")
        report.append("")
        
        report.append("RISK MANAGEMENT:")
        report.append(f"  Current Risk Level: {self.get_risk_adjustment():.2f}x")
        can_trade, reason = self.should_continue_trading()
        if can_trade:
            report.append("  ✅ OK to continue trading")
        else:
            report.append(f"  ⛔ Trading stopped: {reason}")
        report.append("")
        
        report.append("RECOMMENDATIONS:")
        if self.metrics.win_rate < 40 and self.metrics.total_trades > 30:
            report.append("  ⚠️ Low win rate - Review strategy parameters")
        if self.metrics.max_drawdown_percent > 5:
            report.append("  ⚠️ High drawdown - Consider tighter risk management")
        if self.metrics.best_symbol and self.metrics.worst_symbol:
            report.append(f"  💡 Focus on {self.metrics.best_symbol}, avoid {self.metrics.worst_symbol}")
        
        report.append("=" * 50)
        
        return "\n".join(report)

# Global instance
testing_monitor = TestingMonitor(initial_balance=100)