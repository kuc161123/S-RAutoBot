"""
Performance Tracking and Monitoring Module
Tracks bot performance, generates reports, and monitors health
"""
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog
import json

from ..db.database import DatabaseManager

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # PnL metrics
    total_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    risk_reward_achieved: float = 0.0
    sharpe_ratio: float = 0.0
    
    # ML metrics
    ml_accuracy: float = 0.0
    ml_predictions_correct: int = 0
    ml_predictions_total: int = 0
    avg_ml_confidence: float = 0.0
    
    # Market regime performance
    regime_performance: Dict[str, Dict] = field(default_factory=dict)
    
    # Time metrics
    avg_trade_duration: float = 0.0
    longest_trade: float = 0.0
    shortest_trade: float = 0.0
    
    # Symbol performance
    symbol_performance: Dict[str, Dict] = field(default_factory=dict)
    
    # Zone performance
    zone_performance: Dict[str, Dict] = field(default_factory=dict)
    
    # Daily metrics
    daily_metrics: List[Dict] = field(default_factory=list)
    
    # System metrics
    uptime: float = 0.0
    errors_count: int = 0
    signals_generated: int = 0
    signals_executed: int = 0
    
    # Capital metrics
    starting_capital: float = 0.0
    current_capital: float = 0.0
    peak_capital: float = 0.0
    capital_utilization: float = 0.0


class PerformanceTracker:
    """
    Comprehensive performance tracking system
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.metrics = PerformanceMetrics()
        self.start_time = datetime.now()
        self.trade_history = []
        self.daily_pnl = []
        self.equity_curve = []
        
    async def initialize(self, starting_capital: float):
        """Initialize performance tracker"""
        self.metrics.starting_capital = starting_capital
        self.metrics.current_capital = starting_capital
        self.metrics.peak_capital = starting_capital
        
        # Load historical data if available
        await self._load_historical_data()
        
        logger.info(f"Performance tracker initialized with capital: ${starting_capital:.2f}")
    
    async def record_trade(self, trade_data: Dict):
        """Record a completed trade"""
        try:
            # Update trade counts
            self.metrics.total_trades += 1
            
            pnl = trade_data.get('pnl', 0)
            if pnl > 0:
                self.metrics.winning_trades += 1
                self.metrics.avg_win = (
                    (self.metrics.avg_win * (self.metrics.winning_trades - 1) + pnl) /
                    self.metrics.winning_trades
                )
            else:
                self.metrics.losing_trades += 1
                self.metrics.avg_loss = (
                    (self.metrics.avg_loss * (self.metrics.losing_trades - 1) + abs(pnl)) /
                    self.metrics.losing_trades
                )
            
            # Update PnL metrics
            self.metrics.total_pnl += pnl
            self.metrics.total_fees += trade_data.get('fees', 0)
            self.metrics.net_pnl = self.metrics.total_pnl - self.metrics.total_fees
            
            # Update best/worst trade
            if pnl > self.metrics.best_trade:
                self.metrics.best_trade = pnl
            if pnl < self.metrics.worst_trade:
                self.metrics.worst_trade = pnl
            
            # Update win rate
            self.metrics.win_rate = (
                self.metrics.winning_trades / self.metrics.total_trades * 100
                if self.metrics.total_trades > 0 else 0
            )
            
            # Update capital
            self.metrics.current_capital += pnl - trade_data.get('fees', 0)
            if self.metrics.current_capital > self.metrics.peak_capital:
                self.metrics.peak_capital = self.metrics.current_capital
            
            # Calculate drawdown
            drawdown = (self.metrics.peak_capital - self.metrics.current_capital) / self.metrics.peak_capital * 100
            self.metrics.current_drawdown = drawdown
            if drawdown > self.metrics.max_drawdown:
                self.metrics.max_drawdown = drawdown
            
            # Update ML metrics
            if 'ml_prediction' in trade_data:
                self.metrics.ml_predictions_total += 1
                ml_predicted_win = trade_data['ml_success_probability'] > 0.5
                actual_win = pnl > 0
                if ml_predicted_win == actual_win:
                    self.metrics.ml_predictions_correct += 1
                self.metrics.ml_accuracy = (
                    self.metrics.ml_predictions_correct / self.metrics.ml_predictions_total * 100
                )
                
                # Update average ML confidence
                confidence = trade_data.get('ml_confidence', 0)
                self.metrics.avg_ml_confidence = (
                    (self.metrics.avg_ml_confidence * (self.metrics.ml_predictions_total - 1) + confidence) /
                    self.metrics.ml_predictions_total
                )
            
            # Update regime performance
            regime = trade_data.get('market_regime', 'unknown')
            if regime not in self.metrics.regime_performance:
                self.metrics.regime_performance[regime] = {
                    'trades': 0,
                    'wins': 0,
                    'total_pnl': 0,
                    'win_rate': 0
                }
            
            regime_stats = self.metrics.regime_performance[regime]
            regime_stats['trades'] += 1
            if pnl > 0:
                regime_stats['wins'] += 1
            regime_stats['total_pnl'] += pnl
            regime_stats['win_rate'] = regime_stats['wins'] / regime_stats['trades'] * 100
            
            # Update symbol performance
            symbol = trade_data.get('symbol', 'UNKNOWN')
            if symbol not in self.metrics.symbol_performance:
                self.metrics.symbol_performance[symbol] = {
                    'trades': 0,
                    'wins': 0,
                    'total_pnl': 0,
                    'win_rate': 0
                }
            
            symbol_stats = self.metrics.symbol_performance[symbol]
            symbol_stats['trades'] += 1
            if pnl > 0:
                symbol_stats['wins'] += 1
            symbol_stats['total_pnl'] += pnl
            symbol_stats['win_rate'] = symbol_stats['wins'] / symbol_stats['trades'] * 100
            
            # Update zone performance
            zone_type = trade_data.get('zone_type', 'unknown')
            if zone_type not in self.metrics.zone_performance:
                self.metrics.zone_performance[zone_type] = {
                    'trades': 0,
                    'wins': 0,
                    'total_pnl': 0,
                    'win_rate': 0
                }
            
            zone_stats = self.metrics.zone_performance[zone_type]
            zone_stats['trades'] += 1
            if pnl > 0:
                zone_stats['wins'] += 1
            zone_stats['total_pnl'] += pnl
            zone_stats['win_rate'] = zone_stats['wins'] / zone_stats['trades'] * 100
            
            # Update trade duration
            duration = trade_data.get('duration_minutes', 0)
            if duration > 0:
                self.metrics.avg_trade_duration = (
                    (self.metrics.avg_trade_duration * (self.metrics.total_trades - 1) + duration) /
                    self.metrics.total_trades
                )
                if duration > self.metrics.longest_trade:
                    self.metrics.longest_trade = duration
                if self.metrics.shortest_trade == 0 or duration < self.metrics.shortest_trade:
                    self.metrics.shortest_trade = duration
            
            # Add to trade history
            self.trade_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'pnl': pnl,
                'capital': self.metrics.current_capital,
                'drawdown': drawdown
            })
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': datetime.now(),
                'capital': self.metrics.current_capital
            })
            
            # Calculate Sharpe ratio (simplified)
            if len(self.daily_pnl) > 30:
                returns = [d['return'] for d in self.daily_pnl[-30:]]
                avg_return = sum(returns) / len(returns)
                std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                if std_return > 0:
                    self.metrics.sharpe_ratio = (avg_return * 365 ** 0.5) / std_return
            
            logger.info(
                f"📊 Trade recorded: {symbol} PnL=${pnl:.2f} "
                f"Win Rate={self.metrics.win_rate:.1f}% "
                f"Capital=${self.metrics.current_capital:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    async def record_signal(self, signal_data: Dict):
        """Record a generated signal"""
        self.metrics.signals_generated += 1
        
        if signal_data.get('executed', False):
            self.metrics.signals_executed += 1
    
    async def record_error(self, error_type: str, error_message: str):
        """Record an error"""
        self.metrics.errors_count += 1
        
        # Log to database
        self.db_manager.log_error(error_type, error_message)
    
    async def update_daily_metrics(self):
        """Update daily performance metrics"""
        try:
            # Calculate daily PnL
            today_trades = [
                t for t in self.trade_history
                if t['timestamp'].date() == datetime.now().date()
            ]
            
            daily_pnl = sum(t['pnl'] for t in today_trades)
            daily_return = (
                daily_pnl / self.metrics.starting_capital * 100
                if self.metrics.starting_capital > 0 else 0
            )
            
            daily_metric = {
                'date': datetime.now().date(),
                'trades': len(today_trades),
                'pnl': daily_pnl,
                'return': daily_return,
                'capital': self.metrics.current_capital,
                'drawdown': self.metrics.current_drawdown
            }
            
            self.metrics.daily_metrics.append(daily_metric)
            self.daily_pnl.append(daily_metric)
            
            # Update capital utilization
            if self.metrics.current_capital > 0:
                # Simplified: assume max 20% capital per trade, 5 concurrent trades
                max_capital_use = self.metrics.current_capital * 0.2 * 5
                actual_use = sum(
                    t.get('position_value', 0) 
                    for t in self.trade_history[-5:]
                    if t['timestamp'] > datetime.now() - timedelta(hours=1)
                )
                self.metrics.capital_utilization = actual_use / max_capital_use * 100
            
            # Update uptime
            self.metrics.uptime = (datetime.now() - self.start_time).total_seconds() / 3600
            
            logger.info(
                f"📈 Daily Update: PnL=${daily_pnl:.2f} ({daily_return:+.2f}%) "
                f"Trades={len(today_trades)} Capital=${self.metrics.current_capital:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error updating daily metrics: {e}")
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        
        # Calculate additional metrics
        profit_factor = (
            abs(self.metrics.avg_win * self.metrics.winning_trades) /
            abs(self.metrics.avg_loss * self.metrics.losing_trades)
            if self.metrics.losing_trades > 0 and self.metrics.avg_loss > 0 else 0
        )
        
        total_return = (
            (self.metrics.current_capital - self.metrics.starting_capital) /
            self.metrics.starting_capital * 100
            if self.metrics.starting_capital > 0 else 0
        )
        
        # Find best performing symbol
        best_symbol = max(
            self.metrics.symbol_performance.items(),
            key=lambda x: x[1]['total_pnl'],
            default=('N/A', {'total_pnl': 0})
        )
        
        # Find best regime
        best_regime = max(
            self.metrics.regime_performance.items(),
            key=lambda x: x[1]['win_rate'],
            default=('N/A', {'win_rate': 0})
        )
        
        report = {
            'overview': {
                'total_trades': self.metrics.total_trades,
                'win_rate': self.metrics.win_rate,
                'profit_factor': profit_factor,
                'total_return': total_return,
                'net_pnl': self.metrics.net_pnl,
                'max_drawdown': self.metrics.max_drawdown,
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'uptime_hours': self.metrics.uptime
            },
            'trade_stats': {
                'winning_trades': self.metrics.winning_trades,
                'losing_trades': self.metrics.losing_trades,
                'avg_win': self.metrics.avg_win,
                'avg_loss': self.metrics.avg_loss,
                'best_trade': self.metrics.best_trade,
                'worst_trade': self.metrics.worst_trade,
                'avg_duration_minutes': self.metrics.avg_trade_duration
            },
            'ml_performance': {
                'accuracy': self.metrics.ml_accuracy,
                'predictions_correct': self.metrics.ml_predictions_correct,
                'predictions_total': self.metrics.ml_predictions_total,
                'avg_confidence': self.metrics.avg_ml_confidence
            },
            'regime_performance': self.metrics.regime_performance,
            'symbol_performance': self.metrics.symbol_performance,
            'zone_performance': self.metrics.zone_performance,
            'best_performers': {
                'symbol': best_symbol[0],
                'symbol_pnl': best_symbol[1]['total_pnl'],
                'regime': best_regime[0],
                'regime_win_rate': best_regime[1]['win_rate']
            },
            'system_health': {
                'errors_count': self.metrics.errors_count,
                'signals_generated': self.metrics.signals_generated,
                'signals_executed': self.metrics.signals_executed,
                'signal_execution_rate': (
                    self.metrics.signals_executed / self.metrics.signals_generated * 100
                    if self.metrics.signals_generated > 0 else 0
                )
            },
            'capital': {
                'starting': self.metrics.starting_capital,
                'current': self.metrics.current_capital,
                'peak': self.metrics.peak_capital,
                'utilization': self.metrics.capital_utilization
            }
        }
        
        return report
    
    def get_summary_text(self) -> str:
        """Get performance summary as formatted text"""
        
        report = self.get_performance_report()
        
        text = f"""
📊 **PERFORMANCE REPORT**

**Overview:**
• Total Trades: {report['overview']['total_trades']}
• Win Rate: {report['overview']['win_rate']:.1f}%
• Profit Factor: {report['overview']['profit_factor']:.2f}
• Total Return: {report['overview']['total_return']:+.2f}%
• Net PnL: ${report['overview']['net_pnl']:.2f}
• Max Drawdown: -{report['overview']['max_drawdown']:.2f}%
• Sharpe Ratio: {report['overview']['sharpe_ratio']:.2f}

**ML Performance:**
• Accuracy: {report['ml_performance']['accuracy']:.1f}%
• Avg Confidence: {report['ml_performance']['avg_confidence']:.1%}

**Best Performers:**
• Symbol: {report['best_performers']['symbol']} (${report['best_performers']['symbol_pnl']:.2f})
• Market Regime: {report['best_performers']['regime']} ({report['best_performers']['regime_win_rate']:.1f}% win rate)

**Capital:**
• Current: ${report['capital']['current']:.2f}
• Peak: ${report['capital']['peak']:.2f}
• Utilization: {report['capital']['utilization']:.1f}%

**System Health:**
• Uptime: {report['overview']['uptime_hours']:.1f} hours
• Errors: {report['system_health']['errors_count']}
• Signal Execution: {report['system_health']['signal_execution_rate']:.1f}%
"""
        
        return text.strip()
    
    async def _load_historical_data(self):
        """Load historical performance data from database"""
        try:
            # Load recent trades
            recent_trades = self.db_manager.get_closed_trades_today()
            for trade in recent_trades:
                await self.record_trade(trade)
            
            logger.info(f"Loaded {len(recent_trades)} historical trades")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    async def save_report(self):
        """Save performance report to database"""
        try:
            report = self.get_performance_report()
            
            # Save to database or file
            with open('/tmp/performance_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info("Performance report saved")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")


# Export singleton
performance_tracker = None

def get_performance_tracker(db_manager: DatabaseManager) -> PerformanceTracker:
    """Get or create performance tracker instance"""
    global performance_tracker
    if performance_tracker is None:
        performance_tracker = PerformanceTracker(db_manager)
    return performance_tracker