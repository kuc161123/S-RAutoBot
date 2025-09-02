"""
Trade History and Statistics Tracker
Tracks all trades, calculates PnL, and provides performance metrics
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl_usd: float
    pnl_percent: float
    exit_reason: str  # "tp", "sl", "manual"
    leverage: float = 1.0
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        d['entry_time'] = self.entry_time.isoformat()
        d['exit_time'] = self.exit_time.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d):
        """Create from dictionary"""
        d['entry_time'] = datetime.fromisoformat(d['entry_time'])
        d['exit_time'] = datetime.fromisoformat(d['exit_time'])
        return cls(**d)

class TradeTracker:
    """Tracks trade history and calculates statistics"""
    
    def __init__(self, filename: str = "trade_history.json"):
        self.filename = filename
        self.trades: List[Trade] = []
        self.load_trades()
    
    def load_trades(self):
        """Load trade history from file"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.trades = [Trade.from_dict(t) for t in data]
                logger.info(f"Loaded {len(self.trades)} historical trades")
            except Exception as e:
                logger.error(f"Failed to load trade history: {e}")
                self.trades = []
        else:
            self.trades = []
    
    def save_trades(self):
        """Save trade history to file"""
        try:
            with open(self.filename, 'w') as f:
                data = [t.to_dict() for t in self.trades]
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.trades)} trades to history")
        except Exception as e:
            logger.error(f"Failed to save trade history: {e}")
    
    def add_trade(self, trade: Trade):
        """Add a completed trade"""
        self.trades.append(trade)
        self.save_trades()
        logger.info(f"Trade recorded: {trade.symbol} {trade.side} PnL: ${trade.pnl_usd:.2f} ({trade.pnl_percent:.2f}%)")
    
    def calculate_pnl(self, symbol: str, side: str, entry: float, exit: float, 
                     qty: float, leverage: float = 1.0) -> tuple[float, float]:
        """Calculate PnL for a trade"""
        if side == "long":
            pnl_usd = (exit - entry) * qty
            pnl_percent = ((exit - entry) / entry) * 100 * leverage
        else:  # short
            pnl_usd = (entry - exit) * qty
            pnl_percent = ((entry - exit) / entry) * 100 * leverage
        
        return pnl_usd, pnl_percent
    
    def get_statistics(self, days: Optional[int] = None) -> Dict:
        """Calculate comprehensive statistics"""
        
        # Filter trades by time period if specified
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            trades = [t for t in self.trades if t.exit_time >= cutoff]
        else:
            trades = self.trades
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'best_trade': None,
                'worst_trade': None,
                'trading_days': 0,
                'daily_avg': 0
            }
        
        # Calculate metrics
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        
        total_pnl = sum(t.pnl_usd for t in trades)
        win_rate = (len(wins) / len(trades)) * 100 if trades else 0
        
        avg_win = sum(t.pnl_usd for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl_usd for t in losses) / len(losses) if losses else 0
        
        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        best_trade = max(trades, key=lambda t: t.pnl_usd) if trades else None
        worst_trade = min(trades, key=lambda t: t.pnl_usd) if trades else None
        
        # Calculate trading days
        if trades:
            first_trade = min(trades, key=lambda t: t.exit_time)
            last_trade = max(trades, key=lambda t: t.exit_time)
            trading_days = max(1, (last_trade.exit_time - first_trade.exit_time).days + 1)
            daily_avg = total_pnl / trading_days
        else:
            trading_days = 0
            daily_avg = 0
        
        # By symbol statistics
        by_symbol = {}
        for trade in trades:
            if trade.symbol not in by_symbol:
                by_symbol[trade.symbol] = {'trades': 0, 'pnl': 0, 'wins': 0}
            by_symbol[trade.symbol]['trades'] += 1
            by_symbol[trade.symbol]['pnl'] += trade.pnl_usd
            if trade.pnl_usd > 0:
                by_symbol[trade.symbol]['wins'] += 1
        
        # Calculate win rate for each symbol
        for symbol in by_symbol:
            stats = by_symbol[symbol]
            stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
        
        # Sort symbols by PnL
        top_symbols = sorted(by_symbol.items(), key=lambda x: x[1]['pnl'], reverse=True)[:5]
        worst_symbols = sorted(by_symbol.items(), key=lambda x: x[1]['pnl'])[:5]
        
        return {
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'trading_days': trading_days,
            'daily_avg': daily_avg,
            'top_symbols': top_symbols,
            'worst_symbols': worst_symbols,
            'by_symbol': by_symbol
        }
    
    def format_stats_message(self, days: Optional[int] = None) -> str:
        """Format statistics as a readable message"""
        stats = self.get_statistics(days)
        
        if stats['total_trades'] == 0:
            return "📊 *No trades recorded yet*"
        
        period = f"Last {days} days" if days else "All time"
        
        msg = f"📊 *Trading Statistics - {period}*\n"
        msg += "=" * 30 + "\n\n"
        
        # Overall performance
        msg += "*📈 Performance Overview*\n"
        msg += f"Total Trades: {stats['total_trades']}\n"
        msg += f"Wins/Losses: {stats['wins']}/{stats['losses']}\n"
        msg += f"Win Rate: {stats['win_rate']:.1f}%\n"
        msg += f"Total PnL: ${stats['total_pnl']:.2f}\n"
        msg += f"Daily Average: ${stats['daily_avg']:.2f}\n\n"
        
        # Risk metrics
        msg += "*⚖️ Risk Metrics*\n"
        msg += f"Avg Win: ${stats['avg_win']:.2f}\n"
        msg += f"Avg Loss: ${stats['avg_loss']:.2f}\n"
        if stats['profit_factor'] != float('inf'):
            msg += f"Profit Factor: {stats['profit_factor']:.2f}\n\n"
        else:
            msg += "Profit Factor: ∞ (no losses)\n\n"
        
        # Best/Worst trades
        if stats['best_trade']:
            bt = stats['best_trade']
            msg += f"*🏆 Best Trade*\n"
            msg += f"{bt.symbol} {bt.side}: +${bt.pnl_usd:.2f} ({bt.pnl_percent:.1f}%)\n\n"
        
        if stats['worst_trade'] and stats['worst_trade'].pnl_usd < 0:
            wt = stats['worst_trade']
            msg += f"*😔 Worst Trade*\n"
            msg += f"{wt.symbol} {wt.side}: ${wt.pnl_usd:.2f} ({wt.pnl_percent:.1f}%)\n\n"
        
        # Top performing symbols
        if stats['top_symbols']:
            msg += "*🎯 Top Symbols*\n"
            for symbol, data in stats['top_symbols'][:3]:
                msg += f"{symbol}: ${data['pnl']:.2f} ({data['trades']} trades, {data['win_rate']:.0f}% WR)\n"
        
        return msg
    
    def get_recent_trades(self, limit: int = 10) -> List[Trade]:
        """Get most recent trades"""
        return sorted(self.trades, key=lambda t: t.exit_time, reverse=True)[:limit]
    
    def format_recent_trades(self, limit: int = 5) -> str:
        """Format recent trades as a message"""
        recent = self.get_recent_trades(limit)
        
        if not recent:
            return "No recent trades"
        
        msg = "*🕐 Recent Trades*\n"
        for trade in recent:
            emoji = "✅" if trade.pnl_usd > 0 else "❌"
            exit_emoji = {"tp": "🎯", "sl": "🛑", "manual": "👤"}.get(trade.exit_reason, "❓")
            
            msg += f"{emoji} {trade.symbol} {trade.side.upper()}: "
            msg += f"${trade.pnl_usd:.2f} ({trade.pnl_percent:.1f}%) {exit_emoji}\n"
        
        return msg