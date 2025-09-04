"""
Enhanced Trade History and Statistics Tracker with PostgreSQL persistence
Ensures trades are never lost, even on crashes or restarts
"""
import json
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging
from urllib.parse import urlparse

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
        # Remove database-specific fields that aren't part of the Trade class
        d = d.copy()  # Don't modify original
        d.pop('id', None)  # Remove 'id' if it exists
        
        # Convert timestamps
        d['entry_time'] = datetime.fromisoformat(d['entry_time']) if isinstance(d['entry_time'], str) else d['entry_time']
        d['exit_time'] = datetime.fromisoformat(d['exit_time']) if isinstance(d['exit_time'], str) else d['exit_time']
        return cls(**d)

class TradeTrackerPostgres:
    """Enhanced trade tracker with PostgreSQL persistence"""
    
    def __init__(self, fallback_file: str = "trade_history.json"):
        self.fallback_file = fallback_file
        self.trades: List[Trade] = []
        self.conn = None
        self.use_db = False
        
        # Try to connect to database
        if self._connect_db():
            self._init_tables()
            self.load_trades_from_db()
            self.use_db = True
        else:
            logger.warning("Using JSON file fallback for trade history")
            self.load_trades_from_file()
    
    def _connect_db(self) -> bool:
        """Connect to PostgreSQL database"""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                logger.info("No DATABASE_URL found, will use JSON file")
                return False
            
            # Parse the URL
            url = urlparse(database_url)
            
            self.conn = psycopg2.connect(
                host=url.hostname,
                port=url.port,
                database=url.path[1:],  # Remove leading '/'
                user=url.username,
                password=url.password,
                sslmode='require' if url.hostname != 'localhost' else 'prefer'
            )
            
            logger.info("Connected to PostgreSQL for trade tracking")
            return True
            
        except Exception as e:
            logger.warning(f"Could not connect to database: {e}")
            return False
    
    def _init_tables(self):
        """Create tables if they don't exist"""
        try:
            with self.conn.cursor() as cur:
                # Create trades table with all necessary fields
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(50) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        entry_price DECIMAL(20, 8) NOT NULL,
                        exit_price DECIMAL(20, 8) NOT NULL,
                        quantity DECIMAL(20, 8) NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        exit_time TIMESTAMP NOT NULL,
                        pnl_usd DECIMAL(20, 8) NOT NULL,
                        pnl_percent DECIMAL(20, 8) NOT NULL,
                        exit_reason VARCHAR(20) NOT NULL,
                        leverage DECIMAL(10, 2) DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes separately (PostgreSQL syntax)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON trades (symbol)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_exit_time ON trades (exit_time)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_pnl_usd ON trades (pnl_usd)")
                
                # Create statistics cache table for fast dashboard queries
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trade_stats_cache (
                        period VARCHAR(20) PRIMARY KEY,
                        total_trades INT,
                        wins INT,
                        losses INT,
                        win_rate DECIMAL(5, 2),
                        total_pnl DECIMAL(20, 8),
                        avg_win DECIMAL(20, 8),
                        avg_loss DECIMAL(20, 8),
                        profit_factor DECIMAL(10, 2),
                        best_trade_pnl DECIMAL(20, 8),
                        worst_trade_pnl DECIMAL(20, 8),
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                self.conn.commit()
                logger.info("Database tables initialized")
                
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            self.conn.rollback()
    
    def load_trades_from_db(self):
        """Load all trades from database"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM trades 
                    ORDER BY exit_time DESC 
                    LIMIT 10000
                """)
                
                rows = cur.fetchall()
                self.trades = [Trade.from_dict(row) for row in rows]
                logger.info(f"Loaded {len(self.trades)} trades from database")
                
        except Exception as e:
            logger.error(f"Failed to load trades from database: {e}")
            self.trades = []
    
    def load_trades_from_file(self):
        """Load trade history from JSON file (fallback)"""
        if os.path.exists(self.fallback_file):
            try:
                with open(self.fallback_file, 'r') as f:
                    data = json.load(f)
                    self.trades = [Trade.from_dict(t) for t in data]
                logger.info(f"Loaded {len(self.trades)} trades from JSON file")
            except Exception as e:
                logger.error(f"Failed to load trade history from file: {e}")
                self.trades = []
        else:
            self.trades = []
    
    def save_trades_to_file(self):
        """Save trades to JSON file (fallback or backup)"""
        try:
            with open(self.fallback_file, 'w') as f:
                data = [t.to_dict() for t in self.trades]
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.trades)} trades to JSON file")
        except Exception as e:
            logger.error(f"Failed to save trade history to file: {e}")
    
    def add_trade(self, trade: Trade):
        """Add a completed trade to both database and memory"""
        self.trades.append(trade)
        
        if self.use_db:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO trades (
                            symbol, side, entry_price, exit_price, quantity,
                            entry_time, exit_time, pnl_usd, pnl_percent,
                            exit_reason, leverage
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        trade.symbol, trade.side, trade.entry_price, trade.exit_price,
                        trade.quantity, trade.entry_time, trade.exit_time,
                        trade.pnl_usd, trade.pnl_percent, trade.exit_reason, trade.leverage
                    ))
                    self.conn.commit()
                    
                    # Update statistics cache
                    self._update_stats_cache()
                    
                logger.info(f"Trade saved to database: {trade.symbol} {trade.side} PnL: ${trade.pnl_usd:.2f}")
                
                # Also save to file as backup
                self.save_trades_to_file()
                
            except Exception as e:
                logger.error(f"Failed to save trade to database: {e}")
                self.conn.rollback()
                # Fallback to file only
                self.save_trades_to_file()
        else:
            # No database, save to file
            self.save_trades_to_file()
        
        logger.info(f"Trade recorded: {trade.symbol} {trade.side} PnL: ${trade.pnl_usd:.2f} ({trade.pnl_percent:.2f}%)")
    
    def _update_stats_cache(self):
        """Update statistics cache for faster queries"""
        try:
            periods = [
                ('all_time', None),
                ('last_7d', 7),
                ('last_30d', 30)
            ]
            
            for period_name, days in periods:
                stats = self.get_statistics(days)
                
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO trade_stats_cache (
                            period, total_trades, wins, losses, win_rate,
                            total_pnl, avg_win, avg_loss, profit_factor,
                            best_trade_pnl, worst_trade_pnl, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (period) DO UPDATE SET
                            total_trades = EXCLUDED.total_trades,
                            wins = EXCLUDED.wins,
                            losses = EXCLUDED.losses,
                            win_rate = EXCLUDED.win_rate,
                            total_pnl = EXCLUDED.total_pnl,
                            avg_win = EXCLUDED.avg_win,
                            avg_loss = EXCLUDED.avg_loss,
                            profit_factor = EXCLUDED.profit_factor,
                            best_trade_pnl = EXCLUDED.best_trade_pnl,
                            worst_trade_pnl = EXCLUDED.worst_trade_pnl,
                            updated_at = NOW()
                    """, (
                        period_name,
                        stats['total_trades'],
                        stats['wins'],
                        stats['losses'],
                        stats['win_rate'],
                        stats['total_pnl'],
                        stats['avg_win'],
                        stats['avg_loss'],
                        stats['profit_factor'],
                        stats['best_trade'].pnl_usd if stats['best_trade'] else 0,
                        stats['worst_trade'].pnl_usd if stats['worst_trade'] else 0
                    ))
                    
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to update stats cache: {e}")
            self.conn.rollback()
    
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
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'best_trade': None,
                'worst_trade': None,
                'trading_days': 0,
                'daily_avg': 0,
                'top_symbols': [],
                'worst_symbols': [],
                'by_symbol': {}
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
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
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
        
        # Format profit factor nicely
        if stats['profit_factor'] == float('inf'):
            msg += f"Profit Factor: ∞ (no losses)\n\n"
        elif stats['profit_factor'] > 100:
            msg += f"Profit Factor: >100\n\n"
        else:
            msg += f"Profit Factor: {stats['profit_factor']:.2f}\n\n"
        
        # Best/Worst trades
        if stats['best_trade']:
            msg += "*🏆 Best Trade*\n"
            t = stats['best_trade']
            msg += f"{t.symbol} {t.side}: +${t.pnl_usd:.2f} ({t.pnl_percent:.1f}%)\n\n"
        
        if stats['worst_trade'] and stats['worst_trade'].pnl_usd < 0:
            msg += "*😔 Worst Trade*\n"
            t = stats['worst_trade']
            msg += f"{t.symbol} {t.side}: ${t.pnl_usd:.2f} ({t.pnl_percent:.1f}%)\n\n"
        
        # Top performing symbols
        if stats['top_symbols']:
            msg += "*🎯 Top Symbols*\n"
            for symbol, data in stats['top_symbols'][:3]:
                win_rate = data['win_rate']
                msg += f"{symbol}: ${data['pnl']:.2f} ({data['trades']} trades, {win_rate:.0f}% WR)\n"
        
        return msg
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

# Backward compatibility
TradeTracker = TradeTrackerPostgres