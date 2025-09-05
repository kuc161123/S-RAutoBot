"""
Symbol-Specific Data Collector for Future ML Training
Collects comprehensive market data for each symbol to enable advanced ML later
"""
import logging
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import json
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class SymbolProfile:
    """Symbol characteristics and behavior patterns"""
    symbol: str
    category: str  # major/mid/small/meme/defi/gaming/ai
    market_cap_rank: Optional[int] = None
    avg_volume_30d: float = 0.0
    volatility_percentile: float = 0.0
    typical_hold_minutes: float = 0.0
    breakout_success_rate: float = 0.0
    best_session: str = "unknown"  # asian/european/us
    worst_session: str = "unknown"
    fakeout_rate: float = 0.0
    avg_spread_bps: float = 0.0
    correlation_with_btc: float = 0.0
    news_sensitivity: float = 0.5  # 0-1 scale
    manipulation_score: float = 0.0  # 0-1 scale
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class MarketContext:
    """Market-wide context at time of trade"""
    timestamp: datetime
    btc_price: float
    btc_trend: str  # up/down/sideways
    btc_change_24h: float
    total_market_cap: float = 0.0
    btc_dominance: float = 0.0
    fear_greed_index: Optional[int] = None
    funding_rate_avg: float = 0.0
    market_phase: str = "neutral"  # risk_on/risk_off/neutral
    vix_crypto: Optional[float] = None  # Crypto volatility index
    
@dataclass
class TradeContext:
    """Additional context for each trade"""
    symbol: str
    timestamp: datetime
    
    # Market context
    btc_price: float
    btc_trend: str
    market_phase: str
    
    # Time context
    session: str  # asian/european/us/off_hours
    hour_utc: int
    day_of_week: int  # 0=Monday
    day_of_month: int
    is_weekend: bool
    is_month_end: bool
    
    # Symbol-specific metrics
    volume_vs_avg: float  # Current volume / 30d average
    volatility_percentile: float  # Current volatility percentile
    spread_bps: float  # Bid-ask spread in basis points
    
    # Technical context
    distance_from_200ma: float  # % distance from 200 MA
    rsi_at_entry: float
    volume_profile_level: str  # high_volume_node/low_volume_node/neutral
    breakout_attempts: int  # Number of recent failed breakouts
    
    # Correlation context
    correlation_with_btc_30d: float
    sector_performance_24h: float  # How the sector is doing
    
    # Microstructure
    tick_frequency: float  # Price updates per minute
    large_trade_ratio: float  # % of volume from large trades
    order_book_imbalance: float  # Buy vs sell pressure

class SymbolDataCollector:
    """Collects and stores comprehensive symbol data for future ML"""
    
    def __init__(self):
        self.conn = self._init_database()
        self._create_tables()
        self.symbol_categories = self._load_symbol_categories()
        self.btc_price_cache = None
        self.btc_price_cache_time = None
        
    def _init_database(self):
        """Initialize PostgreSQL connection"""
        try:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'trading'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', '')
            )
            logger.info("Connected to PostgreSQL for symbol data collection")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return None
    
    def _create_tables(self):
        """Create necessary tables for symbol data storage"""
        if not self.conn:
            return
            
        create_queries = [
            """
            CREATE TABLE IF NOT EXISTS symbol_profiles (
                symbol VARCHAR(20) PRIMARY KEY,
                category VARCHAR(20),
                market_cap_rank INT,
                avg_volume_30d DECIMAL(20,2),
                volatility_percentile DECIMAL(5,2),
                typical_hold_minutes INT,
                breakout_success_rate DECIMAL(5,2),
                best_session VARCHAR(10),
                worst_session VARCHAR(10),
                fakeout_rate DECIMAL(5,2),
                avg_spread_bps DECIMAL(8,2),
                correlation_with_btc DECIMAL(5,3),
                news_sensitivity DECIMAL(3,2),
                manipulation_score DECIMAL(3,2),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS session_performance (
                symbol VARCHAR(20),
                session VARCHAR(10),
                total_trades INT DEFAULT 0,
                wins INT DEFAULT 0,
                losses INT DEFAULT 0,
                win_rate DECIMAL(5,2),
                avg_pnl DECIMAL(10,2),
                avg_hold_minutes INT,
                best_hour INT,
                worst_hour INT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(symbol, session)
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS symbol_correlations (
                symbol1 VARCHAR(20),
                symbol2 VARCHAR(20),
                correlation_30d DECIMAL(5,3),
                correlation_7d DECIMAL(5,3),
                lead_lag_minutes INT,  -- symbol1 leads symbol2 by X minutes
                stability_score DECIMAL(3,2),  -- How stable the correlation is
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(symbol1, symbol2)
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS trade_context (
                trade_id SERIAL PRIMARY KEY,
                symbol VARCHAR(20),
                timestamp TIMESTAMP,
                
                -- Market context
                btc_price DECIMAL(10,2),
                btc_trend VARCHAR(10),
                market_phase VARCHAR(10),
                btc_dominance DECIMAL(5,2),
                
                -- Time context
                session VARCHAR(10),
                hour_utc INT,
                day_of_week INT,
                day_of_month INT,
                is_weekend BOOLEAN,
                is_month_end BOOLEAN,
                
                -- Symbol metrics
                volume_vs_avg DECIMAL(10,2),
                volatility_percentile DECIMAL(5,2),
                spread_bps DECIMAL(8,2),
                
                -- Technical context
                distance_from_200ma DECIMAL(8,2),
                rsi_at_entry DECIMAL(5,2),
                volume_profile_level VARCHAR(20),
                breakout_attempts INT,
                
                -- Correlation
                correlation_with_btc_30d DECIMAL(5,3),
                sector_performance_24h DECIMAL(8,2),
                
                -- Microstructure
                tick_frequency DECIMAL(10,2),
                large_trade_ratio DECIMAL(5,2),
                order_book_imbalance DECIMAL(5,2),
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS hourly_patterns (
                symbol VARCHAR(20),
                hour_utc INT,
                total_signals INT DEFAULT 0,
                successful_trades INT DEFAULT 0,
                failed_trades INT DEFAULT 0,
                avg_pnl DECIMAL(10,2),
                avg_volume_ratio DECIMAL(5,2),
                breakout_success_rate DECIMAL(5,2),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(symbol, hour_utc)
            )
            """,
            
            """
            CREATE INDEX IF NOT EXISTS idx_trade_context_symbol_time 
            ON trade_context(symbol, timestamp DESC);
            """,
            
            """
            CREATE INDEX IF NOT EXISTS idx_session_performance_symbol 
            ON session_performance(symbol);
            """
        ]
        
        try:
            with self.conn.cursor() as cur:
                for query in create_queries:
                    cur.execute(query)
                self.conn.commit()
                logger.info("Created symbol data collection tables")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            self.conn.rollback()
    
    def _load_symbol_categories(self) -> Dict[str, str]:
        """Load symbol categories for classification"""
        # This can be expanded with actual data from CoinGecko/CMC API
        categories = {
            # Majors
            'BTCUSDT': 'major',
            'ETHUSDT': 'major',
            
            # Large caps
            'BNBUSDT': 'large',
            'SOLUSDT': 'large',
            'XRPUSDT': 'large',
            'ADAUSDT': 'large',
            
            # Meme coins
            'DOGEUSDT': 'meme',
            'SHIBUSDT': 'meme',
            'PEPEUSDT': 'meme',
            'FLOKIUSDT': 'meme',
            'BONKUSDT': 'meme',
            
            # DeFi
            'AAVEUSDT': 'defi',
            'UNIUSDT': 'defi',
            'LINKUSDT': 'defi',
            'MKRUSDT': 'defi',
            
            # Gaming/Metaverse
            'SANDUSDT': 'gaming',
            'MANAUSDT': 'gaming',
            'AXSUSDT': 'gaming',
            'GALAUSDT': 'gaming',
            
            # AI tokens
            'FETUSDT': 'ai',
            'AGIXUSDT': 'ai',
            'OCEANUSDT': 'ai',
        }
        
        # Default to 'mid' for unknown symbols
        return categories
    
    def get_trading_session(self, hour_utc: int) -> str:
        """Determine trading session based on UTC hour"""
        if 0 <= hour_utc < 8:
            return "asian"
        elif 8 <= hour_utc < 16:
            return "european"
        elif 16 <= hour_utc < 24:
            return "us"
        else:
            return "off_hours"
    
    def get_btc_trend(self, current_price: float, lookback_minutes: int = 60) -> str:
        """Determine BTC trend (up/down/sideways)"""
        # This would fetch from exchange or cache
        # Simplified for now
        if not self.btc_price_cache or \
           (datetime.now() - self.btc_price_cache_time).seconds > 60:
            # Update cache (would fetch from exchange)
            self.btc_price_cache = current_price
            self.btc_price_cache_time = datetime.now()
        
        change = (current_price - self.btc_price_cache) / self.btc_price_cache
        if change > 0.01:
            return "up"
        elif change < -0.01:
            return "down"
        else:
            return "sideways"
    
    def calculate_symbol_metrics(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Calculate current symbol-specific metrics"""
        try:
            # Volume analysis
            current_volume = df['volume'].iloc[-1]
            avg_volume_30 = df['volume'].rolling(30 * 96).mean().iloc[-1]  # 30 days of 15-min candles
            volume_vs_avg = current_volume / avg_volume_30 if avg_volume_30 > 0 else 1.0
            
            # Volatility
            returns = df['close'].pct_change()
            current_volatility = returns.rolling(96).std().iloc[-1]  # 24h volatility
            all_volatilities = returns.rolling(96).std().dropna()
            volatility_percentile = (all_volatilities < current_volatility).mean() * 100
            
            # Spread estimate (would need order book data for real spread)
            spread_bps = abs(df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1] * 10000
            
            # Technical indicators
            ma_200 = df['close'].rolling(200).mean().iloc[-1]
            distance_from_200ma = ((df['close'].iloc[-1] - ma_200) / ma_200 * 100) if ma_200 > 0 else 0
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            return {
                'volume_vs_avg': round(volume_vs_avg, 2),
                'volatility_percentile': round(volatility_percentile, 2),
                'spread_bps': round(spread_bps, 2),
                'distance_from_200ma': round(distance_from_200ma, 2),
                'rsi_at_entry': round(rsi, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {e}")
            return {
                'volume_vs_avg': 1.0,
                'volatility_percentile': 50.0,
                'spread_bps': 10.0,
                'distance_from_200ma': 0.0,
                'rsi_at_entry': 50.0
            }
    
    def record_trade_context(self, symbol: str, df: pd.DataFrame, 
                            btc_price: float = None) -> TradeContext:
        """Record comprehensive context for a trade"""
        now = datetime.now()
        hour_utc = now.hour
        
        # Get symbol category
        category = self.symbol_categories.get(symbol, 'mid')
        
        # Calculate metrics
        metrics = self.calculate_symbol_metrics(symbol, df)
        
        # Create context object
        context = TradeContext(
            symbol=symbol,
            timestamp=now,
            
            # Market context
            btc_price=btc_price or 0.0,
            btc_trend=self.get_btc_trend(btc_price) if btc_price else "unknown",
            market_phase="neutral",  # Would determine from broader market data
            
            # Time context
            session=self.get_trading_session(hour_utc),
            hour_utc=hour_utc,
            day_of_week=now.weekday(),
            day_of_month=now.day,
            is_weekend=now.weekday() >= 5,
            is_month_end=now.day >= 28,
            
            # Symbol metrics
            volume_vs_avg=metrics['volume_vs_avg'],
            volatility_percentile=metrics['volatility_percentile'],
            spread_bps=metrics['spread_bps'],
            
            # Technical context
            distance_from_200ma=metrics['distance_from_200ma'],
            rsi_at_entry=metrics['rsi_at_entry'],
            volume_profile_level="neutral",  # Would calculate from volume profile
            breakout_attempts=0,  # Would track from recent history
            
            # Correlation (would calculate from price data)
            correlation_with_btc_30d=0.0,
            sector_performance_24h=0.0,
            
            # Microstructure (would get from tick data)
            tick_frequency=0.0,
            large_trade_ratio=0.0,
            order_book_imbalance=0.0
        )
        
        # Store in database
        self._store_trade_context(context)
        
        return context
    
    def _store_trade_context(self, context: TradeContext):
        """Store trade context in database"""
        if not self.conn:
            return
            
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trade_context (
                        symbol, timestamp, btc_price, btc_trend, market_phase,
                        session, hour_utc, day_of_week, day_of_month, is_weekend, is_month_end,
                        volume_vs_avg, volatility_percentile, spread_bps,
                        distance_from_200ma, rsi_at_entry, volume_profile_level, breakout_attempts,
                        correlation_with_btc_30d, sector_performance_24h,
                        tick_frequency, large_trade_ratio, order_book_imbalance
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s
                    )
                """, (
                    context.symbol, context.timestamp, context.btc_price, context.btc_trend, context.market_phase,
                    context.session, context.hour_utc, context.day_of_week, context.day_of_month, 
                    context.is_weekend, context.is_month_end,
                    context.volume_vs_avg, context.volatility_percentile, context.spread_bps,
                    context.distance_from_200ma, context.rsi_at_entry, context.volume_profile_level, 
                    context.breakout_attempts,
                    context.correlation_with_btc_30d, context.sector_performance_24h,
                    context.tick_frequency, context.large_trade_ratio, context.order_book_imbalance
                ))
                self.conn.commit()
                logger.debug(f"Stored trade context for {context.symbol}")
                
        except Exception as e:
            logger.error(f"Failed to store trade context: {e}")
            self.conn.rollback()
    
    def update_symbol_profile(self, symbol: str, trades_df: pd.DataFrame = None):
        """Update symbol profile with latest statistics"""
        if not self.conn:
            return
            
        try:
            category = self.symbol_categories.get(symbol, 'mid')
            
            # Calculate statistics from trades (if provided)
            if trades_df is not None and len(trades_df) > 0:
                success_rate = (trades_df['pnl_usd'] > 0).mean()
                avg_hold = trades_df['hold_minutes'].mean() if 'hold_minutes' in trades_df else 180
            else:
                success_rate = 0.5
                avg_hold = 180
            
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO symbol_profiles (symbol, category, breakout_success_rate, typical_hold_minutes)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (symbol) 
                    DO UPDATE SET 
                        category = EXCLUDED.category,
                        breakout_success_rate = EXCLUDED.breakout_success_rate,
                        typical_hold_minutes = EXCLUDED.typical_hold_minutes,
                        updated_at = CURRENT_TIMESTAMP
                """, (symbol, category, success_rate * 100, avg_hold))
                self.conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update symbol profile: {e}")
            self.conn.rollback()
    
    def update_session_performance(self, symbol: str, session: str, 
                                  won: bool, pnl: float, hold_minutes: int):
        """Update session performance statistics"""
        if not self.conn:
            return
            
        try:
            with self.conn.cursor() as cur:
                # Insert or update session stats
                cur.execute("""
                    INSERT INTO session_performance (symbol, session, total_trades, wins, losses, avg_pnl)
                    VALUES (%s, %s, 1, %s, %s, %s)
                    ON CONFLICT (symbol, session) 
                    DO UPDATE SET 
                        total_trades = session_performance.total_trades + 1,
                        wins = session_performance.wins + %s,
                        losses = session_performance.losses + %s,
                        avg_pnl = ((session_performance.avg_pnl * session_performance.total_trades) + %s) 
                                  / (session_performance.total_trades + 1),
                        win_rate = (session_performance.wins + %s) * 100.0 / (session_performance.total_trades + 1),
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    symbol, session, 
                    1 if won else 0, 0 if won else 1, pnl,  # INSERT values
                    1 if won else 0, 0 if won else 1, pnl, 1 if won else 0  # UPDATE values
                ))
                self.conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update session performance: {e}")
            self.conn.rollback()
    
    def get_symbol_stats(self, symbol: str) -> Dict:
        """Get comprehensive statistics for a symbol"""
        if not self.conn:
            return {}
            
        try:
            with self.conn.cursor() as cur:
                # Get profile
                cur.execute("SELECT * FROM symbol_profiles WHERE symbol = %s", (symbol,))
                profile = cur.fetchone()
                
                # Get session performance
                cur.execute("SELECT * FROM session_performance WHERE symbol = %s", (symbol,))
                sessions = cur.fetchall()
                
                # Get recent patterns
                cur.execute("""
                    SELECT session, hour_utc, COUNT(*) as count,
                           AVG(volume_vs_avg) as avg_volume_ratio
                    FROM trade_context 
                    WHERE symbol = %s 
                    AND timestamp > NOW() - INTERVAL '30 days'
                    GROUP BY session, hour_utc
                    ORDER BY count DESC
                    LIMIT 5
                """, (symbol,))
                patterns = cur.fetchall()
                
                return {
                    'profile': profile,
                    'sessions': sessions,
                    'patterns': patterns
                }
                
        except Exception as e:
            logger.error(f"Failed to get symbol stats: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Symbol data collector connection closed")

# Singleton instance
_collector_instance = None

def get_symbol_collector() -> SymbolDataCollector:
    """Get or create the symbol data collector instance"""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = SymbolDataCollector()
    return _collector_instance