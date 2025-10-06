"""
PostgreSQL candle storage for Railway deployment
Provides persistent storage across all deployments
"""
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, Optional
from sqlalchemy import create_engine, Column, BigInteger, Float, String, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import json

logger = logging.getLogger(__name__)

Base = declarative_base()

class Candle(Base):
    """Candle data model for PostgreSQL"""
    __tablename__ = 'candles'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(BigInteger, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp', unique=True),
    )

class Candle3m(Base):
    """3m Candle data model stored separately to avoid mixing timeframes"""
    __tablename__ = 'candles_3m'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(BigInteger, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    __table_args__ = (
        Index('idx_symbol_timestamp_3m', 'symbol', 'timestamp', unique=True),
    )

class PostgresCandleStorage:
    # Class-level tracking for summary logging
    _save_summary = {'count': 0, 'symbols': set(), 'last_log_time': 0, 'total_candles': 0}
    _LOG_INTERVAL = 300  # Log summary every 5 minutes

    def __init__(self, database_url: Optional[str] = None):
        """Initialize PostgreSQL storage for candles"""
        # Use environment variable or SQLite fallback
        self.database_url = database_url or os.getenv('DATABASE_URL')
        
        if not self.database_url:
            # Fall back to SQLite if no PostgreSQL available
            logger.warning("No DATABASE_URL found, using SQLite fallback")
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candles.db")
            self.database_url = f"sqlite:///{db_path}"
            self.is_postgres = False
        else:
            # Railway provides postgresql:// but SQLAlchemy needs postgresql+psycopg2://
            if self.database_url.startswith("postgresql://"):
                self.database_url = self.database_url.replace("postgresql://", "postgresql+psycopg2://")
            self.is_postgres = True
        
        try:
            # Create engine with connection pooling disabled for Railway
            self.engine = create_engine(
                self.database_url,
                poolclass=NullPool,  # Disable pooling for Railway
                connect_args={} if self.is_postgres else {"check_same_thread": False}
            )
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            logger.info(f"Database connected: {'PostgreSQL' if self.is_postgres else 'SQLite'}")
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("Database connection test successful")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _update_save_summary(self, symbol: str, saved_count: int):
        """Update save summary and log periodically"""
        current_time = time.time()

        # Update summary statistics
        self._save_summary['count'] += 1
        self._save_summary['symbols'].add(symbol)
        self._save_summary['total_candles'] += saved_count

        # Log summary every LOG_INTERVAL seconds
        if current_time - self._save_summary['last_log_time'] >= self._LOG_INTERVAL:
            if self._save_summary['count'] > 0:
                logger.info(
                    f"Candle Storage Summary: Saved {self._save_summary['total_candles']} candles "
                    f"for {len(self._save_summary['symbols'])} symbols "
                    f"({self._save_summary['count']} operations in last {self._LOG_INTERVAL//60} minutes)"
                )

            # Reset summary
            self._save_summary['count'] = 0
            self._save_summary['symbols'] = set()
            self._save_summary['total_candles'] = 0
            self._save_summary['last_log_time'] = current_time

    def save_candles(self, symbol: str, df: pd.DataFrame) -> bool:
        """Save DataFrame of candles to database"""
        session = self.Session()
        try:
            if df is None or df.empty:
                return False
            
            saved_count = 0
            for idx, row in df.iterrows():
                # Convert timestamp to integer (unix timestamp in ms)
                if isinstance(idx, pd.Timestamp):
                    timestamp = int(idx.timestamp() * 1000)
                else:
                    timestamp = int(pd.Timestamp(idx).timestamp() * 1000)
                
                # Check if candle exists
                existing = session.query(Candle).filter_by(
                    symbol=symbol,
                    timestamp=timestamp
                ).first()
                
                if existing:
                    # Update existing candle
                    existing.open = float(row['open'])
                    existing.high = float(row['high'])
                    existing.low = float(row['low'])
                    existing.close = float(row['close'])
                    existing.volume = float(row['volume'])
                else:
                    # Create new candle
                    candle = Candle(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume'])
                    )
                    session.add(candle)
                saved_count += 1
            
            session.commit()

            # Update summary statistics instead of logging every save
            self._update_save_summary(symbol, saved_count)

            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save candles for {symbol}: {e}")
            return False
        finally:
            session.close()

    def save_candles_3m(self, symbol: str, df: pd.DataFrame) -> bool:
        """Save 3m candles with retry/backoff to handle transient DB issues."""
        if df is None or df.empty:
            return False
        attempts = 0
        delay = 0.5
        max_attempts = 3
        while attempts < max_attempts:
            session = self.Session()
            try:
                saved_count = 0
                for idx, row in df.iterrows():
                    if isinstance(idx, pd.Timestamp):
                        timestamp = int(idx.timestamp() * 1000)
                    else:
                        timestamp = int(pd.Timestamp(idx).timestamp() * 1000)
                    existing = session.query(Candle3m).filter_by(
                        symbol=symbol,
                        timestamp=timestamp
                    ).first()
                    if existing:
                        existing.open = float(row['open'])
                        existing.high = float(row['high'])
                        existing.low = float(row['low'])
                        existing.close = float(row['close'])
                        existing.volume = float(row['volume'])
                    else:
                        candle = Candle3m(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=float(row['volume'])
                        )
                        session.add(candle)
                    saved_count += 1
                session.commit()
                self._update_save_summary(symbol, saved_count)
                return True
            except Exception as e:
                session.rollback()
                attempts += 1
                if attempts >= max_attempts:
                    logger.error(f"Failed to save 3m candles for {symbol}: {e}")
                    return False
                else:
                    logger.warning(f"Retrying 3m save for {symbol} (attempt {attempts}/{max_attempts}) due to: {e}")
                    time.sleep(delay)
                    delay *= 2
            finally:
                session.close()

    def load_candles_3m(self, symbol: str, limit: int = None) -> Optional[pd.DataFrame]:
        """Load 3m candles from database for a symbol"""
        session = self.Session()
        try:
            query = session.query(Candle3m).filter_by(symbol=symbol).\
                order_by(Candle3m.timestamp.desc())
            if limit is not None:
                query = query.limit(limit)
            candles = query.all()
            if not candles:
                logger.debug(f"[{symbol}] No 3m candles found in database")
                return None
            data = []
            for c in candles:
                dt = pd.Timestamp(c.timestamp, unit='ms', tz='UTC')
                data.append({
                    'timestamp': dt,
                    'open': c.open,
                    'high': c.high,
                    'low': c.low,
                    'close': c.close,
                    'volume': c.volume
                })
            df = pd.DataFrame(data).sort_values('timestamp')
            df.set_index('timestamp', inplace=True)
            logger.info(f"[{symbol}] Loaded {len(df)} 3m candles from database")
            return df
        except Exception as e:
            logger.error(f"Failed to load 3m candles for {symbol}: {e}")
            return None
        finally:
            session.close()

    def save_all_frames_3m(self, frames: Dict[str, pd.DataFrame]) -> bool:
        try:
            saved_count = 0
            for symbol, df in frames.items():
                if self.save_candles_3m(symbol, df):
                    saved_count += 1
            logger.info(f"Saved {saved_count}/{len(frames)} symbols to 3m database table")
            return saved_count > 0
        except Exception as e:
            logger.error(f"Failed to save all 3m frames: {e}")
            return False
    
    def load_candles(self, symbol: str, limit: int = None) -> Optional[pd.DataFrame]:
        """Load candles from database for a symbol"""
        session = self.Session()
        try:
            # Query candles
            query = session.query(Candle).filter_by(symbol=symbol)\
                .order_by(Candle.timestamp.desc())
            
            # Apply limit only if specified
            if limit is not None:
                query = query.limit(limit)
                
            candles = query.all()
            
            if not candles:
                logger.info(f"[{symbol}] No historical candles found in database")
                return None
            
            # Convert to DataFrame
            data = []
            for candle in candles:
                # Convert ms timestamp to datetime
                dt = pd.Timestamp(candle.timestamp, unit='ms', tz='UTC')
                data.append({
                    'timestamp': dt,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume
                })
            
            df = pd.DataFrame(data)
            # Sort chronologically (oldest first)
            df = df.sort_values('timestamp')
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"[{symbol}] Loaded {len(df)} candles from database")
            logger.info(f"[{symbol}] Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load candles for {symbol}: {e}")
            return None
        finally:
            session.close()
    
    def save_all_frames(self, frames: Dict[str, pd.DataFrame]) -> bool:
        """Save all symbol DataFrames at once"""
        try:
            saved_count = 0
            for symbol, df in frames.items():
                if self.save_candles(symbol, df):
                    saved_count += 1
            
            logger.info(f"Saved {saved_count}/{len(frames)} symbols to database")
            return saved_count > 0
            
        except Exception as e:
            logger.error(f"Failed to save all frames: {e}")
            return False
    
    def load_all_frames(self, symbols: list) -> Dict[str, pd.DataFrame]:
        """Load all symbols from database"""
        frames = {}
        loaded_count = 0
        
        for symbol in symbols:
            df = self.load_candles(symbol)
            if df is not None and not df.empty:
                frames[symbol] = df
                loaded_count += 1
        
        logger.info(f"Loaded {loaded_count}/{len(symbols)} symbols from database")
        return frames
    
    def get_stats(self) -> dict:
        """Get statistics about stored data"""
        session = self.Session()
        try:
            # Get total candles and symbols
            total_candles = session.query(Candle).count()
            unique_symbols = session.query(Candle.symbol).distinct().count()
            
            # Get database size (PostgreSQL specific)
            if self.is_postgres:
                result = session.execute(text("""
                    SELECT pg_database_size(current_database()) / 1024.0 / 1024.0 as size_mb
                """))
                db_size = result.scalar() or 0
            else:
                # SQLite size
                import os
                db_path = self.database_url.replace("sqlite:///", "")
                db_size = os.path.getsize(db_path) / (1024 * 1024) if os.path.exists(db_path) else 0
            
            return {
                'symbols': unique_symbols,
                'total_candles': total_candles,
                'db_size_mb': db_size,
                'database_type': 'PostgreSQL' if self.is_postgres else 'SQLite'
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
        finally:
            session.close()
    
    def cleanup_old_candles(self, days_to_keep: int = 30):
        """Remove candles older than specified days"""
        session = self.Session()
        try:
            cutoff = datetime.now() - timedelta(days=days_to_keep)
            cutoff_ms = int(cutoff.timestamp() * 1000)
            
            deleted = session.query(Candle).filter(Candle.timestamp < cutoff_ms).delete()
            session.commit()
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} candles older than {days_to_keep} days")
                
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to cleanup old candles: {e}")
        finally:
            session.close()
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            logger.info("Database connection closed")

# For backward compatibility, use the same class name
CandleStorage = PostgresCandleStorage
