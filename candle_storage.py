"""
Persistent candle storage using SQLite
Ensures data survives bot restarts
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)

class CandleStorage:
    def __init__(self, db_path: str = "candles.db"):
        """Initialize SQLite storage for candles"""
        import os
        # Use absolute path to ensure consistent location
        if not os.path.isabs(db_path):
            self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_path)
        else:
            self.db_path = db_path
        self.conn = None
        self._init_db()
        
    def _init_db(self):
        """Create tables if they don't exist"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            
            # Create candles table with all OHLCV data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                ON candles(symbol, timestamp DESC)
            """)
            
            # Create metadata table for tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at INTEGER
                )
            """)
            
            self.conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def save_candles(self, symbol: str, df: pd.DataFrame) -> bool:
        """Save DataFrame of candles to database"""
        try:
            if df is None or df.empty:
                return False
            
            # Prepare data for insertion
            records = []
            for idx, row in df.iterrows():
                # Convert timestamp to integer (unix timestamp in ms)
                if isinstance(idx, pd.Timestamp):
                    timestamp = int(idx.timestamp() * 1000)
                else:
                    timestamp = int(pd.Timestamp(idx).timestamp() * 1000)
                
                records.append((
                    symbol,
                    timestamp,
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume'])
                ))
            
            # Use INSERT OR REPLACE to handle duplicates
            cursor = self.conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO candles 
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, records)
            
            self.conn.commit()
            
            # Update metadata
            self._update_metadata(symbol, len(df))
            
            logger.info(f"[{symbol}] Saved {len(records)} candles to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save candles for {symbol}: {e}")
            return False
    
    def load_candles(self, symbol: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """Load candles from database for a symbol"""
        try:
            cursor = self.conn.cursor()
            
            # Get latest candles
            cursor.execute("""
                SELECT timestamp, open, high, low, close, volume
                FROM candles
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, limit))
            
            rows = cursor.fetchall()
            
            if not rows:
                logger.info(f"[{symbol}] No historical candles found in database")
                return None
            
            # Convert to DataFrame
            data = []
            for row in rows:
                timestamp_ms = row[0]
                # Convert ms timestamp to datetime
                dt = pd.Timestamp(timestamp_ms, unit='ms', tz='UTC')
                data.append({
                    'timestamp': dt,
                    'open': row[1],
                    'high': row[2],
                    'low': row[3],
                    'close': row[4],
                    'volume': row[5]
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
    
    def _update_metadata(self, symbol: str, candle_count: int):
        """Update metadata for tracking"""
        try:
            cursor = self.conn.cursor()
            now = int(datetime.now().timestamp())
            
            cursor.execute("""
                INSERT OR REPLACE INTO metadata (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (f"last_save_{symbol}", str(candle_count), now))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
    
    def get_stats(self) -> dict:
        """Get statistics about stored data"""
        try:
            cursor = self.conn.cursor()
            
            # Get total candles and symbols
            cursor.execute("""
                SELECT COUNT(DISTINCT symbol) as symbols,
                       COUNT(*) as total_candles
                FROM candles
            """)
            
            row = cursor.fetchone()
            if row:
                return {
                    'symbols': row[0],
                    'total_candles': row[1],
                    'db_size_mb': os.path.getsize(self.db_path) / (1024 * 1024)
                }
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def cleanup_old_candles(self, days_to_keep: int = 30):
        """Remove candles older than specified days"""
        try:
            cutoff = datetime.now() - timedelta(days=days_to_keep)
            cutoff_ms = int(cutoff.timestamp() * 1000)
            
            cursor = self.conn.cursor()
            cursor.execute("""
                DELETE FROM candles
                WHERE timestamp < ?
            """, (cutoff_ms,))
            
            deleted = cursor.rowcount
            self.conn.commit()
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} candles older than {days_to_keep} days")
                
                # Vacuum to reclaim space
                cursor.execute("VACUUM")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old candles: {e}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")