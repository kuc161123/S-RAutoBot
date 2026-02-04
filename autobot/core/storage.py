import os
import json
import logging
import psycopg2
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default lifetime stats structure
DEFAULT_LIFETIME_STATS = {
    'start_date': None,
    'starting_balance': 0.0,
    'total_r': 0.0,
    'total_pnl': 0.0,
    'total_trades': 0,
    'wins': 0,
    'best_day_r': 0.0,
    'best_day_date': None,
    'worst_day_r': 0.0,
    'worst_day_date': None,
    'daily_r': {},
    'best_trade_r': 0.0,
    'best_trade_symbol': '',
    'best_trade_date': None,
    'worst_trade_r': 0.0,
    'worst_trade_symbol': '',
    'worst_trade_date': None,
    'max_drawdown_r': 0.0,
    'peak_equity_r': 0.0,
    'current_streak': 0,
    'longest_win_streak': 0,
    'longest_loss_streak': 0
}


class StorageHandler:
    """
    Handles persistence of critical bot state (seen signals + lifetime stats).
    
    Supports:
    1. PostgreSQL (via DATABASE_URL) - Preferred for Railway/Production.
    2. Local JSON file - Fallback for local testing.
    """
    
    def __init__(self, local_file_path: str = 'data/seen_signals.json'):
        self.db_url = os.getenv('DATABASE_URL')
        self.local_file_path = local_file_path
        self.lifetime_stats_file = 'lifetime_stats.json'
        self.use_db = self.db_url is not None
        
        # In-memory cache to minimize IO/DB hits
        self.cache = set()
        
        self._initialize()
        
    def _initialize(self):
        """Initialize storage backend (create tables or load file)"""
        if self.use_db:
            try:
                logger.info("🔌 Connecting to PostgreSQL Database...")
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        # Create seen_signals table if not exists
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS seen_signals (
                                signal_id TEXT PRIMARY KEY,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        """)
                        
                        # Create lifetime_stats table if not exists
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS lifetime_stats (
                                id INTEGER PRIMARY KEY DEFAULT 1,
                                stats_data JSONB NOT NULL,
                                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                CONSTRAINT single_row CHECK (id = 1)
                            )
                        """)
                        
                logger.info("✅ Database connected and tables verified.")
                
                # Load existing signals into cache
                self._load_cache_from_db()
                
            except Exception as e:
                logger.error(f"❌ Database connection failed: {e}")
                logger.warning("⚠️ Falling back to Local JSON storage.")
                self.use_db = False
                self._load_cache_from_file()
        else:
            logger.info("📂 Using Local JSON Storage.")
            self._load_cache_from_file()

    def _load_cache_from_db(self):
        """Load all seen signals from DB to memory"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT signal_id FROM seen_signals")
                    rows = cur.fetchall()
                    self.cache = {row[0] for row in rows}
            logger.info(f"Loaded {len(self.cache)} signals from Database")
        except Exception as e:
            logger.error(f"Failed to load cache from DB: {e}")

    def _load_cache_from_file(self):
        """Load signals from local JSON file"""
        if os.path.exists(self.local_file_path):
            try:
                with open(self.local_file_path, 'r') as f:
                    data = json.load(f)
                    self.cache = set(data.get('signals', []))
                logger.info(f"Loaded {len(self.cache)} signals from Local File")
            except Exception as e:
                logger.error(f"Failed to load local file: {e}")

    def is_seen(self, signal_id: str) -> bool:
        """Check if a signal ID has been seen before"""
        return signal_id in self.cache

    def add_signal(self, signal_id: str):
        """Persist a new signal ID"""
        if signal_id in self.cache:
            return

        # Update cache immediately
        self.cache.add(signal_id)

        if self.use_db:
            try:
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO seen_signals (signal_id) VALUES (%s) ON CONFLICT (signal_id) DO NOTHING",
                            (signal_id,)
                        )
            except Exception as e:
                logger.error(f"Failed to write signal to DB: {e}")
        else:
            # Write to local file
            self._save_to_file()

    def _save_to_file(self):
        """Save cache to local JSON file"""
        try:
            os.makedirs(os.path.dirname(self.local_file_path), exist_ok=True)
            with open(self.local_file_path, 'w') as f:
                json.dump({'signals': list(self.cache)}, f)
        except Exception as e:
            logger.error(f"Failed to save local file: {e}")

    # ========== LIFETIME STATS METHODS ==========
    
    def load_lifetime_stats(self) -> Dict[str, Any]:
        """
        Load lifetime stats from PostgreSQL or local JSON file.
        Returns dict with stats, using defaults for any missing fields.
        """
        stats = DEFAULT_LIFETIME_STATS.copy()
        
        if self.use_db:
            try:
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT stats_data FROM lifetime_stats WHERE id = 1")
                        row = cur.fetchone()
                        if row:
                            loaded = row[0]
                            # Handle daily_r which may be stored differently
                            if 'daily_r' in loaded and isinstance(loaded['daily_r'], str):
                                loaded['daily_r'] = json.loads(loaded['daily_r'])
                            stats.update(loaded)
                            logger.info(f"📊 Loaded lifetime stats from DB: {stats['total_r']:.1f}R, {stats['total_trades']} trades since {stats.get('start_date', 'unknown')}")
                        else:
                            logger.info("📊 No lifetime stats in DB yet - using defaults")
            except Exception as e:
                logger.error(f"Failed to load lifetime stats from DB: {e}")
                # Fall back to local file
                stats = self._load_lifetime_stats_from_file()
        else:
            stats = self._load_lifetime_stats_from_file()
        
        return stats
    
    def _load_lifetime_stats_from_file(self) -> Dict[str, Any]:
        """Load lifetime stats from local JSON file"""
        stats = DEFAULT_LIFETIME_STATS.copy()
        
        if os.path.exists(self.lifetime_stats_file):
            try:
                with open(self.lifetime_stats_file, 'r') as f:
                    loaded = json.load(f)
                    stats.update(loaded)
                logger.info(f"📊 Loaded lifetime stats from file: {stats['total_r']:.1f}R, {stats['total_trades']} trades")
            except Exception as e:
                logger.error(f"Failed to load lifetime stats from file: {e}")
        
        return stats
    
    def save_lifetime_stats(self, stats: Dict[str, Any]) -> bool:
        """
        Persist lifetime stats to PostgreSQL or local JSON file.
        Returns True on success, False on failure.
        """
        if self.use_db:
            try:
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        # Use UPSERT pattern
                        cur.execute("""
                            INSERT INTO lifetime_stats (id, stats_data, updated_at)
                            VALUES (1, %s, CURRENT_TIMESTAMP)
                            ON CONFLICT (id) DO UPDATE SET
                                stats_data = EXCLUDED.stats_data,
                                updated_at = CURRENT_TIMESTAMP
                        """, (json.dumps(stats),))
                logger.debug(f"💾 Lifetime stats saved to DB: {stats['total_r']:.1f}R")
                return True
            except Exception as e:
                logger.error(f"Failed to save lifetime stats to DB: {e}")
                # Fall back to local file
                return self._save_lifetime_stats_to_file(stats)
        else:
            return self._save_lifetime_stats_to_file(stats)
    
    def _save_lifetime_stats_to_file(self, stats: Dict[str, Any]) -> bool:
        """Save lifetime stats to local JSON file"""
        try:
            with open(self.lifetime_stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            logger.debug(f"💾 Lifetime stats saved to file")
            return True
        except Exception as e:
            logger.error(f"Failed to save lifetime stats to file: {e}")
            return False
    
    def reset_lifetime_stats(self, starting_balance: float = 0.0) -> Dict[str, Any]:
        """
        Reset lifetime stats to defaults (for /resetlifetime command).
        Returns the new blank stats.
        """
        stats = DEFAULT_LIFETIME_STATS.copy()
        stats['start_date'] = datetime.now().strftime('%Y-%m-%d')
        stats['starting_balance'] = starting_balance
        stats['daily_r'] = {}
        
        if self.use_db:
            try:
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        # Delete existing and insert fresh
                        cur.execute("DELETE FROM lifetime_stats WHERE id = 1")
                        cur.execute("""
                            INSERT INTO lifetime_stats (id, stats_data, updated_at)
                            VALUES (1, %s, CURRENT_TIMESTAMP)
                        """, (json.dumps(stats),))
                logger.info(f"🔄 Lifetime stats RESET in DB. New start: {stats['start_date']}")
            except Exception as e:
                logger.error(f"Failed to reset lifetime stats in DB: {e}")
                self._save_lifetime_stats_to_file(stats)
        else:
            self._save_lifetime_stats_to_file(stats)
        
        return stats
