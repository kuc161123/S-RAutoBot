import os
import json
import logging
import psycopg2
from datetime import datetime

logger = logging.getLogger(__name__)

class StorageHandler:
    """
    Handles persistence of critical bot state (seen signals).
    
    Supports:
    1. PostgreSQL (via DATABASE_URL) - Preferred for Railway/Production.
    2. Local JSON file - Fallback for local testing.
    """
    
    def __init__(self, local_file_path: str = 'data/seen_signals.json'):
        self.db_url = os.getenv('DATABASE_URL')
        self.local_file_path = local_file_path
        self.use_db = self.db_url is not None
        
        # In-memory cache to minimize IO/DB hits
        self.cache = set()
        
        self._initialize()
        
    def _initialize(self):
        """Initialize storage backend (create table or load file)"""
        if self.use_db:
            try:
                logger.info("🔌 Connecting to PostgreSQL Database...")
                with psycopg2.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        # Create table if not exists
                        # Use TEXT primary key for signal_id
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS seen_signals (
                                signal_id TEXT PRIMARY KEY,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        """)
                logger.info("✅ Database connected and verified.")
                
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
