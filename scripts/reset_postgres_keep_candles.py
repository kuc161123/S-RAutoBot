#!/usr/bin/env python3
"""
Reset PostgreSQL trading data while keeping candle history.

Actions:
- Keeps: candles (15m) and candles_3m tables intact
- Clears: trades, trade_stats_cache (if present)
- Optionally vacuums tables (Postgres only)

Usage:
  DATABASE_URL=postgresql://user:pass@host:5432/db python scripts/reset_postgres_keep_candles.py

Safety:
- No destructive operations on candle tables
- Idempotent; skips missing tables gracefully
"""
import os
import sys
import logging
from sqlalchemy import text

# Reuse existing storage to get an engine correctly configured for Railway
from candle_storage_postgres import PostgresCandleStorage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [DB-RESET] %(levelname)s - %(message)s')
logger = logging.getLogger("db_reset")


def main():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set. Aborting.")
        sys.exit(1)

    # Initialize storage to ensure tables exist and get an engine
    store = PostgresCandleStorage()
    engine = store.engine

    stmts = [
        # Clear executed trades table
        ("TRUNCATE TABLE trades", False),
        # Clear stats cache if present
        ("TRUNCATE TABLE trade_stats_cache", True),
    ]

    with engine.begin() as conn:
        for sql, optional in stmts:
            try:
                conn.execute(text(sql))
                logger.info(f"OK: {sql}")
            except Exception as e:
                if optional:
                    logger.info(f"Skip optional: {sql} ({e})")
                else:
                    logger.warning(f"Failed: {sql} ({e})")

        # Optional maintenance
        try:
            conn.execute(text("VACUUM"))
            logger.info("VACUUM executed")
        except Exception:
            # Likely not Postgres or insufficient privileges; ignore
            pass

    # Close engine cleanly
    try:
        store.close()
    except Exception:
        pass
    logger.info("Database reset complete (kept candles/candles_3m; cleared trades/stats)")


if __name__ == "__main__":
    main()

