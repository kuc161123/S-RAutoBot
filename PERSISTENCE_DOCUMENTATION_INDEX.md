# Trading Bot Data Persistence & Storage - Documentation Index

## Quick Start

If you're in a hurry, start here:
1. **PERSISTENCE_QUICK_REFERENCE.txt** - 1-page summary of all persistence layers
2. **PERSISTENCE_ARCHITECTURE_DIAGRAM.txt** - Visual diagrams of data flow

For detailed understanding:
3. **DATA_PERSISTENCE_ANALYSIS.md** - Complete 1000+ line comprehensive guide

---

## Document Guide

### 1. PERSISTENCE_QUICK_REFERENCE.txt
**Length:** 2 pages | **Format:** Text summary

**Contents:**
- Candle storage (PostgreSQL + SQLite)
- Trade tracker (multi-tier fallback)
- Phantom tracker (ML learning dataset)
- ML model persistence (Redis)
- Timing & performance
- Recovery scenarios
- Operational checklist
- Key commands

**Best for:** Quick lookup, operational decisions, debugging

---

### 2. PERSISTENCE_ARCHITECTURE_DIAGRAM.txt
**Length:** 3 pages | **Format:** ASCII diagrams

**Sections:**
1. Complete data flow diagram (WebSocket → Storage)
2. Signal tracking & phantom lifecycle
3. Trade recording (multi-tier reliability)
4. ML model persistence (Redis state)
5. Phantom trading state (Redis + PostgreSQL)
6. Fallback chain (graceful degradation)
7. Concurrency & write safety
8. Performance characteristics

**Best for:** Understanding system architecture, identifying failure points, capacity planning

---

### 3. DATA_PERSISTENCE_ANALYSIS.md
**Length:** 1071 lines | **Format:** Detailed technical documentation

**Sections:**

#### Part 1: Candle Storage (OHLCV Data)
- Database schema (PostgreSQL + SQLite)
- Connection strategy
- Persistence operations (save/load)
- Data retention & cleanup
- Integration with main bot
- Summary logging

**Key Classes:**
- `PostgresCandleStorage`
- `Candle`, `Candle3m` (SQLAlchemy models)

**Key Methods:**
- `save_candles(symbol, df)`
- `save_all_frames(frames)`
- `save_candles_3m(symbol, df)` (with retry)
- `load_candles(symbol, limit)`
- `cleanup_old_candles(days_to_keep)`

#### Part 2: Trade Tracker (Executed Trade History)
- Trade data model (dataclass)
- PostgreSQL schema
- Multi-tier fallback (PostgreSQL → JSON → Redis → Memory)
- PnL calculation
- Statistics caching (all_time, 7d, 30d)
- Integration with main bot

**Key Classes:**
- `Trade` (dataclass)
- `TradeTrackerPostgres`

**Key Methods:**
- `add_trade(trade)`
- `calculate_pnl(symbol, side, entry, exit, qty, leverage)`
- `get_statistics(days)`
- `format_stats_message(days)`

#### Part 3: Phantom Trade Tracking (ML Learning)
- PhantomTrade dataclass (34 features)
- Redis-based persistence (volatile)
- Phantom lifecycle (record → update → close)
- PostgreSQL audit trail
- ML learning data export
- Statistics & accuracy tracking

**Key Classes:**
- `PhantomTrade` (dataclass)
- `PhantomTradeTracker`

**Key Methods:**
- `record_signal(symbol, signal, ml_score, was_executed, features, strategy_name)`
- `update_phantom_prices(symbol, current_price, df, btc_price, symbol_collector)`
- `_close_phantom(symbol, phantom, exit_price, outcome, ...)`
- `get_phantom_stats(symbol)`
- `get_learning_data()`

#### Part 4: Redis State Persistence
- ML model storage (base64 pickles)
- Namespaced keys (ml:trend:*)
- Legacy key fallback (tml:*)
- Model persistence flow
- Win-rate rolling window
- Blocked signal counters

**Storage Keys:**
- `ml:trend:model`, `ml:trend:scaler`, `ml:trend:calibrator`
- `phantom:active`, `phantom:completed`
- `phantom:wr:trend/mr/scalp`
- `phantom:blocked:YYYYMMDD`

#### Part 5: Phantom Persistence (PostgreSQL Audit)
- Dedicated audit table
- Schema for signal tracking
- Immutable record of all signals
- Features stored as JSON

**Key Tables:**
- `phantom_trades_live`

#### Part 6: Fallback File-Based Storage
- Trade history JSON
- SQLite fallback database
- Limitations and use cases

#### Part 7: Data Flow Architecture
- End-to-end flow (WebSocket → Storage → Statistics)
- Save intervals
- Recovery scenarios

#### Part 8: Database Operations & Reliability
- Connection management (NullPool for Railway)
- Transaction handling (UPSERT pattern)
- Error recovery (three-tier)
- Retry logic (exponential backoff)
- Database monitoring

#### Part 9: Timing & Performance
- Persistence latencies (50-200ms for batch saves)
- Storage growth rates (~25MB/month for 50 symbols)
- Concurrent access patterns

#### Part 10: Recovery Scenarios
- Bot crash during trade
- PostgreSQL unavailable
- Redis down
- Railway restart (ephemeral filesystem)

#### Part 11: Operational Considerations
- Scheduled maintenance
- Backup strategy
- Monitoring points

#### Part 12: Schema Evolution & Migrations
- Current schema versions
- Upgrade paths
- ALTER TABLE safety

#### Part 13: Summary & Recommendations
- System strengths
- Enhancement areas
- Pre-launch checklist

**Best for:** Deep understanding, troubleshooting, enhancement planning, operational runbooks

---

## Architecture Overview

### Persistence Layers (4-Tier Redundancy)

```
TIER 1: PostgreSQL (Primary)
├─ Candles table (15m OHLCV)
├─ Candles_3m table (3m OHLCV)
├─ Trades table (executed history)
├─ Trade_stats_cache (materialized stats)
└─ Phantom_trades_live (audit trail)

TIER 2: SQLite (Fallback - Candles only)
├─ candles.db file
└─ Same schema as PostgreSQL

TIER 3: Redis (Fast State + Backup)
├─ Phantom active/completed tracking
├─ ML model storage (base64 pickles)
├─ Win-rate rolling windows
├─ Trade history backup (last 10000)
└─ Signal blocking counters

TIER 4: JSON Files + Memory (Fallback)
├─ trade_history.json (trade backup)
└─ In-memory dict structures (always loaded)
```

### Key Design Patterns

1. **UPSERT Pattern (Candles)**
   - Check if row exists by (symbol, timestamp)
   - If exists: UPDATE
   - If not: INSERT
   - Atomic commit

2. **Multi-Tier Fallback (Trades)**
   - Primary: PostgreSQL transaction
   - Fallback 1: JSON file write
   - Fallback 2: Redis backup
   - Result: Guaranteed durability

3. **Async Operations (Candle Saves)**
   - Use `loop.run_in_executor()` to avoid blocking
   - Periodic batch saves every 15m
   - Per-symbol saves on update

4. **Lazy Initialization (ML Models)**
   - Load into memory on startup
   - Save after each training
   - Namespace-based Redis keys with legacy fallback

---

## Common Operations

### View Database Statistics
```python
stats = storage.get_stats()
# Returns: {symbols, total_candles, db_size_mb, database_type}
```

### Get Trade Statistics
```python
stats = trade_tracker.get_statistics(days=30)
# Returns: {win_rate, profit_factor, total_pnl, top_symbols, ...}
```

### Manual Candle Cleanup
```python
storage.cleanup_old_candles(days_to_keep=90)
```

### Check ML Training Status
```python
info = ml_scorer.get_retrain_info()
# Returns: {can_train, trades_until_next_retrain, ...}
```

### View Phantom Statistics
```python
stats = phantom_tracker.get_phantom_stats()
# Returns: {total, executed, rejected, ml_accuracy, ...}
```

---

## Performance Characteristics

### Save Latencies
- Candle batch (50 symbols): 50-200ms
- Single trade: 5-20ms
- Phantom record: <1ms
- Phantom close: 20-100ms

### Storage Growth
- Candles: ~5KB/symbol/month (15m) + ~15KB/symbol/month (3m)
- Trades: ~500B/trade (unlimited retention)
- Phantoms: ~2KB/phantom (last 1000 only)
- Total for 50 symbols: ~25MB/month

---

## Troubleshooting Guide

### Problem: "No historical candles found in database"
**Cause:** Cold start, or candles older than 30 days
**Solution:** Wait for new candles, or restore from backup

### Problem: "Failed to save candles for SYMBOL"
**Cause:** PostgreSQL unavailable, SQLite fallback triggered
**Solution:** Check DATABASE_URL, verify SQLite file permissions

### Problem: "Trade not found after position close"
**Cause:** Add_trade() failed, but position closed anyway
**Solution:** Check trade_history.json, recover from Redis if available

### Problem: "ML models not trained"
**Cause:** < 30 executed trades, or Redis unavailable
**Solution:** Execute more trades, or check Redis connection

### Problem: "Phantom stats show 0 trades"
**Cause:** Redis cleared, or Redis down
**Solution:** Phantoms still tracked in memory, stats recover on next trade

---

## Deployment Checklist

Before going to production:

**Database Setup**
- [ ] PostgreSQL accessible via DATABASE_URL
- [ ] SQLite fallback directory writable
- [ ] Database user has CREATE/INSERT/UPDATE permissions
- [ ] Backups configured (especially for Railway)

**Redis Setup (Optional but Recommended)**
- [ ] Redis accessible via REDIS_URL
- [ ] Redis has sufficient memory (1GB+ recommended)
- [ ] Persistence enabled (if using Railway)

**File System**
- [ ] trade_history.json location writable
- [ ] candles.db location writable
- [ ] ML model directory has write permissions
- [ ] 30+ MB free space for candles (per month per 50 symbols)

**Monitoring**
- [ ] Setup alerts for save_all_frames() success rate
- [ ] Monitor PostgreSQL connection availability
- [ ] Track database size growth
- [ ] Monitor add_trade() latency

**Maintenance**
- [ ] Schedule cleanup_old_candles() (cron job)
- [ ] Setup PostgreSQL backup exports (daily)
- [ ] Document phantom retention policy
- [ ] Plan for schema migration procedures

---

## File Locations in Codebase

**Core Persistence Files:**
- `/Users/lualakol/AutoTrading Bot/candle_storage_postgres.py` - Candle storage
- `/Users/lualakol/AutoTrading Bot/trade_tracker_postgres.py` - Trade tracking
- `/Users/lualakol/AutoTrading Bot/phantom_trade_tracker.py` - Phantom tracking
- `/Users/lualakol/AutoTrading Bot/phantom_persistence.py` - Phantom audit table
- `/Users/lualakol/AutoTrading Bot/ml_scorer_trend.py` - ML model persistence

**Integration Points:**
- `/Users/lualakol/AutoTrading Bot/live_bot.py` - Main bot (save calls at lines 2966, 6852, 6864, 6877, 9915, 10644, 7019)
- `/Users/lualakol/AutoTrading Bot/position_mgr.py` - Position tracking
- `/Users/lualakol/AutoTrading Bot/broker_bybit.py` - Trade execution

---

## Related Topics

- **Strategy Implementation:** See `strategy_pullback.py`, `strategy_scalp.py`
- **ML System:** See `ml_scorer_trend.py`, `enhanced_mr_scorer.py`
- **Telegram Monitoring:** See `telegram_bot.py` (/db_stats, /ml_stats, /phantom_stats commands)
- **WebSocket Handling:** See `multi_websocket_handler.py`

---

## Document Metadata

- **Version:** 1.0
- **Last Updated:** 2024-11-10
- **System:** Trading Bot v1.0 with Trend/MR/Scalp strategies
- **Database:** PostgreSQL (Railway) with SQLite fallback
- **Cache:** Redis (optional)

---

## Support Matrix

| Issue Type | Quick Reference | Diagrams | Full Analysis |
|-----------|-----------------|----------|--------------|
| Quick stats | ✓ | | |
| Architecture understanding | ✓ | ✓ | ✓ |
| Troubleshooting | ✓ | ✓ | ✓ |
| Deployment | ✓ | ✓ | ✓ |
| Performance tuning | | ✓ | ✓ |
| Schema changes | | | ✓ |
| Recovery procedures | ✓ | ✓ | ✓ |
| ML integration | | | ✓ |

