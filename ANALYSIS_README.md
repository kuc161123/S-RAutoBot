# Risk and Position Management - Analysis Documentation

This directory contains comprehensive analysis of the AutoTrading Bot's risk management and position tracking system.

## Document Index

### Primary Documentation

1. **RISK_MANAGEMENT_GUIDE.md** (This directory)
   - **Purpose**: Comprehensive reference guide for risk management system
   - **Audience**: Developers, traders, auditors
   - **Contents**: 
     - One-per-symbol enforcement
     - Risk calculation system
     - Leverage determination
     - Position sizing
     - Order execution flow
     - TP/SL modes (Partial vs Full)
     - Position recovery
     - Closed position detection
     - Safety features summary
     - Configuration quick reference
   - **Format**: Markdown with code examples
   - **Length**: ~2000 lines

### Supplementary Documentation (in /tmp)

2. **RISK_POSITION_ANALYSIS.md**
   - **Purpose**: Detailed technical analysis with complete code walkthroughs
   - **Contents**: 10 detailed sections covering all aspects
   - **Format**: Markdown
   - **Length**: 2500+ lines

3. **RISK_POSITION_SUMMARY.txt**
   - **Purpose**: Executive summary and quick reference
   - **Contents**: All key concepts with detailed explanations
   - **Format**: Text file
   - **Length**: 500+ lines

4. **DETAILED_CODE_REFERENCE.txt**
   - **Purpose**: Complete code snippets with exact line numbers
   - **Contents**: Actual function implementations and patterns
   - **Format**: Text file with code blocks
   - **Length**: 2000+ lines

5. **ANALYSIS_COMPLETE.txt**
   - **Purpose**: Summary of analysis findings and recommendations
   - **Contents**: Key findings, architecture insights, testing recommendations
   - **Format**: Text file

## Quick Navigation

### By Topic

**One-Per-Symbol Rule**
- RISK_MANAGEMENT_GUIDE.md: Section 1
- Files: position_mgr.py (lines 21-52), live_bot.py (lines 1365-1370, 7724, 10284)

**Risk Calculation**
- RISK_MANAGEMENT_GUIDE.md: Section 2
- Files: position_mgr.py (RiskConfig), sizer.py (qty_for method)
- Config: config.yaml (lines 59-83)

**Leverage**
- RISK_MANAGEMENT_GUIDE.md: Section 3
- Files: broker_bybit.py (set_leverage), live_bot.py (execution flow)
- Config: config.yaml (symbol_meta section)

**Position Sizing**
- RISK_MANAGEMENT_GUIDE.md: Section 4
- Files: position_mgr.py (round_step), sizer.py (qty_for)

**Order Execution**
- RISK_MANAGEMENT_GUIDE.md: Section 5
- Files: broker_bybit.py (place_market, set_tpsl), live_bot.py

**TP/SL Modes**
- RISK_MANAGEMENT_GUIDE.md: Section 6
- Files: broker_bybit.py (set_tpsl), live_bot.py (lines 1525-1540)

**Position Recovery**
- RISK_MANAGEMENT_GUIDE.md: Section 7
- Files: live_bot.py (recover_positions, lines 5410-5566)

**Closed Position Detection**
- RISK_MANAGEMENT_GUIDE.md: Section 8
- Files: live_bot.py (check_closed_positions, record_closed_trade)

### By File

**position_mgr.py**
- RiskConfig dataclass
- Position dataclass
- Book dataclass
- round_step() function

**sizer.py**
- Sizer class
- qty_for() method
- Risk calculation logic

**broker_bybit.py**
- place_market()
- set_leverage()
- set_tpsl()
- set_sl_only()

**live_bot.py**
- recover_positions()
- check_closed_positions()
- record_closed_trade()
- Execution logic (multiple locations)

**config.yaml**
- Risk settings (trade section)
- Leverage settings (symbol_meta)
- Position settings

## Key Concepts Summary

### One-Per-Symbol Rule
- Enforced by Book dictionary: Dict[symbol -> Position]
- Pre-execution check before EVERY order
- Post-execution addition after fill
- Removal when position closes

### Risk Management
- Default: 1% of account balance per trade
- Formula: qty = (balance * 1%) / R
- R = |entry_price - stop_loss_price|
- Optional: ML-based dynamic scaling (disabled)
- Optional: Fee-aware sizing (0.11% fees)

### Leverage
- Symbol-specific caps (BTC=100x, ETH=100x, ALT=25-50x)
- Set BEFORE market order (critical for TP/SL preservation)
- Fallback chain: [requested, 20x, 10x]
- Automatic retry on API rejection

### Position Execution
- 8-phase process: Checks → Size → Leverage → Market → Readback → TP/SL → Book → Notify
- Market order: IOC, one-way mode
- TP/SL: Partial mode preferred (Limit TP, Market SL)
- Fallback: Full mode if size unavailable

### Position Recovery
- Fetches all positions from Bybit on startup
- Preserves ALL existing TP/SL orders (NO modifications)
- Restores metadata from Redis
- Restarts are safe and non-destructive

### Closed Position Detection
- Every 30 seconds
- Compares book with live positions
- Classifies exit: TP vs SL vs manual (0.3% tolerance)
- Records to trade tracker with full details

## Important Files

### Core Risk Management
- `/Users/lualakol/AutoTrading Bot/position_mgr.py` - Data structures
- `/Users/lualakol/AutoTrading Bot/sizer.py` - Position sizing
- `/Users/lualakol/AutoTrading Bot/broker_bybit.py` - Exchange API
- `/Users/lualakol/AutoTrading Bot/live_bot.py` - Orchestrator
- `/Users/lualakol/AutoTrading Bot/config.yaml` - Configuration

### Related Files
- `/Users/lualakol/AutoTrading Bot/trade_tracker_postgres.py` - Trade recording
- `/Users/lualakol/AutoTrading Bot/trade_tracker.py` - Trade tracking
- `/Users/lualakol/AutoTrading Bot/telegram_bot.py` - User notifications

## Configuration Reference

### Risk Settings (config.yaml, lines 59-83)
```yaml
trade:
  risk_percent: 1.0              # % of balance per trade (ACTIVE)
  use_percent_risk: true         # Percentage mode enabled
  risk_usd: 10                   # Fallback fixed USD
  fee_total_pct: 0.00110         # 0.11% fees for sizing
  use_ml_dynamic_risk: false     # ML scaling disabled
```

### Leverage Settings (config.yaml, symbol_meta section)
```yaml
BTCUSDT: max_leverage: 100.0
ETHUSDT: max_leverage: 100.0
Most altcoins: 25-50x
Default: 10x
```

### Position Settings
```yaml
confirmation_candles: 2
ml_min_score: 70.0
rr: 2.5
```

## Safety Features

✓ One-per-symbol enforced via dictionary structure
✓ Pre-execution checks on every order
✓ Leverage fallback chain (20x, 10x)
✓ Position recovery preserves existing orders
✓ Decimal precision handling (Decimal module)
✓ Multi-layer validation (qty_step, min_qty, min_order_value)
✓ Fee-aware sizing option
✓ Safe iteration patterns (list() wrapper)

## Testing Checklist

- [ ] One-per-symbol enforcement
- [ ] Risk calculation accuracy
- [ ] Leverage fallback chain
- [ ] Position sizing validation
- [ ] Position recovery flow
- [ ] Closed position detection
- [ ] Partial mode TP/SL
- [ ] Fee-aware sizing

## Deployment Checklist

- [ ] Account balance fetching working
- [ ] Symbol metadata loaded correctly
- [ ] Bybit API credentials valid
- [ ] Redis connectivity verified (for recovery)
- [ ] Position recovery tested
- [ ] Trade tracker functional
- [ ] All validations active
- [ ] Logging configured

## Questions and Support

For specific code locations, refer to:
1. RISK_MANAGEMENT_GUIDE.md - Overview and flow diagrams
2. DETAILED_CODE_REFERENCE.txt - Exact line numbers and functions
3. Files directly for latest implementation

---

**Analysis Date**: 2025-10-29
**Analysis Scope**: Comprehensive risk management and position tracking
**Status**: Complete
