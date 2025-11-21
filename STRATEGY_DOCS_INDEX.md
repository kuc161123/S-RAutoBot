# Trading Strategy Documentation Index

This directory contains comprehensive documentation of the AutoTrading Bot's three trading strategies. Use this index to find the information you need.

## Documentation Files

### 1. **STRATEGY_SUMMARY.txt** (START HERE)
**Purpose**: High-level executive overview of all strategies  
**Best for**: Getting a quick understanding of what each strategy does  
**Content**:
- Strategy overview (MR, Scalp, Trend)
- ML system architecture
- Execution flow diagram
- Configuration control points
- Common issues and solutions
- File organization

**Size**: ~7,500 words | **Format**: Plain text with ASCII diagrams

---

### 2. **STRATEGY_QUICK_REFERENCE.md** (QUICK LOOKUP)
**Purpose**: Fast lookup reference for active strategies  
**Best for**: Looking up specific parameters, debugging issues, checking current status  
**Content**:
- Active strategies summary
- Key parameters for each strategy
- Configuration control master switches
- Stop loss calculation methods
- Common configuration adjustments
- Execution flow summary

**Size**: ~3,000 words | **Format**: Markdown with code blocks

---

### 3. **STRATEGY_ANALYSIS.md** (DETAILED REFERENCE)
**Purpose**: Deep technical analysis of all strategy implementations  
**Best for**: Understanding the complete logic, edge cases, and data structures  
**Content**:

#### Per Strategy:
- **Architecture**: Files, ML scorer, phantom tracker, status
- **Entry Conditions**: Multi-phase breakdown with state machines
- **Phase 1**: Detection/Identification logic
- **Phase 2**: Confirmation or breakout detection
- **Phase 3**: Signal formation or pullback detection
- **Phase 4**: Confirmation or validation
- **Phase 5**: Execution logic
- **Stop Loss**: Calculation methods, options, adjustments
- **Take Profit**: Fee-adjusted calculations
- **ML Features**: All features calculated per strategy (12-35+ total)
- **Configuration**: Gates, thresholds, phantom settings
- **State Machine**: Full state flow diagram

#### ML System:
- Phantom tracking architecture
- ML scoring timeline (0-50, 50-200, 200+ trades)
- Quality scoring (Qscore) calculation
- Feature engineering per strategy
- Adaptive flow control

#### Signal Flow:
- Signal generation pipeline
- Phantom recording
- ML scoring gates
- Execution gate checks
- Order placement logic

#### Edge Cases:
- Trend vs Mean Reversion differences
- Scalp-specific considerations
- Phantom learning edge cases

**Size**: ~12,000 words | **Format**: Markdown with Python code blocks and detailed tables

---

## How to Use This Documentation

### I Want To...

#### Understand the Overall System
1. Read **STRATEGY_SUMMARY.txt** (5-10 minutes)
2. Reference **STRATEGY_QUICK_REFERENCE.md** for specifics

#### Change a Configuration Parameter
1. Find parameter name in **STRATEGY_QUICK_REFERENCE.md**
2. Check current value in **config.yaml**
3. Read about implications in **STRATEGY_SUMMARY.txt** section "Common Configuration Adjustments"
4. Make change and test

#### Debug Low Execution Frequency
1. Check **STRATEGY_QUICK_REFERENCE.md** "Execution Flow Summary"
2. Use Telegram `/status` command
3. Review checklist in **STRATEGY_QUICK_REFERENCE.md** "Quick Debug Checklist"
4. Check **STRATEGY_SUMMARY.txt** "Common Configuration Adjustments"

#### Learn How Mean Reversion Works
1. Read "Mean Reversion Strategy" in **STRATEGY_SUMMARY.txt**
2. Review detailed phases in **STRATEGY_ANALYSIS.md** Section 2
3. Check ML features in **STRATEGY_ANALYSIS.md** Section 2.3
4. Review config options in **STRATEGY_QUICK_REFERENCE.md** "Key Parameters by Strategy"

#### Learn How Scalping Works
1. Read "Scalping Strategy" in **STRATEGY_SUMMARY.txt**
2. Review detailed phases in **STRATEGY_ANALYSIS.md** Section 3
3. Review execution gates in **STRATEGY_ANALYSIS.md** Section 3 "Execution Gates"
4. Understand Adaptive Combo Manager and Pro Rules fallback system

#### Understand Phantom Tracking
1. Review "ML System Architecture" in **STRATEGY_SUMMARY.txt**
2. Read "Phantom Learning Edge Cases" in **STRATEGY_ANALYSIS.md** Section 7
3. Check timeout values in **STRATEGY_QUICK_REFERENCE.md** "Key Parameters by Strategy"

#### Deploy or Test the Bot
1. Review "Deployment & Testing" in **STRATEGY_SUMMARY.txt**
2. Check required environment in **CLAUDE.md**
3. Run appropriate test command from list

#### Monitor Performance
1. Check "Performance Metrics & Monitoring" in **STRATEGY_SUMMARY.txt**
2. Use Telegram commands listed in **STRATEGY_QUICK_REFERENCE.md**
3. Review "Health Checks" section for what to monitor

---

## Key Concepts Quick Links

### Stop Loss Methods
- **Trend/MR Hybrid**: See **STRATEGY_ANALYSIS.md** Section 1.4 & 2.3
- **Scalp Mean-Band**: See **STRATEGY_ANALYSIS.md** Section 3.5
- Quick reference: **STRATEGY_QUICK_REFERENCE.md** "Stop Loss Calculation"

### ML & Phantom System
- **Overview**: **STRATEGY_SUMMARY.txt** "ML System Architecture"
- **Details**: **STRATEGY_ANALYSIS.md** Section 6 & 7
- **Configuration**: **STRATEGY_QUICK_REFERENCE.md** Phantom Flow Control

### Execution Flow
- **Diagram**: **STRATEGY_SUMMARY.txt** "Execution Flow"
- **Detailed Steps**: **STRATEGY_ANALYSIS.md** Section 6 "Signal Flow"
- **Quick Reference**: **STRATEGY_QUICK_REFERENCE.md** "Execution Flow Summary"

### Configuration Control
- **Master Switches**: **STRATEGY_QUICK_REFERENCE.md** "Configuration Control"
- **All Settings**: **config.yaml** (master configuration file)
- **Adjustments**: **STRATEGY_SUMMARY.txt** "Common Configuration Adjustments"

---

## Strategy Comparison Table

| Aspect | Trend | Mean Reversion | Scalp |
|--------|-------|----------------|-------|
| **Status** | DISABLED (Phantom) | ENABLED | ENABLED |
| **Timeframe** | 15m | 15m | 3m |
| **Entry Type** | S/R Breakout + HL/LH | Range Edge Touch | EMA Alignment + Multi-Anchor |
| **Win Rate** | 55-65% | 45-55% | 50-60% |
| **Risk/Reward** | 2.5:1 | 2.5:1 | 2.1:1 |
| **Features** | 22 | 35+ | 12 |
| **Setup Frequency** | Low | High | Very High |
| **Phantom Timeout** | 168h (1w) | 36h (1.5d) | 8h |
| **SL Method** | Hybrid 3-option | Hybrid 3-option | Mean-band centered |
| **ML Threshold** | 75 | 65 | 70 |

---

## File References

### Strategy Implementation Files
- `/Users/lualakol/AutoTrading Bot/strategy_mean_reversion.py`
- `/Users/lualakol/AutoTrading Bot/strategy_scalp.py`
- `/Users/lualakol/AutoTrading Bot/strategy_pullback_ml_learning.py`

### ML Scoring Files
- `/Users/lualakol/AutoTrading Bot/ml_scorer_mean_reversion.py`
- `/Users/lualakol/AutoTrading Bot/ml_scorer_scalp.py`
- `/Users/lualakol/AutoTrading Bot/ml_scorer_trend.py`

### Phantom Tracking Files
- `/Users/lualakol/AutoTrading Bot/mr_phantom_tracker.py`
- `/Users/lualakol/AutoTrading Bot/scalp_phantom_tracker.py`
- `/Users/lualakol/AutoTrading Bot/phantom_trade_tracker.py`

### Configuration Files
- `/Users/lualakol/AutoTrading Bot/config.yaml` (Master settings)
- `/Users/lualakol/AutoTrading Bot/CLAUDE.md` (Development guide)

### Documentation Files
- `/Users/lualakol/AutoTrading Bot/STRATEGY_SUMMARY.txt` (This index file)
- `/Users/lualakol/AutoTrading Bot/STRATEGY_ANALYSIS.md`
- `/Users/lualakol/AutoTrading Bot/STRATEGY_QUICK_REFERENCE.md`
- `/Users/lualakol/AutoTrading Bot/STRATEGY_DOCS_INDEX.md` (This file)

---

## Common Tasks Quick Links

| Task | Reference |
|------|-----------|
| Change execution threshold | **STRATEGY_QUICK_REFERENCE.md** → "Configuration Control" |
| Adjust risk percentage | **config.yaml** → trade.risk_percent / scalp.exec.risk_percent |
| Increase phantom flow | **config.yaml** → phantom_flow.daily_target |
| Disable a strategy | **config.yaml** → scalp.enabled / mr.exec.enabled / trend.exec.enabled |
| Change stop loss | **STRATEGY_ANALYSIS.md** → Strategy-specific "Stop Loss Calculation" |
| Monitor ML training | **STRATEGY_QUICK_REFERENCE.md** → "/ml_stats" Telegram command |
| Debug no trades | **STRATEGY_QUICK_REFERENCE.md** → "Quick Debug Checklist" |
| Understand phantom learning | **STRATEGY_ANALYSIS.md** → Section 7 "Phantom Learning Edge Cases" |

---

## Last Updated
- Documentation: 2025-11-10
- Configuration reviewed: Yes (config.yaml current)
- Strategies status: Mean Reversion (ENABLED), Scalp (ENABLED), Trend (PHANTOM-ONLY)

---

**Need help?** Start with **STRATEGY_SUMMARY.txt** for an overview, then use the links above to dive deeper into specific topics.
