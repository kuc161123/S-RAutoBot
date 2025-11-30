# Adaptive Combo System - Implementation Analysis

## Overview

The Adaptive Combo Manager is a **dynamic filtering system** that automatically enables/disables trading pattern combinations based on their historical performance. It learns from phantom trades (signals that didn't execute) and executed trades to identify high-performing patterns.

---

## Architecture

### Core Components

1. **AdaptiveComboManager** (`autobot/strategies/scalp/combos.py`)
   - Analyzes phantom trade outcomes
   - Calculates performance metrics (WR, EV_R)
   - Enables/disables combos based on thresholds
   - Persists state to Redis

2. **Combo Key Generation** (`bot.py:_scalp_combo_key_from_features`)
   - Converts signal features into combo identifiers
   - Uses binning logic to categorize continuous values

3. **Combo Matching** (`bot.py:_scalp_combo_allowed`)
   - Checks if signal matches an enabled combo
   - Routes to Pro Rules fallback if no match

---

## How Combos Work

### 1. Combo Definition

A **combo** is a unique combination of indicator states:

```
Combo Format: "RSI:{bin} MACD:{bull|bear} VWAP:{bin} Fib:{zone} {MTF|noMTF}"
```

**Example:**
```
"RSI:40-60 MACD:bull VWAP:<0.6 Fib:38-50 MTF"
```

### 2. Feature Binning

Continuous indicator values are binned into discrete categories:

**RSI Bins:**
- `<30`: RSI < 30
- `30-40`: 30 ≤ RSI < 40
- `40-60`: 40 ≤ RSI < 60
- `60-70`: 60 ≤ RSI < 70
- `70+`: RSI ≥ 70

**MACD Bins:**
- `bull`: MACD histogram > 0
- `bear`: MACD histogram ≤ 0

**VWAP Bins:**
- `<0.6`: VWAP distance < 0.6 ATR
- `0.6-1.2`: 0.6 ≤ VWAP distance < 1.2 ATR
- `1.2+`: VWAP distance ≥ 1.2 ATR

**Fibonacci Zones:**
- `0-23`, `23-38`, `38-50`, `50-61`, `61-78`, `78-100`

**MTF:**
- `MTF`: 15m timeframe agrees with trade direction
- `noMTF`: No MTF alignment

### 3. Combo Key Generation

```python
def _scalp_combo_key_from_features(feats: dict) -> Optional[str]:
    """Generate combo key from signal features"""
    # Extract and bin features
    rsi_bin = bin_rsi(feats['rsi_14'])
    macd_bin = 'bull' if feats['macd_hist'] > 0 else 'bear'
    vwap_bin = bin_vwap(feats['vwap_dist_atr'])
    fib_zone = feats['fib_zone']
    mtf = 'MTF' if feats['mtf_agree_15'] else 'noMTF'
    
    # Build combo key
    return f"RSI:{rsi_bin} MACD:{macd_bin} VWAP:{vwap_bin} Fib:{fib_zone} {mtf}"
```

**Total Possible Combos:**
- RSI: 5 bins
- MACD: 2 bins
- VWAP: 3 bins
- Fib: 6 zones
- MTF: 2 states
- **Total: 5 × 2 × 3 × 6 × 2 = 360 possible combos per side**

---

## Performance Analysis

### Data Source

The system analyzes **phantom trades** (signals that didn't execute) and **executed trades** from the last 30 days:

1. **Phantom Trades**: Signals that were blocked by gates
2. **Executed Trades**: Signals that passed gates and were executed

### Metrics Calculated

For each combo:

1. **Win Rate (WR)**: `wins / total_trades × 100`
2. **Wilson Lower Bound (WR_LB)**: Statistical confidence interval (95%)
3. **Expected Value in R (EV_R)**: Average R multiple per trade
4. **Sample Size (N)**: Number of trades in the combo

### Performance Breakdown

The system tracks:
- **Total trades** (N)
- **Wins** (W)
- **Executed vs Phantom** breakdown
- **24-hour recent** activity
- **Side-specific** metrics (long/short separate)

---

## Enable/Disable Logic

### Gating Criteria

A combo is **enabled** if ALL of these are true:

1. **Sample Size**: `N ≥ min_sample_size` (default: 30)
2. **Win Rate**: `WR_LB ≥ min_wr_threshold` (default: 45%)
   - Long: `WR_LB ≥ 45%` (configurable)
   - Short: `WR_LB ≥ 45%` (configurable)
3. **Expected Value**: `EV_R ≥ ev_floor_r` (default: -1000.0, effectively disabled)

### Hysteresis (Anti-Flip-Flop)

To prevent rapid enable/disable cycles:

- **Disabling**: Combo stays enabled if `WR_LB ≥ (threshold - 2%)`
- **Enabling**: Combo requires `WR_LB ≥ (threshold + 2%)` to enable

**Example:**
- Threshold: 45%
- Disable threshold: 43% (45% - 2%)
- Enable threshold: 47% (45% + 2%)

### Wilson Lower Bound

Uses **Wilson Score Interval** for statistical confidence:

```
WR_LB = (p + z²/(2n) - z×√(p(1-p)/n + z²/(4n²))) / (1 + z²/n)
```

Where:
- `p = wins/n` (win rate)
- `z = 1.96` (95% confidence)
- `n = sample size`

This provides a **conservative estimate** of true win rate, accounting for sample size uncertainty.

---

## Update Mechanism

### Update Triggers

1. **Trade-based**: Every N completed trades (default: 50)
2. **Time-based**: Every N hours (default: 1 hour)
3. **Manual**: Force update on startup

### Update Process

```
1. Collect phantom trades from last 30 days
2. Group by combo key (RSI, MACD, VWAP, Fib, MTF)
3. Calculate performance metrics for each combo
4. Load previous state from Redis
5. Apply enable/disable logic with hysteresis
6. Save updated state to Redis
7. Send Telegram notifications for changes
```

### State Persistence

- **Redis Keys**: 
  - `adaptive_combos:long` (long combos)
  - `adaptive_combos:short` (short combos)
- **TTL**: 7 days
- **Format**: JSON serialized `ComboPerformance` objects

---

## Integration with Execution

### Execution Flow

```
Signal Detected
    │
    ▼
Build Features
    │
    ▼
Generate Combo Key
    │
    ▼
Check Combo Manager
    │
    ├─► Manager Ready?
    │   │
    │   YES
    │   │
    │   ├─► Combo Enabled?
    │   │   │
    │   │   YES → EXECUTE
    │   │   │
    │   │   NO → Block (record phantom)
    │   │
    │   NO → Fall through to Pro Rules
    │
    ▼
Pro Rules Fallback
    │
    ├─► All rules pass? → EXECUTE
    └─► Any rule fails? → Block (record phantom)
```

### Manager Readiness

The manager is considered "ready" when:
- Manager is initialized
- Manager is enabled
- At least one combo has been analyzed (has data)

### Side Separation

- **Long combos** and **Short combos** are tracked separately
- Each side has independent enable/disable logic
- Prevents cross-contamination of metrics

---

## Strengths of Implementation

### ✅ Well-Implemented Aspects

1. **Statistical Rigor**
   - Uses Wilson Lower Bound for confidence
   - Accounts for sample size uncertainty
   - Prevents overfitting to small samples

2. **Hysteresis System**
   - Prevents flip-flopping
   - Reduces false enable/disable cycles
   - Provides stability

3. **Side Separation**
   - Long/short tracked independently
   - Prevents cross-contamination
   - Allows different thresholds per side

4. **Comprehensive Tracking**
   - Executed vs Phantom breakdown
   - 24-hour recent activity
   - Detailed performance metrics

5. **Robust Data Handling**
   - Handles missing features gracefully
   - Type conversion with fallbacks
   - Error handling throughout

6. **State Persistence**
   - Redis persistence for durability
   - Separate keys for long/short
   - TTL for automatic cleanup

7. **Update Frequency**
   - Trade-based updates (responsive)
   - Time-based updates (periodic)
   - Force updates on startup

8. **Notification System**
   - Telegram alerts for combo changes
   - Detailed change messages
   - Threshold information included

---

## Potential Issues & Improvements

### ⚠️ Areas for Improvement

1. **Combo Key Parsing**
   ```python
   # Current: Returns metadata only, doesn't parse combo key back to constraints
   # Issue: Can't match signals to combos without full feature set
   ```
   **Impact**: Low - system works but could be more efficient

2. **Default Behavior**
   ```python
   # is_combo_enabled() returns True if combo not found (fail open)
   # This means unknown combos are allowed by default
   ```
   **Impact**: Medium - Could allow untested patterns initially

3. **EV_R Floor**
   ```python
   # ev_floor_r = -1000.0 (effectively disabled)
   # No EV_R filtering currently
   ```
   **Impact**: Low - WR filtering is primary gate

4. **Combo Coverage**
   - 360 possible combos per side
   - May take time to build sufficient data for all combos
   - Some combos may never occur in practice

5. **Update Frequency**
   - Default: Every 50 trades or 1 hour
   - May be too frequent for small sample sizes
   - Could cause premature enable/disable decisions

6. **Minimum Sample Size**
   - Default: 30 trades
   - May be too high for rare combos
   - Could prevent good combos from enabling

---

## Code Quality Assessment

### ✅ Excellent Practices

1. **Type Safety**: Uses dataclasses, type hints
2. **Error Handling**: Comprehensive try/except blocks
3. **Logging**: Detailed logging throughout
4. **Documentation**: Good docstrings
5. **Separation of Concerns**: Clear responsibilities
6. **Configuration**: Highly configurable via config.yaml

### ⚠️ Minor Issues

1. **Magic Numbers**: Some hardcoded values (e.g., z=1.96)
2. **Combo Key Parsing**: Could be more robust
3. **Default Behavior**: Fail-open may not be desired
4. **Redis Dependency**: No graceful degradation if Redis unavailable

---

## Performance Characteristics

### Computational Complexity

- **Update**: O(N) where N = number of phantom trades
- **Lookup**: O(1) with Redis hash lookup
- **Memory**: O(C) where C = number of unique combos (~360 per side)

### Scalability

- **Handles**: 360 combos per side = 720 total
- **Data**: 30-day rolling window
- **Updates**: Efficient (only on trigger events)

---

## Recommendations

### Immediate Improvements

1. **Fail-Closed Default**
   ```python
   # Change is_combo_enabled() to return False if combo not found
   # Only enable combos with proven performance
   ```

2. **Adaptive Sample Size**
   ```python
   # Reduce min_sample_size for rare combos
   # Use confidence-based thresholds
   ```

3. **EV_R Filtering**
   ```python
   # Enable EV_R floor (e.g., 0.0 or 0.1)
   # Filter out negative EV combos
   ```

### Future Enhancements

1. **Combo Hierarchies**
   - Group related combos
   - Enable/disable groups together

2. **Time-Based Performance**
   - Track performance by time of day
   - Enable combos only during profitable hours

3. **Symbol-Specific Combos**
   - Track combo performance per symbol
   - Enable combos only for specific symbols

4. **Machine Learning Integration**
   - Use ML to predict combo performance
   - Pre-enable promising combos

---

## Conclusion

### Overall Assessment: **WELL IMPLEMENTED** ✅

The Adaptive Combo Manager is a **sophisticated, production-ready system** with:

- ✅ Strong statistical foundation (Wilson LB)
- ✅ Robust error handling
- ✅ Comprehensive tracking
- ✅ Good separation of concerns
- ✅ Configurable thresholds
- ✅ State persistence
- ✅ Notification system

### Minor Issues

- ⚠️ Default fail-open behavior
- ⚠️ EV_R filtering disabled
- ⚠️ Combo key parsing could be improved

### Verdict

**8.5/10** - Excellent implementation with room for minor improvements. The system is well-designed, statistically sound, and production-ready. The minor issues are easily addressable and don't significantly impact functionality.

---

## Usage Example

```python
# Initialize manager
manager = AdaptiveComboManager(config, redis_client, phantom_tracker)

# Update combos (called periodically)
enabled, disabled, changes = manager.update_combo_filters()

# Check if combo is enabled
active_combos = manager.get_active_combos(side='long')
combo_enabled = any(c['combo_id'] == my_combo_key for c in active_combos)

# Get stats
stats = manager.get_stats_summary()
print(f"Long enabled: {stats['long_enabled']}")
print(f"Short enabled: {stats['short_enabled']}")
```

