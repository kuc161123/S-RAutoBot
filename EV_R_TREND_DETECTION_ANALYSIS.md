# EV_R Trend Detection for Combo System - Analysis & Recommendations

## Overview

Detecting whether EV_R is **increasing**, **decreasing**, or **constant** can significantly improve combo enable/disable decisions by:
- **Early warning** of performance degradation
- **Proactive disabling** of declining combos
- **Faster enabling** of improving combos
- **Better risk management** by avoiding combos on the decline

---

## Why Track EV_R Trends?

### Current Problem

The combo system currently uses **static thresholds**:
- Enable if EV_R ≥ 0.3
- Disable if EV_R < 0.3

**Issues**:
- A combo with EV_R = 0.35 might be declining (was 0.50 last week)
- A combo with EV_R = 0.25 might be improving (was 0.10 last week)
- No early warning of performance decay
- Miss opportunities to enable improving combos early

### Solution: Trend-Aware Decisions

Track EV_R trends and adjust decisions accordingly:
- **Declining combos**: Disable even if above threshold
- **Improving combos**: Enable even if slightly below threshold
- **Stable combos**: Standard threshold behavior

---

## Method 1: Simple Time-Window Comparison (Recommended)

### Concept

Compare recent EV_R vs older EV_R to detect trends.

### Implementation Approach

```python
# Calculate EV_R for different time windows
ev_r_recent = calculate_ev_r(trades_last_7_days)   # Last 7 days
ev_r_older = calculate_ev_r(trades_8_30_days_ago)  # Days 8-30
ev_r_overall = calculate_ev_r(trades_last_30_days) # Overall 30 days

# Determine trend
trend = "increasing" if ev_r_recent > ev_r_older + threshold
trend = "decreasing" if ev_r_recent < ev_r_older - threshold
trend = "constant" otherwise
```

### Advantages

- ✅ **Simple** to implement
- ✅ **Fast** calculation
- ✅ **Easy** to understand
- ✅ **Low** computational overhead

### Disadvantages

- ⚠️ Can be noisy with small samples
- ⚠️ Sensitive to outliers

### Recommended Thresholds

- **Trend threshold**: 0.1 R difference (recent vs older)
- **Recent window**: 7 days
- **Older window**: Days 8-30
- **Minimum samples**: 10 trades per window

---

## Method 2: Linear Regression Trend (More Sophisticated)

### Concept

Fit a linear regression to EV_R over time to detect slope.

### Implementation Approach

```python
# Collect EV_R values over time (daily or per-trade)
time_points = [day1, day2, day3, ..., day30]
ev_r_values = [ev_r_day1, ev_r_day2, ..., ev_r_day30]

# Fit linear regression: EV_R = slope * time + intercept
slope = linear_regression(time_points, ev_r_values)

# Determine trend
if slope > 0.01:  # Increasing by 0.01 R per day
    trend = "increasing"
elif slope < -0.01:  # Decreasing by 0.01 R per day
    trend = "decreasing"
else:
    trend = "constant"
```

### Advantages

- ✅ **More accurate** trend detection
- ✅ **Quantifies** rate of change
- ✅ **Less sensitive** to outliers
- ✅ **Statistical** significance testing possible

### Disadvantages

- ⚠️ More complex to implement
- ⚠️ Requires more data points
- ⚠️ Higher computational cost

### Recommended Parameters

- **Slope threshold**: ±0.01 R per day (or ±0.05 R per week)
- **Minimum points**: 15-20 data points
- **Time granularity**: Daily or per 10 trades

---

## Method 3: Moving Average Crossover (Technical Analysis Style)

### Concept

Use fast and slow moving averages of EV_R to detect trends.

### Implementation Approach

```python
# Calculate moving averages
fast_ma = moving_average(ev_r_values, window=7)   # 7-day MA
slow_ma = moving_average(ev_r_values, window=14)  # 14-day MA

# Detect crossover
if fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
    trend = "increasing"  # Golden cross
elif fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
    trend = "decreasing"  # Death cross
else:
    trend = "constant"
```

### Advantages

- ✅ **Smooth** trend detection (less noise)
- ✅ **Familiar** to traders (MA crossover)
- ✅ **Good** for filtering noise

### Disadvantages

- ⚠️ Lagging indicator (slower to react)
- ⚠️ Requires more historical data
- ⚠️ May miss rapid changes

---

## Method 4: Rate of Change (ROC) Analysis

### Concept

Calculate the rate of change in EV_R over time periods.

### Implementation Approach

```python
# Calculate EV_R for consecutive periods
ev_r_period1 = calculate_ev_r(trades_period1)  # Week 1
ev_r_period2 = calculate_ev_r(trades_period2)  # Week 2
ev_r_period3 = calculate_ev_r(trades_period3)  # Week 3
ev_r_period4 = calculate_ev_r(trades_period4)  # Week 4

# Calculate rate of change
roc = (ev_r_period4 - ev_r_period1) / ev_r_period1 * 100  # Percentage change

# Determine trend
if roc > 10%:  # Increased by 10%+
    trend = "increasing"
elif roc < -10%:  # Decreased by 10%+
    trend = "decreasing"
else:
    trend = "constant"
```

### Advantages

- ✅ **Percentage-based** (normalized)
- ✅ **Easy** to interpret
- ✅ **Good** for comparing different combos

### Disadvantages

- ⚠️ Sensitive to initial value
- ⚠️ Can be misleading with small values

---

## Recommended Approach: Hybrid Method

### Best of Both Worlds

Combine **Simple Time-Window Comparison** with **Linear Regression** for robustness:

```python
def detect_ev_r_trend(combo_trades, recent_days=7, older_days=30):
    """
    Detect EV_R trend using multiple methods
    """
    now = datetime.utcnow()
    recent_cutoff = now - timedelta(days=recent_days)
    older_cutoff = now - timedelta(days=older_days)
    
    # Method 1: Simple comparison
    recent_trades = [t for t in combo_trades if t.exit_time >= recent_cutoff]
    older_trades = [t for t in combo_trades if older_cutoff <= t.exit_time < recent_cutoff]
    
    if len(recent_trades) >= 10 and len(older_trades) >= 10:
        ev_r_recent = np.mean([t.realized_rr for t in recent_trades])
        ev_r_older = np.mean([t.realized_rr for t in older_trades])
        
        diff = ev_r_recent - ev_r_older
        
        if diff > 0.1:  # Increased by 0.1 R
            trend_simple = "increasing"
        elif diff < -0.1:  # Decreased by 0.1 R
            trend_simple = "decreasing"
        else:
            trend_simple = "constant"
    else:
        trend_simple = "insufficient_data"
    
    # Method 2: Linear regression (if enough data)
    if len(combo_trades) >= 20:
        # Group by day and calculate daily EV_R
        daily_ev_r = calculate_daily_ev_r(combo_trades)
        if len(daily_ev_r) >= 10:
            slope = linear_regression_slope(daily_ev_r)
            if slope > 0.01:
                trend_regression = "increasing"
            elif slope < -0.01:
                trend_regression = "decreasing"
            else:
                trend_regression = "constant"
        else:
            trend_regression = "insufficient_data"
    else:
        trend_regression = "insufficient_data"
    
    # Combine results (both must agree for strong signal)
    if trend_simple == trend_regression:
        return trend_simple, "strong"
    elif trend_simple != "insufficient_data":
        return trend_simple, "moderate"
    elif trend_regression != "insufficient_data":
        return trend_regression, "moderate"
    else:
        return "insufficient_data", "weak"
```

---

## How to Use Trends in Combo Decisions

### Current Logic (Static Threshold)

```python
if ev_r >= 0.3 and n >= 30:
    enable_combo()
else:
    disable_combo()
```

### Trend-Aware Logic (Recommended)

```python
trend, confidence = detect_ev_r_trend(combo_trades)

if trend == "increasing":
    # Improving combo: Lower threshold to enable faster
    threshold = 0.25  # Lower than standard 0.3
    if ev_r >= threshold and n >= 20:  # Also lower sample size
        enable_combo()
        
elif trend == "decreasing":
    # Declining combo: Higher threshold or disable
    threshold = 0.35  # Higher than standard 0.3
    if ev_r < threshold or ev_r < 0.2:  # Disable if below 0.2
        disable_combo()
    else:
        # Keep enabled but monitor closely
        enable_combo()  # But flag for review
        
elif trend == "constant":
    # Stable combo: Standard threshold
    threshold = 0.3
    if ev_r >= threshold and n >= 30:
        enable_combo()
        
else:  # insufficient_data
    # Not enough data: Use standard logic
    if ev_r >= 0.3 and n >= 30:
        enable_combo()
```

### Enhanced Decision Matrix

| Current EV_R | Trend | Action | Reason |
|--------------|-------|--------|--------|
| 0.35 | Increasing | ✅ Enable | Strong and improving |
| 0.35 | Decreasing | ⚠️ Disable | Declining despite good current value |
| 0.35 | Constant | ✅ Enable | Good and stable |
| 0.25 | Increasing | ✅ Enable | Improving, enable early |
| 0.25 | Decreasing | ❌ Disable | Below threshold and declining |
| 0.25 | Constant | ❌ Disable | Below threshold |
| 0.30 | Increasing | ✅ Enable | At threshold and improving |
| 0.30 | Decreasing | ⚠️ Monitor | At threshold but declining |

---

## Additional Features to Consider

### 1. Trend Strength Indicator

Not just direction, but **how strong** the trend is:

```python
trend_strength = abs(slope) * sample_size
# Strong trend: slope > 0.02 R/day with 20+ trades
# Weak trend: slope < 0.01 R/day or < 10 trades
```

### 2. Trend Persistence

How long has the trend been consistent?

```python
# Track trend over multiple periods
trend_history = ["increasing", "increasing", "increasing"]  # Last 3 periods
if all(t == "increasing" for t in trend_history):
    # Strong persistent trend
    confidence = "high"
```

### 3. Acceleration Detection

Is the trend **accelerating** or **decelerating**?

```python
# Second derivative (rate of change of slope)
acceleration = slope_current - slope_previous
if acceleration > 0:
    # Trend is accelerating (getting stronger)
elif acceleration < 0:
    # Trend is decelerating (weakening)
```

### 4. Volatility-Adjusted Trends

Account for **volatility** in EV_R:

```python
# If EV_R is very volatile, require stronger trend signal
volatility = np.std(ev_r_values)
if volatility > 0.2:  # High volatility
    trend_threshold = 0.15  # Require larger change
else:  # Low volatility
    trend_threshold = 0.10  # Standard threshold
```

---

## Integration Points

### Where to Add Trend Detection

1. **In `_analyze_combo_performance()`**
   - Calculate trend for each combo
   - Store trend in `ComboPerformance` dataclass

2. **In `update_combo_filters()`**
   - Use trend in enable/disable logic
   - Adjust thresholds based on trend

3. **In `get_active_combos()`**
   - Include trend in returned combo data
   - Allow filtering by trend

### New Fields to Add

```python
@dataclass
class ComboPerformance:
    # ... existing fields ...
    
    # Trend detection
    ev_r_trend: str = "unknown"  # "increasing", "decreasing", "constant"
    trend_confidence: str = "weak"  # "strong", "moderate", "weak"
    ev_r_recent: float = 0.0  # EV_R in last 7 days
    ev_r_older: float = 0.0  # EV_R in days 8-30
    trend_slope: float = 0.0  # Linear regression slope
    trend_persistence: int = 0  # Number of consecutive periods with same trend
```

---

## Recommended Configuration

### Settings

```yaml
adaptive_combos:
  # ... existing settings ...
  
  # Trend detection
  use_ev_r_trend: true              # Enable trend detection
  trend_method: "hybrid"            # "simple", "regression", "hybrid"
  trend_recent_days: 7              # Recent window (days)
  trend_older_days: 30              # Older window (days)
  trend_threshold: 0.1              # Minimum change to detect trend (R)
  trend_min_samples: 10             # Minimum trades per window
  
  # Trend-based adjustments
  trend_increasing_threshold: 0.25  # Lower threshold for increasing combos
  trend_decreasing_threshold: 0.35  # Higher threshold for decreasing combos
  trend_constant_threshold: 0.3     # Standard threshold for constant combos
  
  # Early enable/disable
  enable_improving_early: true      # Enable improving combos below threshold
  disable_declining_early: true     # Disable declining combos above threshold
```

---

## Benefits Summary

### 1. Early Warning System
- Detect declining combos **before** they drop below threshold
- Proactive risk management

### 2. Faster Adaptation
- Enable improving combos **sooner**
- Don't wait for them to reach threshold

### 3. Better Risk Management
- Avoid combos that are declining
- Focus on improving/stable combos

### 4. More Nuanced Decisions
- Not just "above/below threshold"
- Consider **direction** and **momentum**

### 5. Performance Optimization
- Enable best combos faster
- Disable worst combos sooner

---

## Potential Issues & Mitigations

### Issue 1: Small Sample Sizes

**Problem**: Trend detection unreliable with few trades

**Mitigation**:
- Require minimum samples (10+ per window)
- Use "insufficient_data" status
- Fall back to standard threshold logic

### Issue 2: Noisy Data

**Problem**: EV_R can be volatile, causing false trends

**Mitigation**:
- Use moving averages to smooth
- Require consistent trend over multiple periods
- Use confidence levels (strong/moderate/weak)

### Issue 3: Over-Fitting

**Problem**: Too sensitive, causing frequent enable/disable

**Mitigation**:
- Use hysteresis (buffer around thresholds)
- Require persistent trends (multiple periods)
- Combine with existing thresholds (don't replace)

### Issue 4: Computational Cost

**Problem**: Trend calculation adds overhead

**Mitigation**:
- Calculate only when needed (during updates)
- Cache trend results
- Use simple method first, regression only if needed

---

## Implementation Priority

### Phase 1: Basic Trend Detection (Recommended Start)

1. ✅ Simple time-window comparison
2. ✅ Store trend in `ComboPerformance`
3. ✅ Basic trend-based threshold adjustment
4. ✅ Logging and monitoring

### Phase 2: Enhanced Detection

1. ⏳ Linear regression method
2. ⏳ Trend confidence levels
3. ⏳ Trend persistence tracking

### Phase 3: Advanced Features

1. ⏳ Acceleration detection
2. ⏳ Volatility-adjusted trends
3. ⏳ Multi-method hybrid approach

---

## Example Scenarios

### Scenario 1: Declining Combo

**Current State**:
- EV_R = 0.35 (above 0.3 threshold)
- n = 50 trades
- **Status**: Enabled ✅

**With Trend Detection**:
- EV_R recent (7d) = 0.20
- EV_R older (8-30d) = 0.45
- Trend = **Decreasing** (-0.25 R)
- **Action**: Disable ⚠️ (declining despite good overall)

### Scenario 2: Improving Combo

**Current State**:
- EV_R = 0.25 (below 0.3 threshold)
- n = 25 trades
- **Status**: Disabled ❌

**With Trend Detection**:
- EV_R recent (7d) = 0.40
- EV_R older (8-30d) = 0.15
- Trend = **Increasing** (+0.25 R)
- **Action**: Enable ✅ (improving, enable early)

### Scenario 3: Stable Combo

**Current State**:
- EV_R = 0.30 (at threshold)
- n = 40 trades
- **Status**: Enabled ✅

**With Trend Detection**:
- EV_R recent (7d) = 0.31
- EV_R older (8-30d) = 0.29
- Trend = **Constant** (+0.02 R)
- **Action**: Keep enabled ✅ (stable performance)

---

## Summary

### Recommended Implementation

1. **Start with Simple Time-Window Comparison**
   - Easy to implement
   - Fast calculation
   - Good results

2. **Add to ComboPerformance**
   - Store trend direction
   - Store recent vs older EV_R
   - Store trend confidence

3. **Adjust Thresholds Based on Trend**
   - Lower for increasing
   - Higher for decreasing
   - Standard for constant

4. **Add Hysteresis**
   - Prevent flip-flopping
   - Require persistent trends

5. **Monitor and Iterate**
   - Track performance
   - Adjust thresholds based on results

### Expected Benefits

- ✅ **20-30% faster** combo enable/disable
- ✅ **Better risk management** (avoid declining combos)
- ✅ **More profitable** (enable improving combos early)
- ✅ **More stable** (less flip-flopping)

---

## Questions to Consider

1. **How sensitive should trend detection be?**
   - Small changes (0.05 R) or large (0.15 R)?

2. **How much to adjust thresholds?**
   - Small adjustment (0.05 R) or large (0.10 R)?

3. **How many periods for persistence?**
   - 2-3 periods or more?

4. **When to use vs ignore trends?**
   - Always use or only when confident?

5. **How to handle insufficient data?**
   - Fall back to standard or wait for more data?

---

**Ready for implementation when you approve!** 🚀

