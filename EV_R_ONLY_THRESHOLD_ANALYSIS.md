# Using EV_R Only as Combo Threshold - Analysis

## Why EV_R Only Makes Sense

### EV_R is the Ultimate Metric

**EV_R (Expected Value in R)** directly tells you profitability:
- **EV_R > 0**: Profitable
- **EV_R = 0**: Break-even
- **EV_R < 0**: Losing

Unlike Win Rate, EV_R:
- ✅ **Accounts for R:R automatically** (no need to calculate separately)
- ✅ **Direct profitability measure** (tells you exactly how much you'll make per trade)
- ✅ **Works with any R:R ratio** (adapts if you change R:R)
- ✅ **Simpler logic** (one metric instead of WR + EV_R)

### Current Approach (WR + EV_R)

```yaml
min_wr_threshold: 42.0%  # Wilson LB
ev_floor_r: 0.2          # Minimum EV
```

**Problem**: Two gates to manage, can be redundant.

### EV_R Only Approach

```yaml
use_ev_r_only: true
min_ev_r: 0.2            # Only gate needed
min_sample_size: 30      # Still need for confidence
```

**Benefit**: Simpler, more direct.

---

## EV_R Threshold Recommendations

### With R:R = 2.1:1

| EV_R Threshold | Equivalent WR* | Assessment | Recommendation |
|----------------|----------------|------------|----------------|
| 0.0 | 32.3% | Break-even | Too low |
| 0.1 | 35.5% | Minimal profit | Low |
| **0.2** | **38.7%** | **Good profit** | **Recommended** |
| **0.3** | **41.9%** | **Strong profit** | **Premium** |
| 0.4 | 45.2% | Excellent profit | Aggressive |
| 0.5 | 48.4% | Outstanding | Very aggressive |

*Calculated: WR = (EV_R + 1) / 3.1

### Recommended Thresholds

#### Option 1: Conservative (Good Balance)
```yaml
use_ev_r_only: true
min_ev_r: 0.2            # 0.2 R per trade = good profitability
min_sample_size: 30
```

**Rationale**:
- 0.2 R per trade = meaningful profit
- ≈38.7% WR equivalent
- Good balance of quality and quantity

#### Option 2: Balanced (Recommended)
```yaml
use_ev_r_only: true
min_ev_r: 0.3            # 0.3 R per trade = strong profitability
min_sample_size: 30
```

**Rationale**:
- 0.3 R per trade = strong profit
- ≈41.9% WR equivalent
- Higher quality combos

#### Option 3: Aggressive (Maximum Quality)
```yaml
use_ev_r_only: true
min_ev_r: 0.4            # 0.4 R per trade = excellent profitability
min_sample_size: 40      # More samples for confidence
```

**Rationale**:
- 0.4 R per trade = excellent profit
- ≈45.2% WR equivalent
- Only best combos

---

## Statistical Confidence with EV_R

### The Challenge

EV_R can be volatile with small samples:
- 10 trades: EV_R might be misleading
- 30 trades: More reliable
- 100+ trades: Very reliable

### Solution: Confidence Intervals for EV_R

Calculate **EV_R confidence interval**:

```python
def ev_r_confidence_interval(trades: List[Trade], confidence: float = 0.95):
    """
    Calculate confidence interval for EV_R
    - Lower bound: Conservative estimate
    - Use lower bound for gating decisions
    """
    ev_r_values = [t.realized_rr for t in trades]
    mean_ev = np.mean(ev_r_values)
    std_ev = np.std(ev_r_values)
    n = len(ev_r_values)
    
    # t-distribution for small samples
    from scipy import stats
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_critical * (std_ev / np.sqrt(n))
    
    lower_bound = mean_ev - margin
    return lower_bound, mean_ev, mean_ev + margin
```

**Gate on lower bound** instead of raw EV_R.

---

## Implementation Options

### Option A: Simple EV_R Only (Easiest)

```yaml
adaptive_combos:
  use_ev_r_only: true
  min_ev_r: 0.3          # Only gate
  min_sample_size: 30
  # Remove WR thresholds
```

**Pros**: Simple, direct
**Cons**: No confidence adjustment for small samples

### Option B: EV_R with Confidence Interval (Recommended)

```yaml
adaptive_combos:
  use_ev_r_only: true
  use_ev_r_confidence: true    # Use lower bound
  min_ev_r: 0.3                 # Lower bound threshold
  min_sample_size: 30
```

**Pros**: Statistically sound, accounts for uncertainty
**Cons**: Slightly more complex

### Option C: Hybrid (EV_R Primary, WR Secondary)

```yaml
adaptive_combos:
  use_ev_r_primary: true
  min_ev_r: 0.3
  min_wr_threshold: 35.0        # Lower threshold, just as safety
  min_sample_size: 30
```

**Pros**: Safety net, prevents edge cases
**Cons**: Two gates again

---

## Comparison: WR vs EV_R Only

### Scenario: Combo with 40% WR, R:R = 2.1:1

**WR Approach**:
- WR = 40% → Below 42% threshold → **DISABLED** ❌
- But EV_R = 0.24 R → Actually profitable!

**EV_R Only Approach**:
- EV_R = 0.24 R → Above 0.2 threshold → **ENABLED** ✅
- Correctly identifies profitable combo

### Scenario: Combo with 50% WR, R:R = 1.5:1

**WR Approach**:
- WR = 50% → Above 42% threshold → **ENABLED** ✅
- But EV_R = 0.25 R → Profitable

**EV_R Only Approach**:
- EV_R = 0.25 R → Above 0.2 threshold → **ENABLED** ✅
- Same result, but more direct

### Scenario: Combo with 45% WR, R:R = 1.8:1

**WR Approach**:
- WR = 45% → Above 42% threshold → **ENABLED** ✅
- EV_R = 0.26 R → Profitable

**EV_R Only Approach**:
- EV_R = 0.26 R → Above 0.2 threshold → **ENABLED** ✅
- Same result

**Key Insight**: EV_R only is more accurate because it accounts for actual R:R achieved.

---

## Recommended Implementation

### Best Approach: EV_R with Confidence Interval

```yaml
adaptive_combos:
  enabled: true
  use_ev_r_only: true              # Use EV_R as primary/only gate
  use_ev_r_confidence: true        # Use lower bound for confidence
  min_ev_r: 0.3                    # Lower bound EV_R threshold
  min_sample_size: 30               # Minimum samples for confidence
  ev_r_confidence_level: 0.95     # 95% confidence interval
  # WR thresholds become optional/disabled
  min_wr_threshold: 0.0            # Disabled (not used)
  use_wilson_lb: false             # Not needed with EV_R confidence
```

**Benefits**:
1. **Simpler**: One metric instead of two
2. **More accurate**: Accounts for actual R:R achieved
3. **Statistically sound**: Confidence intervals account for uncertainty
4. **Adaptive**: Works with any R:R ratio automatically

---

## Code Changes Needed

### 1. Add EV_R Confidence Calculation

```python
def _ev_r_confidence_lb(self, trades: List[Trade], confidence: float = 0.95) -> float:
    """Calculate EV_R lower bound with confidence interval"""
    if len(trades) < 2:
        return 0.0
    
    ev_r_values = [t.realized_rr for t in trades]
    mean_ev = np.mean(ev_r_values)
    std_ev = np.std(ev_r_values)
    n = len(ev_r_values)
    
    # Use t-distribution for small samples
    from scipy import stats
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_critical * (std_ev / np.sqrt(n))
    
    return mean_ev - margin  # Lower bound
```

### 2. Update Gating Logic

```python
if self.use_ev_r_only:
    # Calculate EV_R lower bound
    ev_r_lb = self._ev_r_confidence_lb(trades, self.ev_r_confidence_level)
    ev_ok = ev_r_lb >= self.min_ev_r
    n_ok = perf.n >= self.min_sample_size
    
    # Only gate on EV_R and sample size
    base_enabled = bool(n_ok and ev_ok)
else:
    # Original WR + EV_R logic
    ...
```

---

## Recommendation

### ✅ YES, EV_R Only Makes Perfect Sense!

**Why**:
1. **More direct**: EV_R directly measures profitability
2. **Simpler**: One gate instead of two
3. **More accurate**: Accounts for actual R:R achieved
4. **Adaptive**: Works with any R:R ratio

### Recommended Configuration:

```yaml
adaptive_combos:
  enabled: true
  use_ev_r_only: true
  use_ev_r_confidence: true        # Use confidence intervals
  min_ev_r: 0.3                    # Lower bound EV_R (strong profit)
  min_sample_size: 30
  ev_r_confidence_level: 0.95      # 95% confidence
```

**This is actually BETTER than WR + EV_R because**:
- Simpler logic
- More accurate (accounts for actual R:R)
- Statistically sound (confidence intervals)
- One clear metric to optimize

---

## Summary

**EV_R Only Approach**:
- ✅ Simpler (one metric)
- ✅ More accurate (accounts for R:R)
- ✅ More direct (profitability measure)
- ✅ Statistically sound (with confidence intervals)

**Recommended Threshold**: **0.3 R** (lower bound with 95% confidence)

This gives you strong profitability while maintaining statistical rigor.

