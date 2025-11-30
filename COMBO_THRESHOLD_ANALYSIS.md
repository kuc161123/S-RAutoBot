# Combo Threshold Analysis & Recommendations

## Current Settings

```yaml
min_wr_threshold: 45.0%
min_wr_threshold_long: 45.0%
min_wr_threshold_short: 45.0%
min_sample_size: 30
use_wilson_lb: false  # Using raw WR
ev_floor_r: 0.0
```

---

## Statistical Analysis

### Break-Even Calculation

With **R:R = 2.1:1**:
- **Break-even WR** = 1 / (1 + R:R) = 1 / (1 + 2.1) = **32.3%**

### Expected Value at Different WR Levels

| Win Rate | EV per Trade (in R) | Annualized* | Assessment |
|----------|---------------------|-------------|------------|
| 32.3% | 0.00 R | 0% | Break-even |
| 40% | 0.24 R | +24% | Acceptable |
| 45% | 0.40 R | +40% | **Good** |
| 50% | 0.55 R | +55% | **Excellent** |
| 55% | 0.71 R | +71% | Outstanding |

*Assuming 100 trades/year

**Formula**: `EV = (WR × R:R) - ((1-WR) × 1)`

---

## Wilson Lower Bound vs Raw WR

### Why Use Wilson LB?

**Wilson Lower Bound** provides statistical confidence accounting for sample size:

| Sample Size | Raw WR | Wilson LB (95% CI) | Difference |
|-------------|--------|-------------------|------------|
| 10 trades | 60% | 31.3% | -28.7% |
| 30 trades | 50% | 33.2% | -16.8% |
| 50 trades | 50% | 36.7% | -13.3% |
| 100 trades | 50% | 40.3% | -9.7% |
| 200 trades | 50% | 43.1% | -6.9% |

**Key Insight**: Small samples need higher raw WR to meet LB thresholds.

### Recommendation: **USE WILSON LB**

- More statistically sound
- Prevents overfitting to small samples
- Accounts for uncertainty
- Better risk management

---

## Threshold Recommendations

### Option 1: Conservative (Recommended for Production)

**Use Wilson LB with moderate thresholds:**

```yaml
use_wilson_lb: true              # Enable Wilson LB
min_wr_threshold_long: 40.0      # Wilson LB ≥ 40% (raw WR ~50% at n=30)
min_wr_threshold_short: 40.0     # Wilson LB ≥ 40% (raw WR ~50% at n=30)
min_sample_size: 30              # Minimum 30 trades
ev_floor_r: 0.2                  # Require positive EV (0.2 R minimum)
```

**Rationale**:
- 40% Wilson LB ≈ 50% raw WR at n=30
- EV_R ≥ 0.2 ensures profitability
- Conservative but statistically sound

### Option 2: Balanced (Good Balance)

**Use Wilson LB with standard thresholds:**

```yaml
use_wilson_lb: true
min_wr_threshold_long: 42.0      # Wilson LB ≥ 42% (raw WR ~52% at n=30)
min_wr_threshold_short: 42.0
min_sample_size: 30
ev_floor_r: 0.3                  # Require 0.3 R minimum
```

**Rationale**:
- Higher quality combos
- Still allows reasonable number of combos
- Good risk/reward balance

### Option 3: Aggressive (Maximum Quality)

**Use Wilson LB with high thresholds:**

```yaml
use_wilson_lb: true
min_wr_threshold_long: 45.0      # Wilson LB ≥ 45% (raw WR ~56% at n=30)
min_wr_threshold_short: 45.0
min_sample_size: 40              # Require more samples
ev_floor_r: 0.4                  # Require 0.4 R minimum
```

**Rationale**:
- Only best combos enabled
- Higher confidence
- Fewer but higher quality trades

---

## Sample Size Considerations

### Adaptive Thresholds Based on Sample Size

For better statistical rigor, consider:

| Sample Size | Recommended WR Threshold (Wilson LB) |
|-------------|--------------------------------------|
| 20-30 | 40-42% (higher due to uncertainty) |
| 30-50 | 42-44% (standard) |
| 50-100 | 40-42% (can be lower with more confidence) |
| 100+ | 38-40% (high confidence, can accept lower) |

**Implementation**: Adjust threshold dynamically based on `n`

---

## Long vs Short Thresholds

### Current: Same for Both (45%)

**Considerations**:
- Shorts often have lower WR in crypto
- Market structure favors longs
- May need different thresholds

### Recommendation:

```yaml
min_wr_threshold_long: 42.0   # Slightly lower (longs easier)
min_wr_threshold_short: 44.0  # Slightly higher (shorts harder)
```

**OR** if data shows similar performance:
```yaml
min_wr_threshold_long: 42.0
min_wr_threshold_short: 42.0  # Same if performance similar
```

---

## EV_R Floor Recommendations

### Current: 0.0 (break-even)

**Options**:

1. **Conservative**: `ev_floor_r: 0.2`
   - Ensures meaningful profitability
   - Filters out barely profitable combos

2. **Balanced**: `ev_floor_r: 0.3`
   - Good profitability threshold
   - Standard recommendation

3. **Aggressive**: `ev_floor_r: 0.4`
   - Only highly profitable combos
   - Maximum quality

**Recommendation**: Start with **0.2** and adjust based on results.

---

## Final Recommendation

### Best Configuration for Your System:

```yaml
adaptive_combos:
  enabled: true
  use_wilson_lb: true              # ✅ Enable Wilson LB (more statistically sound)
  min_wr_threshold_long: 42.0     # Wilson LB ≥ 42% (≈52% raw WR at n=30)
  min_wr_threshold_short: 44.0     # Slightly higher for shorts
  min_sample_size: 30              # Keep at 30 (good balance)
  ev_floor_r: 0.2                  # Require 0.2 R minimum (meaningful profit)
  hysteresis_pct: 2.0              # Keep at 2% (prevents flip-flop)
  lookback_days: 30                # Keep at 30 days
```

### Why This Configuration?

1. **Wilson LB**: Accounts for sample size uncertainty
2. **42% LB for Longs**: ≈52% raw WR at n=30 = 0.39 R per trade
3. **44% LB for Shorts**: Slightly higher due to crypto market structure
4. **EV_R ≥ 0.2**: Ensures meaningful profitability
5. **n ≥ 30**: Good statistical confidence

### Expected Outcomes:

- **Quality**: Only profitable combos enabled
- **Quantity**: Reasonable number of active combos
- **Confidence**: High statistical confidence
- **Risk**: Low risk of false positives

---

## Monitoring & Adjustment

### Track These Metrics:

1. **Number of enabled combos** (should be 10-50 per side)
2. **Average WR of enabled combos** (should be 45-55%)
3. **Average EV_R of enabled combos** (should be 0.3-0.6 R)
4. **Combo enable/disable frequency** (should be stable)

### Adjustment Guidelines:

- **Too few combos enabled** (< 5 per side):
  - Lower thresholds by 2-3%
  - Or reduce min_sample_size to 25

- **Too many combos enabled** (> 50 per side):
  - Raise thresholds by 2-3%
  - Or increase min_sample_size to 35-40

- **High flip-flop rate**:
  - Increase hysteresis_pct to 3.0

- **Low average EV_R** (< 0.2):
  - Increase ev_floor_r to 0.3

---

## Quick Reference

### Minimum Viable Thresholds (Break-Even):
```yaml
use_wilson_lb: true
min_wr_threshold: 35.0  # Wilson LB (≈42% raw at n=30)
ev_floor_r: 0.0
```

### Recommended Thresholds (Good Quality):
```yaml
use_wilson_lb: true
min_wr_threshold_long: 42.0
min_wr_threshold_short: 44.0
ev_floor_r: 0.2
```

### Premium Thresholds (Maximum Quality):
```yaml
use_wilson_lb: true
min_wr_threshold_long: 45.0
min_wr_threshold_short: 47.0
ev_floor_r: 0.4
min_sample_size: 40
```

---

## Summary

**Recommended Settings**:
- ✅ **Enable Wilson LB** (`use_wilson_lb: true`)
- ✅ **42% LB for Longs** (≈52% raw WR)
- ✅ **44% LB for Shorts** (≈54% raw WR)
- ✅ **EV_R ≥ 0.2** (meaningful profit)
- ✅ **n ≥ 30** (good confidence)

This provides a **good balance** of quality, quantity, and statistical confidence.

