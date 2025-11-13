# Trading Strategy Implementation Analysis

## Overview

The AutoTrading Bot implements three primary trading strategies operating on a 15-minute timeframe:

1. **Trend Pullback Strategy** - Currently DISABLED (exec.enabled: false)
2. **Mean Reversion Strategy** - ENABLED for execution
3. **Scalping Strategy** - ENABLED for execution (3m stream)

All strategies use phantom tracking for ML learning and rule-mode quality scoring (Qscore).

---

## 1. TREND PULLBACK STRATEGY

### Architecture
- **File**: `strategy_pullback_ml_learning.py`
- **ML Scorer**: `ml_scorer_trend.py`
- **Phantom Tracker**: `phantom_trade_tracker.py`
- **Quality Adapter**: `ml_qscore_trend_adapter.py`
- **Status**: Currently DISABLED (`trend.exec.enabled: false`)

### Entry Conditions: Zone-Based Breakout → HL/LH → 2-Candle Confirmation

#### Phase 1: Support/Resistance Detection
```
1. Identify Recent Pivot Points (left=3, right=2)
   - Pivot High: highest close within 3 candles left, 2 right
   - Pivot Low: lowest close within 3 candles left, 2 right
   
2. Create S/R Zones (0.3% on each side = 0.6% total)
   - Resistance Zone: [level × 0.997, level × 1.003]
   - Support Zone: [level × 0.997, level × 1.003]
   
3. Optional: Multi-Timeframe S/R Enhancement
   - Validates with 1H/4H levels
   - Minimum strength: 2.8 (confluence tolerance: 0.25%)
   - Clearance requirement: 0.10 ATR beyond HTF level
```

#### Phase 2: Breakout Detection
```
State Transition: NEUTRAL → RESISTANCE_BROKEN or SUPPORT_BROKEN

LONG Setup (Resistance Breakout):
- Close > Resistance Zone Upper (closes OUTSIDE zone)
- Stores: breakout_level, breakout_time, previous_pivot_low
- Updates: breakout_high (tracks swing high)

SHORT Setup (Support Breakout):
- Close < Support Zone Lower (closes OUTSIDE zone)
- Stores: breakout_level, breakout_time, previous_pivot_high
- Updates: breakout_low (tracks swing low)
```

#### Phase 3: Pullback Formation Detection (HL/LH)
```
Higher Low (HL) Detection (for LONG):
- Pullback low must stay ABOVE resistance zone lower boundary
- Current candle close must be > pullback low
- Previous candle close must be > pullback low
- Calculates retracement: (high - min_low) / (high - breakout_level) × 100

Lower High (LH) Detection (for SHORT):
- Pullback high must stay BELOW support zone upper boundary
- Current candle close must be < pullback high
- Previous candle close must be < pullback high
- Calculates retracement: (max_high - low) / (breakout_level - low) × 100
```

#### Phase 4: 2-Candle Confirmation
```
LONG Confirmation:
- State: HL_FORMED → HL_FORMED (increment counter)
- Requirement: Close > Open (bullish candle)
- Metrics logged:
  * Body ratio: (close - open) / (high - low) × 100%
  * Range/ATR: (high - low) / ATR × 100%
  * Volume ratio: current_vol / 20-bar_avg_vol
- Trigger: confirmation_count >= 2

SHORT Confirmation:
- State: LH_FORMED → LH_FORMED (increment counter)
- Requirement: Close < Open (bearish candle)
- Same metrics as LONG
- Trigger: confirmation_count >= 2
```

### Stop Loss Calculation (Hybrid Method)

```python
# Three Options - Choose Most Conservative (most room):
Option 1: Previous Pivot Method
  LONG: sl = previous_pivot_low - (buffer × 0.3 × ATR)
  SHORT: sl = previous_pivot_high + (buffer × 0.3 × ATR)

Option 2: Breakout Method (Larger Buffer)
  LONG: sl = breakout_level - (buffer × 1.6 × ATR)
  SHORT: sl = breakout_level + (buffer × 1.6 × ATR)

Option 3: Pullback Extreme Method
  LONG: sl = pullback_extreme - (ATR × buffer)
  SHORT: sl = pullback_extreme + (ATR × buffer)

# Choose lowest SL for LONG, highest SL for SHORT
# Volatility Adjustment:
  - Normal volatility: buffer = 1.0 × sl_buf_atr (1.5)
  - High volatility (>75th %ile): buffer × 1.3
  - Extreme volatility (>90th %ile): buffer × 1.5

# Minimum Distance Guarantee:
  - At least 1% of entry price
  - Prevents micro-stops
```

### Take Profit Calculation

```python
# Fee-Adjusted Risk/Reward
fee_adjustment = 1.00165  # Compensates for 0.165% total costs
  - Bybit entry (market): 0.06%
  - Bybit exit (limit): 0.055%
  - Slippage estimate: 0.05%

LONG: tp = entry + ((entry - sl) × RR × fee_adjustment)
SHORT: tp = entry - ((sl - entry) × RR × fee_adjustment)

# RR = 2.5 (configured)
```

### ML Features Calculated (22 total)

```
Technical Indicators:
- trend_strength: 20-bar linear regression slope
- higher_tf_alignment: MA50 vs MA20 vs close alignment
- ema_distance_ratio: (close - EMA200) / EMA200
- rsi: 14-period RSI
- bb_position: (close - BB_lower) / BB_width

Volume Features:
- volume_ratio: current_vol / 20-bar_avg
- volume_trend: 5-bar_avg_vol / 20-bar_avg_vol
- breakout_volume: same as volume_ratio
- volume_percentile: % of 100-bar history

ATR Features:
- atr_percentile: % of 100-bar ATR history
- candle_range_atr: (high - low) / ATR
- atr_stop_distance: 1.5 (default)

Structure Features:
- support_resistance_strength: # of zone touches (0-5)
- pullback_depth: retracement %
- confirmation_candle_strength: body% × direction alignment

Risk/Reward:
- risk_reward_ratio: potential_reward / potential_risk
- candle_body_ratio: (close - open) / range
- upper_wick_ratio: upper_wick / range
- lower_wick_ratio: lower_wick / range
- volume_ma_ratio: 5-bar / 20-bar volume

Cluster/Price Tier:
- symbol_cluster: 0-3 (volatility groups)
- cluster_volatility_norm: normalized within cluster
- cluster_volume_norm: normalized within cluster
- btc_correlation_bucket: 0-3 (BTC correlation)
- price_tier: 1-5 (micro to mega)
- near_major_resistance: binary
- near_major_support: binary
- mtf_level_strength: 0-10
```

### Configuration & Gates

**Execution Gates** (apply only to execution; phantoms free):

```yaml
sr (Support/Resistance Gate):
  enabled: true
  min_strength: 2.8         # S/R touch count minimum
  confluence_tolerance_pct: 0.0025  # 0.25%
  min_break_clear_atr: 0.10

btc_gate (BTC Trend Gate):
  enabled: true
  min_trend_strength_15m: 60
  min_trend_strength_60m: 55
  allowed_volatility: [low, normal]

htf_gate (1H/4H Gate):
  enabled: true
  min_trend_strength_1h: 60.0
  min_trend_strength_4h: 55.0
  ema_alignment: true
  adx_min_1h: 20.0
  structure_confluence: true  # HH/HL or LL/LH

regime:
  enabled: true
  min_conf: 0.60
  allowed_vol: [low, normal]
  phantom_on_filter_fail: true  # record phantom if gate fails
```

**Rule-Mode Qscore** (Quality-based execution):

```yaml
execute_q_min: 78      # A-tier: execute when Qscore ≥ this
phantom_q_min: 30      # B-tier: phantom when 30-78
low_weight_below: 65   # C-tier: low-quality phantoms <65

weights (auto-normalized):
  sr: 0.25             # HTF S/R quality
  htf: 0.30            # 1H/4H alignment
  bos: 0.15            # BOS confirmations/body
  micro: 0.10          # 3m micro alignment
  risk: 0.10           # risk geometry
  div: 0.10            # divergence (optional)
```

**Phantom Settings**:
```yaml
min_ml: 0              # No ML filter for phantoms (record all)
timeout_hours: 168    # Auto-close after 1 week
enable_virtual_snapshots: true  # variants at threshold ± delta
```

### State Machine

```
NEUTRAL
  ├─ Price > Resistance Zone Upper → RESISTANCE_BROKEN
  └─ Price < Support Zone Lower → SUPPORT_BROKEN

RESISTANCE_BROKEN
  ├─ HL forms (low stays above zone) → HL_FORMED
  └─ Timeout (4 hours) → NEUTRAL

SUPPORT_BROKEN
  ├─ LH forms (high stays below zone) → LH_FORMED
  └─ Timeout (4 hours) → NEUTRAL

HL_FORMED / LH_FORMED
  ├─ 2 confirmations met → SIGNAL_SENT
  └─ Timeout (4 hours) → NEUTRAL

SIGNAL_SENT
  ├─ Timeout (4 hours) → NEUTRAL
  └─ Position closes → reset to NEUTRAL
```

---

## 2. MEAN REVERSION STRATEGY

### Architecture
- **File**: `strategy_mean_reversion.py`
- **ML Scorer**: `ml_scorer_mean_reversion.py`
- **Phantom Tracker**: `mr_phantom_tracker.py`
- **Quality Adapter**: `ml_qscore_range_adapter.py`
- **Status**: ENABLED (`mr.exec.enabled: true`)

### Entry Conditions: Range Identification → Edge Touch → Reversal Confirmation

#### Phase 1: Range Detection

```python
# Identify Range Boundaries using Pivot Points (left=5, right=5)
upper_range = avg(last_2_pivot_highs)
lower_range = avg(last_2_pivot_lows)

# Range Width Validation (STRICT)
range_width = (upper_range - lower_range) / lower_range
min_range_width: 2.5%    # minimum
max_range_width: 6.0%    # maximum

# Reject if:
- range_width < 2.5%  → Range too narrow
- range_width > 6.0%  → Range too wide
```

#### Phase 2: Range Confidence Calculation

```python
# Heuristic scoring (0.0 - 1.0)
recent_window = 30 candles

1. Touch Score (0.5 weight):
   - Count upper touches within 0.5% of upper_range
   - Count lower touches within 0.5% of lower_range
   - touch_score = min(1.0, total_touches / 6.0)

2. In-Range Ratio (0.3 weight):
   - % of closes within [lower_range, upper_range]
   - in_range_ratio = closes_in_range / window_size

3. Width Component (0.2 weight):
   - Prefers tighter ranges up to 8%
   - width_component = min(1.0, 0.08 / range_width)

range_confidence = (touch_score × 0.5) + (in_range_ratio × 0.3) + (width_component × 0.2)
Range: 0.0 - 1.0
```

#### Phase 3: Entry Signal - LONG (Bounce from Support)

```python
# Trigger Condition:
- abs(current_price - lower_range) < (current_atr × 0.5)  [STRICT 0.5 ATR]
- Confirmation: Close > Open (bullish candle required)

entry = current_price

# Hybrid SL Method (whichever gives most room):
Option 1: Range Boundary Method
  sl = lower_range - (buffer × 0.3 × ATR)

Option 2: Entry-Based ATR Method
  sl = entry - (buffer × 1.0 × ATR)

Option 3: Support Level Method (Largest Buffer)
  sl = lower_range - (ATR × buffer)

# Volatility Adjustment:
- Percentile > 80%: buffer × 1.4
- Percentile > 60%: buffer × 1.2
- Percentile < 20%: buffer × 0.8
- Default: buffer × 1.0

# Minimum Distance: 1% of entry
if abs(entry - sl) < entry × 0.01:
  sl = entry - (entry × 0.01)

# TP Calculation (Fee-Adjusted):
fee_adjustment = 1.00165
tp = entry + ((entry - sl) × RR × fee_adjustment)
RR = 2.5
```

#### Phase 4: Entry Signal - SHORT (Rejection from Resistance)

```python
# Trigger Condition:
- abs(current_price - upper_range) < (current_atr × 0.5)  [STRICT 0.5 ATR]
- Confirmation: Close < Open (bearish candle required)

entry = current_price

# Hybrid SL Method (whichever gives most room):
Option 1: Range Boundary Method
  sl = upper_range + (buffer × 0.3 × ATR)

Option 2: Entry-Based ATR Method
  sl = entry + (buffer × 1.0 × ATR)

Option 3: Resistance Level Method (Largest Buffer)
  sl = upper_range + (ATR × buffer)

# Same volatility adjustment and minimum distance as LONG

# TP Calculation:
tp = entry - ((sl - entry) × RR × fee_adjustment)
```

### ML Features Calculated

```python
Estimated 35+ features including:
- Range quality metrics
- Volatility percentile
- Volume spike detection
- Price location within range
- Trend alignment
- Time-of-day effects
- Cluster-based features
- Multi-timeframe alignment
```

### Configuration

```yaml
mr.exec:
  enabled: true
  slippage_recalc:
    enabled: true
    min_pct: 0.001          # 0.1% threshold
    pivot_buffer_atr: 0.05  # SL buffer beyond pivot
  high_ml_force: 65         # Force exec at ML score 65

mr.explore (Phantom-Only Relaxation):
  enabled: true
  rc_min: 0.50              # was 0.70 (relaxed)
  touches_min: 4            # was 6 (relaxed)
  dist_mid_atr_min: 0.40    # was 0.60 (relaxed)
  rev_candle_atr_min: 0.80  # was 1.0 (relaxed)
  allow_volatility_high: true
  timeout_hours: 36
  enable_virtual_snapshots: true

mr.regime:
  enabled: true
  min_conf: 0.60
  min_persist: 0.60
  phantom_on_filter_fail: true
  execute_only: true        # Regime gates only for execution

mr.promotion (Auto-Execution):
  enabled: false
  promote_wr: 3.0           # Activate at 3%+ win rate
  demote_wr: 0.0            # Deactivate below 0%
  min_recent: 20
  min_total_trades: 50
  block_extreme_vol: true
  daily_exec_cap: 20
```

### Cooldown & State Management

```python
# Prevents multiple signals in quick succession
cooldown_duration = min_candles_between_signals × 15 minutes
# Default behavior: block if time_since_last < cooldown

# State Tracking:
- last_signal_candle_time: datetime of last signal
- State reset after 4 hours automatically
```

---

## 3. SCALPING STRATEGY

### Architecture
- **File**: `strategy_scalp.py`
- **ML Scorer**: `ml_scorer_scalp.py`
- **Phantom Tracker**: `scalp_phantom_tracker.py`
- **Quality Adapter**: `ml_qscore_scalp_adapter.py`
- **Stream**: 3-minute bars (when `stream_3m_only: true`)
- **Status**: ENABLED (`scalp.enabled: true`, `exec.enabled: true`)

### Design Philosophy

**EVWAP Pullback + Trend Continuation**
- EVWAP: Exponential VWAP (faster adaptation than rolling VWAP)
- Multi-anchor means: VWAP OR EMA-band OR BB-midline
- EMA alignment: Fast (8) > Slow (21) for continuation bias
- Momentum burst: High volume + strong wick rejection + tight BB

### Entry Conditions: Trend Alignment → Multi-Anchor Proximity → Wick/Body Rejection

#### Phase 1: Trend Confirmation (EMA Stack)

```python
# LONG Continuation:
ema_aligned_up = (close > ema_fast > ema_slow)

# SHORT Continuation:
ema_aligned_dn = (close < ema_fast < ema_slow)

# Both EMAs missing → No signal
```

#### Phase 2: Multi-Anchor Mean Proximity (K-of-N Gating)

```python
# 3 Possible Anchors:
1. EVWAP (Exponential VWAP):
   - VWAP = EMA(price × volume) / EMA(volume)
   - Distance = |close - EVWAP| / ATR
   - Threshold: 0.70 ATR (signal), 0.50 ATR max for tight bands

2. EMA Slow-Band:
   - Distance = |close - EMA21| / ATR
   - Threshold: 1.5 ATR

3. Bollinger Band Midline (20 SMA):
   - Distance = |close - SMA20| / ATR
   - Threshold: 1.5 ATR

# Signal Generation (EXECUTION):
- VWAP only (means_enabled=false) → tight VWAP proximity required
- Multi-anchor OR logic (means_enabled=true) → VWAP OR (EMA OR BB)

# Phantom Paths (K-of-N):
K-of-N gates (when kofn_enabled=true):
  - Requires K of [means_ok, vol_ok, wick_body, bbw_ok]
  - Default K=2 (accept if 2+ of 4 pass)
  
ATR Fallback (atr_fallback_enabled=true):
  - Accept pullback size 0.5-1.5 ATR from recent swing
  - Independent of mean proximity
  
Near-Miss (near_miss_enabled=true):
  - Accept if exactly 1 gate fails (learning edge cases)
```

#### Phase 3: Volume Spike Detection

```python
vol_ratio = current_volume / 20-bar_average_volume

# Gate: vol_ratio >= vol_ratio_min
Signal threshold: 0.80 (relaxed)
Execution gate: 1.20 (strict)

# High BB Width consideration (>85th %ile):
  - Allows slightly lower volume requirement
  - Wide bands = natural volume expansion expected
```

#### Phase 4: Bollinger Band Width Check

```python
bbw = 20-bar rolling std dev / close
bbw_percentile = (bbw <= current_bbw).mean() × 100

# Gates:
Signal (relax): 55th percentile (0.55)
Execution gate: 60th percentile (0.60)
Allow volume spillover when > 85th percentile

# Rationale:
- Narrow bands (low percentile) = less momentum expected
- Wide bands (high percentile) = volatile, volume-driven move
```

#### Phase 5: Wick & Body Rejection (Pullback Momentum)

```python
# Candle Structure Analysis:
upper_wick = max(open, close) - high
lower_wick = low - min(open, close)
body = abs(close - open)
range = high - low
upper_wick_ratio = upper_wick / range
lower_wick_ratio = lower_wick / range
body_ratio = body / range

# LONG Entry (Bullish Rejection):
- Requires: lower_wick >= max(wick_ratio_min, upper_wick + wick_delta_min)
  * Lower wick must be significant
  * Lower wick must dominate upper wick by delta_min
- And: body_ratio >= body_ratio_min (0.30)
- And: close > open (bullish body)
- Interpretation: Strong rejection from pullback, bullish continuation

# SHORT Entry (Bearish Rejection):
- Requires: upper_wick >= max(wick_ratio_min, lower_wick + wick_delta_min)
  * Upper wick must be significant
  * Upper wick must dominate lower wick by delta_min
- And: body_ratio >= body_ratio_min (0.30)
- And: close < open (bearish body)
- Interpretation: Strong rejection from pullback, bearish continuation

# Thresholds (from config):
wick_ratio_min: 0.40       # signal
wick_delta_min: 0.10       # dominant wick delta
body_ratio_min: 0.30       # meaningful body requirement
```

#### Phase 6: Optional ORB (Open Range Breakout)

```python
if orb_enabled:
  recent_high = max(high[-20:])
  recent_low = min(low[-20:])
  
  LONG: Requires close > recent_high (breakout high)
  SHORT: Requires close < recent_low (breakout low)
  
# Current config: orb_enabled = false (disabled)
```

### Stop Loss & Take Profit

```python
# Both directions similar logic:

LONG:
sl = min(cur_vwap, ema_slow) - buf_mult × ATR
  buf_mult:
  - 0.8 if BBW < 70th %ile
  - 1.0 if 70-85th %ile
  - 1.2 if > 85th %ile (high volatility)

# Enforce minimum distance:
min_dist = max(
  entry × min_r_pct,        # 0.5% of price
  0.6 × ATR,                # 0.6 ATR
  entry × 0.002             # 0.2% safety floor
)

if entry - sl < min_dist:
  sl = entry - min_dist

# Take Profit (RR = 2.1):
tp = entry + 2.1 × (entry - sl)

SHORT:
sl = max(cur_vwap, ema_slow) + buf_mult × ATR
# Same buffer logic
tp = entry - 2.1 × (sl - entry)
```

### Signal Acceptance Paths

```
EXECUTION PATHS (Qscore >= 78):
1. Main Path: All gates pass
   - ema_aligned_up/dn ✓
   - bbw_ok ✓
   - vol_ok ✓
   - wick_body_long/short ✓
   - means_ok ✓
   - orb_ok ✓

PHANTOM PATHS (Qscore 15-78):
2. K-of-N Path: Slope mandatory, K of 4 others pass
   gates = [means_ok, vol_ok, wick_body, bbw_ok]
   if sum(gates) >= K: accept

3. ATR Fallback: EMA aligned + wick_body + size in [0.5-1.5] ATR

4. Near-Miss: EMA aligned + exactly 1 gate fails (informational)
```

### High-WR Multi-Feature Bypass (Execution Fast-Track)

```yaml
# Data-driven from 30d Advanced Combos analysis
# 6 winning combinations with 49-57% win rate (N=538 total)

Combo 1 (57.5% WR, N=73):
  Fast Slope: 0.01-0.03%/bar
  Slow Slope: 0.015%+/bar
  ATR%: <0.5%
  BBW%: <1.2%
  VWAP Distance: 0.6-1.0 ATR
  → Risk: 1%

Combo 2 (56.6% WR, N=76):
  Fast Slope: 0.03%+/bar
  Slow Slope: 0.00-0.015%/bar
  ATR%: <0.5%
  BBW%: <1.2%
  VWAP Distance: 1.0+ ATR
  → Risk: 1%

Combo 3 (54.0% WR, N=87):
  Fast Slope: 0.01-0.03%/bar
  Slow Slope: -0.03-0.00%/bar
  ATR%: <0.5%
  BBW%: <1.2%
  VWAP Distance: 1.0+ ATR
  → Risk: 1%

Combo 4 (51.9% WR, N=104):
  Fast Slope: 0.03%+/bar
  Slow Slope: 0.015%+/bar
  ATR%: <0.5%
  BBW%: <1.2%
  VWAP Distance: 0.6-1.0 ATR
  → Risk: 1%

Combo 5 (50.7% WR, N=146):
  Fast Slope: 0.01-0.03%/bar
  Slow Slope: 0.015%+/bar
  ATR%: <0.5%
  BBW%: <1.2%
  VWAP Distance: <0.6 ATR
  → Risk: 1%

Combo 6 (49.0% WR, N=102):
  Fast Slope: 0.03%+/bar
  Slow Slope: 0.015%+/bar
  ATR%: <0.5%
  BBW%: <1.2%
  VWAP Distance: <0.6 ATR
  → Risk: 1%

# Bypass Behavior:
- Matches ANY combo → immediate execution
- Skips ALL gates (Qscore, ML, volume, etc)
- Overrides phantoms (can cancel active phantom)
- Fixed 1% risk for all combos
```

### Execution Gates

```yaml
scalp.hard_gates:
  apply_to_exec: true
  apply_to_phantoms: false  # Phantoms track all

  # HTF Gate (DISABLED):
  htf_enabled: false
  htf_min_ts15: 70.0

  # Volume Gate (ENABLED):
  vol_enabled: true
  vol_ratio_min_3m: 1.20    # 1.2x 20-bar avg

  # Body Gate (DISABLED - handled in signal generation):
  body_enabled: false
  body_ratio_min_3m: 0.30

  # Wick Alignment Gate (ENABLED):
  wick_enabled: true
  wick_delta_min: 0.10

  # Volatility Regime Gate (ENABLED):
  regime_enabled: true
  allowed_regimes: ['normal', 'high']

  # EMA Slope Gate (ENABLED - INVERTED for Mean Reversion):
  # Data: F<0 × S<0 = 52.2% WR vs F>0.05 = 0-6% WR
  slope_enabled: true
  slope_fast_min_pb: 0.03%        # 0.03% per bar (absolute)
  slope_slow_min_pb: 0.015%       # 0.015% per bar (absolute)

  # Bollinger Width Percentile (ENABLED):
  bbw_exec_enabled: true
  bbw_min_pct: 0.60               # 60th %ile
  bbw_max_pct: 0.90               # 90th %ile

  # VWAP Execution Gate (DISABLED):
  vwap_exec_enabled: false
  vwap_dist_atr_min: 0.40
  vwap_dist_atr_max: 0.80
```

### Off-Hours Auto-Block

```yaml
scalp.exec.off_hours:
  enabled: true
  mode: auto                    # auto | fixed | hybrid

  Auto-Detection Thresholds:
  - atr_pct_max: 0.35           # ATR as % of price
  - bb_width_pct_max: 0.012     # 1.2% BB width
  - vol_ratio_max: 1.05         # volume spike <=1.05x
  - weekend_tighten_pct: 0.15   # Tighten thresholds 15% on weekends

  # When all 3 metrics below thresholds → off-hours block
  # Phantoms continue regardless
```

### Phantom Configuration

```yaml
scalp.explore (Phantom-Only Relaxation):
  relax_enabled: true
  vwap_dist_atr_max: 1.20       # increased from 1.0 (deeper pulls)
  min_bb_width_pct: 0.55        # was 0.60 (accept narrower)
  vol_ratio_min: 1.05           # was 1.10 (lighter volume)
  wick_ratio_min: 0.20          # reduced from 0.25 (smaller wicks)
  timeout_hours: 8              # auto-close after 8 hours

scalp.exec (Execution Config):
  enabled: true
  risk_percent: 1.0             # 1% risk per trade
  daily_cap: 0                  # unlimited executes per day

phantom settings:
  hourly_symbol_budget:
    scalp: 12 phantoms/symbol/hour
  
  daily_caps:
    scalp:
      none: 600         # unlimited volume symbols
      cluster3: 120     # high-volatility group
      offhours: 80      # off-hours phantoms
  
  dedup_enabled: false  # No Redis dedup for phantoms
  dedup_hours: 3        # Would dedup within 3 hours if enabled
```

### ML Scoring for Scalp

```python
Features: 12 total
- atr_pct: ATR as % of price
- bb_width_pct: Bollinger width percentile
- impulse_ratio: volume × body ratio
- ema_slope_fast: fast EMA % per bar
- ema_slope_slow: slow EMA % per bar
- volume_ratio: current / 20-bar average
- upper_wick_ratio: upper wick / range
- lower_wick_ratio: lower wick / range
- vwap_dist_atr: distance to VWAP in ATR
- session: asian | european | us | off_hours
- symbol_cluster: 0-3 (volatility)
- volatility_regime: low | normal | high | extreme

Scoring:
- MIN_TRADES_FOR_ML: 50
- RETRAIN_INTERVAL: 50 trades
- INITIAL_THRESHOLD: 75 (execution score)
- PHANTOM_BOOTSTRAP: unlimited (phantom learning only)

Threshold Logic:
- If ML ready (50+ trades) and score >= 75 → execute
- If score 15-75 → phantom record (learning)
- If ML not ready → heuristic scoring (50-base)
```

---

## 4. STRATEGY CONFIGURATION & FEATURE GATES

### Master Switches

```yaml
modes:
  trend_only: false             # Run only Trend (MR + Scalp muted)
  disable_mr: true              # Hard-disable legacy MR paths
  disable_scalp_fallback: true  # Mute legacy Scalp fallback

strategy_independence:
  enabled: true                 # Run independently
  concurrency: single_per_symbol
  confirmation_timeout_bars: 6  # Max bars wait for HL/LH
  pullback_timeout_bars: 12     # Max bars wait before forget setup

# Stream Configuration:
scalp.stream_3m_only: true      # Use 3m stream for scalps
```

### Phantom Flow Control (Adaptive Daily Targets)

```yaml
phantom_flow:
  enabled: true
  daily_target:
    trend: 200 phantoms/day
    mr: 200 phantoms/day
    scalp: 200 phantoms/day
  
  smoothing_hours: 2            # Rolling window
  
  relax_limits (when below target):
    trend:
      slope: reduce by 1.5 pct-points
      ema_stack: reduce by 15 points
      breakout: reduce by 0.08 ATR
    mr:
      rc: reduce by 0.30
      touches: reduce by 3
      dist_mid_atr: reduce by 0.40
      rev_atr: reduce by 0.40
    scalp:
      vwap_dist_atr: increase by 0.60
      bb_width_pct: reduce by 0.30
      vol_ratio: reduce by 0.40
      wick: reduce by 0.15
  
  wr_guard:
    enabled: true
    window: 40 recent phantoms
    min_wr: 40%                 # threshold
    cap: 0.50                   # max relax when triggered
```

### Fear & Greed Index (Optional Sentiment Filter)

```yaml
fear_greed:
  enabled: false                # DISABLED
  min_value: 20                 # block if index < 20 (extreme fear)
  max_value: 80                 # block if index > 80 (extreme greed)
  apply_to_scalp: true
  apply_to_trend: false
  apply_to_mr: false
```

---

## 5. SIGNAL DATA STRUCTURES

### Unified Signal Object

```python
@dataclass
class Signal:
    side: str              # "long" | "short"
    entry: float           # Entry price
    sl: float              # Stop loss price
    tp: float              # Take profit price
    reason: str            # Human-readable reason
    meta: dict             # Strategy-specific metadata

# Meta Fields (Strategy-Dependent):

# Trend Meta:
{
  "breakout_level": float,
  "pullback_low/high": float,
  "fib_retracement": str (percentage),
  "ml_features": dict (22 features),
  "learning_mode": bool,
  "strategy_name": "trend_pullback"
}

# Mean Reversion Meta:
{
  "range_upper": float,
  "range_lower": float,
  "range_confidence": float,
  "mr_features": dict,
  "strategy_name": "mean_reversion"
}

# Scalp Meta:
{
  "vwap": float,
  "atr": float,
  "bbw_pct": float,
  "vol_ratio": float,
  "dist_vwap_atr": float,
  "means": dict (VWAP/EMA/BB distances),
  "acceptance_path": str (means_or | kofn | atr_fallback | near_miss),
  "strategy_name": "scalp"
}
```

### Phantom Trade Objects

Each strategy maintains phantom trackers:

```python
# Trend Phantom:
{
  symbol: str,
  side: str,
  entry_price: float,
  stop_loss: float,
  take_profit: float,
  signal_time: datetime,
  ml_score: float,
  was_executed: bool,
  features: dict (22 ML features),
  outcome: str (win | loss | timeout),
  realized_rr: float,
  ... (multiple exit tracking fields)
}
```

---

## 6. SIGNAL FLOW & EXECUTION ARBITRATION

### 1. Signal Generation (Parallel)

```
15m Kline Update
│
├─ Trend Analysis (if trend.exec.enabled)
│  ├─ Detect S/R breakout
│  ├─ Detect HL/LH
│  ├─ Confirm 2 candles
│  └─ Return Signal (or null)
│
├─ Mean Reversion Analysis (if mr.exec.enabled)
│  ├─ Identify range
│  ├─ Check edge touch
│  ├─ Confirm reversal
│  └─ Return Signal (or null)
│
└─ (Scalp on 3m stream - parallel handling)
```

### 2. ML Scoring & Phantom Recording

```
Signal Generated
│
├─ PHANTOM TRACKING
│  ├─ Record signal immediately (no score gate)
│  ├─ Calculate all 22/35+ ML features
│  ├─ Store in phantom tracker
│  ├─ Monitor outcome (TP/SL hit)
│  └─ Label as win/loss/timeout
│
├─ ML SCORING (if ml_min_score gate enabled)
│  ├─ If <200 trades: take signal (learning mode)
│  ├─ If >=200 trades: score signal
│  │  ├─ If score < threshold: SKIP (record as rejected)
│  │  └─ If score >= threshold: PROCEED to gates
│  └─ If ML error: FAIL_OPEN (execute)
│
└─ RULE-MODE SCORING (Qscore)
   ├─ Calculate quality components
   ├─ Generate weighted Qscore
   ├─ Map to execution tier (A/B/C)
   └─ Store for learning/calibration
```

### 3. Execution Gate Checks (Execution Only)

```
Signal Passed ML
│
├─ Regime Gate
│  └─ If regime=extreme_vol: BLOCK (phantom_on_fail=true)
│
├─ S/R Gate (Trend only)
│  └─ If sr_strength < min: BLOCK
│
├─ HTF Gate (Strategy-Specific)
│  ├─ Trend: check 1H/4H alignment + ADX
│  ├─ Scalp: check 15m trend_strength
│  └─ If fails: BLOCK or SOFT (penalty to Qscore)
│
├─ BTC Gate (Trend only)
│  └─ Check BTC 15m/60m trend_strength
│
├─ EMA Slope Gate (Scalp only)
│  └─ Check fast/slow slope alignment
│
├─ Volume Gate
│  └─ Check volume spike >= threshold
│
└─ Qscore Decision
   ├─ If Qscore >= execute_q_min: EXECUTE
   ├─ If Qscore < phantom_q_min: PHANTOM only
   └─ If execute_q_min > Qscore >= phantom_q_min: PHANTOM
```

### 4. Order Execution

```
Order Placement
│
├─ Calculate Position Size
│  ├─ Risk Amount: risk_percent × account_balance
│  ├─ Quantity: risk_amount / (entry - sl)
│  ├─ Round to tick/step precision
│  └─ Check minimum quantity
│
├─ Place Market Entry Order
│  ├─ Market order (fast execution)
│  ├─ Retry logic (exponential backoff)
│  └─ Validate filled price
│
├─ Place TP/SL Orders (Partial Mode)
│  ├─ Place both simultaneously
│  ├─ Partial TP at TP1 (if scale-out enabled)
│  │  ├─ TP1_R: 1.6 (50% position)
│  │  ├─ Move SL to breakeven after TP1
│  │  └─ Runner target TP2_R: 3.0
│  └─ Trailing stop optional
│
└─ Monitor Position
   ├─ Track fills
   ├─ Update P&L
   └─ Merge with phantom mirrors
```

---

## 7. KEY DIFFERENCES & EDGE CASES

### Trend vs Mean Reversion

| Feature | Trend | Mean Reversion |
|---------|-------|----------------|
| Regime | Trending markets | Ranging markets |
| Entry Type | Breakout + pullback | Range edge touch |
| Direction | With trend | Against move (reversal) |
| S/R Usage | Confirms breakout | Defines range edges |
| Confirmation | 2 candles (HL/LH) | 1 candle (reversal close) |
| Typical Win % | 55-65% (data-driven) | 45-55% |
| Setup Frequency | Lower (fewer breakouts) | Higher (range touches) |
| Risk/Reward | Fixed 2.5:1 | Fixed 2.5:1 |

### Scalp-Specific Considerations

```
1. Speed: 3-minute timeframe
   - Faster entry/exit detection
   - Tighter stops (0.5-1.5 ATR pullbacks)
   - Higher win% but smaller R values (2.1:1)

2. EMA Inversion:
   - Data shows F<0 × S<0 = 52% WR
   - Negative slopes = counter-trend dips
   - Mean reversion within trend

3. Multi-Anchor Logic:
   - OR-gate on means allows flexibility
   - K-of-N phantoms for learning edge cases
   - Fallback paths for near-misses

4. High-WR Bypass:
   - Data-driven feature combinations
   - Immediate execution (no gates)
   - Fixed 1% risk

5. Off-Hours Dampening:
   - Auto-blocks in low volume periods
   - Protects against fake-outs
   - Phantoms unaffected
```

### Phantom Learning Edge Cases

```
1. Virtual Snapshots:
   - Create variants at threshold ± delta
   - Calibrate around decision boundary
   - Improve model discrimination

2. Near-Miss Phantoms:
   - Accept when exactly 1 gate fails
   - Informational learning (why did it fail?)
   - Helps identify false-negative gates

3. Phantom-Only Exploration:
   - Relaxed filters on phantoms
   - Trend slope relax: 1.5 pct-points
   - Scalp VWAP relax: 0.60 ATR increase
   - MR range confidence relax

4. Timeout Auto-Close:
   - Trend: 168 hours (1 week)
   - MR: 36 hours (1.5 days)
   - Scalp: 8 hours
   - Prevents stale phantom accumulation
```

---

## 8. TESTING COMMANDS

```bash
# Test ML integration
python test_ml_integration.py

# Test phantom collection
python test_phantom_collection.py

# Test enhanced strategy components
python test_enhanced_strategy.py

# Check all symbols connectivity
python check_all_symbols.py

# Verify symbol specifications
python fetch_symbol_specs.py

# Run main bot with ML scoring
python live_bot.py

# Custom risk amount
RISK_USD=10 python live_bot.py

# Percentage risk
RISK_PERCENT=3 python live_bot.py
```

---

## 9. PERFORMANCE MONITORING

### Key Metrics

```
Per-Strategy:
- Phantom Count: (active, completed, timeout)
- Win Rate: (executed trades only)
- Avg Win/Loss Ratio
- Execution Frequency
- Avg R per Trade
- Daily P&L

ML Metrics:
- Model Status: trained | learning
- Threshold: current execution threshold
- Recent Trade Count
- Feature Importance
- Calibration Score
- Decay Rate

Gate Effectiveness:
- Qscore Distribution
- Gate Pass Rate (by gate)
- Execution Rate (by gate)
```

### Telegram Commands

```
/status - View all positions
/stats - Win rate and performance
/ml_stats - ML system metrics
/ml_rankings - Symbol performance
/phantom_stats - Phantom tracking stats
/balance - Account balance
/symbols - Active symbols list
/help - All commands
```

