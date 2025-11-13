# ML Scoring and Phantom Tracking System - Deep Dive Analysis

## Executive Summary

This is a sophisticated multi-strategy ML learning system that tracks ALL trading signals (executed and phantom/rejected) to continuously improve prediction accuracy. It employs ensemble learning, phantom trade simulation, context-based thresholds, and adaptive evolution.

---

## 1. ML SCORER ARCHITECTURE

### 1.1 Trend ML Scorer (`ml_scorer_trend.py`)

**Purpose**: Primary ML scorer for Trend Pullback strategy with immediate learning capability.

**Core Design**:
- **Ensemble approach**: RF + GB + optional NN
- **Redis-backed state**: Persistent model storage
- **Immediate activation**: 70% confidence threshold for day-1 trading
- **Auto-activation**: NN head activates at 300+ trades automatically

**Feature Vector (34+ features)**:
```
Order of feature extraction:
1. atr_pct                    # ATR as percentage of price
2. break_dist_atr             # Distance beyond broken S/R level
3. retrace_depth_atr          # Pullback depth before HL/LH signal
4. confirm_candles            # 2-candle confirmation presence
5. ema_stack_score            # Trend filter (optional)
6. range_expansion            # Day range vs median
7. div_ok                     # Divergence present
8. div_score                  # Divergence quality score
9. div_rsi_delta              # RSI divergence magnitude
10. div_tsi_delta             # TSI divergence magnitude
11. protective_pivot_present  # Boolean flag
12. tp1_hit                   # Lifecycle: TP1 reached
13. be_moved                  # Lifecycle: Break-even moved
14. runner_hit                # Lifecycle: Runner target hit
15. time_to_tp1_sec           # Time to first TP (seconds)
16. time_to_exit_sec          # Time to full exit (seconds)
17. session                   # Trading session (Asian/European/US/Off)
18. symbol_cluster            # Symbol behavior cluster
19. volatility_regime         # Market regime
20-23. Composite HTF metrics  # ts15, ts60, rc15, rc60
24. qscore                    # Rule-mode quality score
... + additional technical features
```

**Model Training** (`_retrain()`):
- Minimum threshold: 30 trades to activate ML
- Retrain interval: Every 50 trades
- Sample weighting:
  - Executed trades: 1.0
  - Phantom trades: 0.8
- Isotonic regression for probability calibration
- All data retained (no caps)

**Scoring Pipeline** (`score_signal()`):
```
1. Check if ML ready (is_ml_ready flag)
2. If ready: ensemble prediction
   - RF.predict_proba() for RF prediction
   - GB.predict_proba() for GB prediction
   - NN.predict_proba() for NN (if available)
   - Average predictions
   - Apply isotonic calibrator if available
3. If not ready: fallback heuristic
   - Slope-based scoring (+10 if |slope| >= 5)
   - EMA stack scoring (+10 if score >= 50)
   - Breakout distance (+10 if >= 0.2 ATR)
   - Range expansion (+10 if >= 1.2x)
   - ATR penalty (-5 if >= 0.5%)
4. Return score 0-100
```

**State Persistence**:
- Redis namespace: `ml:trend:`
- Keys: `completed_trades`, `last_train_count`, `threshold`, `model`, `scaler`, `calibrator`, `ev_buckets`, `nn_enabled`
- Legacy compatibility: Also saves to `tml:*` keys

**Key Methods**:
- `record_outcome()`: Track trade result + increment completed count
- `get_stats()`: Return status for dashboard
- `get_ev_threshold()`: EV-based adaptive threshold per session|volatility bucket
- `get_patterns()`: Feature importance + time/session patterns

---

### 1.2 Immediate ML Scorer (Legacy, `ml_signal_scorer_immediate.py`)

**Status**: Kept for backward compatibility; newer Trend scorer is preferred.

**Architecture**:
- RF + GB + NN ensemble
- 13 simplified features: `trend_strength`, `higher_tf_alignment`, `ema_distance_ratio`, `support_resistance_strength`, etc.
- Threshold adaptation based on recent WR
- Probability calibration with power transformation

**Key difference**: Uses simplified feature set vs Trend scorer's 34+ features

---

### 1.3 Scalp ML Scorer (`ml_scorer_scalp.py`)

**Purpose**: Independent scorer for scalping strategy (Phase 0: phantom-only, can be promoted to execution).

**Design**:
- Separate Redis namespace: `ml:scalp:*`
- 12-feature vector: `atr_pct`, `bb_width_pct`, `impulse_ratio`, EMA slopes, volume, wicks, VWAP distance, session, cluster, volatility
- Unlimited phantom bootstrap: can train purely on phantom data (PHANTOM_BOOTSTRAP_MAX = 10M)
- Isotonic calibration for probability mapping

**Key Features**:
- Starts with heuristic scoring (pre-ML):
  - +10 if impulse_ratio > 1.0
  - +10 if vwap_dist_atr < 0.6
  - +5 if bb_width_pct > 0.7
  - +10 if volume_ratio > 1.3
- Auto-retrain when 50 new trades accumulated
- NN component with adaptive learning rate

---

## 2. PHANTOM TRADE TRACKING SYSTEM

### 2.1 Core Phantom Tracker (`phantom_trade_tracker.py`)

**Purpose**: Shadow-track ALL signals (executed and rejected) to enable comprehensive ML learning.

**Data Structure**:
```python
class PhantomTrade:
    symbol: str
    side: str                 # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    signal_time: datetime
    ml_score: float          # Score at time of signal
    was_executed: bool       # Real trade vs phantom
    features: Dict           # ML features snapshot
    strategy_name: str       # Identifies strategy
    phantom_id: str          # UUID for concurrent tracking
    
    # Outcome tracking (filled post-close)
    outcome: Optional[str]                # "win", "loss", "timeout"
    exit_price: Optional[float]
    pnl_percent: Optional[float]
    exit_time: Optional[datetime]
    max_favorable: Optional[float]       # Best excursion
    max_adverse: Optional[float]         # Worst excursion
    
    # Enriched labels
    one_r_hit: Optional[bool]            # Hit 1R profit target
    two_r_hit: Optional[bool]            # Hit 2R profit target
    realized_rr: Optional[float]         # Actual R:R achieved
    exit_reason: Optional[str]           # "tp", "sl", "timeout"
    
    # Lifecycle flags
    be_moved: bool                       # Break-even moved
    tp1_hit: bool                        # First TP hit
    time_to_tp1_sec: Optional[int]      # Timing metrics
    time_to_exit_sec: Optional[int]
```

**Multi-Signal Tracking**:
- **Active phantoms**: Per symbol, multiple concurrent phantoms allowed
- **Completed phantoms**: Historical record for learning
- Redis storage with JSON serialization

**Gating Logic for Non-Executed Phantoms**:
```python
# Regime gating (volatility check)
if vol == 'extreme':
    SKIP phantom  # Don't track in extreme conditions

# Micro-trend gating (EMA alignment check)
if side == 'long':
    if slope < 0.0 or ema_stack < 40.0:
        SKIP phantom  # Require uptrend alignment
if side == 'short':
    if slope > 0.0 or ema_stack < 40.0:
        SKIP phantom  # Require downtrend alignment
```

**Price Update Processing** (`update_phantom_prices()`):

1. **TP1 Detection** (if applicable):
   - Range strategy: When high/low crosses `range_mid`
   - Trend strategy: When price moves 1.6R from entry
   - Action: Move SL to break-even, set `tp1_hit=True`, notify

2. **Full Exit Detection**:
   - TP hit: Close at exact take_profit level
   - SL hit: Close at exact stop_loss level
   - Timeout: After 36 hours (Trend) or 24 hours (Scalp)

3. **Extremes Tracking**:
   - Long: max_favorable = max price seen, max_adverse = min price
   - Short: max_favorable = min price seen, max_adverse = max price

4. **Lifecycle Enrichment**:
   - R-multiple calculations: realized_rr = (exit - entry) / R
   - 1R/2R hit detection
   - Time-to-outcome calculation

**Learning Integration** (`_close_phantom()`):

After a phantom closes:
1. Update lifecycle metrics in features dict
2. Feed to strategy-specific ML scorer:
   - Trend: `ml_scorer_trend.record_outcome()`
   - Range: `ml_scorer_range.record_outcome()`
3. Check if retrain needed
4. Persist to PostgreSQL (audit trail)
5. Update rolling WR list in Redis (for WR guard)
6. Notify via callback (Telegram, etc.)

**Storage**:
- Redis active: `phantom:active` (dict of symbol -> list)
- Redis completed: `phantom:completed` (last 1000 trades, last 30 days)
- PostgreSQL: Full audit trail via `PhantomPersistence`

---

### 2.2 Scalp Phantom Tracker (`scalp_phantom_tracker.py`)

**Purpose**: Dedicated shadow tracking for scalping strategy with detailed gate analysis.

**Differences from Trend Tracker**:
- Timeout: 24 hours (vs 36 for trend)
- Gate analysis: 26+ variables tracked per phantom
- No micro-trend gating (all phantoms recorded)
- Comprehensive combo analysis (solo/pairs/triplets)

**Gate Status Computation** (`compute_gate_status()`):

Tracks 26+ variables:
```
Original 4 gates:
  - htf: ts15 >= 60.0
  - vol: volume_ratio >= 1.30
  - body: body_ratio >= 0.35 + correct direction
  - align_15m: EMA direction matches side

Body variations (5):
  - body_040, body_045, body_050, body_060: Different thresholds
  - wick_align: Rejection wick in trade direction

VWAP variations (4):
  - vwap_045, vwap_060, vwap_080, vwap_100: Distance ATR thresholds

Volume variations (3):
  - vol_110, vol_120, vol_150: Ratio thresholds

BB Width (3):
  - bb_width_60p, bb_width_70p, bb_width_80p: Percentile-based

Q-Score (4):
  - q_040, q_050, q_060, q_070: Quality score thresholds

Impulse (2):
  - impulse_040, impulse_060: Momentum thresholds

Micro (1):
  - micro_seq: Alignment pattern

HTF variations (2):
  - htf_070, htf_080: Higher timeframe strength levels
```

**Analysis Methods**:

1. `get_gate_analysis()`: Per-gate breakdown, combinations (0000-1111 bitmaps)
2. `get_comprehensive_analysis()`: Solo/pair/triplet analysis with synergy scoring
3. `generate_recommendations()`: Enable/disable suggestions + YAML config
4. `get_monthly_trends()`: Variable performance across months

---

## 3. QSCORE SYSTEM (Adaptive Threshold Learning)

### 3.1 Base Adapter Architecture (`ml_qscore_adapter_base.py`)

**Purpose**: Learn context-aware (session x volatility) thresholds from qscore.

**Context Buckets**:
- Dimension 1: Session (asian, european, us, global)
- Dimension 2: Volatility regime (low, normal, high, extreme)
- Results in ~16 possible buckets per strategy

**Threshold Learning** (`_retrain()`):

```
For each bucket:
  1. Collect all (qscore, outcome, pnl_percent) records
  2. If sufficient samples (>= 20):
     a. Fit IsotonicRegression: qscore -> P(win)
     b. Calculate empirical average win/loss magnitudes
     c. Compute p_min = avg_loss / (avg_loss + avg_win)
     d. Find smallest qscore where P(win) >= p_min
        (This ensures EV > 0)
     e. Store as bucket threshold
  3. Return new threshold dict
```

**Usage** (`get_threshold()`):
```python
threshold = adapter.get_threshold(
    ctx={'session': 'us', 'volatility_regime': 'high'},
    floor=60.0,      # Minimum allowed threshold
    ceiling=95.0,    # Maximum allowed threshold
    default=75.0     # Fallback if no data
)
# Returns: max(floor, min(ceiling, bucket_threshold))
```

**Retrain Trigger**:
- Minimum records: 50
- Retrain interval: Every 50 new records
- Persistent state in Redis: `qcal:{strategy}:thr`

---

### 3.2 Trend Qscore Adapter (`ml_qscore_trend_adapter.py`)

**Implementation**:
- Loads completed phantoms from `phantom:completed` Redis key
- Filters only Trend Pullback phantoms
- Extracts qscore from features dict
- Learns thresholds per (session, volatility) bucket

**Data Loading**:
```python
def _load_training_records():
    # Load phantom:completed from Redis
    # Filter: strategy_name in ('trend', 'trend_pullback', 'trend_breakout')
    # For each phantom:
    #   qscore = features.get('qscore')
    #   outcome = 1 if phantom.outcome == 'win' else 0
    #   pnl = phantom.pnl_percent
    #   ctx = {'session': features['session'], 'volatility': features['volatility_regime']}
    #   yield (qscore, outcome, pnl, ctx)
```

---

## 4. ML EVOLUTION SYSTEM

### 4.1 Symbol-Specific Learning

**Concept**: Eventually evolve to per-symbol models (currently in shadow mode).

**Architecture**:
- Symbol clustering: Group by volatility/behavior
- Per-cluster models: Learn cluster-specific patterns
- Eventually: Per-symbol fine-tuning

**Current Status**: 
- Cluster-level weighting applied during training
- Cluster 3 (high-vol): 0.7x weight penalty
- Foundation for future symbol-specific models

---

### 4.2 Adaptive Threshold Learning

**Strategy-Specific**:
- Trend: Min 30 trades, retrain every 50
- Scalp: Min 50 trades, retrain every 50
- Range: Min 30 trades, retrain every 50

**Threshold Adaptation**:
```python
# Trend ML Scorer
if recent_WR > 70%:
    min_score = min(85, min_score + 2)   # More selective
elif recent_WR < 30%:
    min_score = max(70, min_score - 2)   # Can't go below 70

# Immediate scorer
if recent_WR > 70%:
    min_score += 2  (cap at 85)
elif recent_WR < 30%:
    min_score -= 2  (floor at 70)
```

---

## 5. INTEGRATION POINTS

### 5.1 Signal Flow

```
Strategy generates signal (entry, TP, SL)
    |
    v
Feature extraction (34+ features)
    |
    v
ML scoring: score_signal(signal, features)
    |
    +---> If score >= min_score: EXECUTE
    |        |
    |        v
    |     Record phantom with was_executed=True
    |     Track as executed trade outcome
    |
    +---> If score < min_score: REJECT
         |
         v
      Record phantom with was_executed=False
      Update phantom with price movements
      Wait for TP/SL/timeout
      Record outcome
      Feed to ML scorer
```

### 5.2 Execution Decision

```python
score, reasoning = ml_scorer.score_signal(signal, features)

# Get adaptive threshold
threshold = qscore_adapter.get_threshold(
    ctx={
        'session': features['session'],
        'volatility_regime': features['volatility_regime']
    },
    floor=70.0,
    ceiling=90.0,
    default=ml_scorer.min_score
)

if score >= threshold:
    # EXECUTE
    position_mgr.enter_trade()
    phantom_tracker.record_signal(
        symbol, signal, score, 
        was_executed=True, 
        features
    )
else:
    # PHANTOM TRACK (learning)
    phantom_tracker.record_signal(
        symbol, signal, score,
        was_executed=False,
        features
    )
```

### 5.3 Feedback Loop

```
Trade closes (executed or phantom)
    |
    v
Calculate outcome + P&L
    |
    v
Update phantom trade record
(one_r_hit, two_r_hit, realized_rr, etc.)
    |
    v
Feed to ML scorer:
  ml_scorer.record_outcome(signal, outcome, pnl_percent)
    |
    +---> Check retrain trigger
    |        |
    |        +---> If ready: _retrain()
    |               |
    |               +---> Fit ensemble (RF, GB, NN)
    |               |
    |               +---> Calibrate probabilities
    |               |
    |               +---> Update Qscore adapter
    |
    v
Update performance metrics
(threshold, stats, patterns)
    |
    v
Persist to Redis + PostgreSQL
    |
    v
Notify via Telegram
```

---

## 6. TRAINING PIPELINE

### 6.1 Retrain Trigger Conditions

**Trend ML**:
```python
total = len(training_data)  # executed + phantom
if total >= MIN_TRADES_FOR_ML (30):
    if (total - last_train_count) >= RETRAIN_INTERVAL (50):
        _retrain()
```

**Data Composition**:
- All executed trades: weight = 1.0
- All phantom trades: weight = 0.8
- No caps or sampling (retain full history)

### 6.2 Feature Importance Extraction

```python
# From Random Forest
feature_names = [
    'atr_pct', 'break_dist_atr', 'retrace_depth_atr',
    'confirm_candles', 'ema_stack_score', 'range_expansion',
    'session', 'symbol_cluster', 'volatility_regime',
    ... (34+ total)
]

importances = models['rf'].feature_importances_

# Top 10 features sorted by importance
for feat, imp in sorted(zip(names, importances), 
                        key=lambda x: x[1], reverse=True)[:10]:
    patterns['feature_importance'][feat] = imp * 100
```

### 6.3 Model Validation

**Win Rate Analysis**:
```
Overall WR = (wins / total_trades) * 100
Executed WR = (executed_wins / executed_trades) * 100
Phantom WR = (phantom_wins / phantom_trades) * 100
```

**Calibration Mapping**:
- For 50+ records: IsotonicRegression.fit()
- Maps raw ensemble probabilities to empirical win rates
- Applied during scoring: `score = calibrator.predict(raw_prob)`

**Bucket-Based Analysis**:
```
For each (session, volatility) bucket:
  wins[bucket] = count of wins
  losses[bucket] = count of losses
  avg_win = sum(wins pnl) / wins count
  avg_loss = sum(losses pnl) / losses count
  
  EV = P(win) * avg_win - P(loss) * avg_loss
  
  # Derive optimal threshold where EV >= 0
  p_min = avg_loss / (avg_loss + avg_win)
  threshold = smallest_score_where_P(win) >= p_min
```

---

## 7. FALLBACK BEHAVIOR

### 7.1 Pre-ML Heuristic Scoring

When ML not ready (< 30 trades):
```python
score = 50.0  # Neutral baseline

# Trend slope
if |slope| >= 5:
    score += 10

# EMA stack alignment
if ema_stack >= 50:
    score += 10

# Breakout distance
if breakout_dist >= 0.2 ATR:
    score += 10

# Range expansion
if range_exp >= 1.2:
    score += 10

# Volatility penalty
if atr_pct >= 0.5:
    score -= 5

return max(0, min(100, score))
```

### 7.2 Error Handling

```python
try:
    # ML scoring
    score, method = ml_scorer.score_signal(signal, features)
except Exception as e:
    logger.warning(f"ML scoring failed: {e}")
    # Fallback to heuristic
    score, method = _heuristic_score(signal, features)

# Apply adaptive threshold
threshold = qscore_adapter.get_threshold(...)

# Execute if score meets threshold
if score >= threshold:
    execute_trade()
```

---

## 8. KEY STATISTICS & DASHBOARDS

### 8.1 ML Status Dashboard

```python
ml_stats = {
    'status': '✅ Ready' or '⏳ Training',
    'completed_trades': int,
    'current_threshold': float (70-90),
    'recent_win_rate': float (0-100%),
    'recent_trades': int (last 200),
    'models_active': ['rf', 'gb', 'nn'],  # Ensemble heads
    'executed_count': int,
    'phantom_count': int,
    'total_records': int,  # executed + phantom
}

ml_patterns = {
    'feature_importance': {'atr_pct': 15.2, ...},
    'time_patterns': {
        'best_hours': {'14:00': 'WR 65% (N=12)'},
        'session_performance': {'us': 'WR 62% (N=45)'}
    },
    'market_conditions': {
        'volatility_impact': {'high': 'WR 55% (N=30)', ...},
        'volume_impact': {'high_volume': 'WR 68% (N=25)'},
        'trend_impact': {'strong_trend': 'WR 70% (N=40)'}
    }
}
```

### 8.2 Phantom Stats

```python
phantom_stats = {
    'total': int,
    'executed': int,
    'rejected': int,
    'rejection_stats': {
        'total_rejected': int,
        'would_have_won': int,
        'would_have_lost': int,
        'missed_profit_pct': float,
        'avoided_loss_pct': float
    },
    'ml_accuracy': {
        'correct_rejections': int,
        'wrong_rejections': int,  # Missed wins
        'correct_approvals': int,
        'wrong_approvals': int,   # Bad trades taken
        'accuracy_pct': float     # (correct / total)
    }
}
```

### 8.3 Scalp Gate Analysis

```python
gate_analysis = {
    'baseline_wr': 52.3,  # Overall WR
    'all_gates_pass': {
        'wins': 45,
        'total': 60,
        'wr': 75.0,           # Much higher!
        'delta': 22.7         # vs baseline
    },
    'variable_stats': {
        'body_050': {
            'pass_wins': 40, 'pass_total': 50, 'pass_wr': 80.0,
            'fail_wins': 5, 'fail_total': 10, 'fail_wr': 50.0,
            'delta': 27.7,
            'sufficient_samples': True
        },
        ...
    },
    'sorted_variables': [          # Ranked by impact
        ('body_050', {...}),
        ('vwap_045', {...}),
        ...
    ],
    'top_combinations': [
        {'bitmap': '1111', 'wins': 20, 'total': 24, 'wr': 83.3},  # All gates
        {'bitmap': '1110', 'wins': 18, 'total': 22, 'wr': 81.8},
        ...
    ]
}
```

---

## 9. PERFORMANCE OPTIMIZATIONS

### 9.1 Data Management

**Redis Storage Strategy**:
- Active phantoms: Live in Redis, per-symbol lists
- Completed phantoms: Last 1000 + last 30 days
- Models: Pickle + base64 encoded in Redis
- Scalers: StandardScaler state persisted

**Memory Efficiency**:
- Phantom timeout: Auto-close after 24-36 hours
- Data cleanup: Auto-remove >30 days old
- Training data cap: ~5000 records for ImmediateML

### 9.2 Model Complexity

**Ensemble Design**:
- RF: 150-200 estimators, max_depth=8, min_samples=5
- GB: 150 estimators, max_depth=3, learning_rate=0.08
- NN: (32,16) hidden layers, 400 epochs (Trend) or (64,32) for Scalp

**Training Time**: Typically <5 seconds per retrain on moderate hardware

### 9.3 Feature Engineering

**Dynamic Feature Handling**:
- Missing features default to 0.0
- Session encoding: categorical -> numeric (0-3)
- Volatility regime: categorical -> numeric (0-3)
- All features scaled via StandardScaler

---

## 10. RELIABILITY & SAFETY

### 10.1 Multiple Fallbacks

```
Priority order for threshold:
1. Qscore context-specific bucket threshold
2. Trend ML min_score (70 minimum)
3. EV-based threshold (computed from outcomes)
4. Hardcoded default (70)

Priority order for scoring:
1. ML ensemble (if models ready)
2. Rule-based heuristic
3. Default 50 (neutral)
```

### 10.2 Phantom Health Monitoring

```
Track active phantoms per symbol:
- Alert if > 50 active (elevated)
- Alert if > 100 active (critical - check outcome detection)

Timeout enforcement:
- Trend: 36 hours
- Scalp: 24 hours
- Auto-close and mark as 'timeout' outcome
- Skip timeout phantoms from ML training
```

### 10.3 State Recovery

```
On startup:
1. Load all phantom trades from Redis
2. Load all ML models + scalers from Redis
3. Attempt retrain if sufficient new data (>= RETRAIN_INTERVAL)
4. Resume normal operation
5. Backfill timeout/missing phantoms if needed
```

---

## 11. SUMMARY TABLE

| Component | Strategy | Min Trades | Retrain Interval | Features | Models | Status |
|-----------|----------|-----------|------------------|----------|--------|--------|
| Trend ML | Trend Pullback | 30 | 50 | 34+ | RF/GB/NN | Active |
| Scalp ML | Scalp | 50 | 50 | 12 | RF/GB/NN | Phase 0 (Phantom) |
| Range ML | Range | 30 | 50 | ~20 | RF/GB/NN | Phase 1 |
| Phantom Tracker | All | - | - | Full signal | - | Active |
| Scalp Phantom | Scalp | - | - | Full signal + gates | - | Active |
| Qscore Adapter | Strategy-specific | 50 | 50 | qscore + context | IsotonicRegression | Active |

---

## 12. KEY TAKEAWAYS

1. **Dual-Path Learning**: Execute on high-confidence signals while learning from ALL signals (even rejected ones)
2. **Ensemble Approach**: RF + GB + NN provide diverse predictions, averaged for robustness
3. **Context Awareness**: Thresholds adapt by session, volatility, and performance metrics
4. **Phantom Power**: Learning from hypothetical outcomes dramatically accelerates ML without risk
5. **Adaptive Evolution**: System continuously improves thresholds based on bucket-level empirical data
6. **Safety First**: Multiple fallbacks, timeouts, health checks, and error handling throughout
7. **Full Traceability**: PostgreSQL audit trail + Redis state enables complete recovery/analysis

---

Generated: 2025-11-10
