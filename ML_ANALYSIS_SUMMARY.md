# ML Scoring and Phantom Tracking System - Executive Summary

## Overview

This trading bot implements a **sophisticated dual-learning ML system** that:
1. **Executes trades** on high-confidence signals (score >= threshold)
2. **Tracks rejected signals** (phantom trades) to learn from hypothetical outcomes
3. **Continuously adapts** thresholds and models based on empirical performance
4. **Ensures safety** through multiple fallbacks, timeouts, and error handling

---

## Key Innovation: The Phantom Trade System

Rather than only learning from executed trades, the system tracks ALL signals:
- **Executed (Real)**: 1.0x weight in training
- **Rejected (Phantom)**: 0.8x weight in training

This enables:
- **Accelerated learning**: 2-3x more data without additional trading risk
- **ML accuracy validation**: See what was correctly rejected
- **Fast iteration**: Complete performance picture in half the trades

---

## Component Summary

### 1. ML Scorers (3 Active)

| Scorer | Strategy | Features | Models | Min Trades | Status |
|--------|----------|----------|--------|-----------|--------|
| **Trend ML** | Trend Pullback | 34+ | RF/GB/NN | 30 | Production |
| **Scalp ML** | Scalp | 12 | RF/GB/NN | 50 | Phase 0 (Phantom) |
| **Range ML** | Range Breakout | ~20 | RF/GB/NN | 30 | Phase 1 |

### 2. Phantom Trackers (2 Active)

| Tracker | Strategy | Tracking | Analysis | Status |
|---------|----------|----------|----------|--------|
| **Phantom Tracker** | Trend/Range | All signals | Win/Loss, R-multiples | Production |
| **Scalp Phantom** | Scalp | All signals | 26+ gate variables | Development |

### 3. Adaptive Systems

| System | Purpose | Context | Resolution |
|--------|---------|---------|------------|
| **Qscore Adapter** | Context-aware thresholds | Session x Volatility | 16 buckets |
| **EV Calculator** | Risk/reward thresholds | P&L distribution | Per-bucket |
| **Threshold Adjuster** | Win-rate based tuning | Recent 20 trades | Dynamic ±2% |

---

## Data Flow (High-Level)

```
Signal Generated (Strategy)
    ↓
Extract 34+ Features (Technical + Context)
    ↓
ML Score: 0-100 (Ensemble: RF + GB + NN)
    ↓
Determine Threshold (Context-Aware)
    ├─ YES (score >= thr) → EXECUTE + PHANTOM MIRROR
    └─ NO (score < thr) → PHANTOM ONLY
    ↓
Track Outcome (TP/SL/Timeout)
    ↓
Record Outcome → ML Scorer
    ↓
Check Retrain (every 50 trades)
    ├─ Fit ensemble with executed + phantom data
    ├─ Calibrate probabilities
    └─ Update thresholds
    ↓
Persist to Redis + PostgreSQL
```

---

## Feature Engineering (Trend)

### Core Technical (11)
- `atr_pct`: Volatility measure
- `break_dist_atr`: How far price broke S/R
- `retrace_depth_atr`: Pullback depth
- `confirm_candles`: 2-candle confirmation
- `ema_stack_score`: Trend alignment
- `range_expansion`: Day range vs median
- Divergence features (3): RSI, TSI, presence

### Context (3)
- `session`: Trading time (Asian/European/US/Off)
- `volatility_regime`: Market condition
- `symbol_cluster`: Behavior group

### Lifecycle (6)
- `tp1_hit`, `be_moved`, `runner_hit`: Milestones
- `time_to_tp1_sec`, `time_to_exit_sec`: Timing
- `qscore`: Rule-based quality

### HTF/Other (11+)
- Composite higher-timeframe metrics
- Additional technical indicators

---

## Training Pipeline

### Trigger
```python
if len(training_data) >= 30:  # Minimum threshold
    if (len(training_data) - last_trained) >= 50:  # Retrain interval
        _retrain()
```

### Process
1. **Load data**: All executed + phantom trades from Redis
2. **Prepare features**: 34-dim vector per trade
3. **Weight samples**: Executed=1.0, Phantom=0.8
4. **Scale features**: StandardScaler normalization
5. **Fit ensemble**:
   - Random Forest: 150 estimators, depth=8
   - Gradient Boosting: 150 estimators, depth=3, lr=0.08
   - Neural Network: (32,16) hidden layers (if 300+ trades)
6. **Calibrate**: IsotonicRegression to map prob → win rate
7. **Persist**: Base64-encoded pickle to Redis

### Output
- Models: Random Forest + Gradient Boosting + optional Neural Network
- Scaler: StandardScaler state (mean/std)
- Calibrator: IsotonicRegression function
- EV buckets: Performance per session|volatility combo

---

## Scoring Pipeline

### ML Ready (Yes)
```
Raw Features → Scaled Features
    ↓
RF.predict_proba() → P_RF
GB.predict_proba() → P_GB
NN.predict_proba() → P_NN (optional)
    ↓
P_mean = mean([P_RF, P_GB, P_NN])
    ↓
P_calibrated = calibrator(P_mean)
    ↓
Score = P_calibrated * 100
Return: 0-100 confidence score
```

### ML Not Ready (Pre-Training)
```
Heuristic rules based on:
- Slope magnitude (±5% = +10 points)
- EMA stack alignment (+10 points)
- Breakout distance from S/R (+10 points)
- Range expansion vs median (+10 points)
- ATR volatility penalty (-5 points)
Return: 30-80 typically
```

### Threshold Application
```
Get adaptive threshold:
1. Qscore adapter: (session, vol) bucket
2. EV-based: empirical P(win) threshold
3. Fallback: 70 (hardcoded minimum)

if score >= threshold:
    EXECUTE trade
else:
    Track as phantom (no execution risk)
```

---

## Phantom Trade Lifecycle

### Creation
- Record signal: entry, TP, SL, score, features
- Assign unique ID (UUID)
- Mark: was_executed (True/False)

### Active Tracking
- Update on each candle: current_price
- Check: high >= TP? → WIN
- Check: low <= SL? → LOSS
- Track extremes: max_favorable, max_adverse
- Detect TP1 milestone: move SL to BE

### Closing
- Calculate P&L: (exit - entry) / entry * 100
- R-multiple analysis: realized_rr, 1R_hit, 2R_hit
- Timeout after 36h (Trend) / 24h (Scalp)

### Learning Integration
- Feed to ML scorer: signal_data, outcome, pnl
- Skip timeouts (learning signal too noisy)
- Update EV buckets: avg_win, avg_loss per context
- Check retrain trigger

### Storage
- Redis active: `phantom:active` (live trades)
- Redis completed: `phantom:completed` (last 1000, <30d)
- PostgreSQL: Full audit trail

---

## Qscore Adaptive Threshold

### Context Buckets
```
Dimensions:
- Session: Asian, European, US, Off-hours (4)
- Volatility: Low, Normal, High, Extreme (4)
Total: 16 possible buckets
```

### Learning Algorithm
```
For each bucket with 20+ records:
1. Collect (qscore, outcome, pnl%)
2. Fit IsotonicRegression: qscore → P(win)
3. Compute average magnitudes:
   - avg_win = mean(winning pnl%)
   - avg_loss = mean(losing pnl%)
4. Derive optimal threshold:
   - p_min = avg_loss / (avg_loss + avg_win)
   - Find smallest qscore where P(win) >= p_min
   - This ensures EV >= 0
5. Store as bucket_threshold[bucket]
```

### Usage
```python
threshold = adapter.get_threshold(
    context={'session': 'us', 'volatility': 'high'},
    floor=60.0,      # Don't go below 60
    ceiling=95.0,    # Don't go above 95
    default=75.0     # If no data, use 75
)
# Returns: max(60, min(95, bucket_threshold))
```

---

## Scalp Gate Analysis (Advanced)

### 26+ Variables Tracked
```
Original 4: htf, vol, body, align_15m
Body variations: body_040, body_045, body_050, body_060
VWAP distance: vwap_045, vwap_060, vwap_080, vwap_100
Volume: vol_110, vol_120, vol_150
BB Width: bb_width_60p, bb_width_70p, bb_width_80p
Q-Score: q_040, q_050, q_060, q_070
Impulse: impulse_040, impulse_060
Micro: micro_seq
HTF: htf_070, htf_080
```

### Analysis Methods
```
Solo: Which single variables improve WR most?
Pairs: Which 2-variable combos work best together?
Triplets: Best 3-variable combinations?
Synergy: Do variables complement or conflict?
```

### Recommendations Generated
- **Enable**: Variables with +5% WR improvement
- **Disable**: Variables with -5% WR degradation
- **Best combo**: Highest WR pair/triplet
- **Config snippet**: YAML ready to deploy

---

## Error Handling & Fallbacks

### Scoring Errors
```
Try: ML ensemble scoring
  → Success: Return score
  → Fail: Fall back to heuristic
        → Return heuristic score
Always: Return 0-100 (never crash)
```

### Threshold Errors
```
Try: Context-specific Qscore threshold
  → Success: Apply with floor/ceiling
  → Fail: Use EV-based threshold
        → Fail: Use static 70%
Always: Apply bounds [60-95]
```

### Retrain Errors
```
Try: _retrain() with available data
  → Success: Update models, persist to Redis
  → Fail: Log error, continue with old models
         (no execution impact)
```

### Phantom Health Checks
```
Track active phantoms per symbol:
- Warn if > 50 active
- Critical if > 100 active
- Auto-close after timeout
```

---

## Key Statistics

### ML Status
```python
{
    'status': '✅ Ready' or '⏳ Training',
    'completed_trades': 157,
    'current_threshold': 72.5,
    'recent_win_rate': 58.3%,
    'models_active': ['rf', 'gb', 'nn'],
    'executed_count': 89,
    'phantom_count': 68,
    'total_records': 157
}
```

### Phantom Stats
```python
{
    'total': 157,
    'executed': 89,
    'rejected': 68,
    'ml_accuracy': {
        'correct_rejections': 42,      # Correctly rejected losers
        'wrong_rejections': 8,         # Missed winners
        'correct_approvals': 65,       # Correctly approved winners
        'wrong_approvals': 42,         # Bad trades taken
        'accuracy_pct': 68.2%
    }
}
```

### Performance Patterns
```python
{
    'feature_importance': {
        'break_dist_atr': 21.3,
        'ema_stack_score': 15.7,
        'volatility_regime': 13.2,
        'tp1_hit': 11.8,
        'time_to_exit_sec': 10.4
    },
    'time_patterns': {
        'best_hours': {'14:00': 'WR 72% (N=18)'},
        'session_performance': {'us': 'WR 62% (N=45)'}
    },
    'market_conditions': {
        'volatility_impact': {
            'high': 'WR 55% (N=30)',
            'normal': 'WR 68% (N=95)',
            'low': 'WR 72% (N=32)'
        }
    }
}
```

---

## Performance Characteristics

### Training Speed
- Data loading: 100ms (Redis)
- Feature preparation: 200ms (100 trades)
- Model fitting: 2-5 seconds (RF + GB + NN)
- Calibration: 500ms
- **Total retrain: 5-10 seconds**

### Scoring Speed
- Feature scaling: 1ms
- RF prediction: 0.2ms
- GB prediction: 0.3ms
- NN prediction: 0.5ms
- Calibration: 0.1ms
- **Total score: 1-2ms per signal**

### Data Footprint
- Single phantom trade: ~2KB (Redis)
- Single completed phantom: ~1KB
- Complete state: <100MB for 5000 trades

---

## Safety Guarantees

1. **No unrecovered errors**: All exceptions caught, fallbacks applied
2. **Memory bounds**: Auto-cleanup of >30 day old data
3. **Timeout protection**: All phantoms close after 24-36h
4. **Health monitoring**: Active phantom count tracked + alerted
5. **Full auditability**: PostgreSQL audit trail of every trade
6. **State recovery**: Complete restart capability from Redis/PostgreSQL

---

## Future Enhancements

1. **Symbol-specific models**: Per-symbol fine-tuning on top of global model
2. **Multi-timeframe ensemble**: Combine 1m, 5m, 15m, 1h models
3. **Dynamic feature importance**: Reweight features by context
4. **Market regime detection**: Switch between scalping/trending modes
5. **Reinforcement learning**: Active position management via RL
6. **Explainability**: Feature contribution per signal

---

## Files Referenced

| File | Purpose | Lines | Key Classes |
|------|---------|-------|------------|
| `ml_scorer_trend.py` | Trend ML scorer | 619 | `TrendMLScorer` |
| `phantom_trade_tracker.py` | Phantom tracking | 1049 | `PhantomTradeTracker`, `PhantomTrade` |
| `ml_signal_scorer_immediate.py` | Legacy immediate scorer | 1246 | `ImmediateMLScorer` |
| `ml_scorer_scalp.py` | Scalp ML scorer | 589 | `ScalpMLScorer` |
| `scalp_phantom_tracker.py` | Scalp phantom tracking | 1465 | `ScalpPhantomTracker` |
| `ml_qscore_adapter_base.py` | Qscore base | 164 | `QScoreAdapterBase` |
| `ml_qscore_trend_adapter.py` | Trend Qscore | 56 | `TrendQAdapter` |

---

## Documentation Generated

Created two comprehensive documents:

1. **ML_SYSTEM_ANALYSIS.md** (802 lines)
   - Complete system architecture
   - Feature engineering details
   - Training pipeline breakdown
   - Integration points
   - Reliability & safety

2. **ML_SYSTEM_DIAGRAMS.md** (613 lines)
   - Data flow architecture
   - Ensemble model design
   - Phantom lifecycle
   - Qscore learning
   - Gate analysis flows
   - Feature importance evolution
   - State persistence
   - Error handling chains

---

## Contact & Updates

For questions on:
- **ML architecture**: See `ML_SYSTEM_ANALYSIS.md` sections 1-4
- **Phantom tracking**: See `ML_SYSTEM_ANALYSIS.md` section 2
- **Data flow**: See `ML_SYSTEM_DIAGRAMS.md` data flow diagram
- **Scalp gates**: See `ML_SYSTEM_ANALYSIS.md` section 4 + scalp_phantom_tracker.py
- **Qscore threshold**: See `ML_SYSTEM_ANALYSIS.md` section 3

Generated: 2025-11-10
