# ML System Architecture Diagrams

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRADING SIGNAL GENERATION                          │
│              (Trend Pullback, Range, Scalp strategies)                      │
└────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      v
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE EXTRACTION (34+ features)                     │
│  - Technical: ATR, breakout distance, retrace depth, EMA, divergence        │
│  - Context: session, volatility regime, symbol cluster                      │
│  - Lifecycle: TP1_hit, be_moved, time metrics                               │
└────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      v
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ML SCORING: score_signal()                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Is ML Ready?                                                        │   │
│  │  ├─ YES → Ensemble Prediction (RF + GB + NN)                        │   │
│  │  │         └─ Apply Isotonic Calibration                            │   │
│  │  │         └─ Return score 0-100                                    │   │
│  │  └─ NO → Rule-based Heuristic Scoring                               │   │
│  │          └─ Slope, EMA, breakout distance, range                    │   │
│  │          └─ Return score 0-100                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      v
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THRESHOLD DETERMINATION                                   │
│  1. Qscore Adapter: (session, volatility) bucket context                   │
│  2. EV-based threshold: empirical avg win/loss ratio                        │
│  3. Fallback: static minimum (70%)                                          │
└────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                  score >= threshold      score < threshold
                          │                       │
            ┌─────────────────────┐  ┌────────────────────────┐
            │  EXECUTE TRADE      │  │  PHANTOM TRACK (Learn) │
            └─────────────────────┘  └────────────────────────┘
                          │                       │
                          │                       │
      was_executed=True   │                       │   was_executed=False
                          │                       │
            ┌─────────────────────┐  ┌────────────────────────┐
            │  Real Position      │  │  Virtual Position      │
            │  - Execute order    │  │  - Track prices        │
            │  - Manage risk      │  │  - Simulate TP/SL      │
            │  - Close position   │  │  - Record outcome      │
            └──────────┬──────────┘  └────────────┬───────────┘
                       │                          │
                       v                          v
            ┌──────────────────────┐  ┌─────────────────────────┐
            │ Trade Outcome        │  │ Phantom Outcome         │
            │ - Actual fill price  │  │ - Simulated exit price  │
            │ - Actual P&L         │  │ - Hypothetical P&L      │
            │ - Exit reason (TP/SL)│  │ - Lifecycle flags       │
            └──────────┬───────────┘  └────────────┬────────────┘
                       │                          │
                       └──────────┬───────────────┘
                                  │
                                  v
                    ┌──────────────────────────────┐
                    │  record_outcome()            │
                    │  - Feed to ML Scorer         │
                    │  - Increment trade count     │
                    │  - Check retrain trigger     │
                    │  - Persist to Redis/Postgres │
                    │  - Notify Telegram           │
                    └──────────────┬───────────────┘
                                   │
                                   v
                    ┌──────────────────────────────┐
                    │  Retrain Check               │
                    │  total >= 30 &&              │
                    │  (total - last) >= 50?       │
                    └──────────────┬───────────────┘
                                   │
                       ┌───────────┴───────────┐
                       │                       │
                     YES                      NO
                       │                       │
                       v                       v
             ┌──────────────────┐  ┌────────────────────┐
             │  _retrain()      │  │  Update metrics    │
             │ - Fit RF/GB/NN   │  │ - Update threshold │
             │ - Calibrate      │  │ - Update patterns  │
             │ - Store models   │  │ - Store in Redis   │
             └──────────────────┘  └────────────────────┘
                     │
                     v
          Await next signal...
```

---

## Ensemble Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FEATURE VECTOR (34+ dims)                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┬─────────────┐
         │           │           │             │
         v           v           v             v
    ┌────────┐  ┌────────┐  ┌────────┐   (Missing)
    │   RF   │  │   GB   │  │   NN   │
    │ Forest │  │Boosting│  │Network │
    │        │  │        │  │        │
    │Prob    │  │Prob    │  │Prob    │
    │[0][1]  │  │[0][1]  │  │[0][1]  │
    └────┬───┘  └────┬───┘  └────┬───┘
         │           │           │
         │  P(win)   │  P(win)   │  P(win)
         │           │           │
         └───────────┼───────────┘
                     │
                     v
         ┌──────────────────────┐
         │  Average Ensemble:   │
         │  p = mean(preds)     │
         │  Range: [0.0, 1.0]   │
         └──────────┬───────────┘
                    │
                    v
         ┌──────────────────────────┐
         │ Isotonic Calibration     │
         │ (if 50+ training records)│
         │  p_cal = iso.predict(p)  │
         │  Improves > 70% WR       │
         └──────────┬───────────────┘
                    │
                    v
         ┌──────────────────────────┐
         │  Scale to 0-100          │
         │  score = p_cal * 100     │
         │  Return: 0-100           │
         └──────────────────────────┘
```

---

## Phantom Trade Lifecycle

```
SIGNAL GENERATED
    │
    └─> Feature extraction + ML score
        │
        ├─ score >= threshold
        │   └─> was_executed = TRUE
        │       (will execute real trade)
        │
        └─ score < threshold
            └─> was_executed = FALSE
                (phantom only, for learning)

    │
    v
PHANTOM CREATED
    ├─ Entry: signal['entry']
    ├─ TP: signal['tp']
    ├─ SL: signal['sl']
    ├─ Score: ml_score (at signal time)
    ├─ Features: full feature dict
    ├─ Strategy: strategy_name
    └─ ID: uuid (for concurrent tracking)

    │
    v
ACTIVE TRACKING
    ├─ Update with current price
    │   ├─ Long:  high >= TP? → WIN
    │   │         low <= SL? → LOSS
    │   └─ Short: low <= TP? → WIN
    │           high >= SL? → LOSS
    │
    ├─ TP1 Milestone (if applicable)
    │   ├─ Detect when 1.6R reached
    │   ├─ Move SL to break-even
    │   ├─ Set tp1_hit = True
    │   └─ Record tp1_time
    │
    ├─ Track extremes
    │   ├─ Long: max_favorable = max(high)
    │   │        max_adverse = min(low)
    │   └─ Short: max_favorable = min(low)
    │           max_adverse = max(high)
    │
    └─ Timeout check
        └─ 36h (Trend) or 24h (Scalp)?
           └─ Close at entry_price, mark 'timeout'

    │
    v
PHANTOM CLOSED
    ├─ Calculate P&L
    │   ├─ Long: pnl = (exit - entry) / entry * 100
    │   └─ Short: pnl = (entry - exit) / entry * 100
    │
    ├─ Enrich lifecycle metrics
    │   ├─ realized_rr = (exit - entry) / R
    │   ├─ one_r_hit = max_fav >= entry + R?
    │   ├─ two_r_hit = max_fav >= entry + 2R?
    │   └─ exit_reason: 'tp' | 'sl' | 'timeout'
    │
    └─ Outcome: 'win' | 'loss' | 'timeout'

    │
    v
FEED TO ML (skip if timeout)
    ├─ Reconstruct signal_data
    │   ├─ features (updated with outcome flags)
    │   ├─ outcome ('win' or 'loss')
    │   ├─ pnl_percent
    │   └─ exit_reason
    │
    └─> ml_scorer.record_outcome(signal, outcome, pnl)
        ├─ Increment completed_trades
        ├─ Store in Redis: {ns}:trades
        ├─ Check retrain trigger
        ├─ Update EV buckets (session|volatility)
        └─ Check Qscore adapter retrain

    │
    v
COMPLETED TRADE
    ├─ Stored in completed phantoms
    │   ├─ Redis: phantom:completed (last 1000, <30d)
    │   └─ PostgreSQL: Full audit trail
    │
    └─ Available for pattern analysis
        ├─ Feature importance
        ├─ Time patterns (best hours, sessions)
        ├─ Market conditions (vol, trend, volume)
        └─ Gate combinations (Scalp)
```

---

## Qscore Threshold Learning

```
┌─────────────────────────────────────────────────────────────────┐
│                   COMPLETED PHANTOM TRADES                      │
│  (from Redis: phantom:completed)                                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
        ┌────────────────────────────┐
        │ Filter by context:         │
        │ ├─ Strategy (Trend only)   │
        │ ├─ Session (4 types)       │
        │ └─ Volatility (4 regimes)  │
        │ Result: 16 buckets         │
        └────────────┬───────────────┘
                     │
                     v
        ┌────────────────────────────────────┐
        │ For each bucket with 20+ records:  │
        │                                    │
        │ Extract: (qscore, outcome, pnl)   │
        │                                    │
        │ Fit IsotonicRegression:            │
        │ ├─ Input: qscore                   │
        │ └─ Output: P(win)                  │
        │                                    │
        │ Calculate P&L ratios:              │
        │ ├─ avg_win = mean(win pnl)        │
        │ ├─ avg_loss = mean(loss pnl)      │
        │ └─ R_net = avg_win / avg_loss     │
        │                                    │
        │ Compute EV threshold:              │
        │ ├─ p_min = avg_loss/(avg+loss)    │
        │ │  (ensures EV >= 0)               │
        │ └─ Find: smallest q where          │
        │    P(win|q) >= p_min               │
        │                                    │
        │ Store: bucket_threshold[bucket]    │
        └────────────┬────────────────────────┘
                     │
                     v
        ┌──────────────────────────────┐
        │ Persist to Redis:            │
        │ {ns}:thr = {                 │
        │   "session|vol": threshold,  │
        │   ...                        │
        │ }                            │
        └──────────────────────────────┘


USAGE:
┌─────────────────────────────────────┐
│ signal_context = {                  │
│   'session': 'us',                  │
│   'volatility_regime': 'high'       │
│ }                                   │
│                                     │
│ threshold = adapter.get_threshold(  │
│   context,                          │
│   floor=60.0,                       │
│   ceiling=95.0                      │
│ )                                   │
│                                     │
│ Result: max(60, min(95,             │
│            bucket_threshold))       │
└─────────────────────────────────────┘
```

---

## Scalp Gate Analysis

```
┌──────────────────────────────────────────────────────────────┐
│             SCALP PHANTOM COMPLETED TRADES                   │
│          (from Redis: scalp_phantom:completed)               │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     v
        ┌───────────────────────────────────┐
        │ For each phantom, compute status: │
        │                                   │
        │ 26+ binary variables:             │
        │ ├─ Original 4 gates (htf,vol,...)│
        │ ├─ Body variations (body_040+)   │
        │ ├─ VWAP variations (4)            │
        │ ├─ Volume variations (3)          │
        │ ├─ BB width percentiles (3)       │
        │ ├─ Q-score thresholds (4)         │
        │ ├─ Impulse thresholds (2)         │
        │ ├─ Micro alignment (1)            │
        │ └─ HTF variations (2)             │
        │                                   │
        │ Result: 26+ booleans per phantom  │
        └────────────────┬──────────────────┘
                         │
                         v
        ┌────────────────────────────────┐
        │ Analyze each variable:         │
        │                                │
        │ For var in all_26_variables:   │
        │  ├─ Filter: phantoms where var │
        │  │          == True            │
        │  ├─ WR_pass = wins/total       │
        │  │                              │
        │  ├─ Filter: phantoms where var │
        │  │          == False           │
        │  └─ WR_fail = wins/total       │
        │                                │
        │  delta = WR_pass - WR_fail     │
        │  (+ delta = good filter)        │
        │  (- delta = bad filter)         │
        │                                │
        │  sufficient = (pass >= 20 &&   │
        │               fail >= 20)       │
        └────────────────┬───────────────┘
                         │
                         v
        ┌─────────────────────────────────┐
        │ Rank variables by delta:        │
        │ ├─ Top: body_050 (+27.7%)       │
        │ ├─ ... vwap_045 (+24.3%)        │
        │ └─ Bottom: micro_seq (-5.2%)    │
        │                                 │
        │ Select top 10 for combo analysis│
        └────────────────┬────────────────┘
                         │
                         v
        ┌──────────────────────────────────┐
        │ Pair Analysis (top 10 choose 2): │
        │                                  │
        │ For (v1, v2) in combinations:   │
        │  ├─ Filter: phantoms where BOTH │
        │  │          v1=True AND v2=True │
        │  ├─ WR_combo = wins/total      │
        │  │                              │
        │  ├─ expected = (WR_v1 +        │
        │  │             WR_v2) / 2      │
        │  │                              │
        │  └─ synergy = WR_combo -       │
        │              expected          │
        │                                 │
        │ + Synergy = variables work well │
        │ - Synergy = variables conflict  │
        └────────────────┬────────────────┘
                         │
                         v
        ┌──────────────────────────────────┐
        │ Triplet Analysis (top 10 choose 3)
        │ (similar to pairs)               │
        │                                  │
        │ Result: Top combinations        │
        │ ├─ v1+v2+v3 = 85% WR (n=24)   │
        │ ├─ v1+v2+v4 = 83% WR (n=22)   │
        │ └─ ...                          │
        └──────────────────────────────────┘

RECOMMENDATIONS:
┌──────────────────────────────────────┐
│ Enable:                              │
│ ├─ body_050 (80% WR, +27.7 delta)   │
│ ├─ vwap_045 (79% WR, +24.3 delta)   │
│ └─ ...                               │
│                                      │
│ Disable:                             │
│ ├─ micro_seq (48% WR, -5.2 delta)   │
│ └─ ...                               │
│                                      │
│ Best combo: body_050 + vwap_045      │
│ Recommended WR: 85%                  │
└──────────────────────────────────────┘
```

---

## Feature Importance Evolution

```
Training Data Accumulation:
┌──────────────────────────────────────────────────────────┐
│  Executed Trades    │  Phantom Trades    │   Total Sz    │
├──────────────────────────────────────────────────────────┤
│  10                 │  12                │   22 (ready?) │
│  30 (ML ready)      │  32                │   62 (train)  │
│  60                 │  80                │   140 (train) │
│  100                │  150               │   250 (train) │
│  150                │  220               │   370 (train) │
└──────────────────────────────────────────────────────────┘

Feature Importance (Random Forest):
                ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀

After 30 trades (early):
  atr_pct             ████████░░░░░░░░░░  12.3%
  ema_stack_score     ██████░░░░░░░░░░░░░  9.2%
  breakout_dist_atr   ██████░░░░░░░░░░░░░  8.7%
  volatility_regime   █████░░░░░░░░░░░░░░  7.1%
  ... (noisy, high variance)

After 100 trades (converging):
  break_dist_atr      ████████████░░░░░░░  18.4%
  ema_stack_score     █████████░░░░░░░░░░  13.2%
  volatility_regime   ████████░░░░░░░░░░░  11.5%
  atr_pct             ███████░░░░░░░░░░░░  10.1%
  session             ██████░░░░░░░░░░░░░   8.9%
  ... (more stable)

After 300+ trades (mature):
  break_dist_atr      ██████████████░░░░░  21.3%  ↓↑ Consistent
  ema_stack_score     ██████████░░░░░░░░░  15.7%  ↓↑ Stable
  volatility_regime   █████████░░░░░░░░░░  13.2%  ↓↑ Clear
  tp1_hit             ████████░░░░░░░░░░░  11.8%  ↑ Rising
  time_to_exit_sec    ███████░░░░░░░░░░░░  10.4%  ↑ Rising
  be_moved            ███████░░░░░░░░░░░░   9.9%  ↑ Rising
  ... (strong signal pattern emerges)
```

---

## State Persistence

```
┌─────────────────────────────────────────────────────┐
│              REDIS STATE PERSISTENCE                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ML Scorer State (ml:trend:*)                       │
│  ├─ completed_trades → int                          │
│  ├─ last_train_count → int                          │
│  ├─ threshold → float                               │
│  ├─ model → base64(pickle(RF+GB+NN))                │
│  ├─ scaler → base64(pickle(StandardScaler))         │
│  ├─ calibrator → base64(pickle(IsotonicReg))        │
│  ├─ ev_buckets → base64(pickle(bucket_dict))        │
│  ├─ nn_enabled → '0' or '1'                         │
│  ├─ last_train_ts → datetime string                 │
│  └─ trades → [JSON, JSON, ...]  (all records)       │
│                                                     │
│  Phantom Trades (phantom:*)                         │
│  ├─ active → JSON dict symbol→[phantom...]          │
│  ├─ completed → JSON array [phantom...]             │
│  ├─ wr:trend → list '1','0'... (last 200)          │
│  └─ blocked:YYYYMMDD → count                        │
│                                                     │
│  Qscore Adapter (qcal:{strategy}:*)                 │
│  ├─ thr → JSON {bucket→threshold}                   │
│  └─ last_train_count → int                          │
│                                                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│             POSTGRESQL AUDIT TRAIL                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  phantom_trades table                               │
│  ├─ id (PK)                                         │
│  ├─ symbol, side, entry, sl, tp                     │
│  ├─ signal_time, exit_time                          │
│  ├─ outcome, pnl_percent, exit_reason               │
│  ├─ strategy_name, was_executed                     │
│  ├─ ml_score, realized_rr                           │
│  ├─ features_json (full feature dict)               │
│  └─ created_at, updated_at                          │
│                                                     │
│  Notes:                                             │
│  - Full audit trail of all trades                   │
│  - Can reconstruct any phantom or outcome           │
│  - Queryable for analysis                           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Error Handling & Fallback Chain

```
┌──────────────────────────────────────────┐
│         SCORING ERROR HANDLING           │
└──────────┬───────────────────────────────┘
           │
           v
    ┌──────────────────┐
    │ Try ML Score     │
    │ (ensemble)       │
    └──────────┬───────┘
               │
        ┌──────┴──────┐
        │             │
      SUCCESS       FAIL
        │             │
        │             v
        │    ┌─────────────────┐
        │    │ Fallback to     │
        │    │ Rule-based      │
        │    │ Heuristic       │
        │    └────────┬────────┘
        │             │
        └─────┬───────┘
              │
              v
      ┌──────────────────┐
      │ Score: 0-100     │
      │ (always returned)│
      └────────┬─────────┘
               │
               v
      ┌──────────────────────────┐
      │ Apply Threshold          │
      │ priority:                │
      │ 1. Qscore bucket         │
      │ 2. EV-based              │
      │ 3. Static (70)           │
      └────────┬─────────────────┘
               │
               v
      ┌──────────────────┐
      │ Decision:        │
      │ score >= thr?    │
      │ ├─ YES: EXECUTE  │
      │ └─ NO: PHANTOM   │
      └──────────────────┘

┌────────────────────────────────────────┐
│     RETRAIN ERROR HANDLING             │
└──────────┬─────────────────────────────┘
           │
           v
    ┌──────────────────┐
    │ Data available?  │
    │ (>= 30 trades)   │
    └──────┬───────┬──┘
           │       │
          YES     NO
           │       │
           │       v
           │  ┌────────────────┐
           │  │ Deferred until │
           │  │ threshold met  │
           │  └────────────────┘
           │
           v
    ┌──────────────────┐
    │ Try _retrain()   │
    └──────┬───────┬──┘
           │       │
        SUCCESS  FAIL
           │       │
           │       v
           │  ┌──────────────────┐
           │  │ Log error,       │
           │  │ continue with    │
           │  │ previous models  │
           │  │ (no crash)       │
           │  └──────────────────┘
           │
           v
    ┌──────────────────┐
    │ Update models    │
    │ (if success)     │
    │ Persist to Redis │
    └──────────────────┘
```

