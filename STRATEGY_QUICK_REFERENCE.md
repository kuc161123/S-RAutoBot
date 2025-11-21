# Trading Strategy Quick Reference

## Active Strategies

### 1. Mean Reversion (ENABLED - Execution)
- **Files**: `strategy_mean_reversion.py`, `ml_scorer_mean_reversion.py`
- **Timeframe**: 15 minutes
- **Win Rate**: 45-55%
- **RR**: 2.5:1 (fixed)
- **Key Logic**: 
  - Identify range (2.5%-6% width)
  - Wait for edge touch (±0.5 ATR)
  - Confirm reversal (opposite close)
  - Generate signal with hybrid SL

**Entry Requirements**:
- Range identified and valid
- Price within 0.5 ATR of edge
- Bullish close for LONG, bearish for SHORT
- Volatility-adjusted stops (0.8-1.4x multiplier)

---

### 2. Scalping (ENABLED - Execution)
- **Files**: `strategy_scalp.py`, `ml_scorer_scalp.py`
- **Timeframe**: 3 minutes
- **Win Rate**: 50-60% (data-driven)
- **RR**: 2.1:1
- **Key Logic**:
  - EMA stack alignment (8-candle > 21-candle)
  - Multi-anchor proximity (VWAP OR EMA OR BB)
  - Volume spike detection
  - Wick rejection confirmation

**Entry Requirements** (All Must Pass):
1. EMA Trend: close > ema8 > ema21 (LONG)
2. Multi-Anchor: distance to VWAP/EMA/BB ≤ threshold
3. Volume: current_vol / 20bar_avg ≥ 1.20 (exec) or 0.80 (signal)
4. BBW: percentile ≥ 60% (exec) or 55% (signal)
5. Wick/Body: dominant wick + meaningful body in trend direction
6. Regime: not extreme volatility

**Execution Paths**:
- Adaptive Combo Manager → Enables/disables combos based on 30-day performance
- Pro Rules Fallback → MTF + RSI + MACD + VWAP + Fib checks

---

### 3. Trend Pullback (DISABLED - Phantom Only)
- **Files**: `strategy_pullback_ml_learning.py`, `ml_scorer_trend.py`
- **Timeframe**: 15 minutes
- **Win Rate**: 55-65% (when enabled)
- **RR**: 2.5:1 (fixed)
- **Status**: Execution disabled; phantoms recording for learning

**Entry Flow** (If Re-Enabled):
1. S/R Detection: Pivot points + 0.6% zones
2. Breakout: Close outside zone
3. HL/LH: Pullback stays in zone
4. Confirmation: 2 candles in trend direction
5. Generate signal with hybrid SL

---

## ML System

**Phantom Tracking** (All Strategies):
- Records every signal generated (executed or not)
- Calculates full feature set
- Monitors outcome (TP/SL hit, timeout)
- Labels as win/loss for training

**ML Scoring Threshold**:
- < 50 trades: learning mode (accept all)
- 50-200 trades: heuristic scoring (50-70)
- 200+ trades: ML model active (threshold 70+)

**Quality Scoring (Qscore)**:
- Execution: Qscore ≥ 78
- Phantom: 15 ≤ Qscore < 78
- Weighted components: strategy-specific

---

## Stop Loss Calculation

### Trend & MR (Hybrid 3-Option Method)
```
Option 1: Pivot-based (conservative)
Option 2: Breakout-based (medium)
Option 3: Extreme-based (aggressive)

Choose: Most room (lowest SL for LONG, highest for SHORT)
Volatility adjustment: 0.8x - 1.4x multiplier
Minimum distance: 1% of entry
```

### Scalp (Mean-Band Method)
```
LONG: sl = min(VWAP, EMA21) - 0.8-1.2 × ATR
SHORT: sl = max(VWAP, EMA21) + 0.8-1.2 × ATR

Buffer multiplier based on BBW:
- <70th %ile: 0.8x
- 70-85th %ile: 1.0x
- >85th %ile: 1.2x (high volatility)

Minimum distance: max(0.5% price, 0.6 ATR, 0.2% floor)
```

---

## Configuration Control

### Master Switches
```yaml
scalp.enabled: true              # Enable scalp strategy
scalp.exec.enabled: true         # Allow execution
scalp.stream_3m_only: true       # Use 3m bars

mr.exec.enabled: true            # Allow MR execution

trend.exec.enabled: false        # Disable Trend execution
```

### Execution Gates (All Strategies)
```yaml
regime_enabled: true             # Block extreme volatility
ml_min_score: 70.0              # ML threshold

# Strategy-Specific:
scalp.hard_gates.vol_enabled: true           # Volume gate
scalp.hard_gates.bbw_exec_enabled: true      # BB width gate
scalp.hard_gates.slope_enabled: true         # EMA slope gate

mr.regime.enabled: true          # Range regime filter
```

### Phantom Flow Control
```yaml
phantom_flow.enabled: true       # Adaptive daily targets
phantom_flow.daily_target:
  scalp: 200                     # Target 200 phantoms/day
  mr: 200
  trend: 200

phantom_flow.relax_limits:       # Loosen thresholds if below target
  scalp.vwap_dist_atr: +0.60   # Allow deeper VWAP pulls
  scalp.bb_width_pct: -0.30     # Accept narrower bands
```

---

## Key Parameters by Strategy

### Mean Reversion
```yaml
range_width: 2.5% - 6.0%        # Valid range bounds
edge_touch_atr: 0.5             # Distance to edge for signal
range_confidence_threshold: N/A   # All ranges considered

min_candles_between_signals: N/A # Can back-to-back signals
promotion_enabled: false         # Auto-execution disabled

timeout_hours: 36               # Phantom auto-close
```

### Scalping
```yaml
ema_fast: 8                     # Trend confirmation
ema_slow: 21
atr_len: 7                      # ATR for stop sizing

vwap_window: 100                # EVWAP calculation
bb_period: 20                   # Bollinger bands

vol_ratio_min: 1.20 (exec), 0.80 (signal)
wick_ratio_min: 0.40 (signal), 0.25 (phantom)
body_ratio_min: 0.30
wick_delta_min: 0.10            # Dominant wick requirement

timeout_hours: 8                # Phantom auto-close
daily_cap: 0                    # Unlimited executes/day
risk_percent: 1.0               # 1% per trade
```

### Trend Pullback (If Enabled)
```yaml
pivot_left: 3                   # Pivot detection
pivot_right: 2
zone_width: 0.3%                # Zone size (each side)

confirmation_candles: 2         # HL/LH confirmation
state_timeout_bars: 4 hours     # State reset timeout

ml_min_threshold: 75            # Execution threshold
timeout_hours: 168              # Phantom auto-close
```

---

## Common Configuration Issues

### Phantom Accumulation
- Increase timeout_hours to close stale phantoms faster
- Reduce daily_caps to limit phantom flow
- Enable virtual_snapshots for better labeling

### Low Execution Frequency
- Reduce Qscore thresholds (execute_q_min)
- Enable phantom_flow relaxation when below daily target
- Relax individual gate thresholds

### High Slippage/Fees
- Fee adjustment factor: 1.00165 (0.165% total)
- Auto-calculated in TP for 2.5:1 net R:R
- Check leverage and order size vs available liquidity

### Regime Filter Blocking Trades
- Disable when volatility legitimately high
- Or relax allowed_regimes to include 'high'
- Check ATR% and BBW% thresholds

---

## Performance Monitoring

### Key Metrics to Track
1. **Win Rate**: Executed trades only (target 50%+)
2. **Avg R per Trade**: (Entry - SL) magnitude
3. **Phantom-to-Executed Ratio**: Learning efficiency
4. **Daily P&L**: Cumulative performance
5. **Gate Pass Rate**: Execution quality filtering

### Telegram Commands
```
/stats           - Win rate, P&L, trade count
/status          - Open positions
/ml_stats        - Model training status
/phantom_stats   - Phantom tracking metrics
/balance         - Account equity
/symbols         - Active symbols list
```

---

## Execution Flow Summary

```
1. Kline Update (15m or 3m)
   ↓
2. Strategy Analysis
   - Trend: Detect breakout → HL/LH → Confirm
   - MR: Identify range → Edge touch → Confirm
   - Scalp: EMA align → Multi-anchor → Wick/Body
   ↓
3. Signal Generated? → No: Wait for next candle
   ↓ Yes
4. Record Phantom Immediately (no filtering)
   ↓
5. ML Scoring
   - <200 trades: Accept (learning)
   - ≥200 trades: Score signal; if < threshold: Skip
   ↓
6. Rule-Mode Qscore (A/B/C Tiers)
   ↓
7. Execution Gates
   - Regime check
   - S/R strength (Trend)
   - HTF alignment
   - Volume/volatility checks
   ↓
8. If All Pass → Execute Order
   - Market entry
   - SL + TP placed (Partial mode)
   - Update position tracking
   ↓
9. Monitor Until Exit (TP/SL)
   - Close phantom with outcome
   - Label as win/loss
   - Retrain ML if threshold met
```

---

## File Structure

```
Core Strategies:
├── strategy_scalp.py (3m EVWAP pullback)
├── strategy_mean_reversion.py (Range detection)
└── strategy_pullback_ml_learning.py (Breakout+HL/LH) [DISABLED]

ML & Scoring:
├── ml_scorer_scalp.py (12-feature scorer)
├── ml_scorer_mean_reversion.py (35+ features)
├── ml_scorer_trend.py (22 features)
└── ml_qscore_*_adapter.py (Quality scoring)

Phantom Trackers:
├── scalp_phantom_tracker.py
├── mr_phantom_tracker.py
└── phantom_trade_tracker.py (Trend)

Configuration:
├── config.yaml (Master settings)
└── CLAUDE.md (Development guide)
```

---

## Quick Debug Checklist

- [ ] Check config.yaml for strategy status (enabled/disabled)
- [ ] Verify execution gates aren't all blocking (use /status)
- [ ] Check Qscore calculation (is execute_q_min too high?)
- [ ] Review ML status (/ml_stats - is model trained?)
- [ ] Confirm symbol specs loaded (tick size, min qty, leverage)
- [ ] Monitor phantom count for accumulation
- [ ] Check regime classification (is market extreme vol?)
- [ ] Verify Telegram bot connected (test /status command)
- [ ] Review recent trade logs for rejection reasons
- [ ] Confirm Redis/database connectivity for ML persistence

