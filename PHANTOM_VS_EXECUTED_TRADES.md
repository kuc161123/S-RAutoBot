# Phantom vs Executed Trades in Combo System

## ✅ Yes, Both Are Fed to Combo System

Both **phantom trades** and **executed trades** are collected and analyzed by the combo system.

---

## How It Works

### 1. Data Collection

The combo system collects **all completed trades** from the phantom tracker:

```python
# Line 408: Iterate through all completed trades
for arr in (getattr(scpt, 'completed', {}) or {}).values():
    for p in arr:
        # ... process each trade ...
        
        # Line 475: Check if trade was executed
        is_exec = bool(getattr(p, 'was_executed', False))
        
        # Both phantoms and executed trades are added to items
        items.append((rsi, mh, vwap, fibz, mtf, win, rr, p_side, is_exec, is_recent))
```

**Key Point**: The system doesn't filter - it processes **ALL** completed trades, whether executed or phantom.

---

### 2. Separate Tracking

While both are included, they're tracked **separately** for analytics:

```python
# Lines 555-566: Separate counting
if is_exec:
    agg['n_exec'] += 1           # Executed trade count
    if win:
        agg['w_exec'] += 1       # Executed wins
    if is_recent:
        agg['n_exec_24h'] += 1   # Executed in last 24h
else:
    agg['n_phantom'] += 1        # Phantom trade count
    if win:
        agg['w_phantom'] += 1    # Phantom wins
    if is_recent:
        agg['n_phantom_24h'] += 1 # Phantoms in last 24h
```

---

### 3. Combined Metrics

For combo performance calculations, **both are combined**:

```python
# Total count includes both
n = agg['n']  # = n_exec + n_phantom
w = agg['w']  # = w_exec + w_phantom
wr = (w / n * 100.0)  # Combined win rate
ev_r = (agg['rr'] / n)  # Combined EV_R
```

**The combo enable/disable decision uses the COMBINED metrics** (both phantoms + executed).

---

## Why Both Are Included

### Benefits:

1. **More Data**: Faster learning with larger sample sizes
2. **Better Statistics**: More trades = better confidence
3. **Complete Picture**: See how combos perform in all scenarios
4. **Faster Adaptation**: Enable/disable combos faster with more data

### Example:

- **10 executed trades** + **40 phantom trades** = **50 total trades**
- With 50 trades, you can enable combos faster (vs waiting for 50 executed trades)
- More reliable statistics with larger sample size

---

## Tracking Breakdown

The `ComboPerformance` dataclass tracks both separately:

```python
@dataclass
class ComboPerformance:
    n: int              # Total trades (n_exec + n_phantom)
    wins: int           # Total wins (wins_exec + wins_phantom)
    wr: float           # Combined win rate
    ev_r: float         # Combined EV_R
    
    # Separate tracking
    n_exec: int         # Executed trades
    n_phantom: int      # Phantom trades
    wins_exec: int      # Executed wins
    wins_phantom: int   # Phantom wins
    
    # 24h breakdown
    n_24h: int         # Total in last 24h
    n_exec_24h: int    # Executed in last 24h
    n_phantom_24h: int # Phantoms in last 24h
```

---

## Decision Making

### Combo Enable/Disable Uses Combined Metrics:

```python
# EV_R only mode (current)
ev_r = (agg['rr'] / n)  # Uses ALL trades (exec + phantom)
ev_ok = ev_r >= threshold

# Original mode (WR + EV_R)
wr = (w / n * 100.0)    # Uses ALL trades (exec + phantom)
ev_r = (agg['rr'] / n)  # Uses ALL trades (exec + phantom)
```

**Both phantoms and executed trades contribute equally** to the enable/disable decision.

---

## Why This Makes Sense

### 1. Phantoms Are Real Signals
- Phantoms represent **actual trading signals** that were generated
- They just weren't executed (due to gates, risk limits, etc.)
- They're still valid data points for combo performance

### 2. Faster Learning
- If you only used executed trades, you'd need to wait much longer
- Example: 30 executed trades might take weeks
- But 30 total trades (10 exec + 20 phantom) might take days

### 3. More Accurate
- Larger sample size = better statistics
- Better confidence intervals
- More reliable enable/disable decisions

---

## Summary

✅ **Yes, both phantoms and executed trades are fed to the combo system**

- **Collection**: Both are collected from phantom tracker
- **Tracking**: Tracked separately for analytics
- **Metrics**: Combined for combo performance calculations
- **Decision**: Enable/disable uses combined metrics
- **Benefit**: Faster learning, better statistics, more accurate decisions

The combo system uses **all available data** (phantoms + executed) to make the best decisions about which combos to enable or disable.

