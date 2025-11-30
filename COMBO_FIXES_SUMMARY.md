# Combo System Fixes - Summary

## Issues Fixed

### 1. ✅ Fail-Open Behavior → Fail-Closed

**Problem**: Unknown combos were allowed by default (fail-open behavior)

**Fixed In**: `autobot/strategies/scalp/combos.py`

**Changes**:
- `is_combo_enabled()` now returns `False` for unknown combos
- Only combos explicitly enabled by the manager are allowed
- On error, combos are blocked (fail-closed for safety)

**Before**:
```python
if combo_id and combo_id in state:
    return state[combo_id].get('enabled', False)
return True  # Default to enabled if not found (fail open)
```

**After**:
```python
if combo_id and combo_id in state:
    return state[combo_id].get('enabled', False)
# Fail-closed: Only combos explicitly enabled by manager are allowed
return False
```

### 2. ✅ EV_R Filtering Enabled

**Problem**: EV_R floor was set to -1000.0 (effectively disabled)

**Fixed In**: `config.yaml`

**Changes**:
- Changed `ev_floor_r` from `-1000.0` to `0.0`
- Combos must now have non-negative expected value to be enabled
- Filters out losing combos even if they have good win rate

**Before**:
```yaml
ev_floor_r: -1000.0  # Disable EV floor (set very low)
```

**After**:
```yaml
ev_floor_r: 0.0  # Require non-negative EV_R (break-even minimum)
```

### 3. ✅ Enhanced Logging

**Added**: Better debug logging when combos are blocked

**Location**: `autobot/core/bot.py`

**Changes**:
- Logs when a combo is blocked because it's not in the enabled list
- Shows how many combos are enabled for that side
- Helps with debugging and monitoring

---

## Impact

### Security & Safety
- ✅ **Stricter execution**: Only proven combos can execute
- ✅ **No unknown patterns**: Untested combos are blocked until they prove performance
- ✅ **EV filtering**: Negative EV combos are filtered out

### Behavior Changes
- **Before**: Unknown combos → Allowed (fail-open)
- **After**: Unknown combos → Blocked (fail-closed)

- **Before**: EV_R filtering disabled (all combos allowed regardless of EV)
- **After**: EV_R ≥ 0.0 required (only profitable combos enabled)

### Execution Flow
```
Signal Detected
    │
    ▼
Generate Combo Key
    │
    ▼
Manager Ready?
    │
    YES → Check if combo is in enabled list
    │      │
    │      YES → EXECUTE ✅
    │      │
    │      NO → BLOCK ❌ (combo not enabled)
    │
    NO → Fallback to Pro Rules (if configured)
```

---

## Testing Recommendations

1. **Verify Combo Blocking**:
   - Check logs for "Combo blocked: ... not in enabled list"
   - Verify unknown combos are properly blocked

2. **Verify EV_R Filtering**:
   - Check that combos with negative EV_R are disabled
   - Verify only profitable combos are enabled

3. **Monitor Combo Updates**:
   - Watch for combo enable/disable notifications
   - Verify new combos are properly evaluated

---

## Configuration

The combo system now uses these strict settings:

```yaml
adaptive_combos:
  enabled: true
  min_wr_threshold_long: 45.0
  min_wr_threshold_short: 45.0
  min_sample_size: 30
  ev_floor_r: 0.0  # ✅ Now requires non-negative EV
  use_wilson_lb: false
  hysteresis_pct: 2.0
```

---

## Summary

✅ **Fixed**: Fail-open behavior → Fail-closed  
✅ **Fixed**: EV_R filtering enabled (0.0 minimum)  
✅ **Added**: Enhanced logging for debugging  

The combo system now operates in **strict mode**: only combos that have been analyzed, proven profitable (EV_R ≥ 0), and explicitly enabled by the manager are allowed to execute.

