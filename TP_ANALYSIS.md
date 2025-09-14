# Take Profit (TP) Analysis Report

## 1. TP Calculation Summary

### Entry Price Source
- **Entry price comes from the CLOSE price of the signal candle** (line 522 and 593 in strategy_pullback_ml_learning.py)
- This is NOT the current market price at execution time
- There can be slippage between signal generation and order execution

### TP Calculation Formula
From strategy_pullback_ml_learning.py (lines 552 and 623):
```python
# Fee adjustment to compensate for Bybit fees and slippage
fee_adjustment = 1.00165  # Compensate for 0.165% total costs

# TP calculation for LONG trades
tp = entry + ((entry - sl) * settings.rr * fee_adjustment)

# TP calculation for SHORT trades  
tp = entry - ((sl - entry) * settings.rr * fee_adjustment)
```

### Current Settings
- **Risk:Reward Ratio (rr)**: 2.5 (from config.yaml line 274)
- **Fee Adjustment**: 1.00165 (0.165% compensation)
- **Effective R:R after fees**: ~2.504:1

## 2. Fee Breakdown
The 0.165% fee adjustment compensates for:
- **Bybit Market Order Fee**: 0.06% (entry)
- **Bybit Limit Order Fee**: 0.055% (TP exit)
- **Estimated Slippage**: 0.05%
- **Total**: 0.165%

## 3. TP Distance Analysis

For a typical 2% stop loss:
- **Raw TP distance**: 2% × 2.5 = 5.0%
- **Adjusted TP distance**: 2% × 2.5 × 1.00165 = 5.01%

Example calculations:
- Entry: $100
- Stop Loss: $98 (-2%)
- Take Profit: $105.01 (+5.01%)

## 4. Potential Issues Identified

### 4.1 Entry Price Timing
The entry price used for TP calculation is the **close of the signal candle**, not the actual fill price. This can cause issues:
- If price moves significantly between signal and execution
- The calculated TP might be closer/farther than intended
- This could explain why some TPs seem "too close"

### 4.2 No Symbol-Specific Rounding
- TP values are not rounded to symbol-specific tick sizes
- Bybit might adjust these internally, potentially moving TP closer

### 4.3 Volatility-Adjusted Stops
The strategy uses dynamic stop loss adjustment based on volatility (lines 525-545):
- In high volatility (>75th percentile ATR), stops are widened by 1.3x
- In extreme volatility (>90th percentile ATR), stops are widened by 1.5x
- This increases the TP distance proportionally

### 4.4 Minimum Stop Distance
There's a minimum stop distance of 1% enforced (lines 542-545), which ensures:
- Minimum TP distance: 1% × 2.5 × 1.00165 = 2.5%

## 5. TP Validation in Broker

From broker_bybit.py (lines 145-198):
- TP is set using Bybit's position trading-stop endpoint
- Uses **Partial mode** with **Limit orders** for better fills
- No additional TP validation or adjustment is performed
- Bybit accepts the exact TP value provided

## 6. Recommendations

### 6.1 Use Current Market Price for Entry
Consider using the current market price instead of signal candle close:
```python
# Instead of:
entry = close  # Signal candle close

# Use:
entry = df['close'].iloc[-1]  # Current market price
```

### 6.2 Add Tick Size Rounding
Round TP to symbol-specific tick size:
```python
def round_to_tick(price, tick_size):
    return round(price / tick_size) * tick_size

tp = round_to_tick(tp, symbol_meta['tick_size'])
```

### 6.3 Add TP Distance Validation
Ensure minimum TP distance:
```python
min_tp_distance = entry * 0.02  # 2% minimum
if abs(tp - entry) < min_tp_distance:
    tp = entry + (min_tp_distance * (1 if side == "long" else -1))
```

### 6.4 Log Actual vs Expected TP
Add logging to track TP calculations:
```python
expected_rr = (tp - entry) / (entry - sl) if side == "long" else (entry - tp) / (sl - entry)
logger.info(f"TP Calculation: Entry={entry:.4f}, SL={sl:.4f}, TP={tp:.4f}, Expected R:R={expected_rr:.2f}")
```

## 7. Conclusion

The TP calculation itself appears correct with a 2.5:1 R:R ratio plus fee compensation. The main issue is likely the timing difference between signal generation (using candle close) and order execution (at current market price). This can make the actual TP distance smaller than intended if price has already moved in the favorable direction.

The lack of symbol-specific tick size rounding might also cause minor adjustments by Bybit, but this would be minimal.