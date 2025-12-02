# Adaptive Strategy Discovery - Running

## ✅ Status: RUNNING

**Started**: 12:20 PM
**Symbols**: 400
**Current**: Testing symbol #1 (0GUSDT)

## What's Different

This backtest tests **18 different strategy combinations** per symbol to find what works best:

### Strategy Variations (18 total)

**VWAP Patterns** (3):
- `bounce`: Strict VWAP touch (original)
- `revert`: Price returning to VWAP (broader)
- `means_or`: Any mean proximity (broadest)

**EMA Alignment** (2):
- With EMA requirement
- Without EMA requirement

**Parameter Sets** (3):
- **Strict**: BB=0.45, Vol=0.8, Body=0.25, Wick=0.05
- **Medium**: BB=0.35, Vol=0.7, Body=0.20, Wick=0.03
- **Loose**: BB=0.25, Vol=0.6, Body=0.15, Wick=0.02

## Walk-Forward Validation

**Train**: First 60% of data
**Validate**: Last 40% of data

**Pass Criteria**:
1. Train WR > 45%, N >= 30
2. Validate WR > 45%, N >= 20
3. WR consistency (within 10% between train/validate)
4. Combined WR > 45%

## Anti-Overfitting Safeguards

✅ Separate train/validate sets
✅ Must pass on BOTH sets
✅ Consistency check
✅ Realistic costs applied (0.16%)
✅ Min sample sizes enforced

## Expected Results

- **Symbols passing**: 60-120 (realistic with multiple strategies)
- **Per symbol**: Discovers optimal strategy for longs AND shorts separately
- **Quality**: High (validated, not overfit)

## Timeline

- **Per symbol**: ~3-5 minutes (tests 18 strategies)
- **Total**: 20-30 hours
- **ETA**: Tomorrow afternoon (~6:00 PM Dec 3rd)

## Output Format

```yaml
BTCUSDT:
  long:
    strategy: bounce_ema_medium
    combined_wr: 48.7%
    combined_n: 73
    # Train: WR=52%, N=45 | Val: WR=46%, N=28
  short:
    strategy: revert_noema_loose
    combined_wr: 47.2%
    combined_n: 62
    # Train: WR=50%, N=38 | Val: WR=44%, N=24
```

Each symbol gets its own optimal strategy!
