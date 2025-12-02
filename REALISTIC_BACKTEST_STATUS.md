# Production Bot Backtest - Running

## Configuration

✅ **Using EXACT production bot settings from config.yaml**

**Symbols**: 400
**Criteria**: WR > 45%, N >= 30
**Started**: 2025-12-02 12:08:00

## Bot Settings (Production - NOT Modified)

```yaml
vwap_pattern: bounce              # Strict VWAP bounce
vwap_require_alignment: true      # Requires EMA 8/21 alignment
min_bb_width_pct: 0.45           # 45th percentile BB width
vol_ratio_min: 0.8               # 0.8x volume
body_ratio_min: 0.25             # 25% body minimum
wick_delta_min: 0.05             # 5% wick dominance
vwap_dist_atr_max: 1.5           # Max 1.5 ATR from VWAP
vwap_bounce_band: 0.1-0.6 ATR    # Bounce range
```

## Realistic Additions

- Entry: Next candle open (not signal close)
- Slippage: 0.03% per trade
- Fees: 0.11% round trip
- Spread: 0.02%
- **Total cost**: 0.16% per trade

## Why 45% WR Works

With production bot (2.1:1 R:R):
- Expected Value: (0.45 × 2.1) - (0.55 × 1) = +0.395R
- After costs (~0.08R): +0.315R per trade
- **Still profitable**

## Expected Results

With production settings (strict):
- **Symbols passing**: 20-50 (realistic)
- **Average WR**: 46-50%
- **Quality over quantity**

These will be the REAL symbols that work with your actual bot logic.

## Timeline

- Processing: ~1.5 min/symbol
- Total: ~10 hours  
- ETA: ~10:00 PM
