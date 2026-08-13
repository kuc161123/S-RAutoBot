#!/bin/bash
set -e
cd "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/audit_c"
run() { echo "=== $1 ==="; python3 build_param_universe.py "$@"; }

# 1. pivot width
run --cell piv_2_2 --pivot-left 2 --pivot-right 2 --min-pivot-dist 3 --pivot-stale 10 --rsi-period 14 --max-wait 12
run --cell piv_4_4 --pivot-left 4 --pivot-right 4 --min-pivot-dist 3 --pivot-stale 10 --rsi-period 14 --max-wait 12
run --cell piv_5_5 --pivot-left 5 --pivot-right 5 --min-pivot-dist 3 --pivot-stale 10 --rsi-period 14 --max-wait 12
run --cell piv_3_2 --pivot-left 3 --pivot-right 2 --min-pivot-dist 3 --pivot-stale 10 --rsi-period 14 --max-wait 12
run --cell piv_2_3 --pivot-left 2 --pivot-right 3 --min-pivot-dist 3 --pivot-stale 10 --rsi-period 14 --max-wait 12

# 2. max_wait_candles
run --cell wait_4  --min-pivot-dist 3 --pivot-stale 10 --rsi-period 14 --max-wait 4
run --cell wait_8  --min-pivot-dist 3 --pivot-stale 10 --rsi-period 14 --max-wait 8
run --cell wait_18 --min-pivot-dist 3 --pivot-stale 10 --rsi-period 14 --max-wait 18
run --cell wait_24 --min-pivot-dist 3 --pivot-stale 10 --rsi-period 14 --max-wait 24

# 3. pivot staleness limit
run --cell stale_5  --min-pivot-dist 3 --pivot-stale 5  --rsi-period 14 --max-wait 12
run --cell stale_15 --min-pivot-dist 3 --pivot-stale 15 --rsi-period 14 --max-wait 12
run --cell stale_20 --min-pivot-dist 3 --pivot-stale 20 --rsi-period 14 --max-wait 12

# 4. MIN_PIVOT_DISTANCE
run --cell dist_2 --min-pivot-dist 2 --pivot-stale 10 --rsi-period 14 --max-wait 12
run --cell dist_5 --min-pivot-dist 5 --pivot-stale 10 --rsi-period 14 --max-wait 12
run --cell dist_8 --min-pivot-dist 8 --pivot-stale 10 --rsi-period 14 --max-wait 12

# 5. RSI period
run --cell rsi_7  --min-pivot-dist 3 --pivot-stale 10 --rsi-period 7  --max-wait 12
run --cell rsi_21 --min-pivot-dist 3 --pivot-stale 10 --rsi-period 21 --max-wait 12

# 6. second EMA gate at BOS: drop
run --cell dropema --min-pivot-dist 3 --pivot-stale 10 --rsi-period 14 --max-wait 12 --drop-bos-ema-gate

echo "ALL DONE"
