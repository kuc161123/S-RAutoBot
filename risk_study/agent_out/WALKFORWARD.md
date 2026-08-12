# Walk-Forward Risk-Per-Trade Stability Study

Universe: `risk_study/universe_chopBOS.parquet` (32,929 trades, 2023-06-01..2026-07-25). Cost 34.1bps round trip. Harness: `risk_study/sweep.py::run_one` (live config, taper scaled proportionally to f).

11 rolling 6-month windows, new window starting every 3 months from 2023-06-01. Each window starts fresh at $10,000 and is scored independently (ROI%, maxDD%, ROI/DD). f grid: [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03].

## 1-2. Winning f per window, and the stability question

| window | BTC 6mo ret | BTC ann.vol | trend | vol regime | winning f | winning ROI/DD | winning ROI% | winning maxDD% |
|---|---|---|---|---|---|---|---|---|
| 2023-06-01_2023-12-01 | +39.4% | 36.9% | BULL | LOW | **0.75%** | -0.09 | -6.4% | 73.2% |
| 2023-09-01_2024-03-01 | +135.0% | 42.7% | BULL | LOW | **0.10%** | -0.83 | -14.4% | 17.3% |
| 2023-12-01_2024-06-01 | +79.3% | 53.0% | BULL | HIGH | **1.00%** | -0.95 | -80.6% | 84.9% |
| 2024-03-01_2024-09-01 | -4.3% | 55.7% | SIDEWAYS | HIGH | **1.50%** | 4.68 | +397.0% | 84.8% |
| 2024-06-01_2024-12-01 | +42.6% | 50.5% | BULL | HIGH | **1.50%** | 33.25 | +1439.1% | 43.3% |
| 2024-09-01_2025-03-01 | +43.1% | 50.4% | BULL | HIGH | **0.50%** | 3.60 | +201.1% | 55.8% |
| 2024-12-01_2025-06-01 | +8.5% | 52.7% | SIDEWAYS | HIGH | **0.30%** | 23.47 | +525.4% | 22.4% |
| 2025-03-01_2025-09-01 | +29.1% | 43.2% | BULL | LOW | **1.50%** | 5.50 | +290.9% | 52.9% |
| 2025-06-01_2025-12-01 | -13.5% | 37.1% | SIDEWAYS | LOW | **0.20%** | 3.77 | +157.0% | 41.7% |
| 2025-09-01_2026-03-01 | -38.1% | 46.1% | BEAR | LOW | **0.50%** | 30.04 | +1158.0% | 38.5% |
| 2025-12-01_2026-06-01 | -15.3% | 44.8% | SIDEWAYS | LOW | **2.00%** | 3.31 | +286.6% | 86.7% |

Winning f ranges from **0.10%** to **2.00%** across 11 windows — i.e. the extreme ends of the entire grid both win somewhere. Distribution of winners: 0.10%: 1x, 0.20%: 1x, 0.30%: 1x, 0.50%: 2x, 0.75%: 1x, 1.00%: 1x, 1.50%: 3x, 2.00%: 1x

**Single global f** (the one that maximizes mean ROI/DD across all 11 windows) = **1.50%**. What that global choice gives up vs. each window's own winner:

| window | own-best f | own-best ROI/DD | global f ROI/DD | give-up |
|---|---|---|---|---|
| 2023-06-01_2023-12-01 | 0.75% | -0.09 | -0.66 | 0.57 |
| 2023-09-01_2024-03-01 | 0.10% | -0.83 | -0.94 | 0.11 |
| 2023-12-01_2024-06-01 | 1.00% | -0.95 | -0.96 | 0.01 |
| 2024-03-01_2024-09-01 | 1.50% | 4.68 | 4.68 | 0.00 |
| 2024-06-01_2024-12-01 | 1.50% | 33.25 | 33.25 | 0.00 |
| 2024-09-01_2025-03-01 | 0.50% | 3.60 | -0.50 | 4.10 |
| 2024-12-01_2025-06-01 | 0.30% | 23.47 | 19.18 | 4.30 |
| 2025-03-01_2025-09-01 | 1.50% | 5.50 | 5.50 | 0.00 |
| 2025-06-01_2025-12-01 | 0.20% | 3.77 | 1.71 | 2.06 |
| 2025-09-01_2026-03-01 | 0.50% | 30.04 | 28.03 | 2.01 |
| 2025-12-01_2026-06-01 | 2.00% | 3.31 | 0.31 | 3.00 |

Mean give-up: 1.47 ROI/DD units (median 0.57). Max give-up: 4.30 in window 2024-12-01_2025-06-01.

## 3. Rank stability (Spearman correlation of f-vs-ROI/DD ordering, window pairs)

Mean pairwise Spearman rho across all 55 window pairs: **0.228** (median 0.317). Fraction of pairs with NEGATIVE correlation (one window's best-f is the other's worst): **29.1%**.

Full pairwise rank-correlation matrix (rows/cols = windows, chronological):

```
window                 2023-06-01_2023-12-01  2023-09-01_2024-03-01  2023-12-01_2024-06-01  2024-03-01_2024-09-01  2024-06-01_2024-12-01  2024-09-01_2025-03-01  2024-12-01_2025-06-01  2025-03-01_2025-09-01  2025-06-01_2025-12-01  2025-09-01_2026-03-01  2025-12-01_2026-06-01
window                                                                                                                                                                                                                                                                            
2023-06-01_2023-12-01                   1.00                   0.80                   0.50                   0.17                  -0.20                   0.70                   0.57                   0.20                   0.33                   0.02                  -0.22
2023-09-01_2024-03-01                   0.80                   1.00                   0.60                  -0.07                  -0.40                   0.75                   0.38                   0.30                   0.67                  -0.17                  -0.03
2023-12-01_2024-06-01                   0.50                   0.60                   1.00                   0.38                   0.00                   0.32                   0.18                   0.48                   0.70                   0.10                  -0.52
2024-03-01_2024-09-01                   0.17                  -0.07                   0.38                   1.00                   0.52                   0.33                   0.62                   0.67                   0.28                   0.88                  -0.92
2024-06-01_2024-12-01                  -0.20                  -0.40                   0.00                   0.52                   1.00                  -0.15                   0.22                   0.15                  -0.07                   0.47                  -0.42
2024-09-01_2025-03-01                   0.70                   0.75                   0.32                   0.33                  -0.15                   1.00                   0.83                   0.50                   0.67                   0.42                  -0.25
2024-12-01_2025-06-01                   0.57                   0.38                   0.18                   0.62                   0.22                   0.83                   1.00                   0.62                   0.47                   0.63                  -0.38
2025-03-01_2025-09-01                   0.20                   0.30                   0.48                   0.67                   0.15                   0.50                   0.62                   1.00                   0.68                   0.48                  -0.50
2025-06-01_2025-12-01                   0.33                   0.67                   0.70                   0.28                  -0.07                   0.67                   0.47                   0.68                   1.00                   0.25                  -0.23
2025-09-01_2026-03-01                   0.02                  -0.17                   0.10                   0.88                   0.47                   0.42                   0.63                   0.48                   0.25                   1.00                  -0.77
2025-12-01_2026-06-01                  -0.22                  -0.03                  -0.52                  -0.92                  -0.42                  -0.25                  -0.38                  -0.50                  -0.23                  -0.77                   1.00
```

## 4. Regime split

Trend rule: 6-month BTC return > +20% = BULL, < -20% = BEAR, else SIDEWAYS. Vol rule: 6-month BTC annualized realized vol vs. its cross-window median (46.1%) -> HIGH/LOW.

### Best f by trend regime (mean ROI/DD across that regime's windows)

| trend | best f | mean ROI/DD | n windows contributing |
|---|---|---|---|
| BEAR | 0.50% | 30.04 | 1 |
| BULL | 1.50% | 5.95 | 6 |
| SIDEWAYS | 0.30% | 7.57 | 4 |

Full trend-regime x f table (mean ROI/DD):

```
regime_trend   BEAR  BULL  SIDEWAYS
risk_pct                           
0.10          13.03  1.07      4.45
0.20          19.50  1.50      5.54
0.30          23.07  1.68      7.57
0.50          30.04  2.37      7.33
0.75          27.40  1.77      6.67
1.00          28.95  3.42      5.27
1.50          28.03  5.95      6.47
2.00          13.82  2.74      2.09
3.00          13.87 -0.32      1.39
```

### Best f by volatility regime

| vol regime | best f | mean ROI/DD |
|---|---|---|
| HIGH | 1.50% | 11.13 |
| LOW | 0.50% | 6.29 |

```
regime_vol   HIGH   LOW
risk_pct               
0.10         3.71  3.12
0.20         4.41  4.77
0.30         6.27  5.35
0.50         7.16  6.29
0.75         7.11  4.86
1.00         7.07  5.87
1.50        11.13  5.66
2.00         4.28  2.87
3.00         0.22  2.73
```

## 5. Profitability count per f

11 total windows.

| f | n profitable / n windows | frac profitable | mean ROI% | median ROI% |
|---|---|---|---|---|
| 0.10% | 8/11 | 73% | +43.3% | +39.1% |
| 0.20% | 8/11 | 73% | +103.7% | +82.1% |
| 0.30% | 8/11 | 73% | +172.1% | +126.1% |
| 0.50% | 8/11 | 73% | +267.5% | +192.3% |
| 0.75% | 8/11 | 73% | +293.9% | +80.8% |
| 1.00% | 7/11 | 64% | +341.8% | +187.5% |
| 1.50% | 7/11 | 64% | +455.9% | +138.1% |
| 2.00% | 7/11 | 64% | +247.9% | +71.5% |
| 3.00% | 7/11 | 64% | +116.2% | +42.1% |

## Caveats

BEAR trend regime rests on a **single window** (2025-09-01..2026-03-01) — its 'best f' row in §4 is one data point dressed as a mean, not a regime finding. SIDEWAYS (4 windows) and BULL (6 windows) have more support but are still small-n. None of the ROI% figures above are cost-of-capital adjusted or account for the fact that windows overlap by 3 of their 6 months, so adjacent-window rows in every table are correlated, not independent — the reported window count overstates the number of independent observations.

## Verdict

NO STABLE OPTIMUM. The winning f per window covers 1.90 percentage points (0.10%-2.00%) out of a 2.90pp grid (0.10%-3.00%) — i.e. the apparent winner ranges over essentially the WHOLE tested grid, with every distinct grid value from 0.1% to 2.0% winning at least one of the 11 windows outright. Mean pairwise Spearman rank correlation between windows' f-orderings is only 0.23 (median 0.32), and 29% of the 55 window pairs are NEGATIVELY correlated — meaning for roughly 3 in 10 window pairs, the f that wins in one is literally the worst (or near-worst) choice in the other. A correlation this weak means f-rank in one window carries almost no information about f-rank in the next; no f can be selected from this data and expected to hold. The single global f that maximizes mean ROI/DD across all windows (1.50%) still gives up a median of 0.57 and mean of 1.47 ROI/DD units versus each window's own (unknowable in advance) best choice — including two windows where it gives up essentially all the upside (2024-09-01_2025-03-01: 4.10; 2024-12-01_2025-06-01: 4.30). Even the best f is profitable in only 73% of the 11 windows — roughly 3 of 11 6-month periods lose money regardless of risk sizing, so risk-per-trade cannot rescue a period where the underlying edge itself failed; it can only scale the size of the win or loss that period already had. Bottom line: this data does not support picking any single risk-per-trade value as 'the' setting. The historically best-looking f is a function of which window you happened to measure, not a property of the strategy. Any process that re-tunes f on a recent window is fitting noise.
