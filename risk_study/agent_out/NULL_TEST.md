# NULL_TEST — Criteria 3 & 5 (independent implementation, AGENT-NULL)

Run against `risk_study/PROTOCOL_symbols.md`. A0 = live fitted per-(symbol,div_type)
`(rr, atr_mult)`, A1 = single global `(rr=10, atr_mult=3.0)`. Both use the live `s3_a1`
trailing exit. Cost model: net R = r − 0.00242 / stop_frac (24.2 bps).

## Pairing

- A0: 32,585 trades. A1: 31,784 trades.
- Paired on (symbol, div_type, entry_time): **31,766 rows** (99.9% of the
  smaller arm, 31,784).
- Match rate is healthy; proceeding without further diagnosis.
- Of those, **31,766 rows (100.0%)** are also present in all 8 built
  global (rr, atr_mult) arms — this smaller, fully-aligned set is what criterion 3 uses
  (every candidate global choice must have priced the same trade for the comparison to be
  apples-to-apples). Criterion 5 uses the full A0/A1 paired set (31,766 rows); it only
  concerns A0 vs A1, not the other 6 arms.
- Sanity check: A1's column inside the 8-arm matrix reproduces `a1v` from the A0/A1 merge
  to within 0.0e+00 (float noise) — the two data paths agree.

## Criterion 3 — beat a size-matched RANDOM null

### (a) Random per-pair assignment (2000 draws, seed 42)

For each of the 728 unique (symbol, div_type) pairs in the aligned trade set, draw
one of the 8 built `(rr, atr_mult)` combos uniformly at random, price every trade for that
pair under the drawn combo, and take the portfolio mean net R. Repeat 2000 times.

| | mean net R | percentile in random null | p (random ≥ this) |
|---|---|---|---|
| Random null itself | +0.0949 (sd 0.0054) | — | — |
| **A1 — global rr10/atr3.0 (the fitted global winner)** | +0.1908 | **100.00** | 0.0005 |
| **A0 — LIVE fitted per-symbol config** | +0.0620 | **0.00** | 1.0000 |

Random null 95% range: [+0.0845, +0.1057]

**Key comparison the task asked for:** A0 — the bot's *fitted* per-symbol assignment,
the thing the whole protocol exists to test — sits at the **0.0th percentile**
of 2,000 random-per-pair (rr, atr_mult) draws. A percentile this low means the fitted config is statistically indistinguishable from (or below) a coin-flip parameter assignment — direct evidence the per-symbol fitting is fitting noise, not signal.
A1 (the single global choice) lands at the **100.0th percentile** — comfortably above the random-assignment distribution.

### (b) Best-of-8 selection correction (2000 weekly-block bootstraps, seed 42)

On each resample, price all 8 global arms, take the best-performing one on that resample,
and compare its mean net R to A0's mean net R on the *same* resample (paired, so any
common regime shift cancels).

- Best-of-8 beats A0 in **100.0%** of 2000 resamples.
- Margin (best-of-8 − A0): mean +0.1282, 95% CI [+0.0564, +0.2006].
- 165 weekly blocks were resampled with replacement (block bootstrap, not i.i.d.)
  to respect the cross-symbol correlation in signal timing.

**Reading:** Even after explicitly simulating the 'pick the best of 8 after looking' procedure, the winner beats A0 in 100.0% of resamples with a 95% CI that stays entirely positive — so the advantage is not an artefact of having 8 shots at the null; it would have won the selection race on almost any historical draw.

### Criterion 3 verdict

**FAIL** — turns on: A0 (fitted) percentile in the
random null = **0.0** (protocol bar for "clearly beats random" ≈ 95th
percentile), and best-of-8-beats-A0 rate = **100.0%** with CI lower bound
**+0.0564**.

A0's fitted per-symbol assignment does NOT clear the random-assignment null — it performs like a random draw from the same 8-combo grid, which is the single strongest piece of evidence in this study that the per-symbol (rr, atr_mult) fitting captured noise, not a real, transferable edge. That the *global* winner A1 clears the null while the fitted A0 does not is the whole finding: one global knob, chosen honestly, beats 728 fitted knobs chosen by overfitting.

## Criterion 5 — advantage not concentrated in a handful of trades

Full A0/A1 paired set, n = 31,766. Baseline diff (A1 − A0) mean net R = +0.1287.

| % removed | A0 net R (trimmed) | A1 net R (trimmed) | both-trimmed diff | hostile diff (A1-only trimmed) |
|---|---|---|---|---|
| 0.0% | +0.0620 | +0.1908 | +0.1287 | +0.1287 |
| 0.5% | +0.0181 | +0.1433 | +0.1253 | +0.0813 |
| 1.0% | -0.0164 | +0.1112 | +0.1276 | +0.0491 |
| 2.0% | -0.0697 | +0.0612 | +0.1309 | -0.0008 |
| 5.0% | -0.2089 | -0.0617 | +0.1472 | -0.1238 |

"Both-trimmed" removes the top N trades by net R separately from each arm's own
distribution (the fair test — both arms lose their best trades). "Hostile" removes A1's
top N only and leaves A0's full, untouched distribution — the adversarial version the task
asked for, designed to make A1 look as bad as possible.

**Where the advantage disappears:**
- Both-arms-trimmed: does not disappear through 5% removal.
- Hostile (A1-only trimmed): disappears at 2.0% removed.

### Concentration — share of each arm's total positive net R from its own best trades

| arm | trades | total positive net R | top 1% share | top 5% share |
|---|---|---|---|---|
| A0 | 31,766 | 28,331.3 | 8.8% | 29.2% |
| A1 | 31,766 | 30,075.0 | 8.5% | 26.3% |

### Criterion 5 verdict

**FAIL** — turns on whether the A1-over-A0 net-R diff stays
positive after removing the top 0.5/1/2/5% of trades in both the both-trimmed and hostile
tests. It breaks down at 2.0% removed in the hostile test — the advantage IS meaningfully concentrated in a subset of trades and should be treated with more caution than the headline number suggests.

## Bottom line

| criterion | verdict | number it turns on |
|---|---|---|
| 3 — beats random null | **FAIL** | A0 (fitted) sits at the 0.0th percentile of 2000 random per-pair draws (need ≥95th); best-of-8 beats A0 in 100.0% of resamples, margin CI [+0.0564, +0.2006] |
| 5 — not concentrated | **FAIL** | A1−A0 diff after 5% both-trimmed removal = +0.1472; after 5% hostile (A1-only) removal = -0.1238 |

See the numbers above — the picture is mixed and should be read criterion-by-criterion rather than summarized as a single verdict.
