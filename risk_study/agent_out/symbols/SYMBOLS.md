# Symbol / pair selection — should the universe change, and how often?

Agent: AGENT-SYMBOLS. Data: `risk_study/grid_inuni.parquet` (420,953 rows), CHOP filter
`chop_bos<52` applied, last 21 days of entries trimmed (→ 411,266 rows, then 183,096 after
CHOP), cost model `net R = r_<rr> − 0.00242/stop_frac`. All CIs are weekly block bootstraps
(resample ISO weeks with replacement, 1,500–2,000 draws), because trades cluster in
time and within-week trades are not independent draws.

**Bottom line up front:** the universe should **not** be meaningfully changed on the
evidence here. Symbol-level characteristics (liquidity, stop-tightness) carry no
detectable OOS signal in this data. The one thing that *does* show a robust, repeated
OOS gap is **which divergence types a symbol trades** (the pair-level selection baked
into `config.yaml`) — but that gap looks structural/persistent rather than something a
naive "chase trailing R" refit can rediscover, and a naive refit does not improve on
"never touch it." Refit cadence should stay low (the existing 90-day freeze
recommendation from `OVERFIT_VERDICT.md` is not contradicted by anything here — if
anything it's reinforced). Delisting hazard is real and currently unhandled by the bot
— see §4, but the exact number is genuinely uncertain and I give a range, not a point
estimate.

---

## 0. The survivorship-bias check (done first, as instructed)

Verified directly against the two cache directories, not inferred:

- `cache_3yr_1h/`: 522 parquet files, **509 readable** (13 unreadable — mostly dated
  futures contracts like `ETHUSDT-03APR26.parquet`, not perpetuals, plus a handful of
  corrupt/empty files).
- Of the 509 readable: **277 are the live-enabled symbols** in `config.yaml`, leaving
  **232 non-live**.
- `cache_outside/`: 43 files — these are the 43 non-live symbols a **fresh Bybit pull**
  (as of this study, run separately) confirmed are still listed today.
- 232 − 43 = **189 delisted** (task description said 188; off by one, immaterial —
  confirms the same picture independently).

**So: 232/509 = 45.6% of the readable historical cache is non-live, and 189/232 = 81.5%
of that non-live pool is now delisted.** Any test that asks "should we add more of the
509-symbol cache to the tradeable universe" is contaminated: 81% of the candidates you'd
be adding are things you can no longer trade, and the 43 you still could trade were
presumably passed over by the existing symbol-selection process for a reason (lower
liquidity, shorter history, etc. — not tested here).

**Bias direction:** this survivorship bias only threatens Q1/Q2 if I tried to answer
"should we add non-live symbols." I did not — every test below is done **within the
275–277 already-live symbols** (config membership, or symbol/pair characteristics of the
live set), which sidesteps the delisted-candidate problem entirely. Q4 (delisting hazard)
is the one place the biased pool is the *actual subject* of the estimate, and that
section's honesty requirements are handled separately (§4) with its own caveats.

---

## 1. Pair selection (728 selected vs ~380 rejected vs all 1,108)

**Setup.** From `config.yaml`, 277 enabled symbols carry 728 enabled `(symbol,
divergence_type)` pairs out of 277×4 = 1,108 possible. In the trade data, 724 of those
728 have at least one signal (4 never fired); the pairs present in data but *not*
selected number 376 (task estimated "~380" — matches). Critically: **rejection is almost
entirely within-symbol, not symbol-level** — only 60/277 live symbols have all 4
divergence types enabled; 39 have just 1. So "rejected" mostly means "this symbol trades
fine, but its `REG_BULL` config in particular was judged not worth it," not "this symbol
is bad." That matters for interpreting §2 (liquidity/cost effects are symbol-level and
therefore can't explain the pair-selection gap below).

**Grid-average pair quality metric.** Rather than use each selected pair's own
(possibly overfit) `rr`/`atr_mult`, I built a parameter-agnostic quality score:
mean net R per pair averaged equally across all 20 grid cells (4 `atr_mult` × 5 `rr`).
This isolates "is this pair inherently OK" from "did we pick the best knobs for it,"
and lets rejected pairs (which have no live `rr`/`atr_mult` of their own) be scored on
the same footing as selected ones.

### 1a. Does the actual production selection (config.yaml) transfer OOS?

Split at **t = 2025-11-01** (matches the documented walk-forward's own train/test
boundary in CLAUDE.md §13 — "train < 2025-11-01, test through 2026-05-25"):

| Group | Basis | n trades | mean net R/trade | 95% CI |
|---|---|---|---|---|
| Selected 728, **own live settings** | as actually traded | 10,267 | **+0.157** | [−0.080, +0.412] |
| Selected 728, **grid-avg (apples-to-apples)** | unoptimized | — | **−0.015** | — |
| Rejected 376, grid-avg | unoptimized | 17,384 | **−0.347** | — |
| All 1,100, grid-avg | unoptimized | 58,320 | −0.128 | — |

Read this as two separable effects:

1. **Which-pairs selection, on its own (apples-to-apples, unoptimized settings):**
   selected pairs score −0.015/trade vs. rejected's −0.347/trade OOS. That's a genuine,
   large gap that has **nothing to do with parameter tuning** — it persists even when
   both groups are scored with the same unoptimized grid-average. This gap **repeats**
   at two earlier, independent train/test splits (2024-07-01 and 2025-01-01 — see
   table below), so it isn't a fluke of one boundary date.
2. **Per-pair `rr`/`atr_mult` optimization on top** (going from −0.015 grid-avg to
   +0.157 at the pair's actual live settings): a further +0.17R/trade. This is *not*
   fresh evidence of transfer — the 2025-11-01→2026-05-25 window is the walk-forward's
   own designated **test** window, i.e. the config's `rr`/`atr_mult` choices were already
   validated (not blind-held-out) against most of this same period. Treat the +0.157 as
   a re-confirmation of the existing fit, not new proof.

**Robustness of the which-pairs effect across independent splits** (grid-avg,
apples-to-apples, `prod_selected` = the current 728 vs `prod_rejected` = the current 376,
scored purely on data after each split date):

| split t | selected (grid-avg) | rejected (grid-avg) | gap |
|---|---|---|---|
| 2024-07-01 | +0.105 | −0.201 | +0.306 |
| 2025-01-01 | +0.104 | −0.241 | +0.345 |
| 2025-11-01 | −0.015 | −0.347 | +0.332 |

The gap (≈0.3–0.35R/trade) is remarkably stable across three non-overlapping splits.
**This is the one selection signal in this study that looks structurally real rather
than noise-fit** — whatever process produced `config.yaml`'s divergence-type choices
per symbol identified something that keeps being true out-of-sample, years later, on
data it never saw.

**But a naive "rank by trailing average R, take the top 728" re-derivation does *not*
reproduce this.** Doing exactly that at t=2025-11-01: my own top-728 scores −0.106
OOS, my bottom-372 scores −0.167 — barely different, and worse than even the config's
rejected group. Train-vs-test pair-quality correlation is **effectively zero**
(Pearson 0.03–0.11, Spearman 0.03–0.08 across all three splits; deciles of trailing
quality show no monotonic relationship with OOS quality — see
`_q1_decile_transfer.csv`). Overlap between my naive top-728 and the actual config
selection is only 539/728 (74%). **Conclusion: simple performance-chasing does not
identify future winners in this data — whatever the real selection process used
(presumably combined with additional structure: minimum-sample filters, per-pair
`rr`/`atr_mult` grid search, or other criteria not reproduced here) is doing something a
naive refit cannot replicate.** This is consistent with the existing
`selection-transfer-measured` memory note (rho 0.03–0.06 for per-symbol RR/ATR).

**Genuinely fresh check** (entries after 2026-05-25, the one window neither this study
nor the original walk-forward has touched before): only 6 weeks, 1,736 live-config
trades. Selected mean net R = **−0.257** [CI −0.650, +0.196], rejected (grid-avg)
**−0.369**. Both negative — consistent with the known "edge negative OOS since
May 2026" — but the selected group is still directionally less bad, and the sample is
far too small (6 weeks) to be decisive either way.

**Answer to Q1:** selecting *which divergence types per symbol* to trade beats trading
everything, and that edge looks real (stable across 3 independent splits, ~0.3R/trade).
But that edge is **not** recoverable by simply re-ranking on trailing average R — a
naive refit is noise. Whatever produced the current selection should not be casually
re-run with a simpler heuristic; if it's re-derived, it needs to replicate whatever
structure (sample-size filters, joint parameter search, etc.) made the original
selection work, not just "top-N by trailing R."

---

## 2. Non-performance (characteristic-based) selection

All tests below hold the "which pairs" question fixed at the current 728 live pairs and
their live settings, and ask: should we further restrict *within* that set by a
symbol characteristic? Train/test split at t=2025-11-01, symbol-level characteristic
computed on train only, performance measured on test only.

### 2a. Liquidity (`liq_30d`)

| bucket | symbols | trades | mean net R/trade | 95% CI |
|---|---|---|---|---|
| top 25% by liquidity | 68 | 2,335 | +0.108 | [−0.165, +0.421] |
| top 50% | 137 | 4,876 | +0.163 | [−0.068, +0.436] |
| top 100% (all) | 275 | 10,267 | +0.165 | [−0.050, +0.409] |
| bottom 50% | 138 | 5,391 | +0.162 | [−0.056, +0.393] |
| bottom 25% | 69 | 2,775 | +0.146 | [−0.106, +0.402] |

Train-liquidity vs test-net-R Spearman correlation across all 275 symbols: **−0.058**
(essentially zero, slightly negative if anything). **No liquidity signal within the
live universe.** This is expected — the 277 live symbols already passed a liquidity
screen at the `symbol_rr_mapping.py` stage; further ranking within an already-screened
set has nothing left to find.

### 2b. Volatility / cost (`stop_frac`)

Cost in R is mechanically `0.00242/stop_frac`, so tight-stop symbols are expensive by
construction. Question: does excluding them help, and is any gain just the mechanical
cost effect?

| bucket | symbols | trades | mean **net** R | mean **gross** R |
|---|---|---|---|---|
| all | 275 | 10,267 | +0.165 | +0.341 |
| excl. tightest 10% | 248 | 9,127 | +0.173 | +0.320 |
| excl. tightest 25% | 207 | 7,516 | +0.148 | +0.297 |
| **only the tightest 25%** | 68 | 2,751 | **+0.187** | **+0.445** |

**Excluding tight-stop symbols does not help — if anything it's slightly worse**, and
the direction is counter-intuitive: the tightest-stop quartile (mechanically the most
expensive per trade) has both the highest gross R *and* the highest net R of any
bucket. That means these symbols have enough extra raw edge to more than pay for their
higher mechanical cost. **This is the opposite of what a naive "cut the expensive
symbols" rule would predict — do not add a stop-tightness filter.** (Note: differences
across buckets are all within noise given the CIs on §1's headline numbers — the honest
read is "no detectable effect either way," not "tight-stop symbols are secretly
better," but there is certainly no case for excluding them.)

### 2c. Breadth (fewer symbols vs all 277)

Nested top-N by liquidity, same live pairs/settings, test period:

| top N symbols | trades | mean net R/trade | total net R (sum, proxy only) |
|---|---|---|---|
| 277 (all) | 10,267 | 0.167 | 1,694 |
| 200 | 7,225 | 0.171 | 1,214 |
| 150 | 5,430 | 0.168 | 922 |
| 100 | 3,537 | 0.137 | 480 |
| 50 | 1,654 | 0.121 | 199 |
| 25 | 817 | **0.022** | 17 |

Per-trade quality is flat down to ~150 symbols, then degrades noticeably below 100, and
collapses at 25 (mean R/trade near zero, CI mostly negative). Total-R (no portfolio
engine, so treat this only as a rough scale proxy, not a compounding/DD estimate) falls
off faster than trade count as breadth narrows. **No evidence that concentrating into
fewer symbols helps; spraying across the full live universe is at least as good
per-trade and captures far more total opportunity.** This doesn't test the
diversification/drawdown benefit of breadth (that needs the portfolio engine this study
was told to skip), but on the R-level evidence alone there's no reason to narrow.

**Answer to Q2:** neither liquidity nor stop-tightness carries usable OOS signal within
the already-live 277-symbol set — both are already screened for at the symbol-selection
stage upstream. Narrowing breadth has no R-level upside and a clear downside once you go
below ~100 symbols. None of the three characteristic-based rules should be added.

---

## 3. Refit cadence

Simulated an expanding-window walk-forward: at each refit date, rank all 1,100 pairs by
**trailing** grid-avg net R (same parameter-agnostic metric as §1) using all data before
that date, select the top 728 (matching the current universe size), trade that set until
the next refit, concatenate OOS segments. First refit 2024-06-01 (after ~12 months of
training data), through the trimmed end of data (2026-07-04).

| cadence | # refits | trades captured | mean net R/trade | 95% CI | avg. churn/refit |
|---|---|---|---|---|---|
| never (freeze after 1st selection) | 1 | 85,526 | 0.047 | [−0.091, +0.201] | 0% |
| yearly | 3 | 104,976 | 0.044 | [−0.076, +0.183] | **40%** |
| 6-monthly | 5 | 112,017 | 0.050 | [−0.077, +0.194] | 28% |
| quarterly | 9 | 113,258 | 0.051 | [−0.071, +0.175] | 19% |

**Per-trade quality is statistically indistinguishable across all four cadences** — the
CIs overlap almost completely, and the point estimates (0.044–0.051) move by less than
noise. Trade counts and total-R rise with cadence, but that's an artefact of the
selection tracking a larger/growing pool over time as more history accumulates (more
frequent refits pick up more distinct pairs across the whole span), not evidence that
faster refitting finds better trades — consistent with §1's finding that trailing-R
ranking has ~zero OOS predictive power on its own.

**Churn is high and buys nothing measurable:** yearly refit changes ~40% of the 728-pair
set each time, quarterly ~19% — substantial turnover for a cadence that performs the
same, within noise, as never refitting at all. **This is exactly the "high churn, no OOS
benefit → noise-fitting" pattern the task asked to watch for.**

**Answer to Q3:** given a naive trailing-R selection rule, there is no cadence that beats
"never." This doesn't prove no refit cadence could ever help — it proves that *this*
selection mechanism (simple average-R ranking) shouldn't be run on any cadence, fast or
slow, because it isn't finding real signal to refresh. Combined with §1's finding that
the actual `config.yaml` selection embodies something more structural than trailing-R
ranking, the right takeaway is: don't refit on a schedule using this heuristic. This
does not contradict, and mildly reinforces, `OVERFIT_VERDICT.md`'s standing
recommendation to freeze the current selection for a full quarter and collect a genuine
250+-trade holdout before touching it again.

---

## 4. Delisting hazard (no backtest prices this — order-of-magnitude estimate only)

**Data.** 232 non-live symbols with readable cache; exposure time per symbol = its own
observed cache span (`max(start) − min(start)`), event = "not found in the fresh
`cache_outside` pull" (i.e., delisted). Total exposure: 310.7 symbol-years. 189 events.

**Naive pooled hazard (all 232, exponential MLE):** λ = 189/310.7 = **0.61/symbol-year**
→ implied annual survival 54%, 3-yr survival 16% (vs. the observed 3-yr delisted
fraction of 81.5% — consistent). Applied naively to a 277-symbol book: **~168
delistings/year.**

**This number is almost certainly a large overestimate for the currently-live 277.**
Two reasons, both checked:

1. **Median exposure in the non-live pool is only 0.81 years** — most of these symbols
   were short-lived, low-quality listings that likely never would have passed the bot's
   own liquidity/history screen in the first place. Restricting to symbols that already
   survived ≥1 year before the hazard clock starts still gives a high conditional hazard
   (0.46/symbol-year on 135.5 remaining symbol-years of exposure), and even ≥2 years
   gives 0.39/symbol-year — so tenure alone doesn't fully explain the gap, but the
   population is clearly skewed toward marginal names.
2. **A closer analogue exists in CLAUDE.md itself:** of the *already-live-quality*
   symbol pool, only **11** are noted as disabled specifically for having been delisted
   (`§11`, "11 symbols disabled with no comment ... delisted"). That's roughly
   11/288 ≈ 3.8% cumulative over the bot's operating history — an order of magnitude
   below the raw non-live-pool estimate. This is the better analogue for "a symbol that
   already made the live cut," even though I don't have a precise exposure window for
   those 11 to convert it into a clean annual rate.

**Honest range:** the raw pooled estimate (~0.6/symbol-year, ~168 delistings/year in a
277-book) is a defensible **upper bound** — it's the rate for "a random symbol that was
ever tradeable on Bybit perps in this cache," which is a materially riskier population
than "a symbol that passed the bot's current liquidity/history screen." The live-book's
own limited track record (11/288 delistings) suggests the true forward rate for
already-screened symbols is far lower — plausibly **single-digit to low-double-digit
percent per year**, which for a 277-symbol book would mean **very roughly 10–30 forced
closures/stranded positions per year** (a few per month), not hundreds. I cannot narrow
this further without a proper survival model of the live-quality population specifically
(which this repo doesn't have the data to build), so I'm reporting a range rather than a
false point estimate.

**Operational implication regardless of the exact number:** CLAUDE.md's dead-code and
open-issues sections show **no special handling exists** for a symbol getting delisted
mid-position (`storage.py`, `bot.py` — nothing scans for "position holder's symbol no
longer exists"). Even the conservative end of this range (a handful of events a month)
means this is a live, currently-unmonitored operational risk, not a hypothetical one.

---

## Summary answer

- **Should the universe (which symbols/pairs) change?** Not on characteristics
  (liquidity, stop-tightness) — neither shows any OOS signal within the already-screened
  277-symbol live set, and excluding tight-stop symbols would remove the
  best-performing bucket, not the worst. The existing *pair-level* (divergence-type)
  selection embedded in `config.yaml` does show a real, repeated OOS edge over trading
  everything (~0.3R/trade gap, stable across three independent splits) — so the
  selection that already exists should be **kept**, not discarded in favor of trading
  all 1,108 pairs.
- **Performance-based or characteristic-based selection?** Neither a naive
  performance-based refit (rank by trailing average R) nor the characteristic-based
  rules tested (liquidity, stop-tightness, breadth-narrowing) add anything on top of
  what's already in `config.yaml`. The one selection signal that transfers is whatever
  produced the *current* config — and it is not reproducible by the simple heuristics
  tested here.
- **How often revisited?** Not on a fixed cadence using trailing-R ranking — quarterly,
  6-monthly, and yearly refits all perform identically (within noise) to never
  refitting, while yearly refit alone churns ~40% of the universe per event for zero
  measurable gain. This reinforces, rather than contradicts, the standing
  `OVERFIT_VERDICT.md` recommendation to freeze the current selection and collect a
  genuine 250+-trade fresh holdout (data since the last freeze is thin — only ~6 weeks,
  1,736 trades, past the walk-forward's own test window) before any re-tune.
- **Delisting** is a real, currently-unhandled operational risk of plausibly
  low-tens-per-year magnitude for a 277-symbol book — not priced anywhere in this or any
  other backtest in the repo, and worth separate engineering attention independent of
  the selection question.
