# The short-side tail: what causes it, and what causally cuts it

Data: `risk_study/uni_glob_rr10_am3_same.parquet` (32,102 trades, global rr=10/atr_mult=3.0
+ trailing stop, 2023-06-01 → 2026-07-25). Cost model applied throughout:
`net_r = r_result - 0.00242/stop_frac`. Split at **2026-05-25**: TRAIN = 29,722 trades
(explore freely here), HOLDOUT = 2,380 trades (touched only to score frozen rules).
All rules below are computed from information available strictly before the entry bar.

Working files: `build_features.py` (feature construction, verified causal), `lib.py`
(shared helpers), `00_baseline.py` … `12_final_assemble.py` (numbered analysis steps, run
in order), `trades_enriched.parquet` (cached feature table), `short_results.csv` (full
results table).

## 1. Confirming the diagnosis

Worst 10 exit-days in TRAIN: **-1,004.1 net R total**, of which **short trades contributed
-1,025.2 R and longs contributed +21.1 R** — matches the brief almost exactly. These are
genuine correlated events: the two worst days alone (2025-06-23, 2026-02-25) resolved 162
and 140 short trades respectively. HOLDOUT reproduces the same shape at smaller scale
(worst 10 days = -429.9 R, of which shorts = -381.5 R).

**What actually happens on the worst days**, checked against BTC 1H bars directly
(`06_case_study.py`): the two catastrophic days were BTC rallying **+6.2% in about a day**,
against a backdrop where BTC's trailing-30-day return going in was strongly **negative**
(median -6.8% on the 10 worst days vs +0.2% on other days, Mann-Whitney p=3e-71). This is
the textbook short squeeze: the strategy's short density is highest in and after a
decline (bear-trend divergence signals cluster there), and a sharp relief rally catches
a large, correlated short book at once. But a plain "market has been declining" state is
also the bot's single most productive short regime overall — it can't be gated away
without giving back most of the profit (see §2).

## 2. The existing control (`btc_short_gate`) — verified, and it is not doing its job

Simulated by dropping SHORT rows where `btc_impulse` (ret_30d > +10%, shipped) is true.

| | TRAIN total R | TRAIN worst-10-days | HOLDOUT total R | HOLDOUT worst-10-days |
|---|---|---|---|---|
| gate ON (current live) | 5,720.1 | -964.7 | -275.0 | -429.9 |
| gate OFF | **6,425.9** | -1,004.1 | -275.0 | -429.9 |

Confirmed: gate OFF beats gate ON by **+705.8 R in-sample**, and the gate barely touches
the tail it was built for (-964.7 vs -1,004.1, a 4% change against a 71% of one day's
loss). In the holdout window the gate never fired at all (BTC never ran +10%/30d in
Jun–Jul 2026) — identical numbers ON/OFF. **The shipped gate is dead weight in the current
regime and was costing ~700R historically for essentially no tail protection.** All rule
development below uses gate-OFF as the baseline to beat, per the brief.

## 3. What was tried and rejected

**Family 1 — better BTC state variable.** Swept 30d/7d/14d/60d BTC return, realized vol
(14d/30d), distance from 200-EMA, drawdown-from-high, at ~10-20 thresholds each
(`01_family1_btc_state.py`, `family1_sweep_train.csv`). Aggressive drawdown-based gates
(skip shorts when BTC is >15-30% off its high) cut the tail by 250-320R but destroy
40-85% of total return doing it — they gate away the bot's core bear-market short regime,
not just the squeeze days inside it. Mild realized-vol gates (skip when trailing-14/30d
vol is in the top ~15%) are cheap but only nibble the tail (+9 to +70R of ~1,004R, i.e.
under 7%). **No single BTC-level state variable cleanly separates squeeze days from
ordinary profitable short days** — both occur during BTC declines/high-vol regimes, and
the 30-day-return threshold the bot already ships is about the best of this family, which
is to say not very good.

**Family 2/4 (naive form) — crowding as a blunt cutoff.** Trailing entry-count and
open-short-concurrency are strongly, significantly elevated on tail days (Mann-Whitney
p<1e-100 for `short_concurrency`, `density_short_72h`) — but a moderate threshold on
either is a *terrible* rule. Reason (`05_daily_density_vs_r.py`): correlation between a
day's short trade count and that day's short R is **positive** (+0.32 to +0.48), because
busy days are more often jackpot days than disaster days (e.g. 2025-10-10 fired 160 short
exits worth **+1,136R**; only ~2 of the 15 busiest days in TRAIN are actually bad). A
concurrency cap at 20-100 or a moderate density cutoff cuts the tail only by throwing away
30-70% of total return, because it's a proxy for "the bot is busy," which correlates with
both tails, not just the left one. `short_concurrency` in particular is confounded with
calendar time (r=0.52 with trade age) — it's largely just "the book got bigger," not a
risk signal.

**Family 4 — hard concurrency cap.** Same problem, worse: caps of 10-30 destroy
70-85% of total R for a proportional (not disproportionate) tail cut; in the HOLDOUT
window concurrency is almost always above any cap in the 10-30 range (the book has grown),
so a cap this tight nearly stops short-side trading altogether there. **Rejected.**

**Family 3 — uniform side-asymmetric sizing (always).** Scaling every short's R by
0.5/0.65/0.8 is linear and non-selective: it buys tail reduction exactly proportional to
the return given up (0.8x → -1,109.6R total for +193.2R tail improvement in TRAIN — the
same ratio as just trading a smaller book). It is a capital-allocation decision, not a
causal risk control, and it's dominated by the conditional version below.

## 4. What worked: a *universe-wide* density throttle, not per-symbol crowding

The refinement that flips density from "confounded" to "useful": look at the **extreme
upper tail of the density distribution**, and use the **whole-universe** entry count
(both sides), not just the short-side count. `density_all_72h` = number of entries (any
symbol, any side) in the trailing 72 hours, fully causal.

Fine sweep (`08_add_windows_and_finesweep.py`) shows a sharp knee: below roughly the 90th
percentile, density carries no exploitable signal (matches Family 2 above); above it —
i.e. only the most extreme, rare, all-symbols-firing-at-once moments — skipping shorts
becomes nearly free and disproportionately removes tail risk.

**Recommended rule: skip a SHORT entry when `density_all_72h > 160`** (~92nd percentile;
fires on 8.9% of TRAIN shorts, 10.6% of HOLDOUT shorts — a rare-event gate, not a
day-to-day throttle).

| period | | n trades | mean net R/trade | total net R | worst-10-days R |
|---|---|---|---|---|---|
| TRAIN | baseline (gate-off) | 29,722 | 0.2162 | 6,425.9 | -1,004.1 |
| TRAIN | **rule applied** | 28,038 | 0.2245 | **6,294.5** | **-811.0** |
| Δ | | -1,684 skipped | +0.008 | **-131.4** (-2.0%) | **+193.2** (-19.2% of tail) |
| HOLDOUT | baseline (gate-off) | 2,380 | -0.1155 | -275.0 | -429.9 |
| HOLDOUT | **rule applied** | 2,195 | -0.1121 | **-246.1** | **-408.7** |
| Δ | | -185 skipped | +0.003 | **+28.9** (improvement) | **+21.2** (-4.9% of tail) |

Both metrics move the **same direction in both periods** — total R roughly flat-to-better,
worst-10-days tail smaller — for a threshold chosen on TRAIN and then frozen. The
threshold is not a knife-edge: 140/150/160/170/180 all show the same sign pattern in both
periods (`candidate_eval.csv`), which is the main defense against this being a lucky cut.

**Mechanism check** (`10_bootstrap_and_diagnostics.py`): applying the rule to the exact
original 10 worst TRAIN days removes -288.5R specifically from *those* days (short R on
those days goes from -1,025.2 to -736.7) — i.e. the rule is doing what it claims causally,
not just reshuffling which days are worst.

**Weekly block bootstrap (1,000 resamples, `stat_total_r` / `stat_worst10`)**:

| period | metric | baseline mean [95% CI] | rule mean [95% CI] |
|---|---|---|---|
| TRAIN | total R | 6,466 [2,436, 10,831] | 6,343 [2,603, 10,221] |
| TRAIN | worst-10 | -1,668 [-2,152, -1,245] | -1,472 [-1,889, -1,127] |
| HOLDOUT | total R | -268 [-1,055, 627] | -235 [-988, 550] |
| HOLDOUT | worst-10 | -656 [-906, -469] | -616 [-871, -426] |

The tail CIs shift the right direction and don't cross into "clearly worse," but they
overlap the baseline substantially — the point estimate (~15-19% tail cut in TRAIN, ~5-9%
in the much smaller HOLDOUT window) is real but should be read as **modest, not dramatic,
and better established in TRAIN than HOLDOUT** (HOLDOUT is only ~2 months / 61 days, so
"worst 10 days" there is ~16% of all days, not a rare tail — noisier by construction).

**A softer dial exists** (`family3b` in the CSV): instead of a hard skip, scale short size
by a factor only when `density_all_72h>160`. Half-sizing (factor 0.5) keeps ~72% of the
tail benefit (+140.6R of the +193.2R in TRAIN) for about half the cost (-65.7R vs -131.4R).
This is a strictly worse Sharpe-style trade than the hard skip at this particular
threshold (the marginal R given up buys less marginal tail reduction as factor→1), so the
hard skip (factor=0) is the recommended operating point; the dial is there if a smoother
rollout is preferred operationally.

**Stacking** density with the best Family-1 gate (`rvol_30d_causal>0.0298`, OR'd together)
was also tried: it adds cost (-386.0R vs -131.4R in TRAIN) for essentially no extra tail
benefit (+197.9R vs +193.2R). Not worth the complexity — **density alone is the whole
effect**, the BTC-state gates aren't adding an independent signal on top of it.

## 5. Verdict

- The shipped `btc_short_gate` (BTC 30d-return > 10%) should be **turned off** — it costs
  real return and does not address the measured tail (confirms the brief's suspicion).
- **No BTC-price-level state variable (return over any horizon, realized vol, distance
  from 200-EMA, drawdown-from-high) cleanly separates the catastrophic short days from
  the ordinary profitable ones.** The squeeze days occur inside the bot's best short
  regime (post-decline), not identifiably before it starts.
- **Naive crowding measures (short-side density, open-short concurrency) are actively
  misleading** — busy days are disproportionately the bot's *best* days, not its worst;
  gating on them destroys far more return than tail.
- **One rule survives causally on both sides of the split with the same sign**: skip
  shorts when *universe-wide* (not short-only) trailing-72h entry count exceeds ~160 (a
  rare, extreme, all-symbols-at-once event, ~9-11% of shorts). It cuts the worst-10-day
  tail by roughly 15-20% in-sample and ~5-9% out-of-sample, at a total-return cost that is
  negative-to-zero in both periods (net *positive* in the untouched HOLDOUT). Effect size
  is real but moderate — this trims the tail, it does not eliminate it. Bootstrap CIs
  overlap baseline, so treat this as a genuine-but-modest, well-supported improvement,
  not a solved problem.
- Recommended action: replace `btc_short_gate` with a `density_gate`: skip new SHORT
  entries when the trailing 72h count of new entries across the whole 277-symbol universe
  exceeds 160. Implementation is a single rolling counter, no BTC klines required, and it
  is strictly causal (only counts entries that have already happened).
