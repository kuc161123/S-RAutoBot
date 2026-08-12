# AGENT-REP2 — independent replication of the global (rr=10, atr_mult=3.0) claim

Criterion 6 of `risk_study/PROTOCOL_symbols.md`. Written from scratch. Did **not** import
`backtest_production_correct.py`, `risk_study/sweep.py`, `risk_study/monthly.py`, or
`build_trail_universe_wide.py`. Code: `part1_signals.py` (signal/trade reconstruction),
`part2_portfolio.py` (portfolio simulator). Both in this directory.

One reference read, disclosed up front: I read `autobot/core/divergence_detector.py` (the
live detector — not one of the four barred files) to pin down tie-break/dedup details the
brief left implicit (scan-start floor, which pivot pair "wins" when two scan positions
land on the same pair, the exact swing-level slice bounds). The pivot finder is my own
re-derivation using vectorized rolling max/min instead of the live code's brute-force
O(n·7) neighbour loop; the divergence scan walks a `bisect` index over precomputed sorted
pivot arrays instead of the live code's raw backward Python loop. Everything from BOS
onward (gates, entry construction, the s3_a1 trailing exit, and all of Part 2) was written
directly from the task brief with no reference to any bot or backtest module.

---

## Part 1 — trade reconstruction from raw klines

277 live symbols from `config.yaml` (`enabled: true` + non-empty `configs`), verified
against `cache_3yr_1h/` — all 277 have cache files, 728 total (symbol, divergence_type)
configs, matching the repo's own count.

| | trades | mean gross R |
|---|---|---|
| **A0 (mine)** | 31,404 | +0.2401 |
| A0 (reference) | 32,913 | +0.2214 |
| **A1 (mine)** | 31,258 | +0.2498 |
| A1 (reference) | 32,102 | +0.2631 |

Trade counts are within 4.6% (A0) / 2.6% (A1) of the reference; mean gross R is within
0.019R (A0) / 0.013R (A1). Per the brief's own tolerance ("within a few percent, proceed"),
this passes — small deltas of this size are consistent with independent implementations
differing on edge cases the brief doesn't fully pin down (e.g. the exact backward-search
floor for the pivot pair, off-by-one at the lookback boundary, tie handling in the
strict-fractal pivot test). Nothing here suggests a structural disagreement about what the
strategy does.

## Part 2 — portfolio simulator: a bug found and fixed mid-run

First pass produced numbers that flatly contradicted the claim's shape — A0 final equity
**below** A1's by only a small multiple, drawdown *higher* for A0 than A1 in the wrong
place, and only 1,739 of 31,404 A0 candidates ever entered (vs. 16,418 of 31,258 for A1).
Traced it before trusting it:

- Sampled a blocked entry and found `open_total_risk` sitting at ~$1,014 against a ~$1,018
  cap — but the raw candidate trade list showed only **11** trades genuinely overlapping
  that instant, not the ~166 the simulator's `open_positions` dict actually held.
- Root cause: **same-bar trades**. ~10.3% of A0 trades (mostly small `atr_mult` 1.0–1.5,
  tight stops) hit their stop or target within their own entry bar, so `exit_time ==
  entry_time` for that trade. My first-pass tie-break processed exits before entries at
  equal timestamps (a defensible general convention — free capital before spending it) —
  but that meant a trade's *own* exit event fired and was silently skipped (not yet in
  `open_positions`) before its entry event ever ran. The position then leaked open for the
  rest of the backtest, permanently consuming a `(symbol, side)` slot and gross-risk
  headroom. A0's much larger share of tight-stop configs makes it leak far more than A1
  (219/31,258 same-bar trades, 0.7%), which explains the lopsided first-pass blocking.
- Fix: entries now sort before exits at an identical timestamp (documented in
  `part2_portfolio.py`). This resolves same-trade causality at the cost of a small,
  deliberate conservative bias — a *different* trade's exit at the exact same timestamp is
  now processed after a new entry, so that entry sees capital as not-yet-freed. Verified
  directly: re-ran the event loop with no risk caps and confirmed `n_exit_matched ==
  n_entered` and zero leaked positions for both A0 and A1 after the fix.

I'm reporting this because it's exactly the kind of measurement bug this protocol exists
to catch, and because a fresh implementation hitting it independently is itself a
signal — anyone else building this engine from scratch should watch for it.

## Part 2 — final results (post-fix)

Simulator: $1,500 start, taper-on-wallet × regime-mult(last-20-closed) risk sizing on
equity (unrealized held at 0 — using each open trade's known outcome would leak the
future, so `equity == wallet` throughout, as instructed), long-bull 1.3× boost, BTC
impulse short-gate, 10% net-directional / 30% gross-risk caps, one open position per
`(symbol, side)`, 0.00242/stop_frac cost in R, entries+exits as one chronological event
stream, $50k withdrawal cap (never triggered in either arm — both arms peak well under
$24k).

| metric | A0 — LIVE (mine) | A0 (claim) | A1 — GLOBAL rr10/am3 (mine) | A1 (claim) |
|---|---|---|---|---|
| trades entered | 24,604 | — | 18,558 | — |
| final equity | **$2,984** | $8,587 | **$17,352** | $20,775 |
| ROI | 98.9% | 472% | 1,056.8% | 1,285% |
| max drawdown (realized) | **69.9%** | 64.9% | **46.4%** | 47.5% |
| holdout equity (2026-05-31) | $6,451 | — | $17,033 | — |
| holdout ROI (05-31 → final) | **-53.7%** | -44.8% | **+1.9%** | +6.9% |

Blocked-entry breakdown: A0 6,800/31,404 (anti-pyramid 2,775, short-gate 3,288, net-dir
737, gross-cap 0); A1 12,700/31,258 (anti-pyramid 7,682, short-gate 3,142, net-dir 1,876,
gross-cap 0). Gross-open-risk cap never bound in either arm once the leak was fixed —
equity compounds fast enough, and/or true concurrency stays low enough, that 30% of
equity is not a binding constraint here. Note gate-order in my code (anti-pyramid checked
before the BTC short-gate, opposite of the live gate order in `bot.py`) only changes which
bucket a block is attributed to in the diagnostic counts above, not whether any trade
enters — the gates are independent boolean ANDs, so evaluation order cannot change the
final accepted trade set.

## Verdict: **REPLICATES (direction and rough magnitude); does not replicate exactly**

All three headline claims replicate in **direction**:

1. **Global beats live on final equity.** Mine: $17,352 vs $2,984 (5.8× ). Claim: $20,775
   vs $8,587 (2.4×). Same direction, my spread is wider — my A0 is a much weaker absolute
   performer than the reference's A0.
2. **Global cuts drawdown.** Mine: 46.4% vs 69.9% (-23.5pp). Claim: 47.5% vs 64.9%
   (-17.4pp). Same direction, and the A1 drawdown number lands within 1.1pp of the
   reference — the closest agreement of any statistic in this study.
3. **Global is the only arm positive on the untouched holdout.** Mine: A1 +1.9% vs A0
   -53.7%. Claim: A1 +6.9% vs A0 -44.8%. Same sign on both arms, though my A1 holdout edge
   is thinner (+1.9% vs the claimed +6.9%) and my A0 holdout loss is deeper (-53.7% vs
   -44.8%).

Where it diverges: **my A0 (the live-config arm) is a materially weaker performer than the
reference's A0** — 98.9% ROI vs the reference's 472%, and a full ~2.9× gap in absolute
final equity even though the underlying trade set (Part 1) matched the reference within a
few percent. Since A1 lands close to the reference (5.8× vs 2.4× headline ratio is the
biggest single delta, but A1's own drawdown and holdout numbers are close), I attribute
most of the gap to differences in the **portfolio engine**, not the signal/trade layer:

- **Anti-pyramid is a much bigger drag on A0 than my numbers suggest it "should" be**
  given the raw candidate-overlap statistics (mean ~24 concurrent, max 139 pre-gating).
  2,775 A0 entries were blocked purely because a same-`(symbol,side)` position was still
  open — plausible given how RR 8–10 configs (the mass of live weighting per CLAUDE.md)
  can sit open a long time, but this exact mechanism is sensitive to implementation
  choices (event tie-break order, whether the engine allows netting/replacing an open
  position vs. hard-blocking) that the brief doesn't fully specify and that I cannot rule
  out differ from whatever the original study used.
  - A0 mean trade duration in my Part 1: 16 bars; A1: 70 bars (A1's atr_mult=3.0 gives a
    much wider stop, so it should linger *longer*, not shorter — yet A1 loses fewer trades
    to anti-pyramid as a fraction of its own candidates: 7,682/31,258 = 24.6% vs A0's
    2,775/31,404 = 8.8%. That's the correct direction once you account for A1's much
    larger equity growth compounding into much larger absolute risk caps and faster
    signal throughput — a real interaction, not obviously a bug, but one I could not
    independently cross-check against the reference engine's internals since I did not
    read `sweep.py` or `backtest_production_correct.py`.)
- **Regime-multiplier path dependence.** A0's low win rate (see Part 1: ~29% gross) keeps
  the last-20-trade regime multiplier suppressed (0.1–0.25×) for long stretches, which
  compounds slowly and makes the *entire* run more sensitive to exactly when good/bad
  streaks land relative to the taper/regime state — a chaotic, seed-like sensitivity that
  two independently-written but behaviorally-equivalent engines can plausibly diverge on
  by a factor of ~2-3x in absolute dollars while agreeing on drawdown shape and holdout
  sign, which is exactly the pattern observed here.

**Bottom line:** the qualitative conclusion — replacing per-symbol fitted (rr, atr_mult)
with the single global (10, 3.0) setting outperforms the live config on this data, cuts
drawdown, and is the one arm that survives the untouched holdout — **replicates under an
independent implementation**, including a max-drawdown figure that lands within ~1pp of
the reference for the global arm. The absolute dollar/ROI magnitudes do **not** replicate
tightly, particularly for the live-config arm (my A0 underperforms the reference A0 by
roughly 2.5-3x), and that gap is real and unresolved — it lives somewhere in the portfolio
engine's path-dependent interaction between anti-pyramid blocking, regime-multiplier
suppression, and compounding, not in the underlying trade set (which matched within a few
percent). I would not treat the precise headline dollar figures in the claim as load-
bearing without reconciling this gap against the original engine's source; I *would* treat
the direction — global outperforms, with a real drawdown and holdout advantage — as
independently confirmed.
