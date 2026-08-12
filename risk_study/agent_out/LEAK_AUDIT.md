# Look-ahead / data-leakage audit — AGENT-LEAK, 2026-08-11

Scope: the trade universe built by `replay()` in `backtest_halt_multiwindow.py:54-115` and
`risk_study/build_risk_universe.py:40-103`, using `prepare_data()`/`detect_signals()` from
`backtest_3yr_walkforward.py`, compared against the live bot's
`autobot/core/divergence_detector.py` and `autobot/core/bot.py`. The known, already-fixed
CHOP-at-entry-bar leak (STRATEGY_VERDICT_2026-08-11.md §2.1) is **not** re-litigated here
except where needed as context.

All probe scripts referenced below live in this directory and are re-runnable.

## Ranked findings

### 1. [CONFIRMED, HIGH IMPACT] Right-censoring is real, large, and — by direct control test — is at least partly a pure artifact, not (only) genuine recent decay

`STRATEGY_VERDICT_2026-08-11.md §2.4` already flagged this qualitatively. This audit adds a
clean causal proof and reproduces the magnitude independently.

**Mechanism.** The `for k in range(e, n)` exit loop (both `replay()` implementations) has no
`else`: a trade whose stop/target never triggers before the data ends is silently dropped,
never counted as a loss, a win, or anything. Because target is always 1.5–10x farther than
stop, winners systematically take longer to resolve than losers. Near the data's end, slow
winners haven't had time to resolve and get dropped; fast losers still resolve in time. The
*visible* (resolved) sample near the boundary is therefore loser-selected.

**Full-universe count** (`probe_exit_ambiguity_censoring.py`, 277 live-config symbols, honest
`ch[bos]` gate, 34.1bps fee): 33,995 entered trades, 80 (0.24%) never resolve (silently
dropped) — but 61 of those 80 (76%) sit in the last 21 days of that symbol's own data. The
raw censored-count is small in the full 3-year context, but it concentrates almost entirely
at the boundary.

**Boundary bias, matched to STRATEGY_VERDICT's own framing** (`probe_censoring_bias.py`, OOS
= entries ≥ 2026-05-25, n=2,584):

| | n | resolved | avg net R (resolved only) |
|---|---|---|---|
| OOS, not last 21 days | 1,776 | 99.2% | −0.330 |
| OOS, last 21 days | 808 (31.3% of OOS) | 92.6% | **−0.971** |

This reproduces the doc's "−0.50R vs +0.04R, 32% of OOS sample" claim in shape (same
denominator, same direction, larger magnitude here because this run uses the honest
`ch[bos]` gate and the 34.1bps cost floor rather than the contaminated/18bps numbers the
original claim may have used).

**Causal proof it is (at least partly) a pure artifact, not real decay**
(`probe_censoring_control.py`): pick an artificial cutoff deep in history — 2024-12-01 — far
from the independently-established June-2026 edge break (§1.2 of the verdict doc), so any
effect found here cannot be "the edge broke." Run the identical signal→BOS→entry pipeline on
data truncated at the cutoff (mimicking what a real-time backtest would have seen) and
separately resolve the *same* trades against the real, un-truncated future:

| | n | resolved | avg net R |
|---|---|---|---|
| (a) TRUNCATED at 2024-12-01 (real-time view) | 494 | 94.1% | **−0.094** |
| (b) FULL data (true outcome, same trades) | 494 | 100% | **+0.197** |

A +0.29R swing from resolution horizon alone, with zero possible contribution from real
market regime (both columns are the *same trades*, same signals, same market). The 29 trades
that were censored under truncation, once allowed to resolve, averaged **+4.87R** with a
58.6% win rate — i.e. exactly the large, slow winners the mechanism predicts.

**Conclusion**: right-censoring inflates the appearance of a "recent collapse" at any data
boundary. It does not fully explain the June-2026 break (§1.2's difference-in-differences
design controls for this by comparing selected vs. rejected divergence types over the *same*
period), but it means any number computed on entries within ~21 days of a dataset's end —
including the tail of every study in this repo — is biased pessimistic and should either be
truncated (per `STRATEGY_FINDINGS_2026-08.md:190`'s own mandate, which the builders do not
follow) or reported with this caveat attached.

### 2. [NEW FINDING, MED-HIGH IMPACT] The backtest's signal detector is not the live bot's signal detector — ~11.7% of the trade universe is signals live would never generate. Direction is conservative, not inflating.

This is new — not mentioned in STRATEGY_VERDICT or CLAUDE.md. It directly answers the task's
item 2 ("do the backtest and the live bot detect the SAME signals at the SAME time?").
Answer: **no**, and here is why, quantified.

**Mechanism.** `backtest_3yr_walkforward.detect_signals()` (`backtest_3yr_walkforward.py:237-296`)
only searches for bull-divergence pivots when `curr_price > curr_ema` (line 254) and only
searches for bear-divergence pivots when `curr_price < curr_ema` (line 275) — the EMA
trend-gate happens *before* the pivot pair is examined, so `used_pivots` (the dedup set) is
only ever populated on a bar where the divergence would actually be accepted.

The live `autobot/core/divergence_detector.py::detect_divergences()` (lines 179-361) does the
opposite: it always searches for both bull and bear pivots on every scan bar, evaluates the
divergence math unconditionally, and marks `used_pivots` (lines 277, 295, 341, 359) the
moment the divergence condition is true — regardless of EMA alignment. `daily_trend_aligned`
is recorded as metadata only (line 259/323) and the actual EMA filter is applied afterward,
in `bot.py:1594-1596` (`if not signal.daily_trend_aligned: continue`), *after* the signal
object (and its dedup consumption) already exists.

Net effect: if a pivot pair's divergence condition is first satisfied on a bar where price is
on the wrong side of EMA, **live burns that pivot pair's dedup slot right there** and can
never signal on it again — even after price later crosses the EMA within the same 10-bar
freshness window. The backtest, which never even attempted the pivot pair on the misaligned
bar, is free to (and does) detect it once price crosses. CLAUDE.md §3 describes the
"queue up and become valid when price later crosses the EMA" behavior as deliberate for the
*BOS-wait* stage — but this dedup-order effect operates one stage earlier, at detection, and
is not the intended mechanism; it silently forecloses re-detection instead of allowing it.

**Verified directly** (`probe_dedup_hypothesis.py`, BTCUSDT): 29 of 30 sampled
only-in-backtest signals are preceded, within the pivot's 10-bar freshness window, by a live
signal of the same type/side/pivot with `daily_trend_aligned=False` — i.e. live had already
"seen and rejected" that exact opportunity before the backtest's later, trend-aligned version
of it.

**Magnitude** (`probe_dedup_impact_realconfig.py`, 56 live-config symbols, each symbol's real
per-symbol (rr, atr_mult), 33,399 backtest signals):

| | n signals | share of backtest universe | avg net R |
|---|---|---|---|
| only-in-backtest (live would never generate) | 3,896 | 11.7% | +0.067 |
| shared (both engines agree) | ~29,500 | 88.3% | +0.198 |
| only-in-live (backtest misses) | 16 | ~0% | — |

The mismatch is almost perfectly one-directional (3,896 vs 16) — exactly what the dedup-order
mechanism predicts. Blended backtest-universe avg R = +0.182 vs +0.198 for what live would
actually see (shared only) → **the leaked 11.7% drags the reported average down by about
−0.016 R/trade**. So unlike a typical lookahead leak, **this one makes the backtest slightly
pessimistic relative to the live bot**, not optimistic. It is still a genuine
detect-signals-mismatch defect: every study built on `bt.detect_signals()` — which is nearly
every backtest script in this repo, including both `replay()` implementations audited here —
is not exactly replaying what the live bot would trade, it's replaying a slightly larger,
slightly worse-performing superset of it.

**Fix**, if wanted: gate the pivot search in `detect_signals()` the way live effectively
*intends* to (allow queueing), by moving the EMA check to a point that doesn't retroactively
consume `used_pivots` on a rejected bar — e.g., only add to `used_pivots` when the signal is
actually trend-aligned and emitted, matching what the backtest already does. That would make
*live* match the backtest's (better) behavior, or the backtest could be changed to mimic
live's premature-consumption bug — either way the two are currently not the same algorithm.

### 3. [NOT FOUND] `prepare_data()` — no look-ahead

`backtest_3yr_walkforward.py:207-219`. RSI: `rolling(RSI_PERIOD).mean()`, default
(non-centered) window. ATR: `rolling(14).mean()` on true range, non-centered. EMA:
`close.ewm(span=200, adjust=False).mean()` — `adjust=False` matches the live bot's
`calculate_daily_ema` (`divergence_detector.py:141-143`) exactly. No `.shift(-1)`,
`center=True`, or `.bfill()` anywhere in `backtest_3yr_walkforward.py`,
`backtest_halt_multiwindow.py`, `risk_study/build_risk_universe.py`, or
`autobot/core/divergence_detector.py` (grepped, zero hits).

**Numeric proof of causality** (`probe_prepare_data_causality.py`): recomputed rsi/atr/ema on
BTCUSDT truncated at 5 different cutoff points (500, 5000, 15000, 25000, last row) and
compared against the full-series computation at the same timestamps. All 5×3 = 15 values
matched exactly (`np.isclose`). If any future information leaked in, truncating the input
would change the value at the cutoff row; it did not.

CHOP: `chop_series()` in `backtest_halt_multiwindow.py:44-51` is formula-identical to
`calculate_chop()` in `divergence_detector.py:126-138` (same TR, same rolling(14) sum/range,
same log10(14) normalizer).

### 4. [NOT FOUND, beyond the already-fixed §2.1] BOS/ATR/EMA/CHOP timing is correctly anchored at `bos`

Confirmed by direct read of the *current* `replay()` in both harnesses
(`backtest_halt_multiwindow.py:79-103`, `risk_study/build_risk_universe.py:69-90`):

- BOS test: `c[idx]` for `idx` in `[conf+1, conf+MAX_WAIT]` — always a closed historical bar.
- EMA gate: `c[bos] > ema[bos]` / `c[bos] < ema[bos]` — read at `bos`.
- CHOP gate: `ch[bos]` — read at `bos` (this is the post-fix state; the historical
  `ch[e]` bug is §2.1, already resolved, not reproduced here).
- ATR: `atr[bos]` — read at `bos`.
- Only the fill price `entry = o[e]` reads bar `e`, and it reads only the **open**, which is
  known the instant bar `e` starts — no look-ahead.

Nothing is read at `e` or later except the entry open itself.

### 5. [NOT FOUND / benign] Same-bar-as-entry resolution and stop+target ambiguity

Full-universe instrumented replay, 277 symbols, live configs, honest `ch[bos]` gate
(`probe_exit_ambiguity_censoring.py` + `probe_samebar_winrate.py`):

- 33,995 entered trades, 33,915 resolved (99.76%).
- **10.06%** of resolved trades (3,411) resolve on the entry bar itself (`k == e`).
- **Win rate within same-bar resolutions: 3.52%**, vs **19.93%** for trades that take longer
  to resolve (overall win rate 18.28%). Same-bar resolutions are overwhelmingly *stop-outs*,
  consistent with a stop only 1–2×ATR away being reachable within the fill hour while a
  target 3–10×ATR away essentially never is in a single hour. This is the physically
  expected, conservative shape — **not** evidence of unrealistic same-bar wins.
- Ambiguous bars (both stop **and** target price levels crossed within the resolving bar's
  high/low) are rare: **28 of 33,915 resolved (0.08%)**, of which 8 are on the entry bar
  itself.
- Tie-breaking is confirmed pessimistic in code: `out.append((..., -1.0 if hs else rr, ...))`
  checks `hs` (stop-hit) first, so an ambiguous bar always resolves as a loss — matches the
  "SL wins ties" convention documented repo-wide (`build_risk_universe.py:97` comment).

Both sub-questions in item 4 resolve clean: (a) same-bar entry resolution is realistic in
direction (mostly losses, as physically expected) even though the exact intra-bar path
isn't modeled, and (b) the ambiguous-tie rate is tiny (0.08%) and is resolved pessimistically,
not optimistically.

### 6. [CONFIRMED + new mechanical detail] Cache boundary and data quality

**§2.5's claim, verified precisely.** All 277 live-config symbols: 270/277 end exactly at
2026-07-25; the other 7 end earlier (2026-06-26 to 2026-07-20) — `BTCUSDT-26JUN26` /
`ETHUSDT-26JUN26` are stale dated-futures entries left enabled in `config.yaml` (not
perpetuals, and probably shouldn't be in the live symbol list at all), and
`MBOXUSDT`/`MLNUSDT`/`SOLVUSDT`/`SWARMSUSDT`/`PUMPBTCUSDT` appear delisted before the second
download pass. Non-live universe (245 cached, 232 non-empty): 215/232 end exactly at
2026-05-25 00:00:00, **none run later** — confirms "any cross-symbol comparison spanning that
date is confounded" exactly as claimed.

**New: every one of the 277 live symbols is missing exactly one hourly bar, at
2026-05-25 01:00:00** (a gap from 00:00 to 02:00) — the literal seam between the two download
passes. This is 277/277, i.e. completely systematic, not sporadic. It sits precisely on the
in-sample/OOS split date (`analyze_signal_vs_random.py` and others use 2026-05-25 as the OOS
boundary) that nearly every study in this repo treats as meaningful. Effect on indicators is
small — spot-checked BTCUSDT's rsi/atr/ema around the gap and saw no visible anomaly — but a
rolling(14) window computed across a real 2-hour gap, treated as a 1-step gap by
row-position-based `pandas.rolling`, is not exactly what it claims to be for the ~14 bars
downstream of the gap, in every symbol, at the exact date used to define "OOS." Likely a
second-order effect next to findings #1 and #2, but worth fixing (re-fetch across the seam,
or explicitly reindex to an hourly grid and interpolate/mark the gap) since it costs nothing
to fix and currently silently touches all 277 symbols at once.

Only one other anomaly found in a ~95-symbol timestamp-quality sample: `PUMPBTCUSDT` has two
unrelated 2-hour gaps (2025-06-03, 2025-06-13), plausibly real exchange data gaps around its
listing. No duplicate timestamps and no timezone inconsistency in any sampled file — all
`datetime64[ns]`, tz-naive, consistent across 95 sampled symbols including BTC/ETH/SOL and
several thin/delisted names.

**Survivorship**: delisted symbols (`CAMPUSDT`, `RDNTUSDT`, `TRUUSDT`, `GODSUSDT`, and the
other 7 named in CLAUDE.md §11) are present in the cache with truncated end dates — they were
**not** filtered out, so the study does not survivorship-bias against names that delisted
*during* the two collection passes. Caveat: `fetch_all_symbols()` in
`backtest_3yr_walkforward.py:81-99` only ever queried symbols with `status='Trading'` and
≥$100k/day turnover *at download time* — any symbol that had **already** fully delisted
before either pass ran would never have been fetched at all. This is a structural gap in the
245-symbol "never-selected control" universe that `STRATEGY_VERDICT §1.3b` leans on, though
it's inherent to any point-in-time downloader rather than a bug introduced by this repo's
code, and is likely not practically fixable after the fact (Bybit does not expose historical
klines for symbols it has fully delisted from the current instrument list).

### 7. [MINOR, immaterial] `scan_start`/`min_idx` off by 5 bars, comment is wrong

`divergence_detector.py:220`: `scan_start = max(205, lookback_bars + PIVOT_RIGHT + 1)`, with a
comment claiming "Matches backtest exactly." The actual backtest
(`backtest_3yr_walkforward.py:247`) uses `min_idx = max(EMA_PERIOD + 10, LOOKBACK + PIVOT_RIGHT + 1)`
= `max(210, 54)` = 210, not 205. A 5-bar discrepancy in the earliest scannable index. Immaterial:
affects only bars 205–209 of each symbol's history (within the first ~9 days of data, in
2023-03), and virtually every study in this repo filters entries to `>= GEN_FROM (2023-06-01)`
anyway, so this window is discarded downstream regardless. Flagging only because the comment
is factually wrong, not because it changes any reported number.

## Summary table

| # | Finding | Category | Verdict | Est. R/trade impact |
|---|---|---|---|---|
| 1 | Right-censoring at data boundary | Real bias, proven via control | CONFIRMED, refined | ~0.29R swing isolated in control; −0.64R apparent cliff in real OOS tail (31% of OOS sample) |
| 2 | Backtest vs. live signal-detection mismatch (dedup-order) | New finding, not a lookahead | CONFIRMED | ~−0.016 R/trade drag on blended averages (conservative direction); 11.7% of universe affected |
| 3 | `prepare_data()` causality (RSI/ATR/EMA) | Lookahead check | NOT FOUND | none |
| 4 | BOS/ATR/EMA/CHOP anchored at `bos` vs `e` | Lookahead check | NOT FOUND (beyond already-fixed §2.1) | none |
| 5 | Same-bar entry resolution & tie-breaking | Realism / bias check | NOT FOUND / benign | none (conservative if anything) |
| 6 | Cache boundary, gaps, survivorship | Data-quality audit | CONFIRMED (§2.5) + 1 new mechanical detail (universal 1-bar gap at the seam) | small, localized to ~14 bars/symbol at the OOS split |
| 7 | scan_start/min_idx 5-bar mismatch | Code correctness | Minor, immaterial | ~0 (pre-GEN_FROM window, discarded downstream anyway) |

## Files in this directory

- `probe_prepare_data_causality.py` — causality check for #3
- `probe_signal_parity.py`, `probe_dedup_hypothesis.py`, `probe_dedup_impact.py`,
  `probe_dedup_impact_realconfig.py`, `dedup_realconfig_out.txt` — detection-mismatch finding #2
- `probe_exit_ambiguity_censoring.py`, `exit_ambiguity_censoring_by_symbol.csv`,
  `probe_samebar_winrate.py`, `samebar_winrate_out.txt` — exit-resolution finding #5, and the
  full-universe right-censoring count for #1
- `probe_censoring_bias.py`, `censoring_bias_rows.parquet` — boundary-window bias measurement for #1
- `probe_censoring_control.py`, `censoring_control_rows.parquet` — the artificial-cutoff causal
  control proving #1 is (partly) a pure artifact
- `probe_data_quality.py` — gap/dup/timezone sampling for #6
