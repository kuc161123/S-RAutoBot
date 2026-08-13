# Deep audit — findings

Run 2026-08-12 against the configuration committed the same day (global rr=10 / atr_mult=3.0,
trail on, gross cap 0.10, short-gate off, halt 21d). Five independent agent workstreams plus
orchestrator verification. Method per `AUDIT_PROMPT.md`: split at 2026-05-25, cost 24.2 bps,
`ch[bos]`, 21-day entry trim, weekly block bootstrap, rolling windows, leak-free sizing.

Nothing here has been deployed. `config.yaml` and `autobot/` were not modified during the audit.

---

## 1. REMOVE — ranked by severity

### 1.1 The self-pushing learner — **CRITICAL, act this week**

`autobot/core/unified_learner.py:1105-1113` contains an unconditional
`git add && git commit && git push` inside `_promote_combo()`, with errors swallowed by a
bare `except: pass` — no log, no alert.

The module is **currently unreachable** from `main.py` (verified by static and dynamic import
trace, and only imported by the equally-dead `bot_5m_old.py`). But this repo **auto-deploys
from `main`**, so a code path that can push to git is a loaded gun regardless of whether the
trigger is currently connected. The fix is deletion, not documentation.

Delete together (~7,411 lines, all confirmed unreachable):
`unified_learner.py` · `smart_learner.py` · `combo_learner.py` · `bot_5m_old.py` ·
`divergence_detector_5m_old.py` · `shadow_auditor.py`

### 1.2 Config keys that look like safety features and do nothing — **MEDIUM**

`risk.max_daily_loss: 0.1` and `risk.max_position_size_pct: 0.15` are **read nowhere**.
They sit inside the one config block that is genuinely live-wired, so they create false
confidence: there is no daily-loss circuit breaker and no per-position size cap. Either wire
them up or delete them. Leaving them is the worst option.

Same class: the entire `execution`, `indicators`, `legacy`, `monitoring`, `notifications`
blocks are never read. `legacy.trailing_stop: false` is actively misleading — the real switch
is `risk.trailing_stop.enabled: true`.

### 1.3 `net_directional_cap` is inert — **LOW, informational**

At the current settings, `net_directional_cap` values of 0.10, 0.15, 0.20 and `None` produce
**byte-identical output**. Once `gross_open_risk_cap` is 0.10, the gross cap always binds
first and the directional cap never fires. It costs nothing to leave in place, but it is not
providing the protection its comment claims, and it should not be credited in any future
risk assessment.

### 1.4 Five dead broker methods — **LOW**

`set_tpsl`, `set_sl_only`, `set_trailing_sl`, `place_limit`, `place_reduce_only_limit`
(~316 lines in `bybit.py`), unreferenced from the live path.

---

## 2. FIX — real defects found

### 2.1 The bot's execution telemetry is non-functional — **HIGH**

This is the most consequential engineering finding. `exec_log` was built precisely to measure
execution quality, and all three of its cost columns are broken:

| Column | Status | Cause |
|---|---|---|
| `actual_entry` | byte-identical to `intended_entry` in all 196 rows | `bot.py:2164` reads `avgPrice` off the `/v5/order/create` response, which Bybit v5 **never populates** — fills are async |
| `fee_usd` | null in all 196 rows | `bot.py:2676` reads a `closedFee` field that does not exist on `/v5/position/closed-pnl` |
| `funding_usd` | null | the join to `get_wallet_movement_summary()` was never built |

Consequence: the true fill-price gap — plausibly the largest single cost line — **cannot be
measured from any data in this repo.** The 24.2 bps cost figure this whole study rests on had
to be *inferred* from stop-out shortfall rather than read from fills. Fix: poll
`/v5/order/realtime` after placement, the way `get_order_status` already does elsewhere in
the same file.

### 2.2 `MIN_PIVOT_DISTANCE` is a no-op, and the backtest never enforces it — **MEDIUM**

At the live pivot width (3,3), `MIN_PIVOT_DISTANCE=3` produces trades identical to
unconstrained. Separately, `backtest_3yr_walkforward.detect_signals` never enforces it at all,
unlike the live detector. Harmless today; a latent live/backtest divergence the moment pivot
width is ever changed.

### 2.3 Margin-insufficiency skip is silent and fails open — **MEDIUM**

`bot.py:2098` skips an entry on insufficient margin without a Telegram alert, unlike every
sibling gate — and if the aggregate margin lookup errors it **assumes zero margin used** and
proceeds. Given aggregate initial margin already reaches ~92% of equity at the
99th-percentile hour, this is the gate most likely to be firing unseen.

### 2.4 No delisting handling exists — **MEDIUM**

Expected ~10–30 forced closures/year on a 277-symbol book. There is no code path for a
delisted symbol; an open position falls back to a blind "−1R, $0 PnL" default if no closed-PnL
record matches within 3 retries.

---

## 3. ADD — one idea pays

### 3.1 Skip the widest-stop entries — **+0.03 R/trade on the holdout**

Skip an entry whose `stop_frac` (`ATR × atr_mult / entry_price`, already computed by the bot
and currently unused for filtering) sits in the top ~20% of its own trailing 2,000-trade
distribution.

| | in-sample | holdout |
|---|---|---|
| kept − all, net R | **+0.012** | **+0.032** |
| gross R, kept vs skipped | +0.309 vs +0.200 | +0.005 vs −0.210 |

**It is not a cost artifact**, and the check that proves it is worth stating: cost in R is
`0.00242/stop_frac`, so high `stop_frac` trades are the *cheapest* ones. Skipping them skips
the cheap trades — the opposite of what a cost effect would do. Mean cost_R is 0.030 on the
skipped set versus 0.065 on the kept set, and the gap is still there in **gross** R.

Economically it reads as: when a symbol's ATR is extreme relative to its price, it is in an
erratic state the divergence rule does not handle well.

**Caveat and dissent:** the agent that found this reported +0.082 R/trade on the holdout; my
independent reimplementation reproduced the sign and direction but got **+0.032**. The
discrepancy is implementation detail in the rolling-quantile construction. Take the smaller
number, and treat this as the one candidate worth a follow-up test rather than a deploy.

---

## 4. KEEP — survived the audit

- **The CHOP filter.** Tested at thresholds 38–65 and no-gate, at both R level and through
  the production engine. At the portfolio level the live threshold of 52 wins **8 of 11**
  independent rolling 6-month windows against no gate (mean ROI/DD 4.90 vs 1.95). See §5 for
  a genuine disagreement about this.
- **`gross_open_risk_cap` 0.10.** Independently confirmed: holdout ROI peaks exactly at the
  live value in a clean sweep from 0.05 to none.
- **The taper schedule.** Trades ~7% of total profit for ~2.6× better holdout-segment
  resilience — the tradeoff it was designed to make.
- **The entire signal layer.** Pivot width, `max_wait_candles`, pivot staleness,
  `MIN_PIVOT_DISTANCE`, RSI period all swept. Transfer statistic across four walk-forward
  folds is **rho +0.275**, unstable (two of four folds indistinguishable from zero), against
  the +0.619 benchmark that marks a real effect. Every cell's CI overlaps every other cell's.
  Retuning this layer would be noise-fitting.
- **Execution cost.** Near the floor. A maker take-profit leg is worth only ~0.00005 R/trade,
  because the trailing stop already intercepts **99.6%** of winners before they reach the
  fixed TP — only 128 of 32,102 trades exit at the true TP. Funding is +0.0006 R/trade, CI
  crosses zero.

---

## 5. Unresolved dissent

**Does the CHOP filter earn its 62% block rate?**

- *Orchestrator:* yes at the portfolio level — 8 of 11 rolling windows, mean ROI/DD 4.90 vs
  1.95 for no gate.
- *AUDIT-A:* unproven — out-of-sample, **average net R per trade** is worse with the gate than
  without at every threshold from 38 to 52 (holdout: none −0.46, live 52 −0.51, 38 −0.84),
  and all CIs overlap.

Both are computed correctly and both can be true: the gate can improve portfolio ROI/DD (via
fewer, better-spaced trades interacting with the caps) while not improving per-trade R
out-of-sample. The honest position is that the gate's *portfolio* value holds up and its
*selectivity* claim does not. Neither removing nor re-tuning it is supported. Re-test once
the freeze accrues 250+ fresh trades under today's exact rules.

**Methodological note worth carrying forward:** AUDIT-A found that `search_roidd.py` and
`variations.py` inherit `short_gate=True` from `LIVE_SIM`'s defaults unless explicitly
overridden — which no longer matches the live config. The headline numbers in this audit were
re-run with the flag set correctly, and the gross-cap conclusion was independently reproduced
under correct settings. Any future use of those harnesses must pass `short_gate=False`.

---

## 6. What to do, in order

| # | Action | Type | Confidence | Effort |
|---|---|---|---|---|
| 1 | Delete `unified_learner.py` and the 5 other dead modules | REMOVE | high | low |
| 2 | Fix `exec_log`'s three broken columns | FIX | high | medium |
| 3 | Delete or wire `max_daily_loss` / `max_position_size_pct` | REMOVE | high | low |
| 4 | Alert on margin-skip; stop it failing open | FIX | high | low |
| 5 | Delete the five dead config blocks | REMOVE | high | low |
| 6 | Add delisting detection | ADD | medium | medium |
| 7 | Test the `stop_frac` top-20% filter further before deploying | ADD | medium | low |
| 8 | Leave CHOP, the caps, the taper and the signal layer alone | KEEP | high | none |

Items 1–5 are engineering hygiene with no strategy risk: they remove failure modes and false
confidence without touching a single trading decision. Item 7 is the only one that changes
what the bot trades, and it should clear a second independent test first.

**Nothing in this audit changes the standing verdict.** The strategy's edge remains
unproven out-of-sample; on the walk-forward-rejected pairs the current configuration nets
−0.042 R. This audit found dead weight to remove and one modest filter to consider. It did
not find alpha.
