# DEEP AUDIT PROMPT — what is this bot carrying that it should not, and what is it missing?

Use this prompt verbatim to run a full-depth self-assessment of the AutoTrading Bot. It is
written to be re-runnable: paste it as the task, and the agent running it has standing
authority to spawn as many subagents and parallel workstreams as the environment allows.

---

## Your authority and posture

You are the principal investigator. You have **full authority to spawn as many agents as the
environment permits**, in as many waves as needed, and to have them challenge each other's
findings. Keep every available slot busy. Assign genuinely independent briefs — never two
agents that would produce the same answer.

Your posture is that of an owner who suspects the machine is carrying dead weight and missing
free money, and who would rather learn that now than keep paying for it. Nothing is sacred:
every filter, gate, cap, multiplier, parameter and code path is a suspect until it has earned
its place **on the configuration the bot runs today**, not on the one it ran when the feature
was first validated.

**The single most important rule.** Almost every protection in this bot was validated against
a configuration that no longer exists. Two have already been caught this way — the BTC
short-gate (cost −705.8R under the current parameters, and was removed) and the trailing stop
(validated on fitted per-symbol parameters, behaves oppositely on global ones). Assume every
other feature has the same problem until you have re-tested it against the CURRENT config.
A feature that was validated in 2026-06 and never re-checked is a finding waiting to happen.

## What counts as a finding

Two kinds, and you are hunting both:

**REMOVE** — something the bot carries that costs money, costs nothing but adds risk of
failure, or does literally nothing. Dead config keys, dead code, filters that no longer earn
their block rate, gates validated against a dead configuration, redundant computation,
anything that adds operational surface without measurable benefit.

**ADD or CHANGE** — something absent or misconfigured that would measurably improve
profitability or cut drawdown. Prioritise mechanical, low-assumption wins (execution cost,
timing, sizing arithmetic) over anything that requires a new alpha claim.

A finding is only real if it is quantified in **R per trade** or in **end-to-end dollars
through the production engine**, and if it survives out-of-sample. An idea without a number
is a hypothesis; report it as one, separately, and say what test would settle it.

## Non-negotiable methodology

These are the conventions the repo has learned the hard way. Violating one invalidates the
result.

1. **Split chronologically and honour it.** Explore on data before **2026-05-25**; judge on
   the untouched period after. A result that only works in-sample is a NEGATIVE result and
   must be reported as one, not buried.
2. **Cost is 24.2 bps round trip**, measured from 133 real live stop-outs (`exec_log`). Net R
   per trade is `r - 0.00242/stop_frac`. Do not use the repo's legacy 18 bps constant.
3. **Read CHOP at the BOS bar (`ch[bos]`), never the entry bar.** Reading the entry bar's own
   candle leaks ~0.29 R/trade and flips the gate's sign out-of-sample.
4. **Trim the last 21 days of entries.** Unresolved trades are silently dropped, and slow
   winners resolve last, so any window boundary is biased against winners.
5. **Weekly block bootstrap for every confidence interval.** Signals fire in correlated bursts
   across alts; i.i.d. resampling understates variance ~5×.
6. **Rolling windows, not one path.** A single continuous backtest is one draw. Score across
   independent windows and report how often a change wins, not just its total.
7. **Size on realised wallet (`size_basis='wallet'`).** The engine's equity basis marks open
   positions toward their already-known final PnL — a real lookahead that inflates results
   1.2–1.6× and inflates most whatever has the biggest winners.
8. **Never edit `config.yaml` or anything under `autobot/` during the audit.** This is a live
   bot trading real money. Produce recommendations and exact diffs; deploy nothing.

## The current configuration you are auditing

Committed 2026-08-12. Anything validated before that date against different settings is
automatically suspect.

- Global **rr = 10.0, atr_mult = 3.0** on all 728 (symbol, div_type) pairs, 277 symbols
- Trailing stop **s3_a1** on: arms at MFE ≥ 3R on a bar's own excursion, trails 1.0 ATR
  behind closed candles, never widens, no breakeven
- `risk_per_trade` 0.003, wallet-taper / equity-sizing mismatch, regime multiplier on
- `net_directional_cap` 0.10, `gross_open_risk_cap` **0.10** (binds ~9% of entries)
- `btc_short_gate` **false**, `long_bull_boost` 1.3
- Shadow halt 21d (−300/+200), **advisory** — a human must act on it
- CHOP filter on, regime-aware thresholds `{favorable 52, cautious 45, adverse 52, critical 55}`

## Workstreams — assign these to independent agents

Run them in parallel. Each is a genuinely different question; none should reach the same
answer by a different route.

### A. Feature ablation on the CURRENT config
Every gate, filter, cap and multiplier, removed one at a time and in combination, measured
in-sample and on the holdout, across rolling windows. The CHOP filter is the priority target:
it blocks ~62% of all signals, is the single most load-bearing filter in the system, and its
thresholds were tuned for a configuration that no longer exists. Does it still earn a 62%
block rate at rr=10/atr_mult=3.0? Also: `long_bull_boost` 1.3, `net_directional_cap` 0.10,
the taper schedule, the regime tiers and their exact thresholds.

### B. Execution and cost
The largest mechanical lever in the system. Round trip is 24.2 bps against a gross alpha of
roughly +0.19 R/trade, so cost is a third of the edge. Entries and exits are **all taker
market orders** — establish whether any leg could be maker, and what that is worth in R.
Audit funding: the book is structurally net-short and shorts often *earn* funding in crypto —
is the bot capturing that, is it modelled, and does holding time interact with it? Audit
slippage, fill lag (`exec_log.fill_lag_sec`, median 74s), and the gap between the price the
bracket is computed from (`df.iloc[-1]['open']`) and the price the market order actually
fills at.

### C. The signal layer
Never systematically re-tuned. Pivot fractal width (left=3/right=3), `MIN_PIVOT_DISTANCE`,
the 10-bar staleness limit on the triggering pivot, `max_wait_candles=12` for BOS
confirmation, the EMA-200 gate applied twice, RSI period, the dedup key. Each is a free
parameter that has never been swept honestly. Sweep them with a walk-forward, and be alert
that this is the easiest place in the whole system to overfit — report transfer statistics,
not just in-sample optima.

### D. Dead weight and operational surface
~7,000 of 16,000 lines under `autobot/` are unreachable. Config sections `execution`,
`indicators`, `legacy`, `monitoring`, `notifications` are never read. `max_daily_loss` and
`max_position_size_pct` are never read anywhere. Establish exactly what is dead, what is
merely unused, and what is *dangerous* — specifically the three learner modules that can
`git commit && git push` on promote. Also quantify the operational risks no backtest prices:
delisting (~10–30 forced closures/year on a 277-symbol book), the margin-exhaustion finding
(aggregate initial margin reaches ~92% of equity at the 99th-percentile hour), and the
absence of any max-concurrent-position limit.

### E. What is missing
The additive half. Ideas never tried in this repo, each judged on whether it could plausibly
move R per trade by more than 0.02 before you spend compute on it. Consider at minimum:
portfolio-level volatility targeting; sizing by signal characteristics that are causal at
entry; funding-aware side selection; time-of-day or session effects; whether the 60-second
poll loop and 55-minute staleness guard are losing fills; multi-timeframe confirmation.
Report the ones that pay and the ones that do not, with numbers for both.

## Cross-examination

Every load-bearing finding must be reviewed by a second agent that did not produce it, and
the strongest findings should face an adversarial agent whose explicit job is to destroy
them. Resolve disagreements by going back to the data and re-computing — never by majority
vote or by which write-up reads better. Where a disagreement cannot be resolved, report it as
unresolved dissent with both positions and the number that would settle it.

## Deliverable

A ranked recommendation list, split into REMOVE and ADD/CHANGE, where every row carries:

- the measured effect in R per trade or in dollars, with a confidence interval
- in-sample **and** holdout figures, shown separately
- how often it wins across rolling windows
- the exact config or code change required
- a confidence level, and the single strongest reason it could be wrong

Rank by expected value net of implementation risk. State plainly which findings you would act
on this week, which need more evidence and what evidence specifically, and which are
interesting but not actionable.

**"Nothing found in this workstream" is a valid and valuable result.** Report it plainly
rather than manufacturing a marginal finding to fill space. Do not fabricate numbers, agents,
tests or confidence. If something cannot be tested with available data, say so and say what
would be needed.
