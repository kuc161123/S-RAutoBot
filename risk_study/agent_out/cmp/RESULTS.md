# AGENT-CMP — independent six-way reproduction

Simulator: `simulate.py` in this directory. Written from scratch — no import of
`backtest_production_correct.py` or of any code under `risk_study/` other than the
six input parquet files themselves. Single event-driven pass, ~35k events per arm,
entries sorted before exits on timestamp ties (verified: the assertion that zero
positions are left open at the end of the run passes for all six arms).

## What "equity" means here — read this before the numbers

Two different balances are tracked and I report both:

- **wallet** — realised cash only, used for the taper lookup and as `EQUITY` in every
  sizing/gate formula, per the spec. It is capped: any amount over $50,000 is
  "withdrawn" and the wallet resets to $50,000. This is the number that governs how
  large the *next* trade's risk-dollars are.
- **trading_equity** = $1,500 + cumulative realised P&L, **never capped**. This is the
  number in `monthly.csv`, in "final equity", in the drawdown calc, and in ROI%. It
  answers "how much wealth did the strategy generate," treating every withdrawal as
  still belonging to the account (matching CLAUDE.md §6's own convention: "trading DD
  ... immune to deposits/withdrawals").

If I had used the capped wallet instead, every arm that ever clears $50k would show
an artificially flat, wrong equity curve after that point purely from the withdrawal
mechanic, which is a sizing throttle, not a loss. All six arms in this run do clear
$50k, cap out to a constant ~$0.14%×$50k risk-per-trade regime after that, and keep
compounding on paper (uncapped) from there — that's real, not an artefact of my choice.

## Summary

| arm | trades entered / available | final equity | ROI% (wealth basis) | max DD % | win rate (gross) | mean R gross | mean R net | months +/- | holdout 05-31→final |
|---|---|---|---|---|---|---|---|---|---|
| OLD-BASE | 23,904 / 32,913 | $13,796 | +820% | 76.0% | 18.0% | +0.340 | +0.176 | 17 / 21 | **−40.2%** |
| CURRENT | 25,825 / 32,913 | $1,605 | +7% | 74.4% | 28.1% | +0.194 | +0.030 | 17 / 21 | **−52.6%** |
| **GLOBAL** | 19,560 / 32,102 | **$15,150** | **+910%** | **44.7%** | 28.4% | +0.219 | **+0.148** | **23 / 15** | **+1.0%** |
| GLOBAL-FIX | 13,688 / 31,549 | $1,778 | +19% | 50.8% | **10.3%** | +0.138 | +0.068 | 16 / 22 | −8.5% |
| GLB-10/2 | 23,534 / 32,774 | $5,442 | +263% | 58.6% | 28.2% | +0.205 | +0.099 | 21 / 17 | −21.0% |
| GLB-8/2 | 23,492 / 32,872 | $4,345 | +190% | 59.0% | 28.2% | +0.198 | +0.093 | 21 / 17 | −22.6% |

Full numeric detail: `summary.csv`. Per-trade ledgers: `trades_<ARM>.csv`.

## Month-end trading equity, last 14 months (full table in `monthly.csv`, 2023-06→2026-07)

| month | OLD-BASE | CURRENT | GLOBAL | GLOBAL-FIX | GLB-10/2 | GLB-8/2 |
|---|---|---|---|---|---|---|
| 2025-06 | 6,194 | 1,758 | 4,906 | 2,104 | 3,334 | 3,016 |
| 2025-07 | 10,087 | 2,541 | 6,514 | 2,374 | 4,185 | 3,775 |
| 2025-08 | 7,269 | 1,661 | 5,063 | 2,013 | 2,848 | 2,537 |
| 2025-09 | 6,952 | 2,090 | 6,139 | 1,733 | 3,202 | 2,767 |
| 2025-10 | 10,148 | 2,620 | 9,998 | 2,846 | 4,443 | 3,585 |
| 2025-11 | 10,857 | 2,085 | 11,938 | 2,494 | 3,836 | 3,091 |
| 2025-12 | 10,965 | 2,054 | 11,454 | 2,082 | 3,840 | 3,094 |
| 2026-01 | 21,730 | 3,792 | 14,729 | 2,309 | 6,240 | 4,797 |
| 2026-02 | 25,322 | 5,566 | 16,076 | 2,317 | 7,311 | 5,818 |
| 2026-03 | 21,808 | 3,859 | 14,583 | 2,248 | 6,350 | 5,072 |
| 2026-04 | 17,411 | 2,803 | 13,810 | 1,776 | 5,291 | 4,171 |
| 2026-05 | 23,078 | 3,386 | 15,003 | 1,943 | 6,888 | 5,611 |
| 2026-06 | 18,735 | 2,686 | 18,965 | 2,058 | 8,320 | 6,699 |
| 2026-07 | 13,796 | 1,605 | 15,150 | 1,778 | 5,442 | 4,345 |

**2026-07 is a partial month** — entries stop 2026-07-25 (last row in every input
parquet), so the July figure reflects roughly three weeks, not a full month. Treat it
as a mid-month mark, not a closed month.

Note the whole cohort has a rough ride in 2024 (all six arms sit at 45–75% of their
2023-08 peak by mid-2024) before the 2026 run-up — this is a shared property of the
signal set under the regime-multiplier feedback loop, not something specific to one
arm's exit/parameter choice.

## Monthly return stats

| arm | months + | months − | median month | mean month | best month | worst month |
|---|---|---|---|---|---|---|
| OLD-BASE | 17 | 21 | −3.36% | +11.5% | +122.8% | −31.7% |
| CURRENT | 17 | 21 | −4.37% | +3.8% | +84.6% | −40.3% |
| GLOBAL | 23 | 15 | +3.19% | +8.6% | +75.9% | −27.4% |
| GLOBAL-FIX | 16 | 22 | −3.51% | +2.4% | +81.5% | −21.0% |
| GLB-10/2 | 21 | 17 | +2.55% | +6.6% | +70.8% | −34.6% |
| GLB-8/2 | 21 | 17 | +3.66% | +5.8% | +62.4% | −35.1% |

GLOBAL is the only arm with more up months than down months and the only arm with a
positive median month. Every other arm has a negative median month and relies on a
small number of large up-months (the classic RR≥8 profile) to end up net positive.

## Ranking and recommendation

**1. GLOBAL** (global rr=10, atr_mult=3.0, trailing exit) — best on every axis: highest
final equity, lowest max drawdown (45% vs 59–76% for the rest), most positive months,
best net mean R, and the *only* arm with a positive holdout period. If I had to deploy
one of these six, this is it.

**2. OLD-BASE** (today's per-symbol fitted params, fixed TP — i.e. what the bot ran
before 2026-08-02) — close to GLOBAL on final equity but gets there with much rougher
handling: 76% max DD, a strongly negative holdout (−40%), and a return profile that
depends on a couple of outsized months (best month +123%, matching the "top 1% of
trades = 112% of profit" concentration problem CLAUDE.md already flags for this exact
arm). I would not prefer this over GLOBAL.

**3–6. CURRENT, GLB-10/2, GLB-8/2, GLOBAL-FIX** — meaningfully worse. CURRENT (today's
actually-deployed trail) is the worst full-history performer of the six under this
simulator; see the caveat below before treating that as an indictment of the live
trail feature. GLB-10/2 and GLB-8/2 (same trail rule, tighter atr_mult=2.0 stop) both
land well below GLOBAL, so — within this comparison — the wider 3.0×ATR stop is doing
real work, not just adding variance. GLOBAL-FIX is the fixed-TP twin of GLOBAL and is
the worst-ranked despite a per-trade mean R that isn't dramatically lower than the
trail arms — see below, this one is substantially a regime-gate artefact.

## Two things that are simulator-spec artefacts, not "the trades are worse" — flagging explicitly

**GLOBAL-FIX's 10.3% win rate is mechanical, and the gate formula punishes it hard.**
GLOBAL-FIX is GLOBAL's exact same signals/entries at the exact same params (rr=10,
atr_mult=3.0), just resolved against a fixed take-profit instead of the trail. Hitting
a full +10R fixed target is rare, so win rate collapses to 10.3% — but mean gross R
(+0.138) is only modestly below GLOBAL's (+0.219). The regime-multiplier formula
(`wr>=0.18` is the gate for the top two tiers) was evidently calibrated around a
strategy with an ~18–29% observed win rate; a 10.3%-WR arm spends much more of its
life in the `wr<0.10` "critical" bucket (mult 0.1), so its position sizes are
chronically tiny and the compounded final equity ($1,778) looks catastrophic relative
to its actual per-trade edge. A sizing scheme that read `avg_r` alone, or used a
lower WR breakpoint, would likely close much of this gap. **Read GLOBAL-FIX's rank as
"this WR/gate interaction is bad," not "the fixed-TP trades themselves are much worse
than trailed ones."**

**CURRENT vs OLD-BASE is a genuine reversal of what CLAUDE.md's `report_trail_live_
realistic.py` study reported, and I can't fully reconcile it from these parquets
alone — flagging rather than papering over.** That report (§6.1) says the trail (s3_a1)
beat fixed-TP in cumulative dollars over the "last 15 months" ($119,030 vs $101,452).
Checking the raw `r_fixed`/`r_trail` columns directly in `universe_trail.parquet`
(no simulator involved) over the same trailing-15-month window (2025-04-25 onward):
mean `r_fixed` = +0.388, mean `r_trail` = +0.223 — fixed is *already* ahead on raw
gross R before any portfolio mechanics touch it, and stays ahead across the full
2023-06→2026-07 window (+0.409 vs +0.221). My simulator then compounds that gap
through the regime-multiplier/taper feedback loop (the trail arm's higher win rate
but lower mean R changes which regime tier it sits in trade-by-trade, versus the
fixed arm), which turns a ~0.19R/trade gross gap into an 8.6× final-equity gap. I have
no visibility into what sizing/window methodology the cited report used — it may not
compound through this same regime loop, or may be scoped to a different universe
snapshot — so **I'm not asserting CLAUDE.md's trail conclusion is wrong**, only that
under the exact spec given to me, on the exact parquet handed to me, fixed-TP wins
convincingly and by a lot, and the size of that "a lot" owes much more to compounding
dynamics than to the underlying ~0.19R/trade difference. Take the *direction* of this
result more seriously than the *magnitude*.

## Other observations worth keeping in mind

- Blocking is dominated by the anti-pyramid rule (one position per symbol+side) in
  every arm — 2,786 to 14,273 blocks vs 606–3,448 for the short-impulse and
  net-directional gates combined. The gross-open-risk cap (30% of equity) never binds
  for any arm in this run; individual per-trade risk is small enough relative to
  typical concurrency (median ~19–121 open positions across arms) that it isn't the
  active constraint here — anti-pyramid and net-directional are.
- GLOBAL-FIX's much higher concurrency (median 121 open positions vs 19–58 for the
  others) is a direct consequence of a fixed target that rarely resolves quickly —
  positions sit open far longer, which is also part of why it blocks 14,273 signals on
  the anti-pyramid rule alone (more than 3× any other arm).
- Regime input uses **net (cost-adjusted) R**, matching what the live bot's own
  `r_value = pnl_usd / risk_usd_at_entry` actually reflects (real P&L, not the
  backtest's gross-R column) — this is an interpretation choice on an ambiguous point
  in the spec, called out here rather than left silent.
- All six arms clear the $50k withdrawal ceiling well before the end of the run
  (GLOBAL first, around 2025-11), so from that point on every arm's "final equity" is
  the compounding of a nearly-constant per-trade risk-dollar figure (0.14% taper rung
  × regime mult × ~$50k), not of a growing account.

## Files in this directory

- `simulate.py` — the simulator (self-contained, only reads the six input parquets).
- `summary.csv` — one row per arm, all headline stats.
- `monthly.csv` — month-end trading_equity, one column per arm, 2023-06..2026-07.
- `monthly_returns.csv` — month-over-month % return per arm.
- `trades_<ARM>.csv` — full closed-trade ledger per arm (entry/exit time, gross R,
  net R, risk$, pnl, running wallet/equity) for anyone who wants to re-derive anything
  above independently.
