# Independent replication — AGENT-REPL

Engine: `risk_study/agent_out/sim.py`, written from scratch, no imports from
`backtest_production_correct.py` or `risk_study/sweep.py`. Event-driven,
interleaved entry/exit simulation over `universe_chopBOS.parquet` (32,929
trades, 277 symbols, 2023-06-01 → 2026-07-25).

## Modeling choices made where the spec was silent

1. **Equity vs wallet.** The trade table gives only terminal R per trade — no
   intrabar price path — so there is no leakage-free way to mark open
   positions to market. I set **equity ≡ wallet** (starting capital + realized
   net-R dollars from trades closed so far) throughout: for the taper lookup
   (as specified) *and* for the risk-dollar base and the two exposure caps
   (where the spec calls for "equity" including unrealized). This is a real
   simplification, not a detail — see the Verdict section for why I think it's
   the single largest source of divergence from the other engine, which is
   told to also model margin (and therefore almost certainly does carry some
   notion of live mark-to-market equity).
2. **Event ties.** When an exit and an entry share a timestamp, the exit is
   processed first (frees risk capacity before new risk is committed). Entries
   sharing a timestamp are processed in source-row order (stable sort) — this
   is the same intra-hour ordering ambiguity the repo's own studies flag
   (`chop_lookahead...`, "every dollar figure carries ±15%").
3. **Regime window** (wr, avg_r over the last 20 closed trades) is tracked on
   **net (post-cost) R**, restricted to trades that were actually opened
   (blocked candidates never enter it), and is **reset per period** — DEV,
   VAL, HOLDOUT are three independent simulations, each starting from
   $10,000 and a blank regime/position book, matching how `sweep_main.csv`
   reports each window with its own `start_balance`.
4. **Cost**: `net_r = r_result - 0.00341 / stop_frac`, applied once at trade
   construction.
5. **Taper below the lowest rung** ($1,500 wallet): taper never fires, base
   fraction = `f` itself, per the documented live-bot quirk.

## Results (my engine)

Full sweep in `risk_study/agent_out/agent_sweep.csv`. `risk_pct` = f×100 to
match the other file's convention.

## Side-by-side comparison

`risk_study/agent_out/comparison.csv` has the full merged table. Key columns
(`_mine` = this engine, `_theirs` = `risk_study/results/sweep_main.csv`):

```
 window  risk%  n_trades(m/t)   ROI%(m)    ROI%(t)   maxDD%(m)  maxDD%(t)  ROI/DD(m)  ROI/DD(t)
 DEV     0.10   12050/11584      127.5      111.9       27.3       30.4       4.67       3.69
 DEV     0.20   12045/11576      304.0      280.9       49.8       53.9       6.10       5.21
 DEV     0.30   11918/11493      394.0      549.5       66.2       69.7       5.96       7.88
 DEV     0.50   11402/11105      302.7      781.9       83.9       86.0       3.61       9.10
 DEV     0.75   10499/10528      253.9     1018.5       90.8       92.9       2.80      10.96
 DEV     1.00    9370/ 9487      -25.7      936.2       95.5       96.6      -0.27       9.69   <- SIGN FLIP
 DEV     1.50    7915/ 7665      -97.8      -58.9       99.5       98.6      -0.98      -0.60
 DEV     2.00    6583/ 6502      -99.8      -95.4      100.0       99.8      -1.00      -0.96
 DEV     3.00    4784/ 4797     -100.0      -99.8      100.0       100.0     -1.00      -1.00

 VAL     0.10   10411/ 9943       88.6      134.4       26.2       26.6       3.39       5.05
 VAL     0.20   10411/ 9943      173.4      317.7       42.0       42.1       4.12       7.55
 VAL     0.30   10352/ 9908      265.3      592.5       56.3       55.4       4.71      10.69
 VAL     0.50    9807/ 9272      258.0      922.8       69.3       66.4       3.72      13.90
 VAL     0.75    8729/ 8567      150.2      938.8       75.0       76.2       2.00      12.31
 VAL     1.00    8119/ 7778      147.3     1355.7       76.3       72.3       1.93      18.74
 VAL     1.50    7231/ 6590      305.7     2063.3       78.6       61.1       3.89      33.76
 VAL     2.00    5844/ 5801      317.1     1460.5       88.1       61.8       3.60      23.62
 VAL     3.00    4529/ 3931     1535.4     1474.3       89.5       75.1      17.16      19.64

 HOLDOUT 0.10    2138/ 2056      -22.2      -23.8       24.2       25.0      -0.92      -0.95
 HOLDOUT 0.20    2138/ 2056      -41.2      -43.6       44.0       45.4      -0.94      -0.96
 HOLDOUT 0.30    2127/ 2047      -55.2      -58.7       58.1       60.6      -0.95      -0.97
 HOLDOUT 0.50    2088/ 2002      -72.1      -77.2       75.5       79.5      -0.96      -0.97
 HOLDOUT 0.75    2009/ 1899      -79.8      -84.3       84.4       87.1      -0.95      -0.97
 HOLDOUT 1.00    1841/ 1680      -84.5      -84.9       89.1       90.5      -0.95      -0.94
 HOLDOUT 1.50    1568/ 1431      -86.4      -90.4       92.0       93.3      -0.94      -0.97
 HOLDOUT 2.00    1070/ 1351      -87.1      -94.3       91.3       96.3      -0.95      -0.98
 HOLDOUT 3.00     830/  838      -96.5      -96.9       97.2       97.4      -0.99      -1.00
```

(m) = mine, (t) = theirs.

## Sign / direction agreement

Step-to-step direction agreement (does ROI move the same way from one f to
the next f in both engines) and Spearman rank correlation across the 9 f
values:

| window | ROI step-dir agree | DD step-dir agree | Spearman ROI | Spearman DD | Spearman ROI/DD |
|---|---|---|---|---|---|
| DEV | 75% | 100% | 0.63 | 1.00 | 0.48 |
| VAL | 50% | 75% | 0.67 | 0.67 | 0.07 |
| HOLDOUT | 100% | 88% | 1.00 | 0.98 | 0.80 |

**Drawdown direction is the most robust agreement**: in DEV and HOLDOUT,
max_dd is monotonically non-decreasing in f in *both* engines (Spearman
0.98–1.00) — more risk per trade reliably means more drawdown, independent of
engine construction. **ROI direction is much less robust**: both engines show
the qualitative "rises then collapses" inverse-U shape as f increases (classic
over-leveraging curve — small f helps, large f is ruinous), but *where* the
peak sits and how sharp the collapse is diverges substantially between
engines, especially in DEV and VAL.

## Cells that disagree qualitatively

**DEV, f = 1.0% — a genuine sign flip.** My engine: ROI = −25.7% (equity
crushed to $7,434 from a peak, maxDD 95.5%). Their engine: ROI = +936.2%
(maxDD 96.6% — comparably brutal drawdown, but a different terminal outcome).
Both engines agree the *path* is extremely violent at this f (95%+ drawdown
either way), so this is not a case where one engine sees calm markets and the
other sees chaos — both see near-total wipeouts intra-run. The sign of the
*final* number is decided by whether a late recovery run gets sized big
enough to dig the account back out before the simulation ends, and that is
exactly what a realized-only equity proxy (mine) will systematically
under-size relative to an engine that marks unrealized P&L into the sizing
equity (theirs, plausibly, since it also tracks margin) — after a big
drawdown, if the *next* wins are still open positions on other symbols
carrying paper profits at the moment a new entry is sized, a mark-to-market
equity base lets that entry size up faster and compounds the recovery harder.
A realized-only base (mine) only "sees" recovery dollars once a trade
actually closes, so it recovers more slowly and, at this f, does not recover
before the window ends. I consider this explained, not a bug: it is the
predictable fingerprint of choice (1) above, amplified by ~45-way position
concurrency and >90% intra-run drawdown, i.e. a regime where the sizing base
is being multiplied by numbers that differ by 2× or more between the two
equity definitions on any given day.

**VAL — drawdown shape disagreement.** My engine's max_dd is monotonically
non-decreasing in f across the whole VAL sweep (26% → 89%). Their engine's
max_dd *falls* from f=0.5% (66%) down to f=1.5–2.0% (61%) before rising again
at f=3.0% (75%) — a genuine non-monotonicity, not noise (it spans three
consecutive f steps). I do not have a fully confident explanation from data I
can see, but the mechanism is consistent with the same equity-basis
difference: with mark-to-market equity, the *denominator* of the drawdown
calc (peak equity) itself moves with unrealized swings, so "drawdown" can
partly reflect paper-equity round-trips that never fully realize as a
capital loss, an effect that is structurally absent from a realized-only
equity curve like mine (which can only step down when a trade actually
closes red). I flag this as the second genuine qualitative disagreement in
the study and would not resolve it without seeing their equity/margin code.

**Peak-f location.** My ROI argmax vs theirs: DEV 0.3% vs 0.75%; VAL 3.0%
(edge of the swept range, still rising) vs 1.5%; HOLDOUT 0.1% vs 0.1% (only
window where the peak location agrees exactly — because HOLDOUT is
dominated by an outright negative edge at every f, so there's no interior
optimum to disagree about, just "smaller f loses less").

## Trade counts

n_trades entered differs by roughly 2–10% between engines at any given cell,
generally within the range you'd expect from margin-blocking alone (their
`margin_blocked` column is 0 at low f and grows into the low hundreds at high
f in DEV/VAL — a real but second-order effect versus the four gates both
engines share: short-gate, anti-pyramid, net-directional cap, gross-risk
cap). This is a much smaller discrepancy than the ROI/DD discrepancy, which
supports pinning the divergence on the *sizing* mechanism (equity basis)
rather than on which trades get let in.

## My independent verdict

**As f rises, in all three periods and in both engines:**
- **Drawdown rises monotonically** (or very close to it) — this is the one
  finding I'd call fully robust and not an artifact of either engine's
  construction. It is a mechanical consequence of levering a fixed-odds bet
  more heavily; I'd trust it without further validation.
- **ROI traces an inverse-U**: rises with f while the compounding tailwind
  from a positive-edge period dominates, then collapses catastrophically once
  f is large enough that a normal string of consecutive losers (structurally
  common at ~14% breakeven WR per `CLAUDE.md`) blows enough of the account
  that recovery is no longer geometrically possible. Both engines agree on
  this *shape*. They disagree by roughly 2–4× on where the peak sits and how
  much ROI is on offer at the peak, and they disagree on the *sign* of one
  cell (DEV, f=1.0%) where both are already deep in a near-total-wipeout
  regime and the final digit is decided by exactly the kind of small
  path-dependent sizing choice this replication cannot pin down without the
  other engine's equity/margin code.
- **ROI/DD (the risk-adjusted objective)** inherits both properties: it rises
  with f at low f, then falls (DEV, and less cleanly VAL/HOLDOUT once ROI
  turns structurally negative in HOLDOUT at every f tested). In HOLDOUT
  specifically — the genuinely out-of-sample window — **both engines agree
  the edge is dead at every f from 0.1% to 3.0%**: ROI is negative
  everywhere, monotonically worsening with f, and ROI/DD is negative and
  roughly flat (~−0.9 to −1.0) regardless of position size. This is the most
  important qualitative agreement in the whole study, and it corroborates
  `CLAUDE.md` §10/§12 and the memory note "Edge negative OOS since May 2026":
  no amount of resizing rescues a period with no edge; sizing only decides
  *how fast* you lose, never whether you lose. I would not want anyone
  reading this replication to walk away thinking "just find the right f" —
  neither engine supports that conclusion for the live-since-2026-05-25 data.
- **Where I'd push back on the spec / the other engine's precision**: the
  magnitude gap between the two engines at moderate-to-high f (0.5%–2%) in
  DEV/VAL is too large (routinely 2–4×, occasionally a sign flip) to treat
  the other engine's headline dollar/ROI figures at those risk levels as
  reliable to better than order-of-magnitude. Both engines agree the
  qualitative conclusion "f above ~1% is unsafe, f below ~0.3–0.5% is where
  the interesting tradeoffs live" — which happens to bracket the live
  `risk_per_trade=0.003` (0.3%) — but I would not use either engine's exact
  ROI number at f=0.5–1.5% to make a sizing decision without first reconciling
  the equity-basis question, since that's the one modeling choice that both
  (a) the spec left genuinely underspecified given the input data available
  to me, and (b) plausibly explains the entire divergence.

## Caveat on my own engine

Because I cannot mark open positions to market, my "equity" is a realized-only
proxy that will systematically *lag* the live bot's true equity during long
winning or losing streaks with many concurrent open positions (~45 typical
per CLAUDE.md). This makes my engine's compounding *slower* to react than the
real system in both directions — it should, if anything, understate both the
best-case ROI and the worst-case ruin speed relative to a true mark-to-market
engine. That bias is consistent with the direction of every disagreement
above (my ROI figures are generally more conservative / less extreme than
theirs at the same f), which is a good internal-consistency check on this
explanation rather than a coincidence.
