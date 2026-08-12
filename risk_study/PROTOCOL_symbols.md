# Pre-registered protocol — symbol universe and (RR, ATR) selection

Written **2026-08-11, before any result was computed.** Everything below — arms, metrics,
fold boundaries, decision rule — is fixed at time of writing. Any deviation is recorded in
a DEVIATIONS section at the bottom with a reason, rather than silently applied.

The point of writing this first: the question "what are the right symbols and parameters?"
is the single easiest question in this repo to overfit, and the repo has already overfit it
once. `STRATEGY_VERDICT_2026-08-11.md` §1.2 showed the historical edge came from per-symbol
RR/ATR selection with zero out-of-sample persistence. A study that re-runs the same search
with a better optimiser will get the same artefact back, prettier. So the protocol is built
to *measure whether selection transfers at all*, not to find the best-looking selection.

---

## 1. Question

Should the bot's symbol universe and/or its per-symbol `(divergence_type, rr, atr_mult)`
assignments be changed from what is in `config.yaml` today?

Answer must be one of: **change the universe**, **change the parameters**, **change both**,
or **keep as is** (including "keep as is, and stop trading it" as a valid outcome).

## 2. Data

- **In-universe**: 277 live symbols, `cache_3yr_1h/`, 2023-06-01 → 2026-07-25.
- **Out-of-universe**: 232 symbols the live config does not trade, re-pulled to the same
  end date into `cache_outside/`. Without this the universe question cannot be asked
  out-of-sample at all — the shipped cache stops these symbols at 2026-05-25.
- Signals, BOS and exits reconstructed exactly as in `risk_study/build_risk_universe.py`.

### Fixed measurement decisions (inherited from the risk study, not re-litigated)
| | value | source |
|---|---|---|
| CHOP gate | `ch[bos]`, threshold 52 | lookahead fix, verdict §2.1 |
| Round-trip cost | **24.2 bps** | measured, 133 live stop-outs in `exec_log` |
| Entry truncation | last **21 days** dropped | right-censoring, verdict §2.4 |
| Tie resolution | stop wins | repo-wide pessimistic convention |
| Portfolio engine | `backtest_production_correct.run_simulation`, live overlays | with both 2026-08-11 sort fixes |
| Risk per trade | **0.3%**, unchanged | isolate selection; sizing was settled separately |

## 3. Parameter grid

`rr ∈ {2, 3, 5, 8, 10}` × `atr_mult ∈ {1.0, 1.5, 2.0, 3.0}` × 4 divergence types.
The live config uses rr ∈ {3,5,8,10} and atr_mult ∈ {1.0,1.5,2.0}; the grid extends one
step in each direction so the live choice is interior, not on a boundary.

## 4. Folds

Anchored walk-forward, train → immediately-following test, no gap, no reuse:

| fold | train | test |
|---|---|---|
| F1 | 2023-06-01 → 2024-06-01 | 2024-06-01 → 2024-12-01 |
| F2 | 2023-06-01 → 2024-12-01 | 2024-12-01 → 2025-06-01 |
| F3 | 2023-06-01 → 2025-06-01 | 2025-06-01 → 2025-12-01 |
| F4 | 2023-06-01 → 2025-12-01 | 2025-12-01 → 2026-05-25 |
| **HOLDOUT** | 2023-06-01 → 2026-05-25 | **2026-05-25 → 2026-07-04** |

The holdout is the same untouched window as the risk study, ending 21 days before data end
per the truncation rule. **It is not inspected until every arm is frozen.**

## 5. Arms (pre-declared — no arm may be added after seeing results)

| id | arm | what it tests |
|---|---|---|
| **A0** | LIVE — today's `config.yaml`, frozen | the incumbent |
| **A1** | GLOBAL — one `(rr, atr_mult)` for all symbols/types, picked on train | is per-symbol fitting worth anything over one global choice? |
| **A2** | WF-PERSYMBOL — best `(rr, atr_mult)` per `(symbol, div_type)` on train | the bot's own method, honestly walk-forwarded |
| **A3** | WF-SHRUNK — A2 with a ≥30-trade minimum and shrinkage toward A1's global | does regularisation rescue per-symbol fitting? |
| **A4** | LIQUIDITY — symbols ranked by median hourly turnover only, global params | does a **non-performance** selection rule transfer where a performance one doesn't? |
| **A5** | ALL — every symbol, every div type, global params, no selection at all | the no-selection baseline |
| **C1** | RANDOM — symbols/params drawn at random, size-matched to A2 | the null. If A2 doesn't beat this, selection is noise |
| **C2** | BTC buy-and-hold | market control (verdict §2.6 flags its omission) |
| **C3** | long-only equal-weight basket of the universe | market control |

Every arm is evaluated on the **same test bars** with the **same** cost, CHOP gate,
truncation and portfolio engine. Only the symbol/parameter assignment differs.

## 6. Metrics

Primary: **mean net R per trade on test folds**, and portfolio **ROI / max drawdown**.
Secondary: trade count, win rate, profit factor, ROI/DD, Calmar, per-fold win/loss record.

**Selection-transfer statistic** (the diagnostic that actually decides this):
for each `(symbol, div_type, rr, atr_mult)` cell with ≥30 train trades and ≥30 test trades,
the Spearman rank correlation between train-fold mean net R and test-fold mean net R,
computed within each fold and pooled. Reported with a block-bootstrap CI.

Interpretation, fixed now:
- ρ ≥ 0.30 → selection carries real information; refitting is justified.
- 0.10 ≤ ρ < 0.30 → weak; only heavily shrunk selection defensible.
- ρ < 0.10 → **selection is noise. Do not refit anything.**

## 7. Decision rule (fixed before results)

Recommend **changing the bot** only if a challenger arm satisfies *all* of:

1. beats A0 on mean net R in the **holdout**, and
2. beats A0 in **≥ 3 of 4** walk-forward test folds, and
3. beats **C1 (random)** by more than the bootstrap CI half-width, and
4. beats **C2/C3** (market controls) on ROI/DD, and
5. its advantage is not concentrated in the top 1% of trades (drop-best-trades test), and
6. is independently reproduced by a second agent on an independent implementation.

If no arm clears all six: **keep the bot as is.** If additionally A0 itself is negative on
the holdout and fails the market controls, the honest recommendation is **keep as is and
stop trading it** — not "pick the least-bad arm".

A challenger that wins only in-sample is explicitly a *negative* result for changing the bot.

## 8. Known limitations, stated up front

- The holdout is ~6 weeks after truncation. It can refute a strong claim; it cannot
  establish a weak one.
- Out-of-universe symbols were re-pulled today, so their history is subject to Bybit's
  retention and to survivorship: symbols delisted before today are absent entirely. Any
  A4/A5 result is therefore biased **optimistic** and must be read as an upper bound.
- The backtest signal detector differs from live's by ~11.7% of signals (dedup ordering,
  found by the leakage audit). Every arm inherits the same difference, so comparisons are
  fair, but absolute numbers are not live-exact.
- Risk-per-trade is held at 0.3% throughout. This study does not revisit sizing.

---

## DEVIATIONS

*(appended during execution, with reasons)*
