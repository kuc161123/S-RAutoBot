# AUDIT D — Dead Weight and Operational Surface

Read-only investigation. Nothing under `autobot/` or `config.yaml` was touched. All line
refs verified against the working tree at commit `263a4ff` (2026-08-12 config,
`bot.py` 3305 lines, `bybit.py` 1610 lines — both have grown since the CLAUDE.md doc
date of 2026-07-25, so line numbers there are stale; the ones below are current).

---

## 1. Dead code — verified by import trace from `main.py`

`main.py` imports only `autobot.core.bot.Bot4H` and `dashboard.app`. Traced every file
that imports each allegedly-dead module (`grep -rl`, excluding the file's own
self-references and the `.claude/worktrees/` scratch copy):

| File | Lines | Importers (live tree) | Verdict |
|---|---|---|---|
| `autobot/core/bot_5m_old.py` | 4054 | **none** | dead — nothing imports it, no deploy config invokes it as a script |
| `autobot/core/unified_learner.py` | 1957 | `bot_5m_old.py` (dead) + `analyze_edge.py` (orphan root script, itself imported nowhere) | dead |
| `autobot/core/smart_learner.py` | 543 | **none** | dead |
| `autobot/core/combo_learner.py` | 500 | **none** | dead |
| `autobot/core/divergence_detector_5m_old.py` | 259 | **none** (not even `bot_5m_old.py`) | fully orphaned |
| `autobot/core/shadow_auditor.py` | 98 | `bot_5m_old.py` only | dead, and additionally broken: `shadow_auditor.py:4` does `from autobot.core.divergence_detector import detect_divergence, DivergenceSignal` — the live module only exports `detect_divergences` (plural, `divergence_detector.py:179`). Confirmed `ImportError` on any attempt to load it. |

No dynamic import path exists that could reach these: `grep -rn "importlib\|__import__\|exec(\|eval("` across `autobot/`, `main.py`, `dashboard.py` returns one hit (`telegram_handler.py:2010`, `__import__('datetime')` — unrelated). No `Procfile`, `Dockerfile`, cron file, or systemd unit in the repo references `bot_5m_old.py`. `main.py`'s only `subprocess` use is `pgrep`/`kill` against stale `main.py`/`Bot4H` processes — unrelated to the dead tree.

**Total dead code: ~7,411 lines in the six core modules above**, plus ~316 lines of dead broker methods (below) = **~7,727 of ~18,000 lines under `autobot/` (43%)**. This matches CLAUDE.md's "~7,000 of 16,000" claim; both tree and dead-code count have grown proportionally since that doc was written. Last commit touching any of the six dead files: `6fe43f2`, 2025-12-27 — over seven months stale.

### Dead broker methods in `bybit.py`

Verified caller-by-caller (`grep -n "\.method(" .` across the whole repo, excluding the `def`):

| Method | Lines | Callers |
|---|---|---|
| `set_tpsl` | 1055–1114 (60) | `bot_5m_old.py:2682` only |
| `set_sl_only` | 1198–1281 (84) | `bot_5m_old.py:2518,2527,3034` only |
| `place_limit` | 797–868 (72) | `bot_5m_old.py:2924,3248` only |
| `set_trailing_sl` | 1352–1427 (76) | **zero callers anywhere**, including `bot_5m_old.py` |
| `place_reduce_only_limit` | 1444–1467 (24) | **zero callers anywhere** |

All five are unreachable from the live path (only reachable transitively through the
already-dead `bot_5m_old.py`, and two aren't even called from there). `set_tpsl` still
has the missing-`await` bug CLAUDE.md notes (`bybit.py:1055` region) — harmless because
nothing calls it.

**Verdict: dead and harmless.** None of these six modules or five methods are reachable
from `main.py` → `Bot4H().run()` by any static or dynamic path found. Safe to delete;
low priority (no maintenance cost while untouched, but they will confuse the next person
who greps for "learner" or "set_sl").

---

## 2. The dangerous one — `unified_learner.py` auto git-push

**Confirmed, quoted verbatim, `autobot/core/unified_learner.py:1105-1113`:**

```python
                # Git commit
                try:
                    subprocess.run(['git', 'add', self.OVERRIDE_FILE], 
                                  capture_output=True, timeout=10)
                    subprocess.run(['git', 'commit', '-m', f'Auto-promote: {symbol} {side}'],
                                  capture_output=True, timeout=10)
                    subprocess.run(['git', 'push'], capture_output=True, timeout=30)
                except:
                    pass
```

This lives inside `_promote_combo()` (`unified_learner.py:1082`), called from
`_check_promotion()` when a symbol/side/combo's trailing 30-day Wilson-lower-bound win
rate and EV clear configured thresholds (`PROMOTE_MIN_TRADES`, `PROMOTE_MIN_LOWER_WR`,
`PROMOTE_MIN_EV`, all class constants). `OVERRIDE_FILE = 'symbol_overrides_VWAP_Combo.yaml'`
(`unified_learner.py:137`) — the file it writes and then unconditionally
`add`/`commit`/`push`es to whatever `origin` and branch the deploy checkout is on. A
bare `except: pass` swallows every failure silently — no log line, no Telegram alert, on
either the write or the push.

**Reachability, exhaustively:**
- `UnifiedLearner` is instantiated in exactly two places: `bot_5m_old.py:288` (inside
  `Bot4H`-analogue `__init__`) and `bot_5m_old.py:3445`. Both are inside the dead file.
- `_check_promotion` / `_promote_combo` are only ever called from within
  `unified_learner.py` itself and from `bot_5m_old.py`'s own event loop.
- `main.py` never imports `bot_5m_old` or `unified_learner`, directly or transitively.
- No cron, systemd unit, or CI/deploy config anywhere in the repo invokes
  `bot_5m_old.py` as a standalone script.

**Verdict: currently unreachable from the live trading process, full stop.** But this is
the one item in the whole dead tree that is not merely inert weight — it is a **live
loaded gun**, for three compounding reasons:
1. It writes to disk and pushes to `origin` on its own initiative, with no human in the
   loop and no way to see it happened (bare `except: pass`).
2. This repo — per the audit's own framing — auto-deploys from `main`. A successful push
   from a bot process would ship unreviewed, bot-authored config changes straight to the
   thing trading real money, on the same branch as everything else in this audit.
3. It sits in a file (`unified_learner.py`) that is one plausible refactor away from being
   reconnected — e.g., "let's revive the combo-promotion idea" is a completely reasonable
   sentence for a future session to say, and nothing marks this function as radioactive
   short of this audit and the CLAUDE.md note.

**Recommendation (severity: HIGH, act this week even though currently inert):**
Delete `unified_learner.py`, `smart_learner.py`, `combo_learner.py`, `bot_5m_old.py`,
`divergence_detector_5m_old.py`, `shadow_auditor.py` outright rather than leaving them as
"dead but present." Keeping unreachable-but-loaded code that can commit-and-push on a
trading bot's own credentials is a standing hazard independent of whether today's call
graph reaches it — the safest state is one where the capability does not exist in the
tree at all. If any of these six files are wanted for reference, move the git-push logic
out into a diff/PR-only design (write a proposal file, never call `git push`) before
resurrecting anything, and require the caller to pass an explicit `dry_run=False` with
no default. Do not "fix" this by adding a feature flag inside the dead file — the flag
itself is another thing that has to stay correctly set forever; delete the capability.

---

## 3. Dead config — verified by grep, zero reads anywhere in the live tree

`load_config()` (`autobot/core/bot.py:1083-1114`) reads exactly three top-level
`config.yaml` keys: `strategy` (→ `self.strategy_config`), `risk` (→ `self.risk_config`),
and, separately in `setup_broker()`, `bybit`. `telegram` is read in `setup_telegram()`
(`bot.py:1296`). Nothing else at the top level is ever touched.

Confirmed via `grep -rn "config\.get(['\"]<key>['\"]" autobot/ main.py dashboard.py`
— zero hits in the live tree for every one of these:

| config.yaml section | Lines | Status |
|---|---|---|
| `execution:` (`enabled`, `entry_slippage_pct`, `fee_pct`) | 12-15 | never read |
| `indicators:` (`atr_period`, `ema_periods`, `rsi_period`) | 16-20 | never read — the real periods are hardcoded module constants in `divergence_detector.py` |
| `legacy:` (`partial_take_profit`, `scalp_mode`, `trailing_stop`, `volume_filter`, `vwap_filter`) | 21-26 | never read — note the name collision: `legacy.trailing_stop: false` looks like it should control the real trailing stop, but the live trailing-stop switch is `risk.trailing_stop.enabled` (`config.yaml:164`), a completely different key. This is a booby trap: someone scanning for "how do I turn off trailing" who edits `legacy.trailing_stop` will change nothing. |
| `monitoring:` (`health_check`, `heartbeat_interval_sec`) | 27-29 | never read — there is no heartbeat mechanism in the live bot |
| `notifications:` (`daily_summary`, `entry`, `exit`, `pending_signals`, `startup`) | 30-35 | never read — Telegram notification behavior is unconditional/hardcoded at each call site, not gated by these flags |

Also confirmed dead **inside** the live-read `risk:` block — these two keys exist,
read like safety features, and are read **nowhere**:

| Key | Value | Grep result |
|---|---|---|
| `risk.max_daily_loss` | `0.1` (`config.yaml:37`) | zero references anywhere in `autobot/`, `main.py`, `dashboard.py` — no daily-loss circuit breaker exists |
| `risk.max_position_size_pct` | `0.15` (`config.yaml:38`) | zero references anywhere — no per-position size cap exists |

Every other key checked in `risk:` (`risk_per_trade`, `net_directional_cap`,
`overlay_ramp_min_balance`, `withdrawal_target`, `btc_short_gate`, `short_gate_ret30`,
`gross_open_risk_cap`, `long_bull_boost`, `shadow_gate.*`, `trailing_stop.*`,
`taper_schedule`) has live-tree references and is genuinely wired in. `max_daily_loss`
and `max_position_size_pct` are the only two orphans in that block.

Two more orphans outside the `execution/indicators/legacy/monitoring/notifications`
blocks: `strategy.entry_params.max_wait_candles` and `strategy.signal_params.lookback_bars`
don't exist anywhere in the current `config.yaml` at all (confirmed by grep), so
`bot.py:1093,1096` silently fall back to code defaults (`12`, `50`). Not a "wrong value"
risk since the defaults happen to be the intended ones, but it means editing those keys
in `config.yaml` today — a completely reasonable thing to try — does nothing, silently.

**Why this matters more than plain dead code:** `execution`, `legacy`, `monitoring`,
`notifications` are inert and easy to spot once you know `load_config()` only reads three
top-level keys. `max_daily_loss` / `max_position_size_pct` are worse — they sit inside
the block that genuinely *is* live-wired, styled identically to real risk controls, with
plausible-sounding values (10% daily loss cap, 15% position size cap). An operator
reading `config.yaml` cold has no way to tell these two apart from `net_directional_cap`
or `gross_open_risk_cap` two lines away, which are both very much real. **This is false
safety confidence, not just dead weight.**

**Recommendation (severity: MEDIUM, low effort):** Either wire up `max_daily_loss` and
`max_position_size_pct` (both would be genuinely useful — there is currently no daily
circuit breaker and no per-position cap at all, see §5) or delete them from
`config.yaml` and add one line to CLAUDE.md/README stating plainly that no daily-loss or
per-position-size control exists. Do not leave them present-but-inert; that is strictly
worse than either alternative. Also delete `execution`, `indicators`, `legacy`,
`monitoring`, `notifications` — especially `legacy.trailing_stop`, which actively
misleads about which key controls the real trailing stop.

---

## 4. Shadow layer — cost/benefit

Four modules, all wrapped in bare `try/except` at every call site in `bot.py`, confirmed
never to touch trading state (no shadow output feeds `execute_trade`'s entry gates
except the one deliberate, documented exception in §4c below).

### 4a. What it costs, quantified

Everything below runs from the **60-second** main loop tick (`bot.py:3288`,
`asyncio.sleep(60)`), not the hourly sweep:

- `shadow_logger.resolve_pending(broker)` (`bot.py:3257`) — pulls up to **40** pending
  signal rows per call (`shadow_logger.py:318`, `limit_rows=40`) and issues **one
  `get_klines` call per row** (`shadow_logger.py:343`). Worst case: 40 extra API calls
  *every minute*.
- `trail_shadow.resolve_pending(broker)` (`bot.py:3265`) — up to **25** rows/call
  (`trail_shadow.py:315`), one `get_klines` per row. Worst case: 25 more API calls/minute.
- Combined worst case: **up to 65 extra `get_klines` calls per minute (~3,900/hour)** on
  top of the primary 277-symbol hourly sweep's own several-round-trips-per-symbol
  footprint — and there is no rate limiter anywhere (§5). This is a real, continuous
  contributor to the total request volume the bot generates against Bybit, not a rounding
  error.
- `shadow_scanner.tick(broker)` (`bot.py:3273`) — internally throttled: candidate
  discovery once/day (`DISCOVERY_INTERVAL_S=86400`, `shadow_scanner.py:38`), the
  candidate sweep itself once/~55min (`SCAN_INTERVAL_S=3300`, `shadow_scanner.py:39`),
  one `get_klines` per candidate (up to `CANDIDATE_CAP`, staggered by
  `SCAN_STAGGER_S`) plus one `get_all_tickers` call on discovery days. Cheap relative to
  the above.
- `shadow_logger.log_signal` / `mark_executed` (`bot.py:1824,2232`) — pure DB writes, one
  per BOS-confirmed signal (executed or blocked). Negligible API cost, but this is the
  write path that fills `shadow_signals` continuously — the table this whole layer's
  Postgres footprint is built on.
- `trail_shadow.log_open/log_move/log_close` (`bot.py:2226,2510,2791`) — pure DB writes,
  fired on every entry, every trail ratchet, and every exit. Also negligible per-call but
  continuous.

### 4b. What it actually produces that is used

- `/trailstats`, `/learn`, `/edge` Telegram output — advisory reading only.
- **One genuine load-bearing consumer**: `_shadow_gate_allows_entry()`
  (`bot.py:858-937`) calls `shadow_analyst.weekly_r()` every `recheck_minutes` (default
  20 min) and, in `advisory` mode (the current `config.yaml:127` setting), uses the
  result only to flip `lifetime_stats['shadow_gate_open']` and alert — it **never blocks
  a trade in code** while `mode: advisory` (verified: `_verdict()` at `bot.py:901-902`
  returns `True` unconditionally unless `auto == True`, and `config.yaml:127` sets
  `advisory`). So today, nothing in the shadow layer gates a single live trade.

### 4c. The `/learn week` −315R number — verified, and more nuanced than "just a bug"

`ShadowAnalyst.weekly_r()` (`shadow_analyst.py:211-244`) calls `self._rows(days=..,
family='DIV')` (`shadow_analyst.py:69`), which has no `executed`/`executed_only`
parameter at all — it filters only by `status` (default `win/loss/expired`) and
optionally `family`/`days`. `ShadowLogger.edge_rows()` does support
`executed_only=True` (`shadow_logger.py:432-438`), but `ShadowAnalyst` never calls
`edge_rows` — it has its own `_rows()` that always includes blocked signals. Confirmed:
**`/learn week`'s headline number sums cost-adjusted R over every evaluated DIV signal,
executed or blocked, exactly as CLAUDE.md describes.**

But the docstring on `_shadow_gate_allows_entry` (`bot.py:871-879`) makes clear this is
**deliberate, not an oversight**, for the one place that actually consumes the number:
gating on executed-only would create a lock-in — halt live entries, and the "executed"
sample instantly stops growing, so the gate could never see the edge return and reopen.
Including blocked signals is what lets the gate self-correct. That reasoning is sound and
this audit found no flaw in it.

The problem is narrower than "the metric is wrong" — **it is that the same number is
shown to a human on `/learn week` with no "executed only" companion figure**, so a
human reading "−315R this week" cannot tell whether that reflects the live book's
actual results or mostly reflects a few hundred R of signals the bot correctly declined
to take. CLAUDE.md's own framing ("conflates 'signals we looked at' with 'how the bot
did'") is accurate for the *display*, but should not be read as license to change the
*gate's* calculation — that would undo the anti-lock-in property the docstring
specifically calls out. **These are two different fixes and must not be conflated.**

**Recommendation (severity: LOW — advisory-only, nothing here currently touches
trading):** Keep the shadow layer; it is cheap in dollars (Postgres writes) even if
non-trivial in API call volume, and the trailing-shadow-R gate has real backing evidence
per CLAUDE.md/config.yaml comments. Two concrete, additive fixes, neither of which
touches `_shadow_gate_allows_entry`'s calculation:
1. Add an `executed_only=True` companion line to `/learn week`'s output (call
   `ShadowLogger.edge_rows(executed_only=True)`, which already exists and is unused by
   `ShadowAnalyst` — the plumbing is already there) so a human reading the dashboard can
   tell the two apart.
2. Given the up-to-65-calls/minute API cost in §4a, if Bybit rate limits are ever
   actually hit (nothing in the codebase currently detects this — see §5), the first two
   places to throttle down are `shadow_logger.resolve_pending`'s `limit_rows` and
   `trail_shadow.resolve_pending`'s `limit_rows`, since they cost real API budget for a
   feature that is advisory-only today.

---

## 5. Operational risks no backtest prices

### Delisting

`grep -rn -i "delist" autobot/` returns **zero hits** — there is no delisting-specific
code path anywhere. Handling is entirely incidental, via two generic mechanisms:

- **New entries**: `fetch_4h_data` (`bot.py:1310-1358`) returns `None` if the klines
  endpoint returns no data or errors (`bot.py:1335-1342,1356-1358`); nothing downstream
  can act on a symbol with `df is None`. So a delisted symbol simply stops generating new
  signals once its klines feed dries up — no crash, no retry storm. This part is benign.
- **Open positions**: no code distinguishes "this symbol delisted and Bybit force-settled
  the position" from any other stale-position case. It falls through the generic
  `sync_with_exchange` → `handle_trade_exit` path (`bot.py:1184,2583`). That path is
  reasonably defensive in general (won't mass-close on an empty `get_positions()`,
  §5-margin below), but its terminal fallback for "position confirmed gone, no matching
  `closed-pnl` record after 3 retries" is: **default to a flat −1R loss, `pnl_usd=0`**
  (`bot.py:2716-2720`, `logger.warning("Position confirmed closed, no PnL record —
  defaulting to -1R LOSS")`). A delisting settlement is a forced closure at the
  exchange's mark/settlement price, not a normal SL/TP fill — there is no guarantee
  Bybit's `closed-pnl` endpoint carries it in a form the side/entry-price-within-1%
  matcher (`bot.py:2628-2637`) can recognize in time. If it doesn't match, **the trade's
  true settlement PnL (which could be a partial win, not a full loss) is silently
  recorded as exactly −1R / $0**, corrupting per-trade R stats, the regime multiplier
  (which reads the last 20 trades' WR and avg R), and `daily_r`/`total_r` — while the
  bot's actual USD balance moves by the real settlement amount, so the dollar and R
  tracks would silently diverge for that trade, on top of the seam already documented in
  CLAUDE.md §6 for missed closes generally.

**Verdict: no delisting-aware code exists; the generic fallback is defensive against
crashing/hanging but not against silently mis-recording the settlement value.** Given
the stated ~10–30 forced closures/year on a 277-symbol book (`risk_study/agent_out/symbols/`),
this is a low-frequency, low-severity-per-event, but real and unaddressed data-integrity
gap.

**Recommendation (severity: LOW-MEDIUM, cheap to build):** Detect delisting explicitly —
Bybit's `/v5/market/instruments-info` reports `status` (e.g., `Trading` vs
`Closed`/`Delisted`); a periodic check (even daily) against the enabled-symbol list
could flag a symbol as delisting and (a) alert via Telegram immediately rather than
waiting for the generic stale-trade path, (b) not fall back to a blind −1R assumption for
that specific case.

### Margin exhaustion

Confirmed in `bot.py:2069-2100`: there **is** a per-trade pre-flight margin check.
`required_margin = (raw_qty * entry_price) / leverage`; `available_balance =
account_balance − Σ positionIM` (summed fresh from `get_positions()` at that instant,
`bot.py:2090-2093`); if `required_margin > available_balance` the entry is skipped
(`bot.py:2098-2100`). Two gaps:

1. **Silent relative to its sibling gates.** CHOP, the BTC short-gate, the opposite-side
   guard, the net-directional cap, and the gross-risk cap all send a Telegram message when
   they block an entry. The margin-insufficiency skip does not — it is `logger.warning`
   only (`bot.py:2099`), invisible outside the server log. An operator watching Telegram
   would have no idea the book is margin-constrained from this signal alone.
2. **Fails open on its own aggregate-margin lookup.** The `Σ positionIM` computation is
   wrapped in a bare `except: pass` (`bot.py:2094`) — if `get_positions()` errors, the
   code falls back to `available_balance = account_balance`, i.e. **assumes zero margin
   is currently in use**, which is the least conservative possible fallback for a check
   whose entire purpose is to avoid over-committing margin. In practice a genuinely
   over-margined order would then simply be rejected by the exchange, which **does**
   trigger a Telegram alert (`bot.py:2158-2161`, "TRADE FAILED"), so this degrades to a
   noisier, later failure rather than a silent one — but the pre-flight check's job is
   specifically to avoid needing that fallback.
3. **No portfolio-level heartbeat.** Nothing computes or alerts on "aggregate margin
   utilization is at 92% of equity" as a standing state — only the marginal effect of
   the next candidate trade is checked, and only when a signal actually fires. Per
   `risk_study/agent_out/LIQ_ANALYSIS.md` (this audit's sibling workstream), aggregate
   initial margin already reaches ~92% of equity at the 99th-percentile hour and ~134% at
   the historical max under today's 0.3% risk with the gross-risk cap applied — meaning
   *silent order rejection due to margin is already an occasional live condition*, and
   there is no dashboard line, log aggregate, or alert that surfaces this as a portfolio
   state rather than one skipped symbol at a time.

**Recommendation (severity: MEDIUM):** Add a Telegram alert on the margin-insufficiency
skip (bring it in line with every other gate), and stop failing open on the
`get_positions()` lookup — if the aggregate-margin computation itself fails, treat that as
"unknown, be conservative" (skip) rather than "assume zero used." A periodic (e.g.
hourly) "margin utilization: X% of equity" line on the dashboard would close the
portfolio-level blind spot the per-trade check can't see.

### No max-concurrent-position limit

`grep -rn -i "max_concurrent\|max_open_positions\|concurrent_limit\|position_limit\|max_positions" autobot/`
returns **zero hits**. Confirmed structurally: nothing in `execute_trade`'s ten-gate
sequence (CLAUDE.md §5's table, still accurate) counts open positions and blocks on a
count. The only things that indirectly cap concurrency are the risk-based caps
(`net_directional_cap`, `gross_open_risk_cap`) and margin availability itself — none of
which bound position *count*, only aggregate *risk*/*margin*. With 277 symbols × up to
several configs each, the structural ceiling is bounded only by margin exhaustion (above)
and the risk caps, not by any explicit count. 45+ concurrent positions "is normal" per
CLAUDE.md and nothing here contradicts that.

**Verdict: confirmed, no code change needed to establish this — it's an absence, not a
bug.** Whether a count cap is *worth adding* is an alpha/drawdown question for workstream
A/E, not this one; flagging here only as the structural fact this workstream was asked to
confirm.

### `_get_precisions` fallback — CLAUDE.md claim is STALE, already fixed

CLAUDE.md (§11) states the hardcoded `("0.0001", "0.001")` fallback is cached for the
session. **This is no longer true.** Current code (`bybit.py:1115-1155`) explicitly does
**not** cache the fallback:

```python
        # Fallback - use conservative defaults and log loudly
        # Deliberately NOT cached. Caching a guessed tick size pinned the wrong value
        # for the rest of the session, so every later price rounding for this symbol —
        # including trailing-stop amendments — used it even after the API recovered.
        # Leaving it uncached means the next call retries the real lookup.
        logger.error(f"❌ USING FALLBACK PRECISION for {symbol}: tickSize=0.0001, "
                     f"qtyStep=0.001 (not cached — will retry next call)")
        return ("0.0001", "0.001")
```

This was fixed in commit `4f97030` ("fix: five measurement/integrity bugs — no change to
trading behaviour"), after the CLAUDE.md doc date. Every subsequent call re-attempts the
real API lookup rather than being permanently pinned to a wrong tick size. Residual
severity is now low: a single unlucky call during an API blip still rounds one order to
`0.0001`/`0.001`, which on a cheap alt with a much finer real tick size could round a
tiny quantity to zero or reject on lot size — but it self-heals on the very next call
instead of persisting for the rest of the session. Given the current global
`rr=10/atr_mult=3.0` config produces wider stops and correspondingly smaller quantities
(per the audit prompt's own framing), a zero-quantity order from a bad rounding is a real
if rare failure mode, but no longer a *persistent* one.

Also noted in passing: `preload_all_precisions()` (`bybit.py:1157`) exists specifically to
warm this cache for all symbols at startup, and is well-suited to eliminating the
first-touch "cache miss" warning + extra API round-trip per symbol — but it is **only
called from the dead `bot_5m_old.py:356`**, never from the live `Bot4H`. This is a cheap,
safe addition (not a fix — nothing is broken without it) that would quiet the log and
save one API call per symbol on its first live touch each session.

**Recommendation (severity: LOW, update-CLAUDE.md item + optional cheap addition):**
Correct the stale claim in CLAUDE.md §11. Optionally wire `preload_all_precisions()` into
`Bot4H`'s startup sequence (it already exists, is dead code today, and does exactly the
right thing) to remove the transient bad-rounding window entirely rather than relying on
self-healing.

### No rate limiter

Confirmed: `grep -n "sleep(0.1)\|rate.limit\|RateLimiter\|throttle\|Semaphore"` across
`bybit.py`/`bot.py` finds exactly two `asyncio.sleep(0.1)` calls — one during startup
leverage configuration (`bot.py:1147`) and one inside `bybit.py:55` — and no general
per-request throttle, semaphore, or backoff policy. This compounds with §4a: the shadow
layer alone can generate up to ~65 `get_klines` calls/minute continuously, on top of the
277-symbol hourly sweep's own multi-call-per-symbol footprint (klines, position checks,
balance checks). Nothing in the codebase currently detects or alerts on hitting Bybit's
rate limits (no handling keyed on Bybit's rate-limit retCode anywhere in `bybit.py`) — a
rate-limit rejection would surface only as a generic failed API call, indistinguishable
in the logs from any other transient failure.

**Recommendation (severity: LOW-MEDIUM):** Add a simple token-bucket or semaphore around
`Bybit._request`, and explicitly detect Bybit's rate-limit retCode/HTTP 429 to log it
distinctly from other failures — today a sustained rate-limit condition would be
invisible as a named failure mode even though it's structurally plausible given the call
volume.

---

## Ranked findings

**REMOVE**

1. **[HIGH]** Delete `unified_learner.py`, `smart_learner.py`, `combo_learner.py`,
   `bot_5m_old.py`, `divergence_detector_5m_old.py`, `shadow_auditor.py` (~7,411 lines).
   Reason: `unified_learner.py:1105-1113` contains an unconditional
   `git add && git commit && git push` on its own initiative with errors swallowed
   silently — on a bot whose repo auto-deploys from `main`. Currently unreachable
   (verified — no import path from `main.py`), but "unreachable today" is not a
   substitute for "does not exist," given a five-minute refactor could reconnect it and
   nothing in the tree marks the function as radioactive. This is the one dead-code
   finding that is dangerous, not just inert.
2. **[LOW]** Delete the five dead `bybit.py` broker methods (`set_tpsl`, `set_sl_only`,
   `set_trailing_sl`, `place_limit`, `place_reduce_only_limit`, ~316 lines). Dead and
   harmless (no git/network side effects beyond normal order placement), but pure
   maintenance drag — `set_tpsl` even has a live bug (missing `await`) that a future
   editor could "fix" and reintroduce without realizing it's unreachable.
3. **[MEDIUM]** Delete config.yaml's `execution:`, `indicators:`, `legacy:`,
   `monitoring:`, `notifications:` blocks (config.yaml:12-35). Never read anywhere.
   `legacy.trailing_stop: false` is actively misleading — it looks like the trailing-stop
   kill switch but the real one is `risk.trailing_stop.enabled`.
4. **[MEDIUM]** Either wire up or delete `risk.max_daily_loss` and
   `risk.max_position_size_pct` (config.yaml:37-38). These read as real safety
   controls sitting inside the block that genuinely is live-wired, and are the two keys
   in the whole file most likely to give an operator false confidence. Worse than plain
   dead code because of where they live, not what they do.

**FIX**

5. **[MEDIUM]** Margin-insufficiency skip (`bot.py:2098-2100`) sends no Telegram alert,
   unlike every sibling gate — bring it in line. Also stop failing open on the aggregate
   `Σ positionIM` lookup (`bot.py:2094`, bare `except: pass` defaults to "assume zero
   margin used," the least safe fallback for this specific check).
6. **[LOW-MEDIUM]** No delisting-aware code path exists anywhere (`grep -i delist` = zero
   hits). New-entry behavior is benign (klines dry up, nothing fires). Open-position
   behavior falls back to a blind "−1R, $0 PnL" default (`bot.py:2716-2720`) if no
   closed-PnL record matches within 3 retries — plausible for a forced delisting
   settlement, which would silently mis-record the true settlement value while the real
   dollar balance moves correctly, creating a stats/dollar divergence for that trade.
7. **[LOW]** CLAUDE.md §11's claim that the precision fallback "caches it for the
   session" is stale — fixed in commit `4f97030`; the fallback is now explicitly
   uncached and self-heals on the next call. Update the doc. Residual risk (one
   badly-rounded order during an API blip, more consequential now that wider stops mean
   smaller quantities) is real but no longer persistent.
8. **[LOW]** `/learn week`'s headline number conflates "every signal looked at" with
   "how the bot did" for a human reader (confirmed: `ShadowAnalyst._rows()` has no
   executed-only filter). Do **not** fix this by changing what feeds the shadow gate
   (`_shadow_gate_allows_entry`, `bot.py:858-937`) — that inclusion is deliberate and
   well-reasoned (avoids the gate locking itself shut). Add a second, executed-only line
   using the already-existing `ShadowLogger.edge_rows(executed_only=True)`
   (`shadow_logger.py:432`), which `ShadowAnalyst` currently never calls.
9. **[LOW]** No rate limiter beyond two `sleep(0.1)` calls; no detection of Bybit's
   rate-limit response either. The shadow layer alone can add up to ~65 `get_klines`
   calls/minute (§4a) on top of the primary sweep. Add a request-level throttle and
   distinct logging for rate-limit rejections.

**ADD**

10. **[LOW, cheap]** Wire `Bybit.preload_all_precisions()` (exists, well-designed,
    currently dead — only called from `bot_5m_old.py`) into `Bot4H` startup. Removes the
    per-symbol first-touch cache-miss warning and extra API call, and closes the
    precision-fallback window entirely rather than relying on self-healing.
11. **[LOW]** A portfolio-level margin-utilization heartbeat (e.g., hourly "X% of equity
    committed to initial margin" line on the dashboard). The per-trade pre-flight check
    only ever sees the marginal effect of the next candidate trade; per
    `risk_study/agent_out/LIQ_ANALYSIS.md`, aggregate margin already reaches ~92-134% of
    equity in real historical hours under today's config, and nothing surfaces that as a
    standing state.
12. **[LOW]** Explicit delisting detection via `/v5/market/instruments-info`'s `status`
    field, checked periodically against the enabled-symbol list, with an immediate
    Telegram alert — rather than relying on the generic stale-trade path's blind −1R
    fallback.

No workstream-D item found here is quantifiable in R/trade or dollars — these are
structural/operational findings, not alpha or cost claims, and are reported as such per
the audit's own methodology (an idea without a number is reported as a hypothesis, not
manufactured into one).
