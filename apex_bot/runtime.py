"""Independent cloud loops: data, decisions, reconciliation, reporting, controls."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from dataclasses import replace
from copy import deepcopy
from datetime import datetime, timezone

from .accounting import loss_metrics
from .ai import AIReviewer
from .bybit import BybitClient
from .cache import MarketCache
from .context import ContextService
from .engine import analyze
from .evidence import review_context, context_fingerprint, candidate_fingerprint
from .execution import ExecutionManager, number, client_id
from .models import Opportunity
from .references import ReferenceReader, load_context
from .risk import PROFILES, assess
from .service import Service
from .shadow_observer import ShadowObserver
from .simulation import advance, apply_funding, create_trade
from .storage import NotLeader
from .telegram import TelegramController, TelegramError
from .universe import entry_symbols, refresh_universe

LOG = logging.getLogger(__name__)


class Runtime:
    def __init__(self, config, store, session):
        self.config, self.store, self.session = config, store, session
        self.service = Service(config, store)
        self.client = BybitClient(
            session,
            base_url=config.bybit_url,
            api_key=config.bybit_key,
            api_secret=config.bybit_secret,
        )
        self.cache = MarketCache(config.redis_url)
        self.observer = ShadowObserver(store, self.client, self.cache)
        self.service.observer = self.observer
        self.ai = AIReviewer(
            session,
            store,
            api_key=config.openai_key,
            model=config.openai_model,
            daily_calls=config.ai_daily_calls,
            max_output_tokens=config.ai_max_output_tokens,
        )
        self.execution = ExecutionManager(self.client, store, config.mode)
        self.telegram = (
            TelegramController(
                session,
                config.telegram_token,
                config.telegram_chat_id,
                set(config.telegram_user_ids),
                self.service,
                bot_username=config.telegram_username,
            )
            if config.telegram_token
            else None
        )
        self.references = ReferenceReader(
            session, store, config.drive_credentials_json, config.drive_file_ids
        )
        self.context = ContextService(session)
        self.instruments = {}
        self.stopping = asyncio.Event()
        self._lease_failed = False
        self.started = False
        self.last_loop = {}
        self.worker_tasks = []
        self.decision_lock = asyncio.Lock()
        self.universe_lock = asyncio.Lock()
        self.ai_queue = asyncio.Queue(maxsize=200)
        self.queued = set()

    async def initialize(self):
        await self.store.initialize()
        if not await self.store.lease():
            raise RuntimeError("Another Apex worker holds the database lease")
        await self.observer.initialize()

        def venue(tx):
            recorded = tx.state.get("execution_venue")
            if tx.state["orders"] and recorded != self.config.mode:
                raise RuntimeError(
                    "Existing execution ledger belongs to another mode; use a separate database"
                )
            if self.config.mode != "shadow":
                tx.state["execution_venue"] = self.config.mode
            tx.state["universe_mode"] = self.config.universe_mode
            tx.state["universe_policy"] = self._universe_policy()

        await self.store.update(venue)
        await self.refresh_instruments()
        # Dynamic selection can take >90 seconds during an outage. Start it in
        # the supervised run loop, where the independent lease heartbeat runs.
        if self.config.universe_mode == "static":
            await self.refresh_universe()
        if self.config.mode != "shadow":
            await self.execution.reconcile(self.instruments)
        await self.refresh_risk()

        def started(tx):
            tx.state["instrument_symbols"] = sorted(self.instruments)
            tx.event(
                "boot:" + self.store.owner,
                "startup",
                {"mode": self.config.mode},
                f"🌊 Apex started · {self.config.mode.upper()}\nDaily Elliott structure → closed 4H confirmation.\n"
                "Risk controls and account reconciliation govern new entries. Use /dashboard.",
            )

        await self.store.update(started)
        self.started = True

    async def _health(self, name, error=None):
        if error is None:
            self.last_loop[name] = time.time()

        def write(tx):
            tx.state["health"][name + "_attempt_at"] = time.time()
            if error is None:
                tx.state["health"][name + "_at"] = time.time()
                tx.state["health"].pop(name + "_error", None)
            tx.state["health"]["redis"] = self.cache.status
            if error:
                # Exception type only: clients can embed URLs/tokens in messages.
                text = f"{name}: {type(error).__name__}"
                tx.state["health"]["error"] = text
                tx.state["health"][name + "_error"] = type(error).__name__
                key = "issue:" + name + ":" + str(int(time.time() // 900))
                tx.event(
                    key,
                    "health_issue",
                    {"component": name, "error_type": type(error).__name__},
                    f"⚠️ {name} unavailable ({type(error).__name__}).\nNew decisions needing this component wait; retries continue.",
                )

        await self.store.update(write)

    async def _loop(self, name, callback, seconds):
        while not self.stopping.is_set():
            started = time.monotonic()
            try:
                await callback()
                await self._health(name)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOG.warning("%s unavailable: %s", name, type(exc).__name__)
                try:
                    await self._health(name, exc)
                except Exception:
                    LOG.warning("Cannot persist health; new entries remain blocked")
            delay = max(1, seconds - (time.monotonic() - started))
            try:
                await asyncio.wait_for(self.stopping.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    async def renew(self):
        if self.stopping.is_set():
            return
        try:
            leased = await self.store.lease()
        except Exception:
            if not self.stopping.is_set():
                self._lease_failed = True
            self.stopping.set()
            raise
        if not leased and not self.stopping.is_set():
            self._lease_failed = True
            self.stopping.set()
            raise RuntimeError("Cloud lease lost")

    async def refresh_instruments(self):
        current = await self.client.instruments()
        # Retain current metadata for draining symbols as well as new candidates.
        self.instruments = current
        if self.started:
            await self.store.update(
                lambda tx: tx.state.update(instrument_symbols=sorted(current))
            )
        if (
            self.config.universe_mode == "static"
            and set(self.config.symbols) - current.keys()
        ):
            raise RuntimeError("Some configured instruments are unavailable")

    def _universe_policy(self):
        return {
            "target": self.config.universe_size,
            "min_turnover_usdt": self.config.universe_min_turnover,
            "max_spread_bps": self.config.universe_max_spread_bps,
            "min_depth_usdt": self.config.universe_min_depth,
        }

    def _entry_symbols(self, state, now):
        if self.config.universe_mode == "static":
            return [s for s in self.config.symbols if s in self.instruments]
        return [
            s
            for s in entry_symbols(
                state.get("universe", {}), now, required_policy=self._universe_policy()
            )
            if s in self.instruments
        ]

    @staticmethod
    def _managed_symbols(state):
        symbols = {
            t["symbol"]
            for t in state.get("trades", {}).values()
            if t.get("status") in {"PENDING", "OPEN"}
        }
        symbols.update(
            o["symbol"]
            for o in state.get("orders", {}).values()
            if o.get("status") not in {"CLOSED", "CANCELLED", "REJECTED"}
        )
        return symbols

    @staticmethod
    def _retire_pending(tx, allowed, at):
        """Stop future entry while preserving fills that occurred before removal."""
        for trade in tx.state.get("trades", {}).values():
            if trade.get("status") != "PENDING" or trade["symbol"] in allowed:
                continue
            cutoff = min(trade["expires_at"], at)
            trade.setdefault("original_expires_at", trade["expires_at"])
            trade.setdefault("universe_retired_at", at)
            if cutoff <= trade["created_at"]:
                trade.update(
                    status="EXPIRED", closed_at=at, exit_reason="UNIVERSE_RETIRED"
                )
            else:
                # Replay still resolves any evidenced pre-cutoff fills on recovery.
                trade["expires_at"] = cutoff
        for order in tx.state.get("orders", {}).values():
            if (
                order.get("status") not in {"CLOSED", "CANCELLED", "REJECTED"}
                and order["symbol"] not in allowed
            ):
                # Execution reconciles actual fills and cancels only the remainder.
                order.setdefault("entry_retired_at", at)

    async def refresh_universe(self):
        async with self.universe_lock:
            state = await self.store.read()
            now = time.time()
            previous = state.get("universe", {})
            if self.config.universe_mode == "static":
                selection = {
                    "mode": "static",
                    "as_of": now,
                    "status": "ready",
                    "target": len(self.config.symbols),
                    "active_symbols": list(self.config.symbols),
                }
                allowed = set(self.config.symbols)
            else:
                if (
                    previous.get("mode") == "dynamic"
                    and previous.get("target") == self.config.universe_size
                    and previous.get("policy", {}).get("min_turnover_usdt")
                    == self.config.universe_min_turnover
                    and previous.get("policy", {}).get("max_spread_bps")
                    == self.config.universe_max_spread_bps
                    and previous.get("policy", {}).get("min_depth_usdt")
                    == self.config.universe_min_depth
                    and 0
                    <= now - previous.get("as_of", 0)
                    < self.config.universe_review_seconds
                ):
                    return
                await self.refresh_instruments()
                metadata = dict(self.client.instrument_metadata)
                snapshot = await self.client.tickers()
                # The all-linear endpoint also contains non-USDT contracts.
                tickers = [r for r in snapshot["list"] if r["symbol"] in metadata]
                prior = previous if previous.get("mode") == "dynamic" else {}
                if len(prior.get("active_symbols", [])) > self.config.universe_size:
                    # Explicit config resize: retain strongest ranked members.
                    # History survives and retired positions continue management.
                    prior = deepcopy(prior)
                    prior["active_symbols"] = prior["active_symbols"][
                        : self.config.universe_size
                    ]
                    prior["members"] = {
                        s: prior["members"][s] for s in prior["active_symbols"]
                    }
                policy = dict(
                    target=self.config.universe_size,
                    snapshot_as_of=snapshot["as_of"],
                    min_turnover_usdt=self.config.universe_min_turnover,
                    max_spread_bps=self.config.universe_max_spread_bps,
                    min_depth_usdt=self.config.universe_min_depth,
                )
                # Pure preview screens status, age, turnover and spread BEFORE
                # allocating book requests. It commits neither observations nor
                # replacement streaks. Final selection uses the original state.
                preview = refresh_universe(
                    prior, metadata, tickers, {}, time.time(), **policy
                )
                ranked = sorted(
                    (
                        s
                        for s, d in preview["eligibility"].items()
                        if d["status"] == "blocked_pending_fresh"
                    ),
                    key=lambda s: (-preview["eligibility"][s]["score"], s),
                )
                candidates = set(ranked[: min(150, self.config.universe_size * 3)])
                candidates.update(
                    s for s in previous.get("active_symbols", []) if s in ranked
                )
                semaphore = asyncio.Semaphore(4)

                async def book(symbol):
                    async with semaphore:
                        try:
                            return symbol, await self.client.liquidity_book(symbol)
                        except Exception:
                            return symbol, None

                results = await asyncio.wait_for(
                    asyncio.gather(*(book(s) for s in sorted(candidates))), timeout=90
                )
                books = {s: b for s, b in results if b is not None}
                if candidates and not books:
                    raise RuntimeError("No fresh order books available for selection")
                now = time.time()
                selection = refresh_universe(
                    prior,
                    metadata,
                    tickers,
                    books,
                    now,
                    **policy,
                )
                selection["mode"] = "dynamic"
                selection["added"] = [
                    s
                    for s in selection["active_symbols"]
                    if s not in previous.get("active_symbols", [])
                ]
                selection["removed"] = sorted(
                    set(previous.get("active_symbols", []))
                    - set(selection["active_symbols"])
                )
                selection["candidate_count"] = len(tickers)
                selection["book_checked"] = len(books)
                selection["snapshot_time"] = snapshot["as_of"]
                allowed = set(entry_symbols(selection, now))

            async with self.decision_lock:

                def save(tx):
                    old = tx.state.get("universe", {})
                    tx.state["universe"] = selection
                    tx.state["universe_mode"] = self.config.universe_mode
                    tx.state["universe_policy"] = self._universe_policy()
                    self._retire_pending(tx, allowed, now)
                    for rec in tx.state.get("opportunities", {}).values():
                        op = rec.get("opportunity", {})
                        rec["universe_blocked"] = op.get("symbol") not in allowed
                        if rec["universe_blocked"] and op.get("state") in {
                            "READY",
                            "WAIT",
                        }:
                            rec["decision"] = (
                                "WAIT: symbol is outside the eligible liquidity universe"
                            )
                    changed = (
                        old.get("active_symbols") != selection["active_symbols"]
                        or old.get("status") != selection["status"]
                        or set(old.get("blocked", {}))
                        & set(old.get("active_symbols", []))
                        != set(selection.get("blocked", {}))
                        & set(selection["active_symbols"])
                    )
                    if changed:
                        added = sorted(
                            set(selection["active_symbols"])
                            - set(old.get("active_symbols", []))
                        )
                        removed = sorted(
                            set(old.get("active_symbols", []))
                            - set(selection["active_symbols"])
                        )
                        text = (
                            f"🌐 Liquidity universe · {len(allowed)}/{selection['target']} eligible\n"
                            f"Status: {selection['status']}\n"
                        )
                        if added:
                            text += "Added: " + ", ".join(added) + "\n"
                        if removed:
                            text += (
                                "Retired from new entries: " + ", ".join(removed) + "\n"
                            )
                        text += "Existing trades keep their protection and exit management. /universe"
                        tx.event(
                            "universe:" + str(selection["as_of"]),
                            "universe",
                            {
                                "added": added,
                                "removed": removed,
                                "eligible": sorted(allowed),
                                "status": selection["status"],
                            },
                            text,
                        )

                await self.store.update(save)

    async def refresh_context(self):
        btc = await self.cache.candles(self.client, "BTCUSDT", "D", 500)
        now = time.time()
        context = await self.context.build(btc, now)
        if self.config.macro_feed_url:
            context["external_advisory"] = await load_context(
                self.session, self.config.macro_feed_url, now
            )

        def save(tx):
            previous = tx.state.get("context", {})
            tx.state["context"] = context
            change = (
                previous.get("risk_state"),
                previous.get("event_blackout"),
                previous.get("data_complete"),
            ) != (
                context.get("risk_state"),
                context.get("event_blackout"),
                context.get("data_complete"),
            )
            if change:
                tx.event(
                    "context:" + str(int(now)),
                    "context",
                    context,
                    "🌍 Market context\n"
                    + str(context.get("reason", "Context updated"))[:1500],
                )

        await self.store.update(save)

    async def scan(self):
        await self.store.assert_leader()
        succeeded = 0
        state = await self.store.read()
        eligible = set(self._entry_symbols(state, time.time()))
        symbols = sorted(eligible | self._managed_symbols(state))
        for symbol in symbols:
            if self.stopping.is_set():
                return
            try:
                now = time.time()
                daily, execution = await asyncio.gather(
                    self.cache.candles(self.client, symbol, "D", 500, now),
                    self.cache.candles(self.client, symbol, "240", 500, now),
                )
                daily = await self.store.candle_history(symbol, "D", daily)
                execution = await self.store.candle_history(symbol, "240", execution)
                if self.config.mode != "shadow":
                    async with self.decision_lock:
                        await self.execution.manage_structure(symbol, daily, execution)
                if symbol not in self._entry_symbols(
                    await self.store.read(), time.time()
                ):
                    # Retired positions continue to receive structure/exit updates.
                    succeeded += 1
                    await self._health("market_" + symbol)
                    continue
                opportunities = await asyncio.to_thread(
                    analyze,
                    symbol,
                    daily,
                    execution,
                    now,
                    tier=(
                        1
                        if symbol
                        in {
                            "BTCUSDT",
                            "ETHUSDT",
                            "SOLUSDT",
                            "BNBUSDT",
                            "LINKUSDT",
                            "HYPEUSDT",
                        }
                        else 2
                    ),
                    bucket="majors" if symbol in {"BTCUSDT", "ETHUSDT"} else "alts",
                )
                observed = set()
                for op in opportunities:
                    observed.add(op.id)

                    def save(tx):
                        if op.symbol not in self._entry_symbols(tx.state, time.time()):
                            return False
                        previous = tx.state["opportunities"].get(op.id)
                        record = {
                            "opportunity": op.to_dict(),
                            "updated_at": now,
                            "decision": (
                                previous.get("decision", op.reason)
                                if previous
                                and previous["opportunity"]["state"] == op.state
                                else op.reason
                            ),
                            "ai": previous.get("ai", {}) if previous else {},
                            "last_risk_review": (
                                previous.get("last_risk_review") if previous else None
                            ),
                        }
                        tx.state["opportunities"][op.id] = record
                        if not previous or previous["opportunity"]["state"] != op.state:
                            notify = previous is not None or op.state in {
                                "WAIT",
                                "READY",
                            }
                            message = (
                                f"🔎 {symbol} {'LONG' if op.side=='Buy' else 'SHORT'} · Elliott {op.setup}\n"
                                f"{op.state}: {op.reason}\nEntry {op.entry:g} · invalidation {op.invalidation:g}\n"
                                f"Hard stop {op.stop:g} · T1 {op.target1:g} · T2 {op.target2:g}"
                            )
                            tx.event(
                                "op:" + op.id + ":" + op.state,
                                "opportunity",
                                op.to_dict(),
                                message if notify else None,
                            )
                        return True

                    if not await self.store.update(save):
                        continue
                    if op.state == "READY":
                        await self.consider(op, None)
                        current = await self.store.read()
                        consumed = (
                            "ai_shadow:" + op.id in current["trades"]
                            if self.config.mode == "shadow"
                            else client_id(op.id) in current["orders"]
                        )
                        if (
                            not consumed
                            and op.id not in self.queued
                            and not self.ai_queue.full()
                        ):
                            self.queued.add(op.id)
                            self.ai_queue.put_nowait((op, daily, execution))

                # Expire old views; retain the event history for every transition.
                def expire(tx):
                    for key, rec in tx.state["opportunities"].items():
                        old = rec["opportunity"]
                        if (
                            old["symbol"] == symbol
                            and key not in observed
                            and old["expires_at"] < now
                            and old["state"] != "EXPIRED"
                        ):
                            old["state"] = "EXPIRED"
                            rec["decision"] = (
                                "Candidate expired without a new confirmed entry."
                            )
                            tx.event(
                                "op:" + key + ":EXPIRED",
                                "opportunity_expired",
                                {"id": key},
                            )
                    tx.state["health"]["last_symbol"] = symbol

                await self.store.update(expire)
                await self._health("market_" + symbol)
                succeeded += 1
            except Exception as exc:
                await self._health("market_" + symbol, exc)
        if not succeeded:
            raise RuntimeError("No market scan completed successfully")

    async def review_worker(self):
        while not self.stopping.is_set():
            op, daily, execution = await self.ai_queue.get()
            try:
                state = await self.store.read()
                latest = state["opportunities"].get(op.id, {}).get("opportunity", {})
                if (
                    latest.get("state") != "READY"
                    or time.time() >= op.expires_at
                    or op.symbol not in self._entry_symbols(state, time.time())
                ):
                    continue
                context = review_context(state, op.symbol, time.time())
                review = await self.ai.review(op, daily, execution, context)
                review["context_fingerprint"] = context_fingerprint(
                    state, op.symbol, time.time()
                )
                review["candidate_fingerprint"] = candidate_fingerprint(op)
                fresh = await self.store.read()
                latest = fresh["opportunities"].get(op.id, {}).get("opportunity", {})
                # Reject a review if inputs changed while the external model ran.
                price_fields = (
                    "state",
                    "side",
                    "entry",
                    "stop",
                    "target1",
                    "target2",
                    "invalidation",
                    "expires_at",
                )
                changed = any(
                    latest.get(k) != op.to_dict().get(k) for k in price_fields
                )
                for key in (
                    "evidence_id",
                    "trigger_evidence_id",
                    "trigger_closed_at",
                    "structural_valid",
                    "data_valid",
                ):
                    changed = changed or latest.get("evidence", {}).get(
                        key
                    ) != op.evidence.get(key)
                if (
                    changed
                    or op.symbol not in self._entry_symbols(fresh, time.time())
                    or review_context(fresh, op.symbol, time.time()) != context
                    or time.time() >= op.expires_at
                ):
                    review = {
                        **review,
                        "verdict": "WAIT",
                        "reason": "Market structure or context changed during review; fresh review required.",
                    }

                def save(tx):
                    if op.id in tx.state["opportunities"]:
                        tx.state["opportunities"][op.id]["ai"] = review
                    tx.event(
                        "review:"
                        + op.id
                        + ":"
                        + str(review.get("evidence_hash", review.get("verdict"))),
                        "candidate_review",
                        review,
                        f"🧠 {op.symbol} Elliott {op.setup}\nAI: {review['verdict']}\n{review.get('reason','')}",
                    )

                await self.store.update(save)
                await self.consider(op, review)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._health("ai", exc)
            finally:
                self.queued.discard(op.id)
                self.ai_queue.task_done()

    def _losses(self, state, equity, arm):
        return loss_metrics(state, equity, arm, time.time(), self.config.mode)

    async def _risk_snapshot(self, state, equity, arm):
        key = arm or self.config.mode
        base = self.config.shadow_equity if arm else equity
        if key not in state.get("risk_reference_equities", {}):
            await self.store.update(
                lambda tx: tx.state.setdefault(
                    "risk_reference_equities", {}
                ).setdefault(key, base)
            )
            state = await self.store.read()
        details = loss_metrics(
            state, equity, arm, time.time(), self.config.mode, include_state=True
        )
        metrics = {
            k: details[k] for k in ("daily_loss_pct", "weekly_loss_pct", "drawdown_pct")
        }
        snapshot = {**details, "as_of": time.time()}

        def save(tx):
            circuits = tx.state.setdefault("risk_circuits", {})
            previous = circuits.get(key, {})
            snapshot["strategy_peak"] = max(
                snapshot["strategy_peak"], previous.get("strategy_peak", base)
            )
            snapshot["drawdown_pct"] = max(
                0,
                (snapshot["strategy_peak"] - snapshot["strategy_value"])
                / snapshot["strategy_peak"]
                * 100,
            )
            snapshot["halted"] = (
                snapshot["daily_loss_pct"] >= 2
                or snapshot["weekly_loss_pct"] >= 4
                or snapshot["drawdown_pct"] >= 12
            )
            circuits[key] = snapshot
            if snapshot["halted"] != previous.get("halted", False):
                tx.event(
                    "risk:" + key + ":" + str(int(snapshot["as_of"])),
                    "risk_circuit",
                    snapshot,
                    f"🛡️ {key} · {'new entries halted' if snapshot['halted'] else 'loss circuit cleared'}\n"
                    f"Daily {snapshot['daily_loss_pct']:.2f}% · weekly {snapshot['weekly_loss_pct']:.2f}% · drawdown {snapshot['drawdown_pct']:.2f}%\n"
                    "Existing protection continues. Other entry checks still apply.",
                )
            return {k: snapshot[k] for k in metrics}

        return await self.store.update(save)

    async def refresh_risk(self):
        state = await self.store.read()
        arms = ["baseline_shadow", "ai_shadow"]
        if self.config.mode != "shadow":
            arms.append(None)
        for arm in arms:
            equity = (
                (
                    self.config.shadow_equity
                    + sum(
                        t.get("net_pnl", 0)
                        for t in state["trades"].values()
                        if t.get("arm") == arm and t.get("status") == "CLOSED"
                    )
                )
                if arm
                else state.get("account", {}).get("equity")
            )
            try:
                await self._risk_snapshot(state, equity, arm)
            except (ValueError, TypeError, KeyError):
                key = arm or self.config.mode

                def unavailable(tx):
                    old = tx.state.setdefault("risk_circuits", {}).setdefault(key, {})
                    old.update(
                        as_of=0, error="Valuation evidence incomplete; new entries wait"
                    )

                await self.store.update(unavailable)

    async def consider(self, op, review):
        async with self.decision_lock:
            now = time.time()
            state = await self.store.read()
            if (
                state["settings"]["paused"]
                or op.state != "READY"
                or now >= op.expires_at
                or op.symbol not in self._entry_symbols(state, now)
            ):
                return
            persisted = state["opportunities"].get(op.id, {}).get("opportunity", {})
            if any(
                persisted.get(k) != op.to_dict().get(k)
                for k in (
                    "state",
                    "side",
                    "entry",
                    "stop",
                    "target1",
                    "target2",
                    "expires_at",
                )
            ):
                return
            if review and review.get("context_fingerprint") != context_fingerprint(
                state, op.symbol, now
            ):
                return
            if review and review.get("candidate_fingerprint") != candidate_fingerprint(
                op
            ):
                return
            ticker = await self.client.ticker(op.symbol)
            ask, bid = number(ticker.get("ask1Price")), number(ticker.get("bid1Price"))
            funding = number(ticker.get("fundingRate"))
            inst = self.instruments[op.symbol]
            funding_8h = (
                funding * 480 / inst.funding_interval_minutes
                if funding is not None
                else None
            )
            spread = (
                (ask - bid) / ((ask + bid) / 2) * 100
                if ask and bid and ask >= bid
                else None
            )
            context = state.get("context", {})
            context_current = bool(
                context.get("data_complete")
                and context.get("expires_at", 0) > now
                and now - context.get("as_of", 0) <= 21600
            )
            settings = state["settings"]
            arm = "baseline_shadow" if review is None else "ai_shadow"
            if review is not None and review.get("verdict") != "APPROVE":
                return
            trade_id = arm + ":" + op.id
            decision = "Existing simulated candidate; monitoring fills and exits"
            risk_review = None
            if trade_id not in state["trades"]:
                equity = self.config.shadow_equity + sum(
                    t.get("net_pnl", 0)
                    for t in state["trades"].values()
                    if t["arm"] == arm and t["status"] == "CLOSED"
                )
                exposures = [
                    t
                    for t in state["trades"].values()
                    if t["arm"] == arm and t["status"] in {"OPEN", "PENDING"}
                ]
                losses = await self._risk_snapshot(state, equity, arm)
                sizing = assess(
                    op,
                    inst,
                    equity,
                    profile=settings["profile"],
                    risk_pct=settings["risk_pct"],
                    exposures=exposures,
                    funding_rate_8h=funding_8h,
                    spread_pct=spread,
                    now=now,
                    **losses,
                )
                risk_review = {"arm": arm, "assessed_at": now, "assessment": sizing}
                decision = (
                    "; ".join(sizing["reasons"])
                    if not sizing["allowed"]
                    else "Risk checks passed"
                )
                if review is None and not sizing["allowed"]:
                    try:
                        await self.observer.consider(
                            op,
                            inst,
                            equity,
                            settings["profile"],
                            settings["risk_pct"],
                            losses,
                            funding_8h,
                            spread,
                            now,
                            sizing,
                        )
                    except NotLeader:
                        raise
                    except Exception as exc:
                        # This independent research ledger cannot admit an order
                        # or interrupt the regular strategy's decision pipeline.
                        LOG.warning(
                            "Shadow observer unavailable: %s", type(exc).__name__
                        )
                        try:
                            await self._health("shadow_observer", exc)
                        except NotLeader:
                            raise
                        except Exception as health_error:
                            LOG.warning(
                                "Observer health write unavailable: %s",
                                type(health_error).__name__,
                            )
                if sizing["allowed"]:
                    sizing["qty_step"] = inst.qty_step

                    def create(tx):
                        if (
                            tx.state["settings_version"] != state["settings_version"]
                            or tx.state["settings"]["paused"]
                        ):
                            return False
                        committed_at = time.time()
                        current = (
                            tx.state["opportunities"]
                            .get(op.id, {})
                            .get("opportunity", {})
                        )
                        if (
                            current.get("state") != "READY"
                            or committed_at >= op.expires_at
                            or committed_at - op.evidence.get("as_of", 0) > 300
                            or op.symbol
                            not in self._entry_symbols(tx.state, committed_at)
                        ):
                            return False
                        if any(
                            current.get(k) != op.to_dict().get(k)
                            for k in (
                                "side",
                                "entry",
                                "stop",
                                "target1",
                                "target2",
                                "expires_at",
                            )
                        ):
                            return False
                        if review and review.get(
                            "context_fingerprint"
                        ) != context_fingerprint(tx.state, op.symbol, committed_at):
                            return False
                        trade = create_trade(op, sizing, arm, committed_at)
                        if trade["id"] not in tx.state["trades"]:
                            tx.state["trades"][trade["id"]] = trade
                            tx.event(
                                trade["id"] + ":pending",
                                "shadow_order",
                                trade,
                                f"👻 {arm.replace('_',' ')} · {op.symbol} {op.side}\n"
                                "Simulated limit waiting for a subsequent candle. No exchange order.",
                            )
                        return True

                    await self.store.update(create)
            if self.config.mode != "shadow" and review is not None:
                state = await self.store.read()
                account = state.get("account", {})
                outbox = await self.store.outbox_health()
                reasons = list(account.get("blockers", []))
                if (
                    not context_current
                    or context.get("live_blocked") is True
                    or context.get("policy_authority") is False
                ):
                    reasons.append("Required macro context is missing or stale")
                if context.get("event_blackout") is not False:
                    reasons.append("Official macro event blackout")
                if now - account.get("as_of", 0) > 30:
                    reasons.append("Exchange account snapshot is stale")
                if outbox["oldest_age"] > 300:
                    reasons.append("Telegram delivery delayed over 5 minutes")
                if (
                    not bid
                    or not ask
                    or (op.side == "Buy" and not op.stop < bid < op.target1)
                    or (op.side == "Sell" and not op.target1 < ask < op.stop)
                ):
                    reasons.append(
                        "Current quote has crossed the plan stop/first target or is unavailable"
                    )
                if not reasons:
                    positions = await self.client.position_info(op.symbol)
                    zeros = [p for p in positions if int(p.get("positionIdx", -1)) == 0]
                    leverage = number(zeros[0].get("leverage")) if zeros else None
                    if not leverage or leverage > 5 or leverage > inst.max_leverage:
                        reasons.append(
                            "Configured one-way leverage not verified within 5×"
                        )
                    if any(number(p.get("size"), 0) > 0 for p in positions):
                        reasons.append("Symbol already has an exchange position")
                if not reasons:
                    equity = account["equity"]
                    exposures = [
                        o
                        for o in state["orders"].values()
                        if o["status"] not in {"CLOSED", "CANCELLED", "REJECTED"}
                    ]
                    losses = await self._risk_snapshot(state, equity, None)
                    factor = context.get(
                        "long_multiplier" if op.side == "Buy" else "short_multiplier", 0
                    )
                    sizing = assess(
                        op,
                        inst,
                        equity,
                        profile=settings["profile"],
                        risk_pct=settings["risk_pct"],
                        exposures=exposures,
                        funding_rate_8h=funding_8h,
                        spread_pct=spread,
                        now=now,
                        regime_multiplier=factor,
                        **losses,
                    )
                    risk_review = {
                        "arm": self.config.mode,
                        "assessed_at": now,
                        "assessment": sizing,
                    }
                    reasons.extend(sizing["reasons"] if not sizing["allowed"] else [])
                if not reasons:
                    tiers = await self.client.risk_limits(op.symbol)
                    tier = next(
                        (
                            r
                            for r in sorted(
                                tiers, key=lambda r: float(r["riskLimitValue"])
                            )
                            if float(r["riskLimitValue"]) >= sizing["notional"]
                        ),
                        None,
                    )
                    mm = number(tier.get("maintenanceMargin")) if tier else None
                    # Conservative isolated estimate: omit maintenance deduction,
                    # reserve close fees, and require twice the stop distance.
                    if mm is None:
                        reasons.append("Maintenance-margin tier unavailable")
                    else:
                        buffer = sizing["entry"] * (1 / leverage - mm - 0.002)
                        if buffer < 2 * abs(sizing["entry"] - sizing["stop"]):
                            reasons.append("Estimated liquidation buffer insufficient")
                if not reasons:
                    sizing["qty_step"] = inst.qty_step
                    submitted = await self.execution.submit(
                        op, sizing, state["settings_version"]
                    )
                    decision = (
                        "Exchange intent recorded; awaiting confirmed fills"
                        if submitted
                        else "WAIT: candidate already submitted or risk state changed before reservation"
                    )
                else:
                    decision = "WAIT: " + "; ".join(reasons)

            def record(tx):
                rec = tx.state["opportunities"].get(op.id)
                if rec:
                    old = rec.get("decision")
                    rec["decision"] = decision
                    if risk_review is not None:
                        rec["last_risk_review"] = risk_review
                    if old != decision:
                        key = (
                            "decision:"
                            + op.id
                            + ":"
                            + hashlib.sha256(decision.encode()).hexdigest()
                        )
                        tx.event(
                            key,
                            "decision",
                            {"id": op.id, "reason": decision},
                            f"🎯 {op.symbol}\n{decision}",
                        )

            await self.store.update(record)

    async def _update_shadow_symbols(self, component, symbols, update_symbol):
        """Keep one unavailable symbol from starving the rest of a recovery pass."""
        failed = 0
        for symbol in symbols:
            if self.stopping.is_set():
                return
            name = component + "_" + symbol
            try:
                await update_symbol(symbol)
                await self._health(name)
            except asyncio.CancelledError:
                raise
            except NotLeader:
                # Losing write authority is a worker failure, not a symbol fault.
                raise
            except Exception as exc:
                failed += 1
                LOG.warning("%s unavailable: %s", name, type(exc).__name__)
                try:
                    await self._health(name, exc)
                except NotLeader:
                    raise
                except Exception:
                    LOG.warning("Cannot persist %s health; recovery will retry", name)
        if failed:
            # The supervisor must not mark a partially completed pass healthy.
            # Successful symbols are committed; failed cursors remain retryable.
            raise RuntimeError(f"{component}: {failed} symbol updates failed")

    async def simulate(self):
        now = time.time()

        def retire(tx):
            allowed = set(self._entry_symbols(tx.state, now))
            selection = tx.state.get("universe", {})
            at = now
            if self.config.universe_mode == "dynamic" and selection.get("as_of"):
                at = min(now, selection["as_of"] + 86400)
            self._retire_pending(tx, allowed, at)

        await self.store.update(retire)
        state = await self.store.read()

        async def simulate_symbol(symbol):
            now = time.time()
            active = [
                t
                for t in state["trades"].values()
                if t["symbol"] == symbol and t["status"] in {"PENDING", "OPEN"}
            ]
            first = min(
                (
                    t["last_bar"] + 180000
                    if t.get("last_bar") is not None
                    else math.ceil(t["created_at"] / 180) * 180000
                )
                for t in active
            )
            end = int(now // 180) * 180000
            if first >= end:
                return
            # Bound each recovery pass; no missing candles are silently skipped.
            end = min(end, first + 10000 * 180000)
            bars = await self.client.candle_range(symbol, "3", first, end)
            daily, execution = await asyncio.gather(
                self.cache.candles(self.client, symbol, "D", 500, now),
                self.cache.candles(self.client, symbol, "240", 500, now),
            )
            daily = await self.store.candle_history(symbol, "D", daily)
            execution = await self.store.candle_history(symbol, "240", execution)

            def resolve(tx):
                for key, t in list(tx.state["trades"].items()):
                    if t["symbol"] != symbol or t["status"] not in {"OPEN", "PENDING"}:
                        continue
                    updated = advance(
                        t, bars, end / 1000, daily=daily, execution=execution
                    )
                    if bars and updated.get("last_bar") == bars[-1].open_time:
                        updated.update(mark_price=bars[-1].close, mark_at=end / 1000)
                    tx.state["trades"][key] = updated
                    if t["status"] != updated["status"]:
                        tx.event(
                            key + ":" + updated["status"],
                            "shadow_state",
                            updated,
                            f"👻 {symbol} · {t['arm']}\n{updated['status']} · {updated.get('exit_reason') or 'limit filled'}\n"
                            f"Estimated net before funding ${updated.get('net_pnl_before_funding',0):.2f}",
                        )
                    if updated.get("data_error") or updated.get("data_gap"):
                        tx.event(
                            key + ":data:" + str(int(now // 900)),
                            "shadow_data_issue",
                            {"id": key},
                            f"⚠️ Shadow replay waiting for complete {symbol} candle history; P&L is not final.",
                        )

            await self.store.update(resolve)

        await self._update_shadow_symbols(
            "simulation",
            sorted(
                {
                    t["symbol"]
                    for t in state["trades"].values()
                    if t["status"] in {"PENDING", "OPEN"}
                }
            ),
            simulate_symbol,
        )

    async def funding(self):
        state = await self.store.read()
        pending = [
            t
            for t in state["trades"].values()
            if t["opened_at"] and not t["funding_complete"]
        ]

        async def fund_symbol(symbol):
            selected = [t for t in pending if t["symbol"] == symbol]
            start = min(t["opened_at"] for t in selected)
            end = time.time()
            rates = await self.client.funding_history(
                symbol, int(start * 1000), int(end * 1000)
            )

            def update(tx):
                for t in selected:
                    current = tx.state["trades"][t["id"]]
                    tx.state["trades"][t["id"]] = apply_funding(
                        current, rates, end, covered_from=start, history_complete=True
                    )

            await self.store.update(update)

        await self._update_shadow_symbols(
            "funding", sorted({t["symbol"] for t in pending}), fund_symbol
        )

    async def reconcile(self):
        if self.config.mode != "shadow":
            async with self.decision_lock:
                await self.execution.reconcile(self.instruments)

    async def poll(self):
        if not self.telegram:
            return
        await self.store.assert_leader()
        state = await self.store.read()
        try:
            offset = await self.telegram.poll_once(state["telegram_offset"])
        except TelegramError as exc:
            offset = getattr(exc, "next_offset", None)
            if offset is not None:
                await self.store.update(
                    lambda tx: tx.state.update(
                        telegram_offset=max(tx.state["telegram_offset"], offset)
                    )
                )
            raise
        await self.store.update(lambda tx: tx.state.update(telegram_offset=offset))

    async def deliver(self):
        if not self.telegram:
            return
        for item in await self.store.pending_notifications():
            try:
                message = await self.telegram.send(item["text"])
                await self.store.notification_result(item["key"], message_id=message)
            except Exception as exc:
                retry = getattr(exc, "retry_after", None) or min(
                    300, 2 ** min(item["attempts"] + 1, 8)
                )
                await self.store.notification_result(item["key"], retry_after=retry)
                break

    async def research(self):
        symbols = self._entry_symbols(await self.store.read(), time.time())
        if not symbols:
            return
        result = await self.ai.research(symbols)
        await self.store.update(lambda tx: tx.state.update(research=result))

    async def observe_shadows(self):
        await self.observer.resolve(time.time())

    async def run(self):
        # All exceptions are supervised. Slow model requests never block stops,
        # Telegram polling, heartbeat or exchange reconciliation.
        loops = [
            ("lease", self.renew, 20),
            ("context", self.refresh_context, 900),
            ("risk", self.refresh_risk, 30),
            ("instruments", self.refresh_instruments, 3600),
            ("universe", self.refresh_universe, 300),
            ("scan", self.scan, self.config.scan_seconds),
            ("simulation", self.simulate, 30),
            ("shadow_observer", self.observe_shadows, 30),
            ("funding", self.funding, 900),
            ("reconcile", self.reconcile, 10),
            ("telegram", self.poll, 1),
            ("outbox", self.deliver, 2),
            ("references", self.references.refresh, 1800),
            ("research", self.research, 21600),
        ]
        self.worker_tasks = [
            asyncio.create_task(self._loop(name, fn, seconds), name=name)
            for name, fn, seconds in loops
        ]
        self.worker_tasks.append(
            asyncio.create_task(self.review_worker(), name="ai_reviews")
        )
        try:
            await self.stopping.wait()
        finally:
            for task in self.worker_tasks:
                task.cancel()
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            try:
                await self.cache.close()
            finally:
                await self.store.close()
        if self._lease_failed:
            # __main__.main already maps runtime exceptions to exit status 1.
            # A normal SIGTERM still drains successfully without this exception.
            raise RuntimeError("Cloud lease failed; worker restart required")
