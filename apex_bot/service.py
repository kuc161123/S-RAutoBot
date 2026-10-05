"""Persistent, offline Telegram control plane over the parent's leased Store.

Only profile/risk_pct/paused are mutable. Confirm and cancel compete in the same
transaction, so a consumed/revoked token cannot act after a restart. Snapshots
never fetch account data, reconcile orders, or expose configuration identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import time
from datetime import datetime, timezone
from decimal import Decimal

from . import risk as risk_policy
from .accounting import loss_metrics
from .simulation import comparison, summary
from .universe import entry_symbols

ACCOUNT_MAX_AGE = 30
CONTEXT_MAX_AGE = 21600
CONFIRM_TTL = 120
RISK_MAX_AGE = 120
UNIVERSE_MAX_AGE = 24 * 60 * 60
# These match the hard halt boundaries in risk.assess (strategy MTM accounting).
LOSS_LIMITS = {"daily_loss_pct": 2, "weekly_loss_pct": 4, "drawdown_pct": 12}
CAP_LABELS = {
    "heat_pct": "Portfolio heat",
    "bucket_pct": "Bucket heat",
    "max_positions": "Positions",
    "max_per_bucket": "Positions per bucket",
    "single_notional_pct": "Single-position notional",
    "gross_notional_pct": "Gross notional",
}
ARMS = (("baseline_shadow", "Rules baseline"), ("ai_shadow", "AI-approved shadow"))


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _fresh(stamp, now, limit):
    return _finite(stamp) and stamp > 0 and 0 <= now - stamp <= limit


def _number(value):
    return (
        format(Decimal(str(value)), "f").rstrip("0").rstrip(".")
        if "." in str(value)
        else str(value)
    )


def _cash(equity, percent=100):
    if equity is None:
        return "cash unavailable"
    return f"{Decimal(str(equity)) * Decimal(str(percent)) / 100:,.2f} USDT"


def percentage(profile, override=None):
    return risk_policy.PROFILES[profile]["risk_pct"] if override is None else override


class Service:
    def __init__(self, config, store):
        self.config, self.store = config, store
        self.observer = None

    def _actor(self, actor):
        chat = str(self.config.telegram_chat_id).strip()
        users = {
            uid for uid in self.config.telegram_user_ids if type(uid) is int and uid > 0
        }
        if (
            not isinstance(actor, str)
            or not re.fullmatch(r"-?[1-9][0-9]*", chat)
            or not users
            or actor not in {f"telegram:{chat}:{uid}" for uid in users}
        ):
            raise ValueError("Telegram actor is not authorized")
        return actor

    @staticmethod
    def _validated(key, value):
        if key == "profile":
            if not isinstance(value, str) or value not in risk_policy.PROFILES:
                raise ValueError("Unknown risk profile")
        elif key == "risk_pct":
            if not _finite(value) or not 0.05 <= value <= 1:
                raise ValueError("Custom risk must be between 0.05% and 1.00%")
        elif key == "paused":
            if type(value) is not bool:
                raise ValueError("Pause must be true or false")
        else:
            raise ValueError("That setting cannot be changed through Telegram")
        return value

    @staticmethod
    def _caps(profile):
        # Mirror assess: a profile can tighten, never raise a hard cap.
        caps = {
            key: min(value, risk_policy.PROFILES[profile].get(key, value))
            for key, value in risk_policy.HARD_CAPS.items()
        }
        if any(not _finite(v) or v <= 0 for v in caps.values()):
            raise ValueError("Risk policy unavailable")
        return caps

    def _capital(self, state, now):
        if self.config.mode == "shadow":
            equity = self.config.shadow_equity
            valid = _finite(equity) and equity > 0
            return {
                "equity": equity if valid else None,
                "as_of": None,
                "basis": "Hypothetical starting capital (USDT; not current account equity)",
                "error": None if valid else "Hypothetical starting capital unavailable",
            }
        account = state.get("account", {})
        equity, stamp = account.get("equity"), account.get("as_of")
        valid = (
            self.config.mode in {"live", "testnet"}
            and _finite(equity)
            and equity > 0
            and _fresh(stamp, now, ACCOUNT_MAX_AGE)
            and account.get("currency", "USDT") == "USDT"
        )
        return {
            "equity": equity if valid else None,
            "as_of": stamp,
            "basis": "Reconciled USDT coin equity (not account-total USD)",
            "error": (
                None
                if valid
                else "Exchange equity unavailable, invalid or stale (30s limit); no fallback"
            ),
        }

    @staticmethod
    def _capital_text(capital):
        if capital["error"]:
            return "⚠️ " + capital["error"]
        line = f"{capital['basis']}: {_cash(capital['equity'])}"
        if capital["as_of"] is not None:
            line += (
                "\nAccount as of: "
                + datetime.fromtimestamp(capital["as_of"], timezone.utc).isoformat()
            )
        else:
            line += (
                "\nEach simulated arm sizes from starting capital + its own closed P&L."
            )
        return line

    @staticmethod
    def _context_blockers(context, now):
        stamp = context.get("as_of", context.get("asof", context.get("timestamp")))
        expires = context.get("expires_at")
        blockers = []
        complete = context.get("data_complete") is True or context.get("fresh") is True
        if (
            not complete
            or context.get("data_complete") is False
            or context.get("fresh") is False
            or context.get("live_blocked") is True
            or context.get("policy_authority") is False
            or not _fresh(stamp, now, CONTEXT_MAX_AGE)
            or not _finite(expires)
            or expires <= now
            or (
                "fresh_until" in context
                and (
                    not _finite(context["fresh_until"]) or context["fresh_until"] <= now
                )
            )
        ):
            blockers.append(
                "Required context is missing, stale or lacks policy authority"
            )
        if context.get("risk_state") not in {"risk_on", "neutral", "risk_off"}:
            blockers.append("Context risk state is unavailable")
        if context.get("event_blackout") is not False:
            blockers.append("Event blackout is active or unknown")
        multipliers = [
            context.get(key) for key in ("long_multiplier", "short_multiplier")
        ]
        if not all(_finite(v) and 0 <= v <= 1 for v in multipliers):
            blockers.append("Context sizing multipliers are unavailable")
        elif max(multipliers) == 0:
            blockers.append("Context blocks both trading directions")
        return blockers

    def _resume_blockers(self, state, now):
        blockers = self._context_blockers(state.get("context", {}), now)
        health, account = state.get("health", {}), state.get("account", {})
        if not _fresh(
            health.get("scan_at"), now, max(120, self.config.scan_seconds * 3)
        ):
            blockers.append("Scanner health is missing or stale")
        components = ("scan", "context", "risk") + (
            () if self.config.mode == "shadow" else ("reconcile",)
        )
        if any(health.get(component + "_error") for component in components):
            blockers.append("A required runtime component has an active health error")
        for source in (state, health, state.get("risk", {}), account):
            if source.get("hard_halt") or source.get("hard_halts"):
                blockers.append("A persistent hard halt is active")
            if source.get("blockers"):
                blockers.append("Active service/account blockers require resolution")
            for field, cap in LOSS_LIMITS.items():
                if field in source and (
                    not _finite(source[field])
                    or source[field] < 0
                    or source[field] >= cap
                ):
                    blockers.append(f"{field} is unknown or at its hard halt ({cap}%)")
        capital = self._capital(state, now)
        if capital["error"]:
            blockers.append(capital["error"])
        if self.config.mode != "shadow":
            if not self.config.live_enabled or not self.config.migration_verified:
                blockers.append(
                    "Deployment execution/migration authorization is incomplete"
                )
            if not isinstance(account.get("blockers"), list) or not isinstance(
                account.get("positions"), list
            ):
                blockers.append("Account reconciliation status is unavailable")
        try:
            for arm in (
                [name for name, _ in ARMS] if self.config.mode == "shadow" else [None]
            ):
                equity = capital["equity"]
                if arm and equity is not None:
                    closed = [
                        t
                        for t in state["trades"].values()
                        if t.get("arm") == arm and t.get("status") == "CLOSED"
                    ]
                    if any(not _finite(t.get("net_pnl")) for t in closed):
                        raise ValueError("Incomplete simulated P&L")
                    equity += sum(t["net_pnl"] for t in closed)
                losses = self._metrics(state, equity, arm, now)
                for field, limit in LOSS_LIMITS.items():
                    if losses[field] >= limit:
                        blockers.append(
                            f"{arm or 'exchange'} {field} hard halt: {losses[field]:.2f}%"
                        )
        except (ValueError, TypeError):
            blockers.append(
                "Strategy MTM accounting/risk circuit is unavailable or stale"
            )
        return list(dict.fromkeys(blockers))

    def _metrics(self, state, equity, arm, now):
        key = arm or self.config.mode
        circuit = state.get("risk_circuits", {}).get(key, {})
        if not _fresh(circuit.get("as_of"), now, RISK_MAX_AGE) or any(
            not _finite(circuit.get(field)) or circuit[field] < 0
            for field in LOSS_LIMITS
        ):
            raise ValueError("Fresh strategy MTM risk circuit is required")
        if circuit.get("hard_halt") or circuit.get("halted"):
            raise ValueError("Persistent strategy MTM risk circuit is halted")
        try:
            computed = loss_metrics(state, equity, arm, now, self.config.mode)
        except (KeyError, TypeError, ValueError, ArithmeticError):
            raise ValueError("Strategy MTM accounting unavailable") from None
        if any(not _finite(computed.get(f)) or computed[f] < 0 for f in LOSS_LIMITS):
            raise ValueError("Invalid strategy MTM accounting")
        # A newly recorded MTM loss or a still-active persisted circuit may only
        # tighten the checks; a read never clears a circuit or a reference base.
        return {field: max(computed[field], circuit[field]) for field in LOSS_LIMITS}

    def _binding(self, state, capital):
        """Bind economically relevant preview facts; a fresh timestamp alone may advance."""
        value = {
            "settings": state["settings"],
            "mode": self.config.mode,
            "equity": capital["equity"],
            "basis": capital["basis"],
            "profiles": risk_policy.PROFILES,
            "hard_caps": risk_policy.HARD_CAPS,
            "context_multipliers": [
                state.get("context", {}).get(k)
                for k in ("long_multiplier", "short_multiplier")
            ],
            "risk_reference_equities": state.get("risk_reference_equities"),
            "risk_circuits": {
                key: {
                    field: value for field, value in circuit.items() if field != "as_of"
                }
                for key, circuit in state.get("risk_circuits", {}).items()
            },
            "closed_loss_records": {
                kind: [
                    {
                        k: record.get(k)
                        for k in (
                            "arm",
                            "closed_at",
                            "net_pnl",
                            "net_pnl_before_funding",
                        )
                    }
                    for record in state[kind].values()
                    if record.get("status") == "CLOSED"
                ]
                for kind in ("trades", "orders")
            },
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def _known_budgets(self, state, percent, capital, now):
        """Exact known reductions, explicitly before opportunity-specific factors."""
        if capital["equity"] is None:
            return ["Effective budgets unavailable until fresh equity is available."]
        lines = []
        try:
            if self.config.mode == "shadow":
                for arm, label in ARMS:
                    closed = [
                        t
                        for t in state["trades"].values()
                        if t.get("arm") == arm and t.get("status") == "CLOSED"
                    ]
                    if any(not _finite(t.get("net_pnl")) for t in closed):
                        raise ValueError("Incomplete simulated P&L")
                    equity = capital["equity"] + sum(t["net_pnl"] for t in closed)
                    losses = self._metrics(state, equity, arm, now)
                    multiplier = 0.5 if losses["drawdown_pct"] >= 8 else 1
                    pct = float(Decimal(str(percent)) * Decimal(str(multiplier)))
                    lines.append(
                        f"{label}: hypothetical realized equity {_cash(equity)}; "
                        f"tier-1 budget {_number(pct)}% ({_cash(equity, pct)})"
                    )
            elif self._context_blockers(state.get("context", {}), now):
                lines.append(
                    "Effective directional budgets unavailable until context is verified/fresh."
                )
            else:
                equity = capital["equity"]
                losses = self._metrics(state, equity, None, now)
                drawdown = Decimal("0.5") if losses["drawdown_pct"] >= 8 else Decimal(1)
                for side, field in (
                    ("Long", "long_multiplier"),
                    ("Short", "short_multiplier"),
                ):
                    pct = float(
                        Decimal(str(percent))
                        * Decimal(str(state["context"][field]))
                        * drawdown
                    )
                    lines.append(
                        f"{side} tier-1 budget after context/drawdown: "
                        f"{_number(pct)}% ({_cash(equity, pct)})"
                    )
        except (ValueError, TypeError):
            return [
                "Effective budgets unavailable until strategy MTM circuits/accounting are fresh."
            ]
        lines.append(
            "Before setup-specific reductions; tier 2 halves risk again. Hard halts can block all entries."
        )
        return lines

    def _preview(self, state, new, capital, blockers):
        old = state["settings"]
        old_risk = percentage(old["profile"], old["risk_pct"])
        new_risk = percentage(new["profile"], new["risk_pct"])
        budget = lambda pct: f"{_number(pct)}% ({_cash(capital['equity'], pct)})"
        lines = [
            "⚙️ Old → proposed",
            self._capital_text(capital),
            f"Profile: {old['profile']} → {new['profile']}",
            f"Custom override: {old['risk_pct']} → {new['risk_pct']} (None uses profile)",
            f"Base per-entry risk budget: {budget(old_risk)} → {budget(new_risk)}",
            f"Entries paused: {old['paused']} → {new['paused']}",
            "🛡 Hard caps, old → proposed",
        ]
        before, after = self._caps(old["profile"]), self._caps(new["profile"])
        for key in risk_policy.HARD_CAPS:
            values = [
                budget(v) if key.endswith("_pct") else str(v)
                for v in (before[key], after[key])
            ]
            lines.append(f"{CAP_LABELS.get(key, key)}: {values[0]} → {values[1]}")
        now = time.time()
        lines += [
            "Current effective budget context:",
            *self._known_budgets(state, old_risk, capital, now),
            "Proposed effective budget context:",
            *self._known_budgets(state, new_risk, capital, now),
        ]
        lines += [
            "Prospective effect: future entries only. Selecting a profile clears its custom override.",
            "Pause blocks new simulated and exchange entries. Existing simulated trades continue;",
        ]
        lines += [
            "existing exchange stops/protection and position management stay in place.",
            "Pending orders/trades are not cancelled. Resume never enables live mode.",
            "These are base budgets, not a promised fill loss. Tier 2, context, setup and drawdown "
            "reductions apply per opportunity; current hard loss limits remain 2% daily / 4% weekly / 12% drawdown.",
        ]
        if blockers:
            lines.append("⛔ Entry/resume checks: " + "; ".join(blockers))
        lines.append(
            "Confirm within 120 seconds. Equity, policy and settings are rechecked."
        )
        return self._safe("\n".join(lines))

    async def propose_change(self, key, value, actor):
        actor, value = self._actor(actor), self._validated(key, value)
        token = secrets.token_urlsafe(24)

        def propose(tx):
            now, state = time.time(), tx.state
            capital = self._capital(state, now)
            # Pausing must remain possible during an account/context outage.
            if capital["error"] and (key, value) != ("paused", True):
                raise ValueError(capital["error"])
            blockers = self._resume_blockers(state, now)
            if key == "paused" and value is False and blockers:
                raise ValueError("Resume blocked: " + "; ".join(blockers))
            new = dict(state["settings"], **{key: value})
            if key == "profile":
                new["risk_pct"] = None
            text = self._preview(state, new, capital, blockers)
            # A new preview invalidates that actor's prior previews, not others'.
            state["confirmations"] = {
                k: v
                for k, v in state["confirmations"].items()
                if v["expires"] > now and v["actor"] != actor
            }
            state["confirmations"][token] = {
                "key": key,
                "value": value,
                "actor": actor,
                "expires": now + CONFIRM_TTL,
                "version": state["settings_version"],
                "binding": self._binding(state, capital),
            }
            return {"token": token, "summary": text}

        return await self.store.update(propose)

    async def confirm_change(self, token, actor):
        actor = self._actor(actor)

        def confirm(tx):
            now, state = time.time(), tx.state
            pending = state["confirmations"].get(token)
            if not pending or pending["actor"] != actor:
                return "⌛ Confirmation unavailable. Request a new preview."
            # Rejections consume the caller's token too. Returning commits revocation.
            del state["confirmations"][token]
            if (
                pending["expires"] <= now
                or pending["version"] != state["settings_version"]
            ):
                return (
                    "⌛ Preview expired or settings changed. Request a fresh preview."
                )
            key, value = pending["key"], self._validated(
                pending["key"], pending["value"]
            )
            capital = self._capital(state, now)
            if (key, value) != ("paused", True):
                if capital["error"] or pending.get("binding") != self._binding(
                    state, capital
                ):
                    return "⌛ Equity, policy or context changed/unavailable. Request a fresh preview."
            if key == "paused" and value is False:
                blockers = self._resume_blockers(state, now)
                if blockers:
                    return self._safe("⛔ Resume blocked: " + "; ".join(blockers))
            state["settings"][key] = value
            if key == "profile":
                state["settings"]["risk_pct"] = None
            state["settings_version"] += 1
            text = (
                "⏸ New simulated/exchange entries paused. Existing protection continues; pending entries remain."
                if key == "paused" and value
                else f"✅ Saved: {key} = {value}. Future entries remain subject to all safety checks."
            )
            tx.event(
                "setting:" + token,
                "setting_change",
                {"key": key, "value": value, "actor": actor},
                text,
            )
            return text

        return await self.store.update(confirm)

    async def cancel_change(self, token, actor):
        actor = self._actor(actor)

        def cancel(tx):
            pending = tx.state["confirmations"].get(token)
            if not pending or pending["actor"] != actor:
                return "⌛ Preview unavailable or already handled. Check /status."
            del tx.state["confirmations"][token]
            tx.event("cancel:" + token, "setting_cancelled", {"actor": actor})
            return "✖️ Preview revoked. Its confirmation can no longer apply a change."

        return await self.store.update(cancel)

    async def change_setting(self, key, value, actor):
        # Preserve the parent's explicit emergency pause API, never a bypass for resume.
        actor = self._actor(actor)
        self._validated(key, value)
        if key != "paused" or value is not True:
            raise ValueError("Use a preview and confirmation for this change")

        def pause(tx):
            tx.state["settings"]["paused"] = True
            tx.state["settings_version"] += 1
            text = "⏸ New simulated/exchange entries paused. Existing protection continues; pending entries remain."
            tx.event(
                "pause:" + str(tx.state["settings_version"]),
                "setting_change",
                {"actor": actor},
                text,
            )
            return text

        return await self.store.update(pause)

    def _safe(self, text):
        # Whitelisted presentation fields below, plus defense for free-form reasons/research.
        text = re.sub(r"telegram:-?[0-9]+:[0-9]+", "[private actor]", str(text))
        for field in (
            "telegram_token",
            "bybit_key",
            "bybit_secret",
            "openai_key",
            "drive_credentials_json",
        ):
            value = getattr(self.config, field, "")
            if value:
                text = text.replace(str(value), "[redacted]")
        ids = [
            str(self.config.telegram_chat_id),
            *(str(v) for v in self.config.telegram_user_ids),
        ]
        for value in ids:
            if value:
                text = re.sub(
                    r"(?<![\w.])" + re.escape(value) + r"(?![\w.])",
                    "[private ID]",
                    text,
                )
        return text

    async def snapshot(self, view, symbol=None):
        await self.store.assert_leader()
        return self._safe(await self._snapshot(view, symbol))

    async def assert_leader(self):
        await self.store.assert_leader()

    async def _snapshot(self, view, symbol):
        state, now = await self.store.read(), time.time()
        if view == "observer":
            if self.observer is None:
                return "🔬 Capacity observer is unavailable."
            data = await self.observer.summary()
            rate = f"{data['wr']:.1f}%" if data["wr"] is not None else "N/A"
            mean = f"{data['mean_r']:+.2f}R" if data["mean_r"] is not None else "N/A"
            return "\n".join(
                [
                    "🔬 CAPACITY OBSERVER · SHADOW ONLY",
                    "Valid setups blocked by portfolio capacity are followed independently.",
                    f"Captured: {data['total']} · Waiting: {data['pending']}",
                    f"Open: {data['open']} · Closed: {data['closed']} · Unfilled: {data['expired']}",
                    f"Reconciled closes: {data['complete_closed']}",
                    f"✅ Wins {data['wins']} · ❌ Losses {data['losses']} · ➖ Flat {data['breakeven']}",
                    f"Win rate: {rate} · Mean net outcome: {mean}",
                    f"Incomplete closes: {data['incomplete_closed']} · Data issues: {data['data_issues']}",
                    "R = original entry-to-stop price risk; modeled fees and estimated funding included.",
                    "Overlapping observations are correlated; these are not portfolio returns.",
                    "Execution limits stay in force. This collects evidence; it does not train or change the strategy automatically.",
                ]
            )
        if view == "universe":
            return self._universe(state, now)
        settings, health = state["settings"], state.get("health", {})
        risk = percentage(settings["profile"], settings["risk_pct"])
        capital = self._capital(state, now)
        account = state.get("account", {})
        budget = f"{_number(risk)}% ({_cash(capital['equity'], risk)})"
        if view in {"dashboard", "start", "status"}:
            opps = list(state["opportunities"].values())
            eligible = self._universe_entries(state, now)
            ready = sum(
                rec["opportunity"]["state"] == "READY"
                and rec["opportunity"].get("symbol") in eligible
                and _finite(rec["opportunity"].get("expires_at"))
                and rec["opportunity"]["expires_at"] > now
                for rec in opps
            )
            blockers = self._resume_blockers(state, now)
            lines = [
                "🌊 APEX · ELLIOTT WAVE",
                f"Mode: {self.config.mode.upper()}",
                f"New entries: {'⏸ Paused' if settings['paused'] else 'Subject to checks'}",
                f"Profile: {settings['profile']} · base budget {budget}",
                self._capital_text(capital),
                *self._universe_summary(state, now),
                "Daily structure / 4H entry",
                f"Last scan: {self._age(health.get('scan_at'))}",
                f"Candidates: {len(opps)} · Ready: {ready}",
            ]
            if self.config.mode == "shadow":
                lines.append("👻 Hypothetical simulation; exchange execution is off.")
            from .release import LIVE_APPROVED

            if not LIVE_APPROVED:
                lines.append("🔬 Strategy validation pending · live release blocked.")
            if blockers:
                lines.append("⛔ " + "; ".join(blockers))
            lines.extend(self._circuit_lines(state, now))
            lines += ["", self._performance(state)]
            return "\n".join(lines)
        if view in {"risk", "profile", "settings"}:
            lines = [
                "🎚 Risk controls",
                self._capital_text(capital),
                f"Current: {settings['profile']} · base risk {budget}",
                "Profiles (base budgets):",
            ]
            for name, policy in risk_policy.PROFILES.items():
                pct = policy["risk_pct"]
                lines.append(
                    f"• {name}: {_number(pct)}% ({_cash(capital['equity'], pct)})"
                )
                lines.extend(self._cap_lines(name, capital["equity"]))
            lines += [
                "",
                "Custom override: " + str(settings["risk_pct"]),
                "Custom risk: /risk 0.15 means 0.15%. Profile selection clears the override.",
                "Base budgets include the sizing cost allowance. Tier/context/setup/drawdown reductions "
                "apply per opportunity; effective risk can be lower or blocked.",
                "Hard loss halts: daily 2% · weekly 4% · drawdown 12% (strategy MTM, not full-account performance).",
                "Prospective effects only; existing stops are never widened.",
            ]
            lines.extend(self._known_budgets(state, risk, capital, now))
            blockers = self._resume_blockers(state, now)
            if blockers:
                lines.append("⛔ " + "; ".join(blockers))
            lines.extend(self._circuit_lines(state, now))
            return "\n".join(lines)
        if view in {"opportunities", "why", "waves", "radar"}:
            records = sorted(
                state["opportunities"].values(),
                key=lambda r: r.get("updated_at", 0),
                reverse=True,
            )
            if symbol:
                symbol = symbol.upper()
                if not symbol.endswith("USDT"):
                    symbol += "USDT"
                records = [r for r in records if r["opportunity"]["symbol"] == symbol]
            lines = ["🔎 Elliott opportunities"]
            eligible = self._universe_entries(state, now)
            for rec in records[:8]:
                op = rec["opportunity"]
                display_state = op["state"]
                decision = str(rec.get("decision", op["reason"]))
                if op["symbol"] not in eligible:
                    decision = (
                        "WAIT: symbol outside the current eligible universe. "
                        + decision
                    )
                    if display_state == "READY":
                        display_state = "WAIT (stored READY)"
                elif not _finite(op.get("expires_at")) or op["expires_at"] <= now:
                    decision = (
                        "WAIT: candidate expired or expiry unavailable. " + decision
                    )
                    if display_state == "READY":
                        display_state = "WAIT (stored READY)"
                lines.extend(
                    [
                        "",
                        f"{op['symbol']} · {op['side']} · Type {op['setup']} · {display_state}",
                        f"Entry {op['entry']:g} · stop {op['stop']:g} · T1 {op['target1']:g} · T2 {op['target2']:g}",
                        "Decision: " + decision[:350],
                        f"AI: {rec.get('ai', {}).get('verdict', 'Not reviewed')} · {self._age(rec.get('updated_at'))}",
                    ]
                )
                review = rec.get("last_risk_review") or {}
                sizing = review.get("assessment") or {}
                if all(
                    _finite(sizing.get(key))
                    for key in (
                        "rr_target1",
                        "rr_blended",
                        "rr_required_target1",
                        "rr_required_blended",
                    )
                ):
                    lines.extend(
                        [
                            f"Last sizing review: {review.get('arm', 'unknown')} · {self._age(review.get('assessed_at'))}",
                            f"📐 Net R:R · T1 {sizing['rr_target1']:.2f}R (need {sizing['rr_required_target1']:g}R)",
                            f"50/50 targets {sizing['rr_blended']:.2f}R (need {sizing['rr_required_blended']:g}R)",
                        ]
                    )
                    if (
                        _finite(sizing.get("rr_actual_split"))
                        and abs(sizing["rr_actual_split"] - sizing["rr_blended"]) > 1e-8
                    ):
                        lines.append(
                            f"Lot-rounded target split: {sizing['rr_actual_split']:.4f}R"
                        )
                    if _finite(sizing.get("rr_entry_bound")):
                        relation = (
                            "≤"
                            if sizing.get("rr_entry_relation") == "at_or_below"
                            else "≥"
                        )
                        lines.append(
                            f"R:R-only entry bound: {relation} {sizing['rr_entry_bound']:g}"
                        )
                        lines.append(
                            "Diagnostic only; a new valid trigger and every risk check are still required."
                        )
                    if sizing.get("rr_zone_compatible") is False:
                        lines.append(
                            "⚠️ No price in this entry zone meets the nominal R:R hurdles."
                        )
                if view == "waves":
                    evidence = op.get("evidence", {})
                    lines.append(
                        f"Trigger: {evidence.get('trigger_kind', 'unavailable')} · "
                        f"structural validity: {evidence.get('structural_valid', 'unknown')}"
                    )
            if not records:
                lines.append(
                    "No stored candidate yet; waiting for confirmed structure."
                )
            if len(records) > 8:
                lines.append(f"{len(records) - 8} more; use /why SYMBOL.")
            return "\n".join(lines)
        if view in {"positions", "trades"}:
            lines = ["🏦 Exchange positions"]
            if self.config.mode == "shadow":
                lines.append("No private exchange data in SHADOW mode.")
            elif capital["error"] or not isinstance(account.get("positions"), list):
                lines.append(
                    "⚠️ Exchange snapshot unavailable/stale; do not interpret this as flat."
                )
            else:
                lines.append(self._capital_text(capital))
                for p in account["positions"]:
                    lines.append(
                        f"{p['symbol']} {p.get('side')} · qty {p.get('size')}\n"
                        f"Entry {p.get('avgPrice')} · SL {p.get('stopLoss')} · TP {p.get('takeProfit')}"
                    )
                if not account["positions"]:
                    lines.append("No positions in the last reconciled snapshot.")
            lines.append("\n👻 Simulated positions")
            active = [
                t
                for t in state["trades"].values()
                if t["status"] in {"OPEN", "PENDING"}
            ]
            for t in active[:12]:
                lines.append(
                    f"{t['symbol']} {t['side']} · {t['arm']} · {t['status']}\n"
                    f"Entry {t.get('entry') or t['limit']:g} · SL {t['stop']:g}"
                )
            if not active:
                lines.append("None.")
            return "\n".join(lines)
        if view in {"performance", "stats", "shadowstats"}:
            return self._performance(state)
        if view in {"research", "mlpatterns"}:
            research = state.get("research", {})
            return (
                "🧠 AI research\n"
                + str(
                    research.get("summary", research.get("reason", "No result yet."))
                )[:2600]
                + f"\nReviews stored: {len(state['reviews'])}. AI research is advisory, not a measured win probability."
            )
        if view in {"system", "telemetry", "debug"}:
            outbox = await self.store.outbox_health()
            lines = [
                f"⚙️ System · {self.config.mode}",
                f"Database: {'Postgres' if self.config.database_url else 'SQLite (local)'}",
                f"Scan: {self._age(health.get('scan_at'))} · Reconcile: {self._age(health.get('reconcile_at'))}",
                f"Telegram pending: {outbox['pending']} · oldest {int(outbox['oldest_age'])}s",
                f"AI configured: {'yes' if self.config.openai_key and self.config.openai_model else 'no'}",
                f"Reference documents: {len(state['references'])}",
                f"Issue recorded: {'yes' if health.get('error') else 'no'}",
                "UTC accounting days.",
            ]
            lines.extend(self._circuit_lines(state, now))
            if view == "telemetry":
                lines += ["", self._performance(state)]
            return "\n".join(lines)
        return "Unknown view. Use /help."

    @staticmethod
    def _universe_symbols(value):
        if not isinstance(value, (list, tuple)):
            return []
        return list(dict.fromkeys(s for s in value if isinstance(s, str) and s))

    def _universe_entries(self, state, now):
        if getattr(self.config, "universe_mode", "static") != "dynamic":
            return set(self.config.symbols)
        universe = state.get("universe")
        if not isinstance(universe, dict):
            return set()
        stamp = universe.get("as_of")
        if not _finite(stamp) or stamp <= 0 or not 0 <= now - stamp < UNIVERSE_MAX_AGE:
            return set()
        required_policy = {
            "target": getattr(self.config, "universe_size", 50),
            "min_turnover_usdt": getattr(
                self.config, "universe_min_turnover", 20_000_000
            ),
            "max_spread_bps": getattr(self.config, "universe_max_spread_bps", 10),
            "min_depth_usdt": getattr(self.config, "universe_min_depth", 25_000),
        }
        eligible = set(entry_symbols(universe, now, required_policy=required_policy))
        if isinstance(state.get("instrument_symbols"), list):
            eligible.intersection_update(state["instrument_symbols"])
        return eligible

    def _universe_summary(self, state, now):
        dynamic = getattr(self.config, "universe_mode", "static") == "dynamic"
        universe = state.get("universe")
        universe = universe if isinstance(universe, dict) else {}
        active = self._universe_symbols(
            universe.get("active_symbols") if dynamic else self.config.symbols
        )
        managed = {
            record["symbol"]
            for ledger in ("trades", "orders")
            for record in state.get(ledger, {}).values()
            if isinstance(record, dict)
            and isinstance(record.get("symbol"), str)
            and (
                record.get("status") in {"OPEN", "PENDING"}
                if ledger == "trades"
                else record.get("status") not in {"CLOSED", "CANCELLED", "REJECTED"}
            )
        }
        draining = f"Draining: {len(managed - set(active))} · managed outside active membership"
        if not dynamic:
            return [f"Universe: {len(active)} configured · manual (static)", draining]
        target = universe.get("target", getattr(self.config, "universe_size", 50))
        lines = [f"Universe: {len(active)}/{target} active · dynamic"]
        stamp = universe.get("as_of")
        eligible = len(self._universe_entries(state, now))
        lines.append(
            f"Eligible for entry: {eligible}/{len(active)} · other checks still apply"
        )
        if not active:
            lines.append("Awaiting selection · entry wait")
        desired = {
            "target": getattr(self.config, "universe_size", 50),
            "min_turnover_usdt": getattr(
                self.config, "universe_min_turnover", 20_000_000
            ),
            "max_spread_bps": getattr(self.config, "universe_max_spread_bps", 10),
            "min_depth_usdt": getattr(self.config, "universe_min_depth", 25_000),
        }
        if active and any(
            universe.get("policy", {}).get(k) != v for k, v in desired.items()
        ):
            lines.append("Settings changed · waiting for liquidity recheck")
        if isinstance(state.get("instrument_symbols"), list):
            missing = set(active) - set(state["instrument_symbols"])
            if missing:
                lines.append(
                    f"Fresh instrument rules unavailable: {len(missing)} members · entry wait"
                )
        lines.append(f"Selection status: {universe.get('status') or 'unavailable'}")
        health = state.get("health")
        error = health.get("universe_error") if isinstance(health, dict) else None
        if error:
            lines.append("⚠️ Universe refresh error: " + self._safe(error)[:200])
        if not _finite(stamp) or not 0 < stamp <= now:
            lines.append(
                "Selection as of: unavailable · freshness unknown · entry wait"
            )
        else:
            age = int(now - stamp)
            if age >= 3600:
                elapsed = f"{age // 3600}h {(age % 3600) // 60}m"
            elif age >= 60:
                elapsed = f"{age // 60}m {age % 60}s"
            else:
                elapsed = f"{age}s"
            freshness = (
                "STALE · entry wait" if now - stamp >= UNIVERSE_MAX_AGE else "fresh"
            )
            lines.append(
                f"Selection as of: {datetime.fromtimestamp(stamp, timezone.utc):%Y-%m-%d %H:%M UTC}"
                f" · {elapsed} ago · {freshness}"
            )
        lines.append(draining)
        return lines

    def _universe_policy(self, policy):
        if not isinstance(policy, dict) or not policy:
            return ["Policy: " + self._universe_description(policy)]
        labels = {
            "min_turnover_usdt": "24h turnover ≥{} USDT",
            "max_spread_bps": "Spread ≤{} bps",
            "min_listing_days": "Listing age ≥{} days",
            "min_depth_usdt": "Depth ≥{} USDT/side",
            "depth_band_bps": "Depth band: {} bps",
            "replacement_ratio": "Challenger ≥{}× incumbent",
            "confirmation_observations": "Confirm across {} observations",
            "max_daily_replacements": "Healthy replacements ≤{}/day",
            "daily_max_replacements": "Healthy replacements ≤{}/day",
            "review_hours": "Review every {}h",
            "max_book_age": "Book age ≤{}s",
            "max_snapshot_age": "Snapshot age ≤{}s",
        }
        details = []
        for key, value in policy.items():
            if key in {"review_seconds", "review_interval_seconds"} and _finite(value):
                details.append(f"Review every {value / 3600:g}h")
            elif key in labels and _finite(value):
                details.append(labels[key].format(self._universe_description(value)))
            else:
                details.append(
                    f"{str(key).replace('_', ' ')}: {self._universe_description(value)}"
                )
        return [
            "Policy (persisted):",
            *(" · ".join(details[i : i + 2]) for i in range(0, len(details), 2)),
        ]

    @staticmethod
    def _universe_description(value):
        if value is None or value == "" or value == {}:
            return "unavailable (not persisted)"
        if isinstance(value, dict):
            return "; ".join(
                f"{str(key).replace('_', ' ')}: {Service._universe_description(item)}"
                for key, item in value.items()
            )
        if isinstance(value, list):
            return ", ".join(str(item) for item in value) or "none"
        if _finite(value):
            for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
                if abs(value) >= scale:
                    return f"{value / scale:.3g}{suffix}"
            return f"{value:g}"
        return str(value)

    def _universe(self, state, now):
        lines = ["🌐 Universe · read-only", *self._universe_summary(state, now)]
        dynamic = getattr(self.config, "universe_mode", "static") == "dynamic"
        universe = state.get("universe")
        universe = universe if dynamic and isinstance(universe, dict) else {}
        active = self._universe_symbols(
            universe.get("active_symbols") if dynamic else self.config.symbols
        )
        members = universe.get("members")
        members = members if isinstance(members, dict) else {}
        if active:
            lines += [
                "",
                (
                    "Active symbols (stored rank when available):"
                    if dynamic
                    else "Configured symbols:"
                ),
            ]
            labels = []
            for symbol in active:
                member = members.get(symbol)
                rank = member.get("rank") if isinstance(member, dict) else None
                labels.append(
                    f"{rank}. {symbol}" if type(rank) is int and rank > 0 else symbol
                )
            lines.extend(
                " · ".join(labels[i : i + 2]) for i in range(0, len(labels), 2)
            )
        if not dynamic:
            return "\n".join(lines)
        lines += [
            "",
            "Ranking basis: " + self._universe_description(universe.get("basis")),
            *self._universe_policy(universe.get("policy")),
            "Replacements today: "
            + self._universe_description(universe.get("day_replacements")),
            "Added: " + self._universe_description(universe.get("added")),
            "Removed: " + self._universe_description(universe.get("removed")),
        ]
        for key, label in (
            ("candidate_count", "Candidates checked"),
            ("book_checked", "Order books checked"),
        ):
            if key in universe:
                lines.append(f"{label}: {self._universe_description(universe[key])}")
        if "snapshot_time" in universe:
            stamp = universe["snapshot_time"]
            snapshot_time = (
                f"{datetime.fromtimestamp(stamp, timezone.utc):%Y-%m-%d %H:%M UTC}"
                if _finite(stamp) and 0 < stamp <= now
                else "unavailable"
            )
            lines.append(f"Market snapshot: {snapshot_time}")

        # Compact ranges retain all tickers without a fifty-row price/metrics table.
        def compact(value):
            for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
                if abs(value) >= scale:
                    return f"{value / scale:.3g}{suffix}"
            return f"{value:.3g}"

        lines += ["", "Stored liquidity ranges (active members):"]
        for key, label, unit in (
            ("score", "Score", ""),
            ("turnover24h", "24h turnover", " USDT"),
            ("spread_bps", "Spread", " bps"),
            ("bid_depth_usdt", "Bid depth", " USDT"),
            ("ask_depth_usdt", "Ask depth", " USDT"),
            ("sample_count", "History samples", " per member"),
        ):
            values = [
                member[key]
                for symbol in active
                if isinstance((member := members.get(symbol)), dict)
                and _finite(member.get(key))
            ]
            detail = "unavailable"
            if values:
                detail = (
                    f"{compact(min(values))}–{compact(max(values))}{unit}"
                    f" ({len(values)}/{len(active)})"
                )
            lines.append(f"{label}: {detail}")
        blocked = universe.get("blocked")
        if isinstance(blocked, dict):
            active_blocked = [symbol for symbol in active if symbol in blocked]
            lines.append(
                f"Blocked: {len(blocked)} total · {len(active_blocked)} active"
            )
            if active_blocked:
                lines.append("Entry wait for blocked active members.")
            ordered = active_blocked + [
                symbol for symbol in blocked if symbol not in active_blocked
            ]
            for symbol in ordered[:5]:
                reason = blocked[symbol]
                lines.append(f"• {symbol}: {self._universe_description(reason)[:200]}")
            if len(blocked) > 5:
                lines.append(f"+{len(blocked) - 5} more blocked symbols")
        else:
            lines.append("Blocked reasons: unavailable (not persisted)")
        return "\n".join(lines)

    def _circuit_lines(self, state, now):
        lines = [
            "🛡 Strategy MTM risk gates (not full-account performance)",
            "Daily/weekly gates include the entire current open loss, even from earlier periods; "
            "these are not exact calendar-period returns. Fixed reference capital is tracked per arm/mode.",
        ]
        for key in (
            [arm for arm, _ in ARMS]
            if self.config.mode == "shadow"
            else [self.config.mode]
        ):
            circuit = state.get("risk_circuits", {}).get(key, {})
            if not _fresh(circuit.get("as_of"), now, RISK_MAX_AGE) or any(
                not _finite(circuit.get(f)) or circuit[f] < 0 for f in LOSS_LIMITS
            ):
                lines.append(f"{key}: unavailable/stale; resume blocked")
            else:
                halted = (
                    circuit.get("hard_halt")
                    or circuit.get("halted")
                    or any(circuit[f] >= limit for f, limit in LOSS_LIMITS.items())
                )
                label = " · HALTED" if halted else ""
                lines.append(
                    f"{key}: daily {circuit['daily_loss_pct']:.2f}% · weekly {circuit['weekly_loss_pct']:.2f}% · "
                    f"drawdown {circuit['drawdown_pct']:.2f}% · {self._age(circuit['as_of'])}{label}"
                )
        return lines

    def _cap_lines(self, profile, equity):
        lines = []
        for key, value in self._caps(profile).items():
            text = (
                f"{_number(value)}% ({_cash(equity, value)})"
                if key.endswith("_pct")
                else str(value)
            )
            lines.append(f"  {CAP_LABELS.get(key, key)} ≤ {text}")
        return lines

    @staticmethod
    def _age(stamp):
        return (
            f"{int(time.time() - stamp)}s ago"
            if _fresh(stamp, time.time(), 10**10)
            else "unavailable"
        )

    @staticmethod
    def _breakdown(records, net_field):
        lines = []
        for side, label in (("Buy", "Long"), ("Sell", "Short")):
            selected = [r for r in records if r.get("side") == side]
            wins = sum(r.get(net_field, 0) > 1e-9 for r in selected)
            losses = sum(r.get(net_field, 0) < -1e-9 for r in selected)
            lines.append(
                f"{label}: {len(selected)} closed · {wins}W/{losses}L/{len(selected)-wins-losses}BE"
            )
        setups = sorted(
            {
                str(r.get("setup", r.get("opportunity", {}).get("setup", "unknown")))
                for r in records
            }
        )
        for setup in setups:
            selected = [
                r
                for r in records
                if str(r.get("setup", r.get("opportunity", {}).get("setup", "unknown")))
                == setup
            ]
            long = sum(r.get("side") == "Buy" for r in selected)
            short = sum(r.get("side") == "Sell" for r in selected)
            lines.append(
                f"Setup {setup}: {len(selected)} closed · long {long} / short {short}"
            )
        return lines

    def _performance(self, state):
        lines = ["📊 Forward performance"]
        trades = list(state["trades"].values())
        for arm, label in ARMS:
            result = summary(trades, arm)
            wr = "—" if result["wr"] is None else f"{result['wr']:.1f}%"
            lines += [
                "",
                f"👻 {label}",
                f"{result['wins']} wins · {result['losses']} losses · {result['breakeven']} breakeven",
                f"WR {wr} (valid closed) · Closed {result['closed']} · Open {result['open']} · Pending {result['pending']}",
                f"Expired unfilled {result['expired']} · Estimated net {_cash(result['net_pnl'])}",
                f"Funding pending on {result['funding_pending']} closed trades",
                f"Invalid closed outcomes excluded: {result['invalid_outcomes']}",
                f"Management incomplete: {result['management_incomplete']} · Data gaps/errors: {result['data_gaps']}",
            ]
            closed = [
                t
                for t in trades
                if t.get("arm") == arm
                and t.get("status") == "CLOSED"
                and _finite(t.get("net_pnl"))
                and not t.get("data_error")
            ]
            lines.extend(self._breakdown(closed, "net_pnl"))
        lines.extend(self._paired_comparison(trades))
        closed = [o for o in state["orders"].values() if o.get("status") == "CLOSED"]
        valid = [
            o
            for o in closed
            if _finite(o.get("net_pnl_before_funding")) and not o.get("data_error")
        ]
        lines += [
            "",
            f"🏦 Exchange-confirmed: {len(closed)} closed",
            f"Net before funding: {_cash(sum(o['net_pnl_before_funding'] for o in valid))}",
            f"Invalid closed outcomes excluded: {len(closed)-len(valid)}",
        ]
        lines.extend(self._breakdown(valid, "net_pnl_before_funding"))
        lines += [
            "Shadow estimates include assumed costs; exchange funding reconciliation is separate.",
            "No closed trades = no measured win rate. Legacy results remain separate.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _paired_comparison(trades):
        lines = ["", "🔬 Rules baseline vs AI-selected candidates"]
        # Do not invent missing decision times or identities for historical rows.
        if any(
            not isinstance(t.get(k), str) or not t[k]
            for t in trades
            for k in ("id", "opportunity_id")
        ) or any(not _finite(t.get("created_at")) for t in trades):
            return lines + [
                "Paired comparison unavailable: decision/identity metadata incomplete."
            ]
        result = comparison(trades)
        matches = result["matches"]
        invalid_ids = {
            t["id"]
            for t in trades
            if t.get("status") == "CLOSED"
            and (not _finite(t.get("net_pnl")) or t.get("data_error"))
        }
        invalid_pairs = sum(
            m["baseline_id"] in invalid_ids or m["ai_id"] in invalid_ids
            for m in matches
        )
        lines += [
            f"Matched opportunities: {result['matched_count']}",
            f"Unmatched candidates: baseline {result['baseline_unselected_count']} · AI {result['ai_without_baseline_count']}",
        ]
        # comparison currently counts finite data_error outcomes as closed. Keep
        # these from appearing as a valid paired result until repaired upstream.
        if invalid_pairs:
            lines.append(
                f"Paired results unavailable: {invalid_pairs} matched opportunities have invalid outcomes."
            )
        else:
            lines.append(
                f"Matched opportunities closed in both arms: {result['paired_closed_count']}"
            )
            net = (
                _cash(result["matched_net_delta"])
                if result["paired_closed_count"]
                else "— (no closed pairs)"
            )
            lines.append(f"Paired AI − baseline net: {net}")
        if matches:
            latency = [m["decision_latency_seconds"] for m in matches]
            lines += [
                f"AI decision delay vs baseline: {min(latency):g}–{max(latency):g}s",
                f"Same entry window: {sum(m['same_entry_window'] for m in matches)}/{len(matches)} pairs",
                "Differences across matched candidates: "
                + " · ".join(
                    f"{label} {sum(key in m['confounds'] for m in matches)}"
                    for key, label in (
                        ("entry_latency", "entry timing"),
                        ("portfolio_sizing_or_levels", "sizing/levels"),
                        ("simulation_assumptions", "simulation assumptions"),
                    )
                ),
                f"Funding incomplete on {sum(m['both_closed'] and not m['funding_complete'] for m in matches)} closed pairs",
            ]
        lines += [
            "Paired comparisons exclude unmatched/unfilled candidates; they do not measure selection skill alone.",
            result["note"],
        ]
        return lines
