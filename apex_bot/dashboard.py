"""Read-only Telegram presentation. No network, settings or ledger mutations."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import urlsplit

from .simulation import summary


ARM_NAMES = {"baseline_shadow": "Rules baseline", "ai_shadow": "AI-approved shadow"}
PROFILE_NAMES = {
    "ultra_cautious": "Ultra cautious",
    "cautious": "Cautious",
    "balanced": "Balanced",
    "aggressive": "Aggressive",
    "extreme": "Extreme",
}


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def age(stamp, now):
    if not finite(stamp) or stamp <= 0 or stamp > now:
        return "unavailable"
    seconds = int(now - stamp)
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m ago"
    return f"{seconds // 86400}d ago"


def price(value):
    if not finite(value):
        return "unavailable"
    # Ten significant digits remove binary arithmetic tails while retaining
    # tiny nonzero prices. Expand scientific notation for readable phone cards.
    text = format(Decimal(format(value, ".10g")), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def side(value):
    return {"Buy": "🟢 LONG", "Sell": "🔴 SHORT"}.get(value, "Direction unavailable")


def words(value):
    """A readable fallback, never a fabricated explanation for an unknown code."""
    return str("unavailable" if value is None or value == "" else value).replace(
        "_", " "
    )


def reason(value):
    from .notifications import describe_reason

    return describe_reason(str(value or "No decision recorded yet"))


def pages(text, limit=3000):
    """Lossless paragraph-first pages bounded by UTF-16 units, including emojis."""
    blocks = re.findall(r".*?(?:\n\n|\Z)", text, re.S)
    result, current = [], ""
    units = lambda s: len(s.encode("utf-16-le")) // 2
    for block in blocks:
        if not block:
            continue
        if units(block) > limit:
            if current:
                result.append(current)
                current = ""
            for line in block.splitlines(keepends=True):
                for char in line:
                    if units(current) + units(char) > limit:
                        result.append(current)
                        current = ""
                    current += char
        elif units(current) + units(block) > limit:
            result.append(current)
            current = block
        else:
            current += block
    if current:
        result.append(current)
    return result or ["No information available."]


class Dashboard:
    def __init__(self, service, state, now):
        self.service, self.state, self.now = service, state, now
        self.config = service.config
        self.health = state.get("health", {})
        self.eligible = service._universe_entries(state, now)
        self.scan_limit = max(180, self.config.scan_seconds * 3)

    def words(self, value):
        # Humanization must not transform a secret before exact-match redaction.
        return words(
            self.service._safe("unavailable" if value is None or value == "" else value)
        )

    def data_health(self):
        """Payload evidence is separate from a loop completing successfully."""
        context = self.state.get("context", {})
        context_ok = (
            context.get("data_complete") is True
            and context.get("policy_authority") is not False
            and self.fresh(context.get("as_of"), 21600)
            and finite(context.get("expires_at"))
            and context["expires_at"] > self.now
            and context.get("risk_state") in {"risk_on", "neutral", "risk_off"}
            and type(context.get("event_blackout")) is bool
            and all(
                finite(context.get(k)) and 0 <= context[k] <= 1
                for k in ("long_multiplier", "short_multiplier")
            )
        )
        result = {
            "Market context data": (
                "Available" if context_ok else "Unavailable / stale or incomplete"
            )
        }
        from .service import LOSS_LIMITS

        for key in ARM_NAMES if self.config.mode == "shadow" else (self.config.mode,):
            circuit = self.state.get("risk_circuits", {}).get(key, {})
            current = (
                not circuit.get("error")
                and self.fresh(circuit.get("as_of"), 120)
                and all(finite(circuit.get(k)) and circuit[k] >= 0 for k in LOSS_LIMITS)
            )
            halted = current and (
                circuit.get("hard_halt")
                or circuit.get("halted")
                or any(circuit[k] >= cap for k, cap in LOSS_LIMITS.items())
            )
            result[f"Risk data · {ARM_NAMES.get(key, self.words(key))}"] = (
                "HALTED"
                if halted
                else "Available" if current else "Unavailable / stale or incomplete"
            )
        return result

    def ai_review(self, rec):
        review = rec.get("ai") or {}
        if not review:
            return "AI: Not reviewed"
        stamp = review.get("created_at")
        label = self.words(review.get("verdict", "Unknown result"))
        suffix = ""
        if not self.fresh(stamp, 21600):
            suffix = " · historical / age unavailable"
        return f"AI: {label} · reviewed {age(stamp, self.now)}{suffix}"

    def fresh(self, stamp, limit):
        return finite(stamp) and stamp > 0 and 0 <= self.now - stamp <= limit

    def current(self, rec):
        op = rec.get("opportunity", {})
        return (
            op.get("state") in {"READY", "WAIT"}
            and op.get("symbol") in self.eligible
            and finite(op.get("expires_at"))
            and op["expires_at"] > self.now
            and self.fresh(rec.get("updated_at"), self.scan_limit)
        )

    def candidates(self):
        records = list(self.state.get("opportunities", {}).values())
        current = [r for r in records if self.current(r)]
        return records, current

    def errors(self):
        return {k[:-6]: v for k, v in self.health.items() if k.endswith("_error") and v}

    def research_status(self):
        data = self.state.get("research", {})
        configured = bool(self.config.openai_key and self.config.openai_model)
        if not configured:
            return "Not configured"
        if data.get("status") == "COMPLETE":
            return "Available" if self.fresh(data.get("created_at"), 86400) else "Stale"
        return "Unavailable" if data else "Awaiting first result"

    def home(self):
        from .service import _cash, percentage

        settings = self.state["settings"]
        records, current = self.candidates()
        ready = sum(r["opportunity"]["state"] == "READY" for r in current)
        long = sum(r["opportunity"].get("side") == "Buy" for r in current)
        short = sum(r["opportunity"].get("side") == "Sell" for r in current)
        errors = self.errors()
        scanner = self.fresh(self.health.get("scan_at"), self.scan_limit)
        data_issues = any(v != "Available" for v in self.data_health().values())
        status = (
            "🟢 Running"
            if scanner and not errors and not data_issues
            else "🟠 Needs attention"
        )
        mode = (
            "🧪 SHADOW · simulated trades"
            if self.config.mode == "shadow"
            else f"🏦 {self.config.mode.upper()}"
        )
        risk = percentage(settings["profile"], settings.get("risk_pct"))
        capital = self.service._capital(self.state, self.now)
        target = (
            self.config.universe_size
            if getattr(self.config, "universe_mode", "static") == "dynamic"
            else len(self.config.symbols)
        )
        lines = [
            "🌊 APEX · CONTROL CENTRE",
            mode,
            f"{status} · scan {age(self.health.get('scan_at'), self.now)}",
            f"Updated {datetime.fromtimestamp(self.now, timezone.utc):%d %b %H:%M UTC}",
            "",
            f"🔎 Universe: {len(self.eligible)}/{target} eligible symbols",
            "Daily structure → closed 4H confirmation",
            f"Candidates now: {len(current)} · {long} long / {short} short",
            f"Ready for checks: {ready} · Waiting: {len(current)-ready}",
            "",
            f"🛡 {PROFILE_NAMES.get(settings['profile'], self.words(settings['profile']))} · base risk {risk:g}%",
        ]
        if self.config.mode == "shadow":
            lines += [
                f"Starting capital: {_cash(capital['equity'])} / simulation",
                "Exchange execution: OFF",
            ]
        else:
            lines.append(self.service._capital_text(capital))
            lines.append(f"Base budget: {_cash(capital['equity'], risk)}")
        blockers = self.service._resume_blockers(self.state, self.now)
        if settings.get("paused"):
            lines.append("New entries: ⏸ Paused")
        else:
            lines.append("Entry control: monitoring · checks apply")
        halted = any(
            c.get("hard_halt")
            or c.get("halted")
            or any(
                finite(c.get(k)) and c[k] >= cap
                for k, cap in (
                    ("daily_loss_pct", 2),
                    ("weekly_loss_pct", 4),
                    ("drawdown_pct", 12),
                )
            )
            for key, c in self.state.get("risk_circuits", {}).items()
            if key
            in (ARM_NAMES if self.config.mode == "shadow" else {self.config.mode})
        )
        if halted:
            lines.append("⛔ Risk circuit HALTED · see Risk")
        lines += ["", "📊 SIMULATED PORTFOLIOS"]
        for arm, label in ARM_NAMES.items():
            data = summary(list(self.state["trades"].values()), arm)
            valid = data["closed"] - data["invalid_outcomes"]
            wr = f"{data['wr']:.1f}%" if data["wr"] is not None else "—"
            lines += [
                f"{label} · Open {data['open']} / Pending {data['pending']}",
                f"Closed {data['closed']} · "
                + (
                    f"W {data['wins']} / L {data['losses']} / Flat {data['breakeven']} · WR {wr}"
                    if valid
                    else "Win rate: — (no valid closed trades)"
                ),
            ]
            incomplete_closes = any(
                t.get("arm") == arm
                and t.get("status") == "CLOSED"
                and not t.get("management_complete")
                for t in self.state["trades"].values()
            )
            if (
                data["funding_pending"]
                or data["invalid_outcomes"]
                or data["data_gaps"]
                or incomplete_closes
            ):
                lines.append(
                    "⚠️ Incomplete outcome data · results provisional · see Performance"
                )
        alerts = []
        if len(self.eligible) < target:
            alerts.append("Below universe target; liquidity rules apply.")
        if self.research_status() != "Available":
            alerts.append(
                f"AI research: {self.research_status().lower()} · advisory only."
            )
        if errors:
            alerts.append(f"{len(errors)} active component issue(s) · see System.")
        if data_issues:
            alerts.append("Context or risk data needs attention · see System.")
        if blockers:
            alerts.append("Resume prerequisites need attention · Status / Risk.")
        if alerts:
            lines += ["", *("⚠️ " + item for item in alerts[:3])]
        return "\n".join(lines)

    def status(self):
        records, current = self.candidates()
        counts = Counter(r["opportunity"].get("state") for r in current)
        lines = [
            "📡 OPERATING STATUS",
            f"Mode: {self.config.mode.upper()}",
            f"Last completed scan: {age(self.health.get('scan_at'), self.now)}",
            f"Last symbol checked: {self.health.get('last_symbol', 'unavailable')}",
            "",
            f"Current candidates: {len(current)}",
            f"Ready for risk checks: {counts['READY']} · Waiting / entry conditions unmet: {counts['WAIT']}",
            f"Historical / inactive records: {len(records)-len(current)}",
            "Historical records are not current signals.",
        ]
        lines += ["", *self.service._universe_summary(self.state, self.now)]
        if self.config.mode != "shadow":
            capital = self.service._capital(self.state, self.now)
            lines += ["", self.service._capital_text(capital)]
            if capital["error"]:
                lines.append("Base budget: cash unavailable")
        blockers = self.service._resume_blockers(self.state, self.now)
        lines += ["", "ENTRY CONTROL & RESUME PREREQUISITES"]
        if self.state["settings"].get("paused"):
            lines.append("⏸ New entries paused; existing management continues.")
        lines += ["⛔ " + self.words(b) for b in blockers] or [
            "Resume prerequisites pass; every candidate still needs individual checks."
        ]
        lines.append(
            "These are prerequisites for the Resume control, not a claim that every simulated setup is blocked."
        )
        context = self.state.get("context", {})
        lines += [
            "",
            "MARKET CONTEXT",
            f"Risk regime: {self.words(context.get('risk_state'))}",
            f"Updated: {age(context.get('as_of'), self.now)}",
            "Event blackout: "
            + {True: "Active", False: "Not active"}.get(
                context.get("event_blackout"), "Unknown"
            ),
            "Calendar covers scheduled US CPI, jobs and FOMC events; coverage is not all market news.",
            "",
            "Research is advisory. A ready candidate is not an executed trade.",
        ]
        return "\n".join(lines)

    def opportunities(self, view, symbol):
        from .notifications import setup_label

        records, current = self.candidates()
        if symbol:
            symbol = symbol.upper()
            if not symbol.endswith("USDT"):
                symbol += "USDT"
            records = [
                r for r in records if r.get("opportunity", {}).get("symbol") == symbol
            ]
        records.sort(
            key=lambda r: (
                not self.current(r),
                r.get("opportunity", {}).get("state") != "READY",
                -r.get("updated_at", 0),
            )
        )
        visible = records if symbol else [r for r in records if self.current(r)]
        lines = [
            "🔎 OPPORTUNITY RADAR" if not symbol else f"🌊 {symbol} · WAVE REVIEW",
            "Planned setups, not trade fills.",
            f"Current: {len([r for r in visible if self.current(r)])} · Stored history: {len(records)}",
            "Daily structure; confirmation uses closed 4H candles.",
        ]
        if not visible:
            lines += [
                "",
                (
                    "No current qualifying setup in the latest scan. Waiting for confirmed structure."
                    if self.fresh(self.health.get("scan_at"), self.scan_limit)
                    and not self.health.get("scan_error")
                    else "No current qualifying setup can be confirmed. Scanner data is stale or unavailable; see System."
                ),
                "Use /why SYMBOL to inspect a symbol's stored history.",
            ]
        for rec in visible:
            op = rec["opportunity"]
            state = op.get("state", "Unknown")
            label = {
                "READY": "Ready for risk checks",
                "WAIT": "Waiting · entry conditions unmet",
                "INVALID": "Invalidated",
                "EXPIRED": "Expired",
            }.get(state, self.words(state))
            if not self.current(rec):
                label = "Historical / inactive · " + label
            lines += [
                "",
                f"{op.get('symbol')} · {side(op.get('side'))}",
                f"{label} · {setup_label(self.service._safe(op.get('setup')))}",
                f"Planned entry {price(op.get('entry'))}",
                f"Stop {price(op.get('stop'))} · Invalidation {price(op.get('invalidation'))}",
                f"T1 {price(op.get('target1'))} · T2 {price(op.get('target2'))}",
                "Next / decision: "
                + reason(self.service._safe(rec.get("decision") or op.get("reason")))[
                    :450
                ],
                self.ai_review(rec),
                f"Setup observed: {age(rec.get('updated_at'), self.now)}",
            ]
            if symbol or view in {"why", "waves"}:
                for arm, name in ARM_NAMES.items():
                    decision = rec.get("per_arm_decisions", {}).get(arm)
                    if isinstance(decision, dict):
                        lines.append(
                            f"{name}: "
                            + reason(self.service._safe(decision.get("decision")))[:350]
                        )
                        lines.append(
                            "Decision recorded: "
                            + age(decision.get("updated_at"), self.now)
                        )
            if symbol or view in {"why", "waves"}:
                review = rec.get("last_risk_review") or {}
                sizing = review.get("assessment") or {}
                if all(
                    finite(sizing.get(k))
                    for k in (
                        "rr_target1",
                        "rr_blended",
                        "rr_required_target1",
                        "rr_required_blended",
                    )
                ):
                    lines += [
                        f"Last sizing review: {ARM_NAMES.get(review.get('arm'), self.words(review.get('arm')))} · {age(review.get('assessed_at'), self.now)}",
                        f"📐 Net R:R · T1 {sizing['rr_target1']:.2f}R (need {sizing['rr_required_target1']:g}R)",
                        f"50/50 targets {sizing['rr_blended']:.2f}R (need {sizing['rr_required_blended']:g}R)",
                    ]
                    if (
                        finite(sizing.get("rr_actual_split"))
                        and abs(sizing["rr_actual_split"] - sizing["rr_blended"]) > 1e-8
                    ):
                        lines.append(
                            f"Lot-rounded target split: {sizing['rr_actual_split']:.4f}R"
                        )
                    if finite(sizing.get("rr_entry_bound")):
                        relation = (
                            "≤"
                            if sizing.get("rr_entry_relation") == "at_or_below"
                            else "≥"
                        )
                        lines += [
                            f"R:R-only entry bound: {relation} {price(sizing['rr_entry_bound'])}",
                            "Diagnostic only; a valid trigger and every risk check are still required.",
                        ]
                    if sizing.get("rr_zone_compatible") is False:
                        lines.append(
                            "⚠️ No price in this entry zone meets the nominal R:R hurdles."
                        )
                evidence = op.get("evidence", {})
                lines += [
                    f"Trigger: {self.words(evidence.get('trigger_kind'))}",
                    f"Structural validity: {self.words(evidence.get('structural_valid', 'unknown'))}",
                ]
            else:
                lines.append(f"Details: /why {op.get('symbol')}")
        return "\n".join(lines)

    def positions(self):
        lines = ["📂 POSITIONS & WAITING ENTRIES"]
        capital = self.service._capital(self.state, self.now)
        account = self.state.get("account", {})
        if self.config.mode == "shadow":
            lines += [
                "Exchange: not queried in SHADOW mode.",
                "This screen does not confirm the Bybit account is flat.",
            ]
        elif capital["error"] or not isinstance(account.get("positions"), list):
            lines.append(
                "⚠️ Exchange snapshot unavailable/stale; do not interpret this as flat."
            )
        else:
            lines += [self.service._capital_text(capital)]
            for p in account["positions"]:
                lines += [
                    "",
                    f"🏦 {p.get('symbol')} · {side(p.get('side'))}",
                    f"Quantity {p.get('size')} · Entry {p.get('avgPrice')}",
                    f"Stop {p.get('stopLoss')} · Take profit {p.get('takeProfit')}",
                ]
            if not account["positions"]:
                lines.append("No positions in the last reconciled snapshot.")
        for arm, name in ARM_NAMES.items():
            active = sorted(
                [
                    t
                    for t in self.state["trades"].values()
                    if t.get("arm") == arm and t.get("status") in {"OPEN", "PENDING"}
                ],
                key=lambda t: t.get("created_at", 0),
                reverse=True,
            )
            lines += ["", f"🧪 {name} · {len(active)} active"]
            if not active:
                lines.append("No open trades or waiting entries.")
            for t in active:
                opened = t.get("status") == "OPEN"
                lines += [
                    "",
                    f"Portfolio: {name}",
                    f"{t.get('symbol')} · {side(t.get('side'))}",
                    (
                        "Simulated position OPEN"
                        if opened
                        else "Waiting for simulated fill · not open"
                    ),
                    ("Filled entry " if opened else "Planned entry ")
                    + price(t.get("entry") if opened else t.get("limit")),
                    f"Stop {price(t.get('stop'))}",
                    f"T1 {price(t.get('target1'))} · T2 {price(t.get('target2'))}",
                    f"Quantity {price(t.get('qty'))} · Remaining {price(t.get('remaining'))}",
                    f"{'Opened' if opened else 'Created'} {age(t.get('opened_at') if opened else t.get('created_at'), self.now)}",
                ]
                if t.get("data_error") or t.get("data_gap"):
                    lines.append(
                        "⚠️ Price history incomplete; outcome updates may be delayed."
                    )
                if opened:
                    lines += [
                        "First target: "
                        + ("Partially exited" if t.get("tp1_done") else "Not reached"),
                        f"Last valuation: {price(t.get('mark_price'))} · {age(t.get('mark_at'), self.now)}",
                    ]
                    if not self.fresh(t.get("mark_at"), 600):
                        lines.append("⚠️ Valuation is stale or unavailable.")
        return "\n".join(lines)

    def research(self):
        from .notifications import research_diagnosis

        data = self.state.get("research", {})
        lines = [
            "🧠 AI RESEARCH",
            f"Result: {self.research_status()}",
            f"Updated: {age(data.get('created_at'), self.now)}",
            f"Research date: {data.get('asof', 'unavailable')}",
            "Advisory information; it cannot override trading rules.",
        ]
        if data.get("status") != "COMPLETE":
            diagnosis = research_diagnosis(data)
            lines += [
                "",
                "Reason: "
                + (
                    diagnosis
                    or self.words(
                        data.get("reason", "No validated research result yet")
                    )
                ),
                "No validated conclusion is available. This does not mean markets are safe.",
            ]
        else:
            facts = data.get("facts", [])
            if not facts:
                lines += [
                    "",
                    "No validated facts returned. This is not evidence of safety.",
                ]
            for fact in facts:
                lines += [
                    "",
                    str(fact.get("symbol", "Market")),
                    self.service._safe(fact.get("fact", ""))[:1200],
                ]
                url = str(fact.get("source_url", ""))
                parsed = urlsplit(url)
                if (
                    parsed.scheme == "https"
                    and parsed.hostname
                    and not parsed.username
                    and not parsed.password
                ):
                    lines.append("Source: " + url)
                lines += [
                    f"Published: {fact.get('published') or 'unknown'}",
                    f"As of: {fact.get('asof', 'unknown')}",
                    "Uncertainty: "
                    + self.service._safe(fact.get("uncertainty", "unavailable"))[:400],
                ]
        if data.get("summary"):
            lines += ["", self.service._safe(data["summary"])[:2000]]
        jobs = Counter(
            r.get("kind", "unknown") for r in self.state.get("reviews", {}).values()
        )
        lines += [
            "",
            f"AI jobs stored: {len(self.state.get('reviews', {}))}",
            f"Candidate review jobs: {jobs['review']} · Research jobs: {jobs['research']}",
            f"Reference documents loaded: {len(self.state.get('references', {}))}",
            "Model output is not a measured win probability.",
        ]
        return "\n".join(lines)

    def system(self, outbox):
        lines = [
            "⚙️ SYSTEM HEALTH",
            f"Updated {datetime.fromtimestamp(self.now, timezone.utc):%d %b %H:%M UTC}",
            f"Mode: {self.config.mode.upper()}",
            f"Database: {'Postgres' if self.config.database_url else 'SQLite (local)'} · snapshot read succeeded",
            f"Redis: {self.words(self.health.get('redis', 'not observed'))} · disposable cache",
            "",
            "LOOP FRESHNESS · data validity is shown separately",
        ]
        components = (
            ("lease", "Worker ownership", 90),
            ("scan", "Scanner", self.scan_limit),
            ("instruments", "Exchange instrument rules", 10800),
            ("universe", "Universe checks", 900),
            ("context", "Market context", 2700),
            ("risk", "Risk circuits", 120),
            ("simulation", "Shadow positions", 120),
            ("shadow_observer", "Capacity observer", 120),
            ("funding", "Funding history", 2700),
            ("telegram", "Telegram commands", 120),
            ("outbox", "Telegram delivery", 120),
        )
        errors = self.errors()
        for key, name, limit in components:
            stamp = self.health.get(key + "_at")
            state = (
                "⚠️ Error"
                if key in errors
                else (
                    "✅ Current"
                    if self.fresh(stamp, limit)
                    else "⚠️ Stale / not observed"
                )
            )
            lines.append(f"{name}: {state} · {age(stamp, self.now)}")
        lines.append(
            "Private exchange: not used in shadow mode"
            if self.config.mode == "shadow"
            else f"Exchange reconciliation: {age(self.health.get('reconcile_at'), self.now)}"
        )
        lines += ["", "DATA VALIDITY"]
        lines += [
            f"{'✅' if value == 'Available' else '⚠️'} {name}: {value}"
            for name, value in self.data_health().items()
        ]
        context = self.state.get("context", {})
        if context.get("event_blackout") is True:
            lines.append(
                "Context restriction: event blackout active or required while data is unavailable."
            )
        if self.data_health()["Market context data"] != "Available" and context.get(
            "reason"
        ):
            lines.append("Context detail: " + self.words(context["reason"])[:350])
        lines += [
            "",
            f"Telegram pending: {outbox['pending']} · oldest {int(outbox['oldest_age'])}s",
            f"AI configured: {'yes' if self.config.openai_key and self.config.openai_model else 'no'}",
            f"AI research result: {self.research_status()}",
            f"Reference documents: {len(self.state.get('references', {}))}",
            "",
            f"Active component errors: {len(errors)}",
        ]
        for key, error in sorted(errors.items()):
            lines.append(f"• {self.words(key)}: {self.words(error)}")
        if self.health.get("error"):
            lines.append(
                "Last recorded issue (historical): "
                + self.words(self.service._safe(self.health["error"]))[:200]
            )
            if not errors:
                lines.append(
                    "No active error flags. Current timestamps above determine freshness."
                )
        issue = self.health.get("last_issue", {})
        if isinstance(issue, dict) and issue:
            lines.append(
                "Last issue occurred: " + age(issue.get("occurred_at"), self.now)
            )
            if issue.get("recovered_at"):
                lines.append(
                    "Recovery observed: " + age(issue.get("recovered_at"), self.now)
                )
        research = self.state.get("research", {})
        if research.get("status") == "UNAVAILABLE":
            from .notifications import research_diagnosis

            lines += [
                "",
                "⚠️ Advisory research unavailable: "
                + (research_diagnosis(research) or self.words(research.get("reason"))),
                "Research loop completion does not mean a validated AI result was produced.",
            ]
        lines += [
            "",
            "Times and accounting boundaries use UTC. Refresh reads saved state.",
        ]
        return "\n".join(lines)


GUIDE = """📖 APEX · READING THE DASHBOARD

🧪 Shadow mode
Prices drive simulated trades. No exchange orders are placed. Each portfolio
starts with hypothetical capital; this is not your Bybit balance.

🌊 A setup is not a trade
Waiting: entry conditions are unmet. The decision explains whether this is a
missing closed 4H confirmation, symbol eligibility, conflicting signals or data.
Ready: the technical trigger formed; risk checks and the AI portfolio's approvals
still apply. A planned entry becomes open only after a simulated/confirmed fill.
Invalidated/expired: the candidate no longer qualifies. /why SYMBOL gives details.

📊 Results
Rules baseline and AI-approved shadow are separate portfolios. The Observer tracks
extra capacity-rejected opportunities and is not a fundable portfolio.
WR = wins / valid closed trades, including breakeven in the denominator.
— means no measured result; zero means a measured value of zero.
Net shadow results include modeled costs. Funding-pending results are provisional.
R = original entry-to-stop price risk; net R:R compares reward to risk after costs.
Target 1 and Target 2 are planned partial exits, not promised returns.

🛡 Controls
Base risk is reduced by setup/context and portfolio rules. Pause blocks new
decisions; existing positions and previously pending entries continue management.
Profile/risk/resume changes require a fresh preview and confirmation.
Telegram cannot enable live execution or bypass the release lock.

🧠 AI and context
Configured does not mean working. Unavailable research is no conclusion.
Research is advisory; deterministic rules retain authority. The calendar covers
scheduled US CPI, jobs and FOMC events, not every possible market event.
The performance comparison is observational, not proof of AI-caused improvement.
"""
