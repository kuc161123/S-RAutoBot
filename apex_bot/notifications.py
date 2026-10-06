"""Pure, plain-text notification formatting; never changes a trading decision."""

from __future__ import annotations

import math
import re


RESEARCH_ERROR_CODES = frozenset(
    {
        "billing_not_active",
        "insufficient_quota",
        "rate_limit_exceeded",
        "invalid_api_key",
        "model_not_found",
    }
)


_SETUPS = {
    "1": "Elliott impulse pullback",
    "1S": "Elliott impulse pullback",
    "2": "Elliott wave 3 pullback",
    "2S": "Elliott wave 3 pullback",
    "2X": "Elliott extended wave 3 pullback",
    "2XS": "Elliott extended wave 3 pullback",
    "5S": "Elliott corrective reversal",
    "5L": "Elliott corrective reversal",
}

_REASONS = {
    "AWAIT_CLOSED_4H_TRIGGER": "Waiting for confirmation from a closed 4-hour candle.",
    "SWING_BREAK": "A closed 4-hour candle confirmed a break of the corrective swing.",
    "BREAKOUT_RETEST": "A closed 4-hour candle confirmed the breakout retest.",
    "DATA_STALE_OR_MISSING": "Required candle data is missing or stale.",
    "TIER_NOT_ELIGIBLE": "This symbol is not eligible for this setup under current rules.",
    "CONFLICTING_DIRECTIONS": "Long and short candidates conflict; direction is unresolved.",
    "HARD_STOP_BREACHED": "Price breached the planned hard stop.",
    "DAILY_INVALIDATION": "A daily candle closed beyond the structural invalidation level.",
    "STRUCTURE_INVALIDATED": "Price breached the structural invalidation level.",
    "TARGET_ALREADY_PASSED": (
        "Price already reached the first target; this setup is no longer eligible."
    ),
    "ENTRY_OUTSIDE_ZONE": "Price closed outside the permitted entry zone.",
    "TRIGGER_OUTSIDE_ZONE": "The confirmation occurred outside the permitted entry zone.",
    "CHASING": "Price moved too far beyond the planned entry; chasing is blocked.",
    "EXPIRED": "The candidate's entry window expired.",
    "RETEST_EXPIRED": "The breakout retest window expired.",
    "FAILED_FIRST_RETEST": "The first breakout retest failed to hold.",
    "UNIVERSE_RETIRED": "The symbol left the eligible universe; new entry is retired.",
    "TARGET_BEFORE_ENTRY": "Price reached the target before a simulated entry filled.",
    "UNFILLED": "The simulated entry window expired without a fill.",
    "STOP": "The simulated stop was reached.",
    "TP1": "The first simulated target was reached.",
    "TP2": "The final simulated target was reached.",
}


def sanitize_text(text, *, secrets=()) -> str:
    """Redact whole values before callers truncate, paginate or change case."""
    text = str(text)
    for secret in sorted(
        (s for s in secrets if isinstance(s, str) and s), key=len, reverse=True
    ):
        text = text.replace(secret, "[redacted]")
    text = re.sub(r"telegram:-?[0-9]+:[0-9]+", "[private actor]", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[redacted]", text)
    return re.sub(r"[0-9]+:[A-Za-z0-9_-]{8,}", "[redacted]", text)


def setup_label(setup) -> str:
    return _SETUPS.get(str(setup), "Elliott setup")


def describe_reason(reason, *, secrets=()) -> str:
    """Translate known codes, preserving already-readable decision explanations."""
    if not isinstance(reason, str) or not reason.strip():
        return "Reason unavailable."
    reason = sanitize_text(reason, secrets=secrets)
    if reason in _REASONS:
        return _REASONS[reason]
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", reason):
        return reason.replace("_", " ").capitalize()
    return reason


def side_label(side) -> str:
    return {"Buy": "LONG", "Sell": "SHORT", "LONG": "LONG", "SHORT": "SHORT"}.get(
        side, "Direction unavailable"
    )


def _number(value) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "unavailable"
    try:
        return f"{value:.10g}" if math.isfinite(value) else "unavailable"
    except OverflowError:
        return "unavailable"


def opportunity_alert(op, *, secrets=()) -> str:
    state = op.state
    status = {
        "WAIT": "WAIT · Setup spotted; confirmation pending",
        "READY": "READY · Candidate ready, not executed",
        "INVALID": "Setup invalidated",
        "EXPIRED": "Candidate expired",
    }.get(state, "Candidate state unavailable")
    if state == "READY":
        meaning = "Risk and AI checks still apply. This alert is not a fill."
        next_step = (
            "Risk assessment and AI review; any entry needs separate fill confirmation."
        )
    elif state == "WAIT":
        meaning = "No trade opened by this alert."
        next_step = {
            "DATA_STALE_OR_MISSING": (
                "Wait for complete, fresh daily and 4-hour candle data, then reassess."
            ),
            "TIER_NOT_ELIGIBLE": "Wait for an eligible setup under the existing symbol rules.",
            "CONFLICTING_DIRECTIONS": "Wait for the conflicting direction signals to resolve.",
        }.get(
            op.reason,
            "Wait for a qualifying closed 4-hour trigger within the entry zone.",
        )
        if (
            op.reason == "AWAIT_CLOSED_4H_TRIGGER"
            and _number(op.confirmation) != "unavailable"
            and op.confirmation > 0
        ):
            next_step += (
                f" Watch confirmation level {_number(op.confirmation)}; "
                "the level alone is insufficient."
            )
    else:
        meaning = "This setup alert does not confirm a trade entry or exit."
        next_step = "Wait for a new valid setup."
    levels_label = (
        "Previous planned entry" if state in {"INVALID", "EXPIRED"} else "Planned entry"
    )
    lines = [
        f"🔎 {op.symbol} · {side_label(op.side)}",
        setup_label(op.setup),
        status,
        meaning,
        describe_reason(op.reason, secrets=secrets),
        f"{levels_label}: {_number(op.entry)}",
        f"Planned stop: {_number(op.stop)}",
        f"Targets: 1 → {_number(op.target1)} · 2 → {_number(op.target2)}",
        f"Structural invalidation: {_number(op.invalidation)}",
        f"Next: {next_step}",
        f"Details: /why {op.symbol} · /waves {op.symbol}",
    ]
    return sanitize_text("\n".join(lines), secrets=secrets)


def review_alert(op, review, *, secrets=()) -> str:
    verdict = review.get("verdict")
    status = {
        "APPROVE": "AI review approved; risk and entry checks still apply.",
        "WAIT": "AI review waiting; no AI-approved entry from this review.",
        "REJECT": "AI review rejected this candidate.",
    }.get(verdict, "AI review unavailable; no approval established.")
    return sanitize_text(
        f"🧠 {op.symbol} · {side_label(op.side)} · {setup_label(op.setup)}\n"
        f"{status}\n{describe_reason(review.get('reason'), secrets=secrets)}\n"
        f"This review is not a fill. Details: /why {op.symbol}",
        secrets=secrets,
    )


def shadow_alert(trade, *, secrets=()) -> str:
    arm = {
        "baseline_shadow": "Rules baseline",
        "ai_shadow": "AI-approved shadow",
    }.get(trade.get("arm"), "Shadow simulation")
    state = trade.get("status")
    status = {
        "PENDING": "Simulated entry pending · no trade opened yet",
        "OPEN": "Simulated entry filled · trade open",
        "CLOSED": "Simulated trade closed",
        "EXPIRED": "Simulated entry expired · no fill",
    }.get(state, "Simulation state unavailable")
    lines = [
        f"👻 {trade.get('symbol', 'Symbol unavailable')} · {side_label(trade.get('side'))}",
        arm,
        status,
        "No exchange order or exchange fill is confirmed by this simulation.",
    ]
    if state == "PENDING":
        lines.extend(
            [
                f"Planned entry: {_number(trade.get('limit'))} · stop: {_number(trade.get('stop'))}",
                f"Targets: 1 → {_number(trade.get('target1'))} · 2 → {_number(trade.get('target2'))}",
                "Next: a subsequent eligible candle must evidence the simulated limit fill.",
            ]
        )
    if trade.get("exit_reason"):
        lines.append(describe_reason(trade["exit_reason"], secrets=secrets))
    if state in {"OPEN", "CLOSED"}:
        amount = _number(trade.get("net_pnl_before_funding"))
        amount = amount if amount == "unavailable" else "$" + amount
        lines.append(f"Realized net estimate before funding: {amount}")
        if trade.get("data_error") or trade.get("data_gap"):
            lines.append("Incomplete replay data; the result is not final.")
    return sanitize_text("\n".join(lines), secrets=secrets)


def research_failure_details(attempt) -> dict:
    """Allowlisted diagnostics only; never extract a provider's free-form error."""
    status = attempt.get("http_status")
    provider_code = attempt.get("error_code")
    code = attempt.get("error")
    if not isinstance(code, str) or code not in {
        "HTTP_ERROR",
        "TIMEOUT_OUTCOME_UNKNOWN",
        "INVALID_RESPONSE_OR_TRANSPORT_ERROR",
    }:
        code = "VALIDATION_FAILED" if attempt.get("valid") is False else None
    if isinstance(provider_code, str) and provider_code in RESEARCH_ERROR_CODES:
        code = provider_code
    return {
        "http_status": status if type(status) is int and 100 <= status <= 599 else None,
        "error_code": code,
    }


def research_diagnosis(result) -> str:
    """HTTP 429 cannot distinguish a rate limit from a quota limit."""
    status = result.get("http_status")
    code = result.get("error_code")
    verified = {
        "billing_not_active": "API billing is not active",
        "insufficient_quota": "API quota is insufficient",
        "rate_limit_exceeded": "API rate limit exceeded",
        "invalid_api_key": "API credentials were rejected",
        "model_not_found": "Requested API model is unavailable",
    }.get(code if isinstance(code, str) else None)
    if verified:
        suffix = (
            f" (HTTP {status})" if type(status) is int and 400 <= status <= 599 else ""
        )
        return verified + suffix + "."
    if type(status) is int:
        known = {
            401: "API credentials were rejected",
            403: "API access was denied",
            404: "API model or endpoint path is unavailable",
            429: "API rate or quota limit",
        }
        if status in known:
            return f"{known[status]} (HTTP {status})."
        if 500 <= status <= 599:
            return f"API provider failure (HTTP {status})."
        if 400 <= status <= 499:
            return f"API request failed (HTTP {status})."
    return {
        "TIMEOUT_OUTCOME_UNKNOWN": "API request timed out; its outcome is unknown.",
        "INVALID_RESPONSE_OR_TRANSPORT_ERROR": (
            "The API response or connection was unavailable or invalid."
        ),
        "VALIDATION_FAILED": "The research response failed source or content validation.",
        "HTTP_ERROR": "The API request failed; a usable HTTP status is unavailable.",
    }.get(code if isinstance(code, str) else None, "")


def research_failure(attempt) -> str:
    """Readable safe diagnosis for a legacy completed attempt, without its archive."""
    return research_diagnosis(research_failure_details(attempt)) or (
        "Research unavailable; the failure cause was not recorded."
    )


def research_alert(result, *, secrets=()) -> str:
    """Only COMPLETE results may present facts as validated; preserve source URLs."""
    lines = [
        "🧠 Apex daily research · advisory only",
        f"As of: {result.get('asof') or 'unavailable'}",
    ]
    if result.get("status") != "COMPLETE":
        lines.append("UNAVAILABLE")
        diagnosis = research_diagnosis(result)
        if diagnosis:
            lines.append(f"Diagnosis: {diagnosis}")
        lines.extend(
            [
                f"Reason: {describe_reason(result.get('reason'), secrets=secrets)}",
                "No validated research result is available. "
                "This is not evidence of no news or no risk.",
            ]
        )
    else:
        facts = result.get("facts", [])
        lines.extend(["COMPLETE", f"Validated facts: {len(facts)}"])
        if not facts:
            lines.append("The validated response returned no qualifying facts.")
        for fact in facts:
            lines.append(
                f"{fact['symbol']}: {fact['fact']}\n{fact['source_url']}\n"
                f"Published: {fact['published'] or 'unknown'}; as of: {fact['asof']}\n"
                f"Uncertainty: {fact['uncertainty']}"
            )
    lines.append("Research does not establish safety or approve a trade.")
    if result.get("uncertainty"):
        lines.append(f"Coverage uncertainty: {result['uncertainty']}")
    lines.append("Details: /research")
    # The durable Telegram transport splits unexpectedly long content losslessly.
    return sanitize_text("\n".join(lines), secrets=secrets)
