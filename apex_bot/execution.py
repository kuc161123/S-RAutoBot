"""Restart-safe, evidence-driven execution; importing this module makes no calls.

Only explicit live/testnet mode permits venue access. The runtime must use one
manager and serialize account decisions; this manager also serializes submit,
reconcile and structural updates. Store.update must enforce the leader lease.
Persisted intents are consumed before every order POST, including reduce orders.
An absent/ambiguous response NEVER authorizes a replay. Terminal partial IOC
reductions may use a new, bounded residual intent after actual fills reconcile.

REST has no accounting completeness watermark. Completion here requires every
owned trade in the transaction ledger, closed-PnL quantities, and a settlement
record at every crossed funding boundary for the recorded instrument interval.
Missing/ambiguous evidence stays pending. Anonymous full-position TP/SL orders
require an order-ID lookup plus sole-owned inventory from a verified flat start;
same-symbol fills alone never establish ownership. A >7-day execution-history
gap requires operator reconciliation.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import math
import time

from .engine import DAY, H4, _closed, confirmed_zigzag
from .accounting import loss_metrics
from .evidence import candidate_fingerprint, context_fingerprint
from .risk import HARD_CAPS, PROFILES
from .universe import entry_symbols
from .models import Instrument

TERMINAL = {"CANCELLED", "CLOSED", "REJECTED"}
VENUE_TERMINAL = {
    "Filled",
    "Cancelled",
    "PartiallyFilledCanceled",
    "Rejected",
    "Deactivated",
}
MAX_REDUCTIONS = 3
ACCOUNT_MAX_AGE = 30
HISTORY_WINDOW = 7 * DAY - 60


def client_id(opportunity_id: str, action: str = "entry") -> str:
    return (
        "apx-" + hashlib.sha256(f"{opportunity_id}:{action}".encode()).hexdigest()[:28]
    )


def number(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _d(value):
    return Decimal(str(value))


def _near(left, right, step=1):
    return abs(left - right) <= max(1e-9, abs(step) * 1e-6)


def _round(value, step, up=False):
    return float(
        (_d(value) / _d(step)).to_integral_value(
            rounding=ROUND_CEILING if up else ROUND_FLOOR
        )
        * _d(step)
    )


def _sign(order):
    return 1 if order["side"] == "Buy" else -1


def _opposite(order):
    return "Sell" if order["side"] == "Buy" else "Buy"


def _terminal(row):
    return bool(
        row
        and row.get("orderStatus") in VENUE_TERMINAL
        and number(row.get("leavesQty")) == 0
    )


def _matches(row, order, link, side, order_id=None):
    return (
        isinstance(row, dict)
        and row.get("symbol") == order["symbol"]
        and row.get("side") == side
        and row.get("orderLinkId") == link
        and bool(row.get("orderId"))
        and (not order_id or row["orderId"] == order_id)
        and number(row.get("positionIdx")) == 0
    )


def _exit_kind(row):
    kind = row.get("stopOrderType")
    created = row.get("createType")
    if kind in {"StopLoss", "PartialStopLoss", "TrailingStop"}:
        return "SL"
    if kind in {"TakeProfit", "PartialTakeProfit"}:
        return "TP2"
    if created in {
        "CreateByStopLoss",
        "CreateByPartialStopLoss",
        "CreateByTrailingStop",
    }:
        return "SL"
    if created in {
        "CreateByTakeProfit",
        "CreateByPartialTakeProfit",
        "CreateByTrailingProfit",
    }:
        return "TP2"
    if created in {"CreateByUser", "CreateByClosing"}:
        return "MANUAL"
    return "UNKNOWN"


class _EvidenceError(ValueError):
    """Fixed, non-sensitive internal reconciliation reason."""


class ExecutionManager:
    def __init__(self, client, store, mode="shadow"):
        self.client, self.store, self.mode = client, store, mode
        self._lock = asyncio.Lock()
        self.instruments = {}
        self.management_instruments = {}

    @staticmethod
    def _saved_instrument(order):
        try:
            inst = Instrument(**order["instrument"])
            if inst.symbol != order["symbol"]:
                raise ValueError("Mismatched saved instrument")
            for key in (
                "qty_step",
                "min_qty",
                "max_qty",
                "tick_size",
                "max_leverage",
                "funding_interval_minutes",
            ):
                value = getattr(inst, key)
                if number(value) is None or isinstance(value, bool) or value <= 0:
                    raise ValueError("Invalid saved instrument")
            if (
                number(inst.min_notional) is None
                or inst.min_notional < 0
                or inst.min_qty > inst.max_qty
                or inst.qty_step > inst.max_qty
            ):
                raise ValueError("Invalid saved instrument bounds")
            return inst
        except (TypeError, ValueError, KeyError):
            raise _EvidenceError(
                "saved management instrument metadata unavailable"
            ) from None

    async def _get(self, link):
        return (await self.store.read())["orders"][link]

    async def _patch(self, link, **fields):
        def change(tx):
            current = tx.state["orders"][link]
            old_status = current["status"]
            current.update(
                copy.deepcopy(fields)
            )  # Never replace the children/action maps.
            if current["status"] != old_status:
                tx.event(
                    link + ":state:" + current["status"],
                    "order_state",
                    current,
                    f"{self.mode.upper()} · {current['symbol']} {current['side']} · {current['status']}",
                )

        await self.store.update(change)

    def _reservation_allowed(self, state, op, sizing, settings_version):
        now = time.time()
        if state.get("universe_mode") == "dynamic" and op.symbol not in entry_symbols(
            state.get("universe", {}), now, required_policy=state.get("universe_policy")
        ):
            return False
        settings, account = state["settings"], state.get("account", {})
        rec = state.get("opportunities", {}).get(op.id, {})
        current, supplied = copy.deepcopy(rec.get("opportunity", {})), op.to_dict()
        for candidate in (current, supplied):
            candidate.get("evidence", {}).pop("as_of", None)
        approval = rec.get("ai", {})
        evidence_hash = approval.get("evidence_hash")
        if (
            current != supplied
            or current.get("state") != "READY"
            or approval.get("verdict") != "APPROVE"
            or not isinstance(evidence_hash, str)
            or len(evidence_hash) != 64
            or any(c not in "0123456789abcdef" for c in evidence_hash)
            or approval.get("candidate_fingerprint") != candidate_fingerprint(current)
            or approval.get("context_fingerprint")
            != context_fingerprint(state, op.symbol, now)
        ):
            return False
        if (
            settings.get("paused") is not False
            or state["settings_version"] != settings_version
            or op.state != "READY"
            or op.side not in {"Buy", "Sell"}
            or not number(op.expires_at)
            or now >= op.expires_at
            or sizing.get("allowed") is not True
            or settings.get("profile") not in PROFILES
            or account.get("blockers") != []
            or not isinstance(account.get("positions"), list)
            or not isinstance(account.get("open_orders"), list)
        ):
            return False
        stamp, equity = number(account.get("as_of")), number(account.get("equity"))
        if (
            stamp is None
            or not 0 <= now - stamp <= ACCOUNT_MAX_AGE
            or equity is None
            or equity <= 0
        ):
            return False
        fields = (
            "qty",
            "qty_step",
            "entry",
            "stop",
            "target1",
            "target2",
            "risk_cash",
            "notional",
            "risk_pct",
        )
        if any(number(sizing.get(k)) is None or number(sizing[k]) <= 0 for k in fields):
            return False
        q, step, entry, stop = (
            number(sizing[k]) for k in ("qty", "qty_step", "entry", "stop")
        )
        sign = 1 if op.side == "Buy" else -1
        inst = self.instruments.get(op.symbol)
        if (
            inst is None
            or not _near(step, inst.qty_step)
            or not inst.min_qty <= q <= inst.max_qty
        ):
            return False
        for name in ("entry", "stop", "target1", "target2"):
            expected = _round(
                getattr(op, name),
                inst.tick_size,
                up=(sign > 0 if name == "entry" else sign < 0),
            )
            if not _near(sizing[name], expected, inst.tick_size):
                return False
        if sizing["notional"] < inst.min_notional:
            return False
        if (
            not _near(q, _round(q, step), step)
            or sign * (entry - stop) <= 0
            or sign * (sizing["target1"] - entry) <= 0
            or sign * (sizing["target2"] - sizing["target1"]) <= 0
            or not _near(sizing["notional"], q * entry)
            or sizing["risk_cash"] + 1e-8 < q * abs(entry - stop)
        ):
            return False
        profile = PROFILES[settings["profile"]]
        pct = (
            profile["risk_pct"]
            if settings.get("risk_pct") is None
            else number(settings["risk_pct"])
        )
        if pct is None or not 0.05 <= pct <= 1:
            return False
        pct *= 0.5 if op.tier == 2 else 1
        multiplier = number(op.evidence.get("risk_multiplier", 1))
        if multiplier is None or not 0 < multiplier <= 1:
            return False
        pct *= min(multiplier, 0.5 if op.setup in {"2X", "2XS"} else 1)
        context = state.get("context", {})
        if (
            context.get("data_complete") is not True
            or number(context.get("as_of")) is None
            or not 0 <= now - number(context["as_of"]) <= 21600
        ):
            return False
        if context:
            factor = number(
                context.get("long_multiplier" if sign > 0 else "short_multiplier")
            )
            if (
                factor is None
                or not 0 < factor <= 1
                or context.get("live_blocked") is True
                or number(context.get("expires_at"), 0) <= now
                or context.get("event_blackout") is True
            ):
                return False
            pct *= factor
        active = [o for o in state["orders"].values() if o["status"] not in TERMINAL]
        if any(o["symbol"] == op.symbol or o.get("mode") != self.mode for o in active):
            return False
        if any(number(p.get("size")) is None for p in account["positions"]):
            return False
        for p in account["positions"]:
            if number(p["size"]) > 0 and number(p.get("unrealisedPnl")) is None:
                return False
            if number(p["size"]) > 0 and (
                p.get("symbol") == op.symbol
                or not any(
                    o["symbol"] == p.get("symbol")
                    and o["side"] == p.get("side")
                    and o.get("ownership_verified")
                    for o in active
                )
            ):
                return False
        if any(o.get("symbol") == op.symbol for o in account["open_orders"]):
            return False
        try:
            losses = loss_metrics(state, equity, None, now, self.mode)
        except (ValueError, TypeError, KeyError, OverflowError):
            return False
        circuit = state.get("risk_circuits", {}).get(self.mode)
        if circuit is not None:
            stamp = number(circuit.get("as_of"))
            if stamp is None or not 0 <= now - stamp < 60:
                return False
            for metric in losses:
                observed = number(circuit.get(metric))
                if observed is None or observed < 0:
                    return False
                losses[metric] = max(losses[metric], observed)
        if (
            losses["daily_loss_pct"] >= 2
            or losses["weekly_loss_pct"] >= 4
            or losses["drawdown_pct"] >= 12
        ):
            return False
        if losses["drawdown_pct"] >= 8:
            pct *= 0.5
        if (
            sizing["risk_pct"] > pct + 1e-9
            or sizing["risk_cash"] > equity * pct / 100 + 1e-8
        ):
            return False
        if any(
            number(o.get("risk_cash")) is None
            or number(o.get("notional")) is None
            or o["risk_cash"] < 0
            or o["notional"] <= 0
            for o in active
        ):
            return False
        caps = {k: min(v, profile.get(k, v)) for k, v in HARD_CAPS.items()}
        bucket = [o for o in active if o.get("bucket", "").lower() == op.bucket.lower()]
        return (
            len(active) < caps["max_positions"]
            and len(bucket) < caps["max_per_bucket"]
            and sum(o["risk_cash"] for o in active) + sizing["risk_cash"]
            <= equity * caps["heat_pct"] / 100
            and sum(o["risk_cash"] for o in bucket) + sizing["risk_cash"]
            <= equity * caps["bucket_pct"] / 100
            and sizing["notional"] <= equity * caps["single_notional_pct"] / 100
            and sum(o["notional"] for o in active) + sizing["notional"]
            <= equity * caps["gross_notional_pct"] / 100
        )

    async def submit(self, op, sizing, settings_version):
        from .release import LIVE_APPROVED, REASON

        if self.mode == "live" and not LIVE_APPROVED:
            raise RuntimeError(REASON)
        if self.mode not in {"testnet", "live"}:
            raise RuntimeError("Exchange execution is unavailable in shadow mode")
        if op.evidence.get("entry_style") in (
            "resting_limit",
            "monitored_zone",
        ) or op.evidence.get("trigger_kind") in (
            "ZONE_LIMIT",
            "ZONE_ARRIVAL",
            "ZONE_WATCH",
        ):
            raise RuntimeError(
                "Zone entries are research-only; exchange submission is disabled"
            )
        async with self._lock:
            link = client_id(op.id)

            def reserve(tx):
                try:
                    allowed = self._reservation_allowed(
                        tx.state, op, sizing, settings_version
                    )
                except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
                    allowed = False
                if link in tx.state["orders"] or not allowed:
                    return False
                record = {
                    "id": link,
                    "symbol": op.symbol,
                    "side": op.side,
                    "opportunity": op.to_dict(),
                    "sizing": copy.deepcopy(sizing),
                    "instrument": asdict(self.instruments[op.symbol]),
                    "status": "INTENT",
                    "created_at": time.time(),
                    "settings_version": settings_version,
                    "mode": self.mode,
                    "children": {},
                    "risk_cash": sizing["risk_cash"],
                    "bucket": op.bucket,
                    "notional": sizing["notional"],
                    "desired_stop": sizing["stop"],
                    "tp1_done": False,
                    "funding_complete": False,
                    "flat_verified_at": tx.state["account"]["as_of"],
                    "executions": {},
                    "attached": {},
                    "ownership_verified": False,
                    "funding_interval_minutes": self.instruments[
                        op.symbol
                    ].funding_interval_minutes,
                }
                tx.state["orders"][link] = record
                tx.event(
                    link + ":intent",
                    "order_intent",
                    record,
                    f"{self.mode.upper()} intent · {op.symbol} {op.side}",
                )
                return True

            if not await self.store.update(reserve):
                return False
            await self.store.assert_leader()
            try:
                ack = await self.client.submit_limit(
                    op.symbol,
                    op.side,
                    sizing["qty"],
                    sizing["entry"],
                    sizing["stop"],
                    sizing["target2"],
                    link,
                )
                if not ack.get("orderId") or ack.get("orderLinkId") != link:
                    raise _EvidenceError("ambiguous acknowledgment")
                await self._patch(
                    link,
                    status="ACKNOWLEDGED",
                    ack={"orderId": ack["orderId"], "orderLinkId": link},
                )
            except Exception:
                await self._patch(link, status="UNKNOWN")
            return True

    async def _entry(self, order):
        row = await self.client.order(order["symbol"], order["id"])
        if row is None:
            row = (
                order.get("exchange_order")
                if _terminal(order.get("exchange_order"))
                else None
            )
        if row is None:
            raise _EvidenceError("entry outcome unknown; no resubmission")
        if not _matches(
            row, order, order["id"], order["side"], order.get("ack", {}).get("orderId")
        ):
            raise _EvidenceError("entry identity mismatch")
        qty = number(row.get("cumExecQty"))
        if (
            qty is None
            or qty < number(order.get("filled_qty"), 0)
            or qty > order["sizing"]["qty"] + 1e-9
        ):
            raise _EvidenceError("entry quantity inconsistent")
        if (
            row.get("reduceOnly") is not False
            or not _near(
                number(row.get("qty"), -1),
                order["sizing"]["qty"],
                order["sizing"]["qty_step"],
            )
            or not _near(number(row.get("price"), -1), order["sizing"]["entry"])
            or row.get("orderStatus") == "Filled"
            and not _near(qty, order["sizing"]["qty"], order["sizing"]["qty_step"])
        ):
            raise _EvidenceError("entry order differs from persisted intent")
        await self._patch(order["id"], exchange_order=row)
        return row

    async def _cancel_entry(self, order):
        def reserve(tx):
            current = tx.state["orders"][order["id"]]
            if current.get("cancel_intent"):
                return False
            current["cancel_intent"] = {"created_at": time.time(), "status": "UNKNOWN"}
            tx.event(order["id"] + ":cancel", "cancel_intent", {"parent": order["id"]})
            return True

        if await self.store.update(reserve):
            await self.store.assert_leader()
            try:
                await self.client.cancel(order["symbol"], order["id"])
            except Exception:
                pass
        row = await self._entry(await self._get(order["id"]))
        if _terminal(row):
            await self._patch(
                order["id"],
                cancel_intent={"status": "VERIFIED", "verified_at": time.time()},
            )
        return row

    async def _children(self, order):
        for action, child in order.get("children", {}).items():
            if _terminal(child.get("exchange_order")):
                continue
            row = await self.client.order(order["symbol"], child["link"])
            if row is None:
                continue
            if not _matches(
                row, order, child["link"], _opposite(order), child.get("orderId")
            ):
                raise _EvidenceError("child identity mismatch")
            if row.get("reduceOnly") is not True:
                raise _EvidenceError("child is not reduce-only")
            if (
                not _near(
                    number(row.get("qty"), -1),
                    child["qty"],
                    order["sizing"]["qty_step"],
                )
                or number(row.get("cumExecQty"), -1) < 0
                or number(row["cumExecQty"]) > child["qty"] + 1e-9
            ):
                raise _EvidenceError("child quantity differs from intent")

            def save(tx):
                c = tx.state["orders"][order["id"]]["children"][action]
                c.setdefault("action", action.split(":")[0])
                c.update(
                    exchange_order=row,
                    orderId=row["orderId"],
                    status=row["orderStatus"],
                )

            await self.store.update(save)

    async def _reduce(self, order, action, qty):
        """One intent per attempt, at most 3 confirmed residual attempts/action."""

        def reserve(tx):
            current = tx.state["orders"][order["id"]]
            children = current.setdefault("children", {})
            # Any outstanding reduction blocks another; terminal fills must also
            # be visible in our execution ledger before sizing a residual.
            for child in children.values():
                row = child.get("exchange_order")
                if not _terminal(row):
                    return None
                seen = sum(
                    number(f["execQty"], 0)
                    for f in current.get("exit_fills", [])
                    if f["orderId"] == row["orderId"]
                )
                if not _near(
                    seen,
                    number(row.get("cumExecQty"), -1),
                    current["sizing"]["qty_step"],
                ):
                    return None
            count = sum(c["action"] == action for c in children.values())
            if (
                count >= MAX_REDUCTIONS
                or qty <= 0
                or not current.get("ownership_verified")
            ):
                return None
            name = action if count == 0 else action + ":" + str(count + 1)
            link = client_id(order["id"], name)
            children[name] = {
                "link": link,
                "action": action,
                "status": "UNKNOWN",
                "qty": qty,
                "created_at": time.time(),
            }
            tx.event(
                link + ":intent",
                "reduce_intent",
                {"parent": order["id"], "action": action, "qty": qty},
            )
            return name, link

        reserved = await self.store.update(reserve)
        if reserved is None:
            return False
        name, link = reserved
        await self.store.assert_leader()
        try:
            ack = await self.client.reduce_market(
                order["symbol"], _opposite(order), qty, link
            )
            if ack.get("orderId") and ack.get("orderLinkId") == link:

                def save(tx):
                    tx.state["orders"][order["id"]]["children"][name].update(
                        orderId=ack["orderId"], status="ACKNOWLEDGED"
                    )

                await self.store.update(save)
        except Exception:
            pass
        return True

    async def _collect(self, order, open_orders):
        checked = time.time()
        start = order.get("execution_checked_at", order["created_at"])
        if checked - start > HISTORY_WINDOW or order.get("history_gap"):
            await self._patch(order["id"], history_gap=True)
            raise _EvidenceError(
                "execution history gap; manual reconciliation required"
            )
        rows = await self.client.executions(
            order["symbol"], int(max(order["created_at"], start - 60) * 1000)
        )
        executions = copy.deepcopy(order.get("executions", {}))
        for row in rows:
            key = row.get("execId")
            if not key or row.get("symbol") != order["symbol"]:
                raise _EvidenceError("execution identity missing")
            if key in executions and executions[key] != row:
                raise _EvidenceError("execution evidence changed")
            executions[key] = row
        attached = copy.deepcopy(order.get("attached", {}))
        for row in open_orders:
            if (
                row.get("symbol") == order["symbol"]
                and row.get("parentOrderLinkId") == order["id"]
                and row.get("side") == _opposite(order)
                and number(row.get("positionIdx")) == 0
                and row.get("orderId")
            ):
                attached[row["orderId"]] = row
        await self._patch(
            order["id"],
            executions=executions,
            attached=attached,
            execution_checked_at=checked,
        )
        return await self._get(order["id"])

    async def _resolve_protective(self, order, open_orders):
        """Resolve anonymous full-position protection using documented order IDs.

        These rows are only committed after _ledger and _position establish that
        ALL entry/exit fills and the remaining position belong to this intent.
        No parentOrderLinkId is required; a contradictory parent/link is rejected.
        """
        known = {order["exchange_order"]["orderId"]} | set(order.get("attached", {}))
        known.update(c.get("orderId") for c in order.get("children", {}).values())
        child_links = {c["link"] for c in order.get("children", {}).values()}
        protective = copy.deepcopy(order.get("protective_orders", {}))
        candidates = set(protective)
        for f in order.get("executions", {}).values():
            if f.get("execType") == "Funding":
                continue
            if (
                f.get("orderId") in known
                or f.get("orderLinkId") in child_links
                or f.get("parentOrderLinkId") == order["id"]
            ):
                continue
            if (
                not f.get("orderId")
                or f.get("side") != _opposite(order)
                or f.get("orderLinkId") not in (None, "")
                or f.get("parentOrderLinkId") not in (None, "")
            ):
                raise _EvidenceError("foreign or unattributed same-symbol execution")
            candidates.add(f["orderId"])
        for row in open_orders:
            if (
                row.get("symbol") == order["symbol"]
                and row.get("side") == _opposite(order)
                and row.get("orderId")
                and row.get("orderId") not in known
                and row.get("orderLinkId") in (None, "")
                and _exit_kind(row) in {"SL", "TP2"}
            ):
                candidates.add(row["orderId"])
        if not candidates:
            return order
        origin = number(order.get("flat_verified_at"))
        state = await self.store.read()
        active = [
            o
            for o in state["orders"].values()
            if o["status"] not in TERMINAL and o["symbol"] == order["symbol"]
        ]
        if (
            origin is None
            or not 0 <= order["created_at"] - origin <= ACCOUNT_MAX_AGE
            or len(active) != 1
            or active[0]["id"] != order["id"]
        ):
            raise _EvidenceError(
                "anonymous protection lacks sole-owned flat-start evidence"
            )
        for order_id in sorted(candidates):
            fills = [
                f
                for f in order.get("executions", {}).values()
                if f.get("orderId") == order_id and f.get("execType") != "Funding"
            ]
            qty = sum(number(f.get("execQty"), -1) for f in fills)
            row = protective.get(order_id)
            if not (
                _terminal(row)
                and _near(
                    number(row.get("cumExecQty"), -1), qty, order["sizing"]["qty_step"]
                )
            ):
                lookup = getattr(self.client, "order_by_id", None)
                if lookup is None:
                    raise _EvidenceError("protective order-ID lookup unavailable")
                row = await lookup(order["symbol"], order_id)
            if (
                not isinstance(row, dict)
                or row.get("orderId") != order_id
                or row.get("symbol") != order["symbol"]
                or row.get("side") != _opposite(order)
                or number(row.get("positionIdx")) != 0
                or row.get("orderLinkId") not in (None, "")
                or row.get("parentOrderLinkId") not in (None, "", order["id"])
                or row.get("reduceOnly") is not True
                or row.get("closeOnTrigger") is not True
                or row.get("tpslMode") != "Full"
                or row.get("orderType") != "Market"
                or _exit_kind(row) not in {"SL", "TP2"}
            ):
                raise _EvidenceError("anonymous protective order identity unverified")
            stop_kind = _exit_kind({"stopOrderType": row.get("stopOrderType")})
            create_kind = _exit_kind({"createType": row.get("createType")})
            if (
                stop_kind != "UNKNOWN"
                and create_kind != "UNKNOWN"
                and stop_kind != create_kind
                or row.get("createType") in {"CreateByUser", "CreateByClosing"}
            ):
                raise _EvidenceError("protective trigger evidence conflicts")
            created, updated, total, cumulative, leaves = (
                number(row.get(k))
                for k in (
                    "createdTime",
                    "updatedTime",
                    "qty",
                    "cumExecQty",
                    "leavesQty",
                )
            )
            if (
                None in (created, updated, total, cumulative, leaves)
                or not int(order["created_at"] * 1000)
                <= created
                <= updated
                <= time.time() * 1000
                or total <= 0
                or cumulative < 0
                or cumulative > total
                or leaves < 0
                or leaves > total
                or not _near(cumulative, qty, order["sizing"]["qty_step"])
                or row.get("orderStatus")
                not in VENUE_TERMINAL
                | {"New", "PartiallyFilled", "Untriggered", "Triggered"}
                or row["orderStatus"] in VENUE_TERMINAL
                and not _terminal(row)
                or row["orderStatus"] == "Filled"
                and not _near(cumulative, total, order["sizing"]["qty_step"])
                or row["orderStatus"] not in VENUE_TERMINAL
                and not _near(cumulative + leaves, total, order["sizing"]["qty_step"])
                or any(
                    not created <= number(f.get("execTime"), -1) <= updated
                    for f in fills
                )
            ):
                raise _EvidenceError("protective order quantity/timing incomplete")
            protective[order_id] = row
        return {**order, "protective_orders": protective}

    def _ledger(self, order):
        entry_id = order["exchange_order"]["orderId"]
        children = order.get("children", {})
        entry, exits, reasons = [], [], {}
        for f in sorted(
            order.get("executions", {}).values(),
            key=lambda f: (number(f.get("execTime"), 0), f["execId"]),
        ):
            if f.get("execType") == "Funding":
                continue
            if f.get("execType") != "Trade":
                raise _EvidenceError(
                    "non-trade execution requires manual reconciliation"
                )
            q, price, fee, stamp = (
                number(f.get(k))
                for k in ("execQty", "execPrice", "execFee", "execTime")
            )
            if (
                q is None
                or q <= 0
                or price is None
                or price <= 0
                or fee is None
                or stamp is None
                or stamp < order["created_at"] * 1000
                or stamp > time.time() * 1000
            ):
                raise _EvidenceError("invalid execution evidence")
            if f.get("orderId") == entry_id:
                if (
                    f.get("orderLinkId") not in ("", order["id"])
                    or f.get("side") != order["side"]
                ):
                    raise _EvidenceError("entry fill identity mismatch")
                if number(f.get("closedSize"), 0) != 0:
                    raise _EvidenceError("entry closed foreign exposure")
                entry.append(f)
                continue
            child = next(
                (
                    c
                    for c in children.values()
                    if f.get("orderId") == c.get("orderId")
                    or f.get("orderLinkId") == c["link"]
                ),
                None,
            )
            attached = order.get("attached", {}).get(f.get("orderId"))
            protective = order.get("protective_orders", {}).get(f.get("orderId"))
            parent_match = f.get("parentOrderLinkId") == order["id"]
            if not (child or attached or protective or parent_match) or f.get(
                "side"
            ) != _opposite(order):
                raise _EvidenceError("foreign or unattributed same-symbol execution")
            if protective and (
                f.get("orderLinkId") not in (None, "")
                or f.get("parentOrderLinkId") not in (None, "", order["id"])
            ):
                raise _EvidenceError("protective fill identity mismatch")
            if child and (
                (child.get("orderId") and f.get("orderId") != child["orderId"])
                or f.get("orderLinkId") not in ("", child["link"])
            ):
                raise _EvidenceError("child fill identity mismatch")
            closed = number(f.get("closedSize"))
            if closed is None or not _near(closed, q, order["sizing"]["qty_step"]):
                raise _EvidenceError("exit closing quantity unverified")
            exits.append(f)
            reasons[f["execId"]] = (
                child["action"].upper()
                if child
                else _exit_kind(protective or attached or f)
            )
        eq, xq = sum(number(f["execQty"]) for f in entry), sum(
            number(f["execQty"]) for f in exits
        )
        if (
            not _near(
                eq,
                number(order["exchange_order"]["cumExecQty"]),
                order["sizing"]["qty_step"],
            )
            or xq > eq + 1e-9
        ):
            raise _EvidenceError("entry/exit fill totals incomplete")
        for child in children.values():
            qty = sum(
                number(f["execQty"])
                for f in exits
                if f.get("orderId") == child.get("orderId")
                or f.get("orderLinkId") == child["link"]
            )
            if qty > child["qty"] + 1e-9:
                raise _EvidenceError("child execution exceeds reserved reduction")
        for row in order.get("protective_orders", {}).values():
            if (
                number(row["qty"]) > eq + 1e-9
                or not _terminal(row)
                and not _near(
                    number(row["leavesQty"]), eq - xq, order["sizing"]["qty_step"]
                )
            ):
                raise _EvidenceError("protective order exceeds owned inventory")
        balance = 0.0
        entry_ids = {f["execId"] for f in entry}
        for f in sorted(
            entry + exits,
            key=lambda r: (int(r["execTime"]), r["execId"] not in entry_ids),
        ):
            balance += number(f["execQty"]) * (1 if f["execId"] in entry_ids else -1)
            if balance < -1e-9:
                raise _EvidenceError("exit predates owned exposure")
        return entry, exits, reasons, eq, eq - xq

    def _position(self, order, positions, remaining, average, instrument):
        relevant = [
            p
            for p in positions
            if p.get("symbol") == order["symbol"] and number(p.get("size"), -1) != 0
        ]
        if not relevant and _near(remaining, 0, instrument.qty_step):
            return None
        if len(relevant) != 1:
            raise _EvidenceError("owned position missing or ambiguous")
        p = relevant[0]
        if (
            p.get("side") != order["side"]
            or number(p.get("positionIdx")) != 0
            or not _near(number(p.get("size"), -1), remaining, instrument.qty_step)
            or not _near(number(p.get("avgPrice"), -1), average, instrument.tick_size)
        ):
            raise _EvidenceError("position ownership/quantity/entry mismatch")
        return p

    async def _protect(self, order, pos, price, instrument, blockers):
        sign, desired = _sign(order), order.get("desired_stop", order["sizing"]["stop"])
        stop, target = number(pos.get("stopLoss"), 0), number(pos.get("takeProfit"), 0)
        eps = instrument.tick_size * 1e-6
        valid_stop = (
            lambda s: s > 0
            and sign * (price - s) > eps
            and sign * (s - desired) >= -eps
        )
        valid_target = (
            lambda t: _near(t, order["sizing"]["target2"], instrument.tick_size)
            and sign * (t - price) > eps
        )
        if valid_stop(stop):
            desired = max(desired, stop) if sign > 0 else min(desired, stop)
            if desired != order.get("desired_stop"):
                await self._patch(order["id"], desired_stop=desired)
        if valid_stop(stop) and valid_target(target):
            await self._patch(
                order["id"],
                protection_verified_at=time.time(),
                protection_pending=False,
            )
            return True
        if sign * (price - desired) <= eps:
            blockers.append(
                order["symbol"] + ": desired stop crossed; emergency reduction pending"
            )
            await self._reduce(order, "emergency", number(pos["size"]))
            return False
        target = order["sizing"]["target2"]
        if sign * (target - price) <= eps:
            blockers.append(order["symbol"] + ": target2 reduction pending")
            await self._reduce(order, "tp2", number(pos["size"]))
            return False
        await self._patch(order["id"], protection_pending=True, desired_stop=desired)
        await self.store.assert_leader()
        try:
            await self.client.protect(order["symbol"], desired, target)
        except Exception:
            pass
        try:
            refreshed = await self.client.position_info(order["symbol"])
        except Exception:
            blockers.append(
                order["symbol"]
                + ": CRITICAL protection read unavailable; emergency reduction pending"
            )
            await self._reduce(order, "emergency", number(pos["size"]))
            return False
        check = self._position(
            order, refreshed, number(pos["size"]), order["avg_entry"], instrument
        )
        if (
            check
            and valid_stop(number(check.get("stopLoss"), 0))
            and valid_target(number(check.get("takeProfit"), 0))
        ):
            await self._patch(
                order["id"],
                protection_pending=False,
                protection_verified_at=time.time(),
            )
            return True
        blockers.append(
            order["symbol"]
            + ": CRITICAL protection unverified; emergency reduction pending"
        )
        await self._reduce(order, "emergency", number(pos["size"]))
        return False

    async def _accounting(self, order, instrument):
        if instrument.funding_interval_minutes != order.get("funding_interval_minutes"):
            return False  # Historical settlement cadence needs explicit reconciliation.
        start = int(order["opened_at"] * 1000)
        end = int(order["closed_at"] * 1000)
        closed = await self.client.closed_pnl(
            order["symbol"], int(order["created_at"] * 1000)
        )
        logs = await self.client.transaction_log(start, end)
        await self._patch(
            order["id"],
            accounting_raw={"closed_pnl": closed, "transaction_log": logs},
            funding_complete=False,
        )
        owned_fills = {
            f["execId"]: f for f in order["entry_fills"] + order["exit_fills"]
        }
        exit_qty = {}
        for f in order["exit_fills"]:
            exit_qty[f["orderId"]] = exit_qty.get(f["orderId"], 0) + number(
                f["execQty"]
            )
        pnl_rows = {r.get("orderId"): r for r in closed if r.get("orderId") in exit_qty}
        if any(
            k not in pnl_rows
            or not _near(
                number(pnl_rows[k].get("closedSize"), -1), q, instrument.qty_step
            )
            or pnl_rows[k].get("symbol") != order["symbol"]
            for k, q in exit_qty.items()
        ):
            return False
        seen, funding_rows, unique = {}, [], set()
        for row in logs:
            if row.get("symbol") != order["symbol"]:
                continue
            if (
                not row.get("id")
                or row["id"] in unique
                or row.get("currency") != "USDT"
                or row.get("category") != "linear"
                or row.get("transSubType") not in (None, "")
            ):
                return False
            unique.add(row["id"])
            if row.get("type") == "TRADE":
                f = owned_fills.get(row.get("tradeId"))
                if (
                    not f
                    or row["tradeId"] in seen
                    or row.get("orderId") != f["orderId"]
                    or row.get("side") != f["side"]
                ):
                    return False
                if not _near(
                    number(row.get("qty"), -1),
                    number(f["execQty"]),
                    instrument.qty_step,
                ):
                    return False
                seen[row["tradeId"]] = row
            elif row.get("type") == "SETTLEMENT":
                funding_rows.append(row)
            else:
                return False
        if set(seen) != set(owned_fills):
            return False
        # Require actual settlement rows even for zero funding; absence is unknown.
        interval = (
            order.get("funding_interval_minutes", instrument.funding_interval_minutes)
            * 60000
        )
        expected = {}
        for stamp in range((start // interval + 1) * interval, end + 1, interval):
            if any(int(f["execTime"]) == stamp for f in owned_fills.values()):
                return False  # Ordering at the settlement boundary is ambiguous.
            qty = sum(
                number(f["execQty"]) * (1 if f["side"] == order["side"] else -1)
                for f in owned_fills.values()
                if int(f["execTime"]) < stamp
            )
            if qty > 1e-9:
                expected[stamp] = qty
        actual = {}
        for r in funding_rows:
            stamp = number(r.get("transactionTime"))
            if (
                stamp not in expected
                or stamp in actual
                or r.get("side") != order["side"]
                or not _near(
                    number(r.get("size"), math.inf),
                    expected[stamp] * _sign(order),
                    instrument.qty_step,
                )
            ):
                return False
            actual[stamp] = r
        if set(actual) != set(expected):
            return False
        gross = fees = funding = net = 0.0
        for r in list(seen.values()) + funding_rows:
            cash, fee, change = (
                number(r.get(k)) for k in ("cashFlow", "fee", "change")
            )
            fund = (
                0.0
                if r.get("type") == "TRADE" and r.get("funding") in (None, "")
                else number(r.get("funding"))
            )
            if None in (cash, fee, change, fund) or not _near(
                change, cash + fund - fee
            ):
                return False
            gross += cash
            fees += fee
            funding += fund
            net += change
        if not _near(gross, order["gross_pnl"]) or not _near(fees, order["fees"]):
            return False
        await self._patch(
            order["id"],
            funding=funding,
            funding_complete=True,
            net_pnl=net,
            accounting_verified_at=time.time(),
            status="CLOSED",
        )
        return True

    async def _one(self, link, positions, open_orders, instrument, blockers):
        order = await self._get(link)
        if order.get("mode") != self.mode:
            raise _EvidenceError("order belongs to another execution mode")
        if order.get("structure_gap"):
            blockers.append(
                order["symbol"] + ": structural history incomplete; review required"
            )
        if order["status"] == "CLOSING":
            if not await self._accounting(order, instrument):
                blockers.append(
                    order["symbol"] + ": accounting/funding evidence pending"
                )
            return
        row = await self._entry(order)
        state = await self.store.read()
        universe_blocked = state.get("universe_mode") == "dynamic" and order[
            "symbol"
        ] not in entry_symbols(
            state.get("universe", {}),
            time.time(),
            required_policy=state.get("universe_policy"),
        )
        if not _terminal(row) and (
            time.time() >= order["opportunity"]["expires_at"]
            or order.get("exit_request")
            or order.get("entry_retired_at") is not None
            or universe_blocked
            or order["symbol"] not in self.instruments
        ):
            row = await self._cancel_entry(order)
            if not _terminal(row):
                blockers.append(order["symbol"] + ": entry cancellation unconfirmed")
        await self._children(await self._get(link))
        order = await self._collect(await self._get(link), open_orders)
        order = await self._resolve_protective(order, open_orders)
        if any(
            not _terminal(c.get("exchange_order"))
            for c in order.get("children", {}).values()
        ):
            blockers.append(
                order["symbol"] + ": reduction outcome pending; no resubmission"
            )
        entry, exits, reasons, filled, remaining = self._ledger(order)
        average = balance = realized = 0.0
        entry_ids = {f["execId"] for f in entry}
        for f in sorted(
            entry + exits,
            key=lambda r: (int(r["execTime"]), r["execId"] not in entry_ids),
        ):
            qty, price = number(f["execQty"]), number(f["execPrice"])
            if f["execId"] in entry_ids:
                average = (average * balance + qty * price) / (balance + qty)
                balance += qty
            else:
                realized += _sign(order) * (price - average) * qty
                balance -= qty
        fees = sum(number(f["execFee"]) for f in entry + exits)
        pos = self._position(order, positions, remaining, average, instrument)
        fields = dict(
            entry_fills=entry,
            exit_fills=exits,
            exit_classifications=reasons,
            filled_qty=filled,
            remaining_qty=remaining,
            avg_entry=average,
            ownership_verified=True,
            protective_orders=order.get("protective_orders", {}),
            gross_pnl=realized,
            fees=fees,
            net_pnl_before_funding=realized - fees,
            funding_interval_minutes=order.get(
                "funding_interval_minutes", instrument.funding_interval_minutes
            ),
        )
        if filled:
            fields["opened_at"] = order.get(
                "opened_at", min(int(f["execTime"]) for f in entry) / 1000
            )
        await self._patch(link, **fields)
        order = await self._get(link)
        if not filled:
            await self._patch(
                link,
                status=(
                    ("REJECTED" if row["orderStatus"] == "Rejected" else "CANCELLED")
                    if _terminal(row)
                    else "PENDING"
                ),
            )
            return
        if pos is None:
            if not _terminal(row):
                await self._cancel_entry(order)
                blockers.append(
                    order["symbol"] + ": flat but entry remainder not yet finalized"
                )
                return
            gross = _sign(order) * (
                sum(number(f["execPrice"]) * number(f["execQty"]) for f in exits)
                - sum(number(f["execPrice"]) * number(f["execQty"]) for f in entry)
            )
            fees = sum(number(f["execFee"]) for f in entry + exits)
            last = max(exits, key=lambda f: int(f["execTime"]))
            await self._patch(
                link,
                status="CLOSING",
                gross_pnl=gross,
                fees=fees,
                net_pnl_before_funding=gross - fees,
                closed_at=int(last["execTime"]) / 1000,
                exit_reason=reasons[last["execId"]],
                funding_complete=False,
            )
            if not await self._accounting(await self._get(link), instrument):
                blockers.append(
                    order["symbol"] + ": accounting/funding evidence pending"
                )
            return
        await self._patch(link, status="OPEN", position=pos)
        price = number((await self.client.ticker(order["symbol"])).get("lastPrice"))
        if price is None or price <= 0:
            raise _EvidenceError("current LastPrice unavailable")
        sign = _sign(order)
        tp1_filled = sum(
            number(f["execQty"]) for f in exits if reasons[f["execId"]] == "TP1"
        )
        target_qty = order.get("tp1_qty")
        if target_qty and tp1_filled >= target_qty - 1e-9 and not order.get("tp1_done"):
            desired = (
                max(order["desired_stop"], average)
                if sign > 0
                else min(order["desired_stop"], average)
            )
            desired = _round(desired, instrument.tick_size, up=sign > 0)
            await self._patch(
                link,
                tp1_done=True,
                tp1_filled_at=max(
                    int(f["execTime"]) / 1000
                    for f in exits
                    if reasons[f["execId"]] == "TP1"
                ),
                desired_stop=desired,
            )
        order = await self._get(link)
        if (
            order.get("tp1_done")
            and not order.get("trail_activated_at")
            and sign
            * (price - (order["sizing"]["target1"] + order["sizing"]["target2"]) / 2)
            > 0
        ):
            await self._patch(link, trail_activated_at=time.time())
            order = await self._get(link)
        if not await self._protect(order, pos, price, instrument, blockers):
            return
        if order.get("exit_request"):
            blockers.append(order["symbol"] + ": structural exit pending")
            if _terminal(row):
                await self._reduce(order, order["exit_request"]["action"], remaining)
            return
        liquidation = number(pos.get("liqPrice"))
        if liquidation and (
            sign * (average - liquidation) <= 0
            or abs(average - liquidation) < 2 * abs(average - order["sizing"]["stop"])
        ):
            blockers.append(order["symbol"] + ": liquidation buffer breached")
            await self._reduce(order, "emergency", remaining)
            return
        if not order.get("tp1_done") and (
            target_qty or sign * (price - order["sizing"]["target1"]) >= 0
        ):
            if not _terminal(row):
                await self._cancel_entry(
                    order
                )  # Freeze eventual TP1 base quantity first.
                return
            if target_qty is None:
                units = int((_d(filled) / _d(instrument.qty_step)).to_integral_value())
                if units <= 1:
                    # Charter's one-lot convention: no partial child; observed T1
                    # only tightens the stop. This is not a claimed TP1 fill.
                    await self._patch(
                        link,
                        tp1_done=True,
                        tp1_single_lot=True,
                        tp1_observed_at=time.time(),
                        desired_stop=_round(average, instrument.tick_size, up=sign > 0),
                    )
                    await self._protect(
                        await self._get(link), pos, price, instrument, blockers
                    )
                    return
                target_qty = _round(filled / 2, instrument.qty_step, up=True)
                await self._patch(link, tp1_qty=target_qty)
            if not await self._reduce(
                await self._get(link), "tp1", min(remaining, target_qty - tp1_filled)
            ):
                blockers.append(
                    order["symbol"]
                    + ": TP1 outcome unresolved or residual attempt limit reached"
                )

    async def reconcile(self, instruments):
        if self.mode not in {"testnet", "live"}:
            return {
                "as_of": time.time(),
                "positions": [],
                "open_orders": [],
                "blockers": ["Shadow mode: exchange execution disabled"],
            }
        async with self._lock:
            await self.store.assert_leader()

            def invalidate(tx):
                # A failed/aborted read must not leave the previous account
                # snapshot apparently fresh enough to authorize a new entry.
                tx.state.setdefault("account", {})["as_of"] = 0

            await self.store.update(invalidate)
            self.instruments = dict(instruments)
            self.management_instruments = dict(instruments)
            positions = await self.client.positions()
            open_orders = await self.client.open_orders()
            state = await self.store.read()
            blockers = []
            for link, order in state["orders"].items():
                if order["status"] in TERMINAL:
                    continue
                try:
                    instrument = instruments.get(order["symbol"])
                    if instrument is not None:
                        if order.get("instrument") != asdict(instrument):
                            await self._patch(link, instrument=asdict(instrument))
                    else:
                        instrument = self._saved_instrument(order)
                    self.management_instruments[order["symbol"]] = instrument
                    await self._one(
                        link,
                        positions,
                        open_orders,
                        instrument,
                        blockers,
                    )
                except Exception as exc:
                    # Only fixed internal errors are exposed, never client messages.
                    reason = (
                        str(exc)
                        if type(exc) is _EvidenceError
                        else "exchange evidence unavailable"
                    )
                    blockers.append(order["symbol"] + ": " + reason)
                    await self._patch(
                        link, ownership_verified=False, reconciliation_error=reason
                    )
            # Fresh snapshot after mutations, not pre-reduction positions.
            positions = await self.client.positions()
            open_orders = await self.client.open_orders()
            state = await self.store.read()
            active = [
                o for o in state["orders"].values() if o["status"] not in TERMINAL
            ]
            for p in positions:
                size = number(p.get("size"))
                if size is None or size < 0:
                    blockers.append("Malformed exchange position")
                elif size and not any(
                    o["symbol"] == p.get("symbol")
                    and o["side"] == p.get("side")
                    and o.get("ownership_verified")
                    for o in active
                ):
                    blockers.append(
                        "Unmanaged exchange position " + str(p.get("symbol", "?"))
                    )
                if size and number(p.get("unrealisedPnl")) is None:
                    blockers.append("Open position unrealized P&L unavailable")
            links = set(state["orders"])
            links.update(
                c["link"] for o in active for c in o.get("children", {}).values()
            )
            for row in open_orders:
                if row.get("orderLinkId") not in links and not any(
                    row.get("symbol") == o["symbol"]
                    and row.get("side") == _opposite(o)
                    and (
                        row.get("parentOrderLinkId") == o["id"]
                        or o.get("ownership_verified")
                        and row.get("orderId") in o.get("protective_orders", {})
                        and row == o["protective_orders"][row["orderId"]]
                    )
                    for o in active
                ):
                    blockers.append(
                        "Unmanaged open order " + str(row.get("symbol", "?"))
                    )
            try:
                equity = await self.client.equity()
                if number(equity) is None or equity <= 0:
                    raise _EvidenceError()
            except Exception:
                equity = None
                blockers.append("Sizing equity unavailable; entries blocked")
            account = await self.client.account_info()
            if account.get("marginMode") != "ISOLATED_MARGIN":
                blockers.append("Exchange account is not verified isolated margin")
            snapshot = {
                "as_of": time.time(),
                "equity": equity,
                "positions": positions,
                "open_orders": open_orders,
                "blockers": list(dict.fromkeys(blockers)),
                "account": account,
            }

            def save(tx):
                previous = tx.state.get("account", {})
                tx.state["account"] = snapshot
                if equity is not None:
                    tx.state.setdefault("risk_reference_equity", equity)
                    tx.state.setdefault("risk_reference_equities", {}).setdefault(
                        self.mode, equity
                    )
                if previous.get("blockers") != snapshot["blockers"]:
                    key = hashlib.sha256(
                        repr(snapshot["blockers"]).encode()
                    ).hexdigest()
                    tx.event(
                        "account:" + key + ":" + str(time.time()),
                        "account_health",
                        snapshot,
                        "Reconciliation: "
                        + (
                            "; ".join(snapshot["blockers"])
                            or "Exchange evidence reconciled"
                        ),
                    )

            await self.store.update(save)
            return snapshot

    async def manage_structure(self, symbol, daily, execution):
        """Queue structural requests; reconcile verifies ownership before sending.

        Aligned with simulation._manage: daily-close invalidation; count 10/20
        subsequent daily closes only without a favorable closed 4H zone break.
        Same-time 4H evidence precedes the daily timeout decision. TP1 precedes
        midpoint activation; newly confirmed higher lows/lower highs ratchet by
        a 0.1% pivot buffer, quantized conservatively. No synthetic fills.
        Supply contiguous closed series covering every boundary since the last
        persisted cursor, with a fixed history origin for the pivot warmup.
        """
        if self.mode not in {"testnet", "live"}:
            return []
        async with self._lock:
            now = time.time()
            d, e = _closed(daily, DAY, now), _closed(execution, H4, now)
            pivots = confirmed_zigzag(execution, H4, now, atr_multiple=1)
            daily_at = {b.open_time / 1000 + DAY: b for b in d}
            execution_at = {b.open_time / 1000 + H4: b for b in e}
            changed = []

            def manage(tx):
                for order in tx.state["orders"].values():
                    if (
                        order["symbol"] != symbol
                        or order["status"] != "OPEN"
                        or order.get("mode") != self.mode
                        or not order.get("opened_at")
                    ):
                        continue
                    sign, opened = _sign(order), order["opened_at"]
                    inst = self.management_instruments.get(
                        symbol
                    ) or self.instruments.get(symbol)
                    if inst is None:
                        inst = self._saved_instrument(order)
                    for stamp in sorted(set(daily_at) | set(execution_at)):
                        if stamp <= opened:
                            continue
                        bar = execution_at.get(stamp)
                        cursor = order.get("structure_checked_at", opened)
                        if bar and stamp > cursor:
                            if stamp != (int(cursor // H4) + 1) * H4:
                                order["structure_gap"] = (
                                    "Missing closed 4H history; time/trail management blocked"
                                )
                            else:
                                order["structure_checked_at"] = stamp
                                edge = number(
                                    order["opportunity"]
                                    .get("evidence", {})
                                    .get("zone_high" if sign > 0 else "zone_low")
                                )
                                if edge is None:
                                    order["structure_gap"] = (
                                        "Entry zone evidence missing"
                                    )
                                elif sign * (bar.close - edge) > 0:
                                    order.update(
                                        favorable_4h_close=True, time_review_due=False
                                    )
                                midpoint = (
                                    order["sizing"]["target1"]
                                    + order["sizing"]["target2"]
                                ) / 2
                                extreme = bar.high if sign > 0 else bar.low
                                tp1_at = order.get(
                                    "tp1_filled_at",
                                    order.get("tp1_observed_at", math.inf),
                                )
                                if (
                                    order.get("tp1_done")
                                    and bar.open_time / 1000 >= tp1_at
                                    and sign * (extreme - midpoint) > 0
                                ):
                                    order.setdefault("trail_activated_at", stamp)
                                if (
                                    inst
                                    and order.get("tp1_done")
                                    and order.get("trail_activated_at")
                                    and not order.get("structure_gap")
                                ):
                                    kind = "low" if sign > 0 else "high"
                                    known = [
                                        p
                                        for p in pivots
                                        if p.kind == kind and p.available_at <= stamp
                                    ]
                                    if len(known) >= 2:
                                        previous, pivot = known[-2:]
                                        if (
                                            pivot.available_at == stamp
                                            and stamp >= order["trail_activated_at"]
                                            and pivot.open_time / 1000 >= opened
                                            and sign * (pivot.price - previous.price)
                                            > 0
                                        ):
                                            stop = _round(
                                                pivot.price * (1 - sign * 0.001),
                                                inst.tick_size,
                                                up=sign < 0,
                                            )
                                            if (
                                                stop > 0
                                                and sign * (bar.close - stop) > 0
                                                and sign
                                                * (stop - order["desired_stop"])
                                                > 0
                                            ):
                                                order.update(
                                                    desired_stop=stop,
                                                    trail_pivot_id=pivot.id,
                                                    trail_pivot_at=stamp,
                                                )
                                                tx.event(
                                                    order["id"] + ":trail:" + pivot.id,
                                                    "trail_request",
                                                    {"stop": stop, "pivot": pivot.id},
                                                )
                        bar = daily_at.get(stamp)
                        cursor = order.get("daily_checked_at", opened)
                        if bar and stamp > cursor:
                            if (
                                sign
                                * (bar.close - order["opportunity"]["invalidation"])
                                <= 0
                            ):
                                order.setdefault(
                                    "exit_request",
                                    {"action": "daily_invalidation", "at": stamp},
                                )
                            if stamp != (int(cursor // DAY) + 1) * DAY:
                                order["structure_gap"] = (
                                    "Missing closed daily history; time/trail management blocked"
                                )
                                continue
                            order["daily_checked_at"] = stamp
                            order["daily_bars_held"] = (
                                order.get("daily_bars_held", 0) + 1
                            )
                            if (
                                not order.get("favorable_4h_close")
                                and not order.get("structure_gap")
                                and order.get("structure_checked_at", 0) >= stamp
                            ):
                                if order["daily_bars_held"] >= 20:
                                    order.setdefault(
                                        "exit_request",
                                        {"action": "timeout", "at": stamp},
                                    )
                                elif order["daily_bars_held"] >= 10 and not order.get(
                                    "ten_day_reviewed"
                                ):
                                    order.update(
                                        ten_day_reviewed=stamp, time_review_due=True
                                    )
                                    tx.event(
                                        order["id"] + ":10day",
                                        "structure_review",
                                        {"parent": order["id"]},
                                        f"10 daily closes without a favorable 4H zone break · {symbol}: review due",
                                    )
                    changed.append(order["id"])

            await self.store.update(manage)
            return changed
