"""Offline executor integration tests: real isolated SQLite, fake venue only."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

from apex_bot import execution as ex
from apex_bot.engine import Pivot
from apex_bot.execution import ExecutionManager, client_id
from apex_bot.models import Candle, Instrument, Opportunity
from apex_bot.storage import Store

NOW = 1_791_158_401.0


class FakeBybit:
    def __init__(self, store):
        self.store = store
        self.rows, self.fills, self.pos, self.external = {}, [], [], []
        self.calls = []
        self.price = 105
        self.entry_error = self.reduce_error = None
        self.cancel_confirms = self.protect_works = True
        self.reduce_fraction = 1
        self.hidden = set()
        self.protection_read_error = False
        self.equity_error = False
        self.logs_enabled = True
        self.pnl_enabled = True
        self.settlements = []
        self.interval = 480

    async def submit_limit(self, symbol, side, qty, price, stop, target, link):
        saved = (await self.store.read())["orders"][link]
        assert saved["status"] == "INTENT"
        self.calls.append(("submit", link))
        if self.entry_error:
            raise self.entry_error
        self.rows[link] = {
            "orderId": "id-" + link,
            "orderLinkId": link,
            "symbol": symbol,
            "side": side,
            "positionIdx": 0,
            "qty": str(qty),
            "leavesQty": str(qty),
            "cumExecQty": "0",
            "orderStatus": "New",
            "reduceOnly": False,
            "price": str(price),
            "stopLoss": str(stop),
            "takeProfit": str(target),
        }
        return {"orderId": "id-" + link, "orderLinkId": link}

    def fill(self, link, qty, price=None, terminal=None, stamp=None):
        row = self.rows[link]
        price = float(row.get("price", self.price)) if price is None else price
        stamp = int(ex.time.time() * 1000) if stamp is None else stamp
        row["cumExecQty"] = str(float(row["cumExecQty"]) + qty)
        left = max(0, float(row["qty"]) - float(row["cumExecQty"]))
        row.update(
            leavesQty=str(left),
            orderStatus="Filled" if left == 0 else "PartiallyFilled",
        )
        if terminal:
            row.update(orderStatus=terminal, leavesQty="0")
        if "updatedTime" in row:
            row["updatedTime"] = str(stamp)
        row["avgPrice"] = str(price)
        record = {
            "execId": "exec-" + str(len(self.fills)),
            "orderId": row["orderId"],
            "orderLinkId": row["orderLinkId"],
            "symbol": row["symbol"],
            "side": row["side"],
            "execQty": str(qty),
            "execPrice": str(price),
            "execFee": str(qty * 0.01),
            "execTime": str(stamp),
            "execType": "Trade",
            "closedSize": str(qty if row["reduceOnly"] else 0),
        }
        self.fills.append(record)
        if not self.pos and not row["reduceOnly"]:
            self.pos = [
                {
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "size": "0",
                    "positionIdx": 0,
                    "avgPrice": str(price),
                    "stopLoss": row["stopLoss"],
                    "takeProfit": row["takeProfit"],
                    "markPrice": str(self.price),
                    "unrealisedPnl": "0",
                    "liqPrice": "50" if row["side"] == "Buy" else "160",
                }
            ]
        if self.pos:
            current = float(self.pos[0]["size"])
            self.pos[0]["size"] = str(current + (-qty if row["reduceOnly"] else qty))
            if abs(float(self.pos[0]["size"])) < 1e-9:
                self.pos = []
        return record

    async def order(self, symbol, link):
        self.calls.append(("order", link))
        return copy.deepcopy(None if link in self.hidden else self.rows.get(link))

    async def order_by_id(self, symbol, order_id):
        self.calls.append(("order_by_id", symbol, order_id))
        if order_id in self.hidden:
            return None
        return copy.deepcopy(
            next((r for r in self.rows.values() if r["orderId"] == order_id), None)
        )

    async def cancel(self, symbol, link):
        saved = (await self.store.read())["orders"][link]
        assert saved["cancel_intent"]["status"] == "UNKNOWN"
        self.calls.append(("cancel", link))
        if self.cancel_confirms:
            self.rows[link].update(orderStatus="Cancelled", leavesQty="0")
        return {"orderId": self.rows[link]["orderId"], "orderLinkId": link}

    async def reduce_market(self, symbol, side, qty, link):
        state = await self.store.read()
        assert any(
            c["link"] == link
            for o in state["orders"].values()
            for c in o["children"].values()
        )
        self.calls.append(("reduce", link, qty))
        self.rows[link] = {
            "orderId": "id-" + link,
            "orderLinkId": link,
            "symbol": symbol,
            "side": side,
            "positionIdx": 0,
            "qty": str(qty),
            "leavesQty": str(qty),
            "cumExecQty": "0",
            "orderStatus": "New",
            "reduceOnly": True,
        }
        if self.reduce_fraction:
            q = qty * self.reduce_fraction
            self.fill(
                link,
                q,
                self.price,
                terminal="Filled" if q == qty else "PartiallyFilledCanceled",
            )
        if self.reduce_error:
            raise self.reduce_error
        return {"orderId": "id-" + link, "orderLinkId": link}

    async def protect(self, symbol, stop, target):
        self.calls.append(("protect", stop, target))
        if self.protect_works:
            self.pos[0].update(stopLoss=str(stop), takeProfit=str(target))
        return {}

    async def positions(self):
        self.calls.append(("positions",))
        return copy.deepcopy(self.pos)

    async def position_info(self, symbol):
        self.calls.append(("position_info",))
        if self.protection_read_error:
            raise RuntimeError("secret transport details")
        return copy.deepcopy(
            self.pos or [{"symbol": symbol, "size": "0", "positionIdx": 0}]
        )

    async def open_orders(self):
        self.calls.append(("open_orders",))
        rows = [
            r for r in self.rows.values() if r["orderStatus"] not in ex.VENUE_TERMINAL
        ]
        return copy.deepcopy(rows + self.external)

    async def executions(self, symbol, start_ms):
        self.calls.append(("executions", start_ms))
        assert start_ms >= (ex.time.time() - 7 * ex.DAY) * 1000
        return copy.deepcopy(
            [
                f
                for f in self.fills
                if f["symbol"] == symbol and int(f["execTime"]) >= start_ms
            ]
        )

    async def ticker(self, symbol):
        return {"lastPrice": str(self.price)}

    async def equity(self):
        if self.equity_error:
            raise RuntimeError("totalEquity not available in isolated margin")
        return 10000

    async def account_info(self):
        return {"marginMode": "ISOLATED_MARGIN"}

    async def closed_pnl(self, symbol, start_ms):
        self.calls.append(("closed_pnl", start_ms))
        if not self.pnl_enabled:
            return []
        quantities = {}
        for f in self.fills:
            if float(f["closedSize"]) > 0:
                quantities[f["orderId"]] = quantities.get(f["orderId"], 0) + float(
                    f["execQty"]
                )
        return [
            {"orderId": k, "symbol": symbol, "closedSize": str(q)}
            for k, q in quantities.items()
        ]

    async def transaction_log(self, start_ms, end_ms):
        self.calls.append(("transaction_log", start_ms, end_ms))
        if not self.logs_enabled:
            return []
        entry = next(f for f in self.fills if float(f["closedSize"]) == 0)
        rows = []
        for f in self.fills:
            if not start_ms <= int(f["execTime"]) <= end_ms:
                continue
            cash = (
                (float(f["execPrice"]) - float(entry["execPrice"]))
                * float(f["execQty"])
                * (1 if entry["side"] == "Buy" else -1)
                if float(f["closedSize"])
                else 0
            )
            fee = float(f["execFee"])
            rows.append(
                {
                    "id": "tx-" + f["execId"],
                    "tradeId": f["execId"],
                    "orderId": f["orderId"],
                    "symbol": f["symbol"],
                    "side": f["side"],
                    "qty": f["execQty"],
                    "type": "TRADE",
                    "category": "linear",
                    "currency": "USDT",
                    "transactionTime": f["execTime"],
                    "fee": str(fee),
                    "cashFlow": str(cash),
                    "funding": "",
                    "change": str(cash - fee),
                }
            )
        return copy.deepcopy(rows + self.settlements)


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_resting_research_cannot_submit_even_to_testnet(self):
        op = replace(
            self.op,
            evidence={
                **self.op.evidence,
                "entry_style": "resting_limit",
                "trigger_kind": "ZONE_LIMIT",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "research-only"):
            await self.manager.submit(op, self.sizing, 0)
        self.assertEqual(self.calls("submit"), [])

    async def test_research_markers_block_both_sides_and_venues_before_any_io(self):
        markers = (
            {"entry_style": "resting_limit"},
            {"trigger_kind": "ZONE_LIMIT"},
            {"entry_style": "monitored_zone"},
            {"trigger_kind": "ZONE_ARRIVAL"},
            {"trigger_kind": "ZONE_WATCH"},
            {
                "entry_style": "monitored_zone",
                "trigger_kind": "SWING_BREAK",
                "trigger_closed_at": NOW,
                "trigger_price": 100,
                "trigger_evidence_id": "forged-confirmation",
            },
            {"entry_style": "confirmed", "trigger_kind": "ZONE_ARRIVAL"},
            {"entry_style": "resting_limit", "trigger_kind": "ZONE_WATCH"},
        )
        before = await self.store.read()
        for mode in ("testnet", "live"):
            manager = ExecutionManager(self.client, self.store, mode)
            for side in ("Buy", "Sell"):
                for evidence in markers:
                    with self.subTest(mode=mode, side=side, evidence=evidence):
                        op = replace(self.op, side=side, evidence=evidence)
                        # Isolate this guard from the unchanged release gate.
                        # A future live approval must still not permit research.
                        with (
                            patch("apex_bot.release.LIVE_APPROVED", True),
                            patch.object(self.store, "update") as update,
                            patch.object(self.store, "read") as read,
                            patch.object(self.store, "assert_leader") as leader,
                            patch.object(manager, "client") as client,
                        ):
                            with self.assertRaisesRegex(RuntimeError, "research-only"):
                                await manager.submit(op, self.sizing, 0)
                            update.assert_not_called()
                            read.assert_not_called()
                            leader.assert_not_called()
                            self.assertEqual(client.mock_calls, [])
        self.assertEqual(self.client.calls, [])
        self.assertEqual(await self.store.read(), before)

    async def test_live_release_gate_still_precedes_monitored_guard(self):
        from apex_bot.release import REASON

        manager = ExecutionManager(self.client, self.store, "live")
        op = replace(self.op, evidence={"entry_style": "monitored_zone"})
        with patch("apex_bot.release.LIVE_APPROVED", False):
            with self.assertRaises(RuntimeError) as raised:
                await manager.submit(op, self.sizing, 0)
        self.assertEqual(str(raised.exception), REASON)
        self.assertEqual(self.client.calls, [])
        self.assertEqual((await self.store.read())["orders"], {})

    async def asyncSetUp(self):
        self.now = NOW
        self.clock = patch.object(ex.time, "time", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.store = Store(sqlite_path=":memory:")
        await self.store.initialize()
        await self.store.lease(ttl=10**8)
        self.addAsyncCleanup(self.store.close)
        self.client = FakeBybit(self.store)
        self.manager = ExecutionManager(self.client, self.store, "testnet")
        self.inst = Instrument("BTCUSDT", 0.1, 0.1, 1000, 0.1, 5)
        self.instruments = {
            "BTCUSDT": self.inst,
            "ETHUSDT": replace(self.inst, symbol="ETHUSDT"),
            "SOLUSDT": replace(self.inst, symbol="SOLUSDT"),
        }
        self.manager.instruments = self.instruments
        self.op = Opportunity(
            "candidate",
            "BTCUSDT",
            "Buy",
            "2",
            "READY",
            100,
            90,
            120,
            140,
            92,
            99,
            self.now - ex.DAY,
            self.now + 100,
            "trigger",
            {"risk_multiplier": 1, "zone_low": 95, "zone_high": 110},
        )
        self.sizing = dict(
            allowed=True,
            qty=2,
            qty_step=0.1,
            entry=100,
            stop=90,
            target1=120,
            target2=140,
            risk_cash=21,
            notional=200,
            risk_pct=0.25,
        )
        self.link = client_id(self.op.id)
        await self.snapshot()
        await self.approve(self.op)

    async def snapshot(self):
        def save(tx):
            tx.state["account"] = {
                "as_of": self.now,
                "equity": 10000,
                "positions": [],
                "open_orders": [],
                "blockers": [],
            }
            tx.state["context"] = {
                "data_complete": True,
                "as_of": self.now,
                "expires_at": self.now + 21600,
                "long_multiplier": 1,
                "short_multiplier": 1,
                "event_blackout": False,
            }

        await self.store.update(save)

    async def approve(self, op):
        def save(tx):
            tx.state["opportunities"][op.id] = {
                "opportunity": op.to_dict(),
                "ai": {
                    "verdict": "APPROVE",
                    "evidence_hash": "a" * 64,
                    "candidate_fingerprint": ex.candidate_fingerprint(op),
                    "context_fingerprint": ex.context_fingerprint(
                        tx.state, op.symbol, self.now
                    ),
                },
            }

        await self.store.update(save)

    async def record(self):
        return (await self.store.read())["orders"][self.link]

    async def submit(self):
        self.assertTrue(await self.manager.submit(self.op, self.sizing, 0))

    async def opened(self, qty=2):
        await self.submit()
        self.now += 1
        self.client.fill(self.link, qty)
        return await self.manager.reconcile(self.instruments)

    async def reconcile(self):
        self.now += 1
        return await self.manager.reconcile(self.instruments)

    def calls(self, name):
        return [c for c in self.client.calls if c[0] == name]

    async def test_shadow_default_has_no_client_calls(self):
        manager = ExecutionManager(self.client, self.store)
        with self.assertRaises(RuntimeError):
            await manager.submit(self.op, self.sizing, 0)
        self.assertTrue((await manager.reconcile(self.instruments))["blockers"])
        self.assertEqual(await manager.manage_structure("BTCUSDT", [], []), [])
        self.assertEqual(self.client.calls, [])

    async def test_intent_before_post_and_id_never_reused(self):
        await self.submit()
        self.assertEqual((await self.record())["status"], "ACKNOWLEDGED")
        self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        self.assertEqual(len(self.calls("submit")), 1)
        self.assertLessEqual(len(self.link), 36)

    async def test_verified_new_order_becomes_pending_without_executions(self):
        await self.submit()
        snapshot = await self.reconcile()
        order = await self.record()
        self.assertEqual(snapshot["blockers"], [])
        self.assertEqual(order["status"], "PENDING")
        self.assertEqual(order["filled_qty"], 0)
        self.assertEqual(order["entry_fills"], [])
        self.assertTrue(self.calls("order"))
        self.assertTrue(self.calls("executions"))
        self.assertEqual(self.calls("reduce"), [])
        self.assertEqual(self.calls("protect"), [])
        self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        self.assertEqual(len(self.calls("submit")), 1)

    async def test_missing_queried_price_keeps_acknowledgment_and_blocks(self):
        await self.submit()
        del self.client.rows[self.link]["price"]
        snapshot = await self.reconcile()
        order = await self.record()
        self.assertEqual(order["status"], "ACKNOWLEDGED")
        self.assertFalse(order["ownership_verified"])
        self.assertEqual(
            order["reconciliation_error"], "entry order differs from persisted intent"
        )
        self.assertIn(
            "BTCUSDT: entry order differs from persisted intent", snapshot["blockers"]
        )
        self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        self.assertEqual(len(self.calls("submit")), 1)
        self.assertEqual(self.calls("reduce"), [])
        self.assertEqual(self.calls("protect"), [])
        # A later authoritative read can recover the same intent without a POST.
        self.client.rows[self.link]["price"] = str(self.sizing["entry"])
        self.assertEqual((await self.reconcile())["blockers"], [])
        self.assertEqual((await self.record())["status"], "PENDING")
        self.assertEqual(len(self.calls("submit")), 1)

    async def test_ambiguous_entry_never_resubmits_after_restart(self):
        self.client.entry_error = TimeoutError()
        await self.submit()
        self.assertEqual((await self.record())["status"], "UNKNOWN")
        self.manager = ExecutionManager(self.client, self.store, "testnet")
        self.manager.instruments = self.instruments
        self.assertTrue((await self.reconcile())["blockers"])
        self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        self.assertEqual(len(self.calls("submit")), 1)

    async def test_crash_after_reserved_intent_before_post_no_replay(self):
        with patch.object(
            self.store, "assert_leader", side_effect=asyncio.CancelledError()
        ):
            with self.assertRaises(asyncio.CancelledError):
                await self.manager.submit(self.op, self.sizing, 0)
        self.assertEqual((await self.record())["status"], "INTENT")
        self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        self.assertEqual(self.calls("submit"), [])

    async def test_settings_freshness_and_sizing_rechecked_atomically(self):
        for change in (
            "paused",
            "version",
            "freshness",
            "equity",
            "risk",
            "foreign",
            "sizing",
        ):
            with self.subTest(change=change):
                state = await self.store.read()

                def modify(tx):
                    if change == "paused":
                        tx.state["settings"]["paused"] = True
                    if change == "version":
                        tx.state["settings_version"] = 1
                    if change == "freshness":
                        tx.state["account"]["as_of"] = self.now - 31
                    if change == "equity":
                        tx.state["account"]["equity"] = 100
                    if change == "risk":
                        tx.state["settings"]["risk_pct"] = 0.1
                    if change == "foreign":
                        tx.state["account"]["positions"] = [
                            {"symbol": "ETHUSDT", "size": "1", "side": "Buy"}
                        ]

                await self.store.update(modify)
                sizing = (
                    {**self.sizing, "stop": 80} if change == "sizing" else self.sizing
                )
                self.assertFalse(await self.manager.submit(self.op, sizing, 0))
                await self.store.update(lambda tx: tx.state.update(state))
        self.assertEqual(self.calls("submit"), [])

    async def test_reservation_requires_exact_current_ready_opportunity(self):
        original = (await self.store.read())["opportunities"][self.op.id]
        changes = [
            {"state": "WAIT"},
            {"state": "INVALID"},
            {"entry": 101},
            {"stop": 91},
            {"target1": 121},
            {"target2": 141},
            {"invalidation": 93},
            {"confirmation": 100},
            {"side": "Sell"},
            {"symbol": "ETHUSDT"},
            {"setup": "2X"},
            {"expires_at": self.now + 10},
            {"tier": 2},
            {"bucket": "different"},
            {"evidence": {**self.op.evidence, "risk_multiplier": 0.5}},
        ]
        for change in changes:
            with self.subTest(change=change):

                def mutate(tx):
                    tx.state["opportunities"][self.op.id] = copy.deepcopy(original)
                    tx.state["opportunities"][self.op.id]["opportunity"].update(change)

                await self.store.update(mutate)
                self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        await self.store.update(lambda tx: tx.state["opportunities"].clear())
        self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        self.assertEqual(self.calls("submit"), [])
        self.assertEqual((await self.store.read())["orders"], {})

    async def test_candidate_gate_reads_transaction_state_not_prior_snapshot(self):
        update = self.store.update

        async def changed_at_reservation(callback, **kwargs):
            def interleave(tx):
                if callback.__name__ == "reserve":
                    tx.state["opportunities"][self.op.id]["opportunity"][
                        "state"
                    ] = "INVALID"
                return callback(tx)

            return await update(interleave, **kwargs)

        with patch.object(self.store, "update", side_effect=changed_at_reservation):
            self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        self.assertEqual(self.calls("submit"), [])
        self.assertEqual((await self.store.read())["orders"], {})

    async def test_polling_clock_alone_does_not_invalidate_candidate_gate(self):
        await self.store.update(
            lambda tx: tx.state["opportunities"][self.op.id]["opportunity"][
                "evidence"
            ].update(as_of=self.now + 1)
        )
        await self.submit()

    async def test_atomic_candidate_gate_requires_persisted_ai_and_context_hash(self):
        approval = (await self.store.read())["opportunities"][self.op.id]["ai"]
        for changes in (
            {"verdict": "WAIT"},
            {"verdict": "REJECT"},
            {"evidence_hash": ""},
            {"evidence_hash": "unknown"},
            {"evidence_hash": "z" * 64},
            {"candidate_fingerprint": None},
            {"candidate_fingerprint": "b" * 64},
            {"context_fingerprint": None},
            {"context_fingerprint": "b" * 64},
        ):
            await self.store.update(
                lambda tx: tx.state["opportunities"][self.op.id].update(
                    ai={**approval, **changes}
                )
            )
            self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        await self.store.update(
            lambda tx: tx.state["opportunities"][self.op.id].pop("ai")
        )
        self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        self.assertEqual(self.calls("submit"), [])

    async def test_updated_supplied_and_stored_candidate_cannot_retain_old_approval(
        self,
    ):
        changed = replace(self.op, entry=101)
        sizing = {**self.sizing, "entry": 101, "notional": 202, "risk_cash": 23}
        await self.store.update(
            lambda tx: tx.state["opportunities"][self.op.id].update(
                opportunity=changed.to_dict()
            )
        )
        self.assertFalse(await self.manager.submit(changed, sizing, 0))
        self.assertEqual(self.calls("submit"), [])
        self.assertEqual((await self.store.read())["orders"], {})
        # Missing binding also blocks. Only an approval of the changed candidate
        # can authorize the same otherwise-valid candidate and sizing.
        await self.store.update(
            lambda tx: tx.state["opportunities"][self.op.id]["ai"].pop(
                "candidate_fingerprint"
            )
        )
        self.assertFalse(await self.manager.submit(changed, sizing, 0))
        await self.approve(changed)
        self.assertTrue(await self.manager.submit(changed, sizing, 0))
        self.assertEqual(len(self.calls("submit")), 1)
        self.assertEqual((await self.record())["opportunity"]["entry"], 101)

    async def test_context_provenance_change_and_reference_expiry_invalidate_approval(
        self,
    ):
        reference = {
            "current": True,
            "last_success_at": self.now - 3599,
            "snapshot": {
                "revision": "v1",
                "sha256": "source-hash",
                "content": {"text": "Source evidence"},
            },
        }
        await self.store.update(
            lambda tx: tx.state.update(references={"fixture": reference})
        )
        await self.approve(self.op)
        await self.store.update(
            lambda tx: tx.state["references"]["fixture"]["snapshot"].update(
                revision="v2"
            )
        )
        self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        await self.approve(self.op)
        self.now += 2
        self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        self.assertEqual(self.calls("submit"), [])

    async def test_concurrent_bucket_reservations_stop_at_current_cap(self):
        ops = [
            replace(self.op, id=str(i), symbol=s)
            for i, s in enumerate(self.instruments)
        ]
        for op in ops:
            await self.approve(op)
        managers = [ExecutionManager(self.client, self.store, "testnet") for _ in ops]
        for m in managers:
            m.instruments = self.instruments
        outcomes = await asyncio.gather(
            *(m.submit(o, self.sizing, 0) for m, o in zip(managers, ops))
        )
        self.assertEqual(sum(outcomes), 2)
        self.assertEqual(len(self.calls("submit")), 2)

    async def test_fresh_circuit_higher_losses_and_stale_circuit_block(self):
        base = {
            "as_of": self.now,
            "daily_loss_pct": 0,
            "weekly_loss_pct": 0,
            "drawdown_pct": 0,
        }
        for changes in (
            {"daily_loss_pct": 2},
            {"weekly_loss_pct": 4},
            {"drawdown_pct": 12},
            {"drawdown_pct": 8},
            {"as_of": self.now - 60},
            {"daily_loss_pct": None},
        ):
            with self.subTest(changes=changes):
                await self.store.update(
                    lambda tx: tx.state.update(
                        risk_circuits={"testnet": {**base, **changes}}
                    )
                )
                self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        self.assertEqual(self.calls("submit"), [])
        await self.store.update(
            lambda tx: tx.state.update(
                risk_circuits={"testnet": {**base, "drawdown_pct": 8}}
            )
        )
        smaller = {
            **self.sizing,
            "qty": 1,
            "risk_cash": 10.5,
            "risk_pct": 0.125,
            "notional": 100,
        }
        self.assertTrue(await self.manager.submit(self.op, smaller, 0))

    async def test_actual_open_mtm_required_and_higher_loss_not_ignored(self):
        await self.opened()
        op = replace(self.op, id="eth", symbol="ETHUSDT")
        await self.approve(op)
        for value in (None, "nan", "-250"):

            def set_mtm(tx):
                tx.state["account"]["positions"][0]["unrealisedPnl"] = value
                tx.state["risk_circuits"] = {
                    "testnet": {
                        "as_of": self.now,
                        "daily_loss_pct": 0,
                        "weekly_loss_pct": 0,
                        "drawdown_pct": 0,
                    }
                }

            await self.store.update(set_mtm)
            self.assertFalse(await self.manager.submit(op, self.sizing, 0))
        self.assertEqual(len(self.calls("submit")), 1)

    async def test_context_complete_current_and_unexpired_required(self):
        original = (await self.store.read())["context"]
        for context in (
            {},
            {**original, "data_complete": False},
            {**original, "as_of": self.now - 21601},
            {**original, "as_of": self.now + 1},
            {**original, "expires_at": self.now},
        ):
            await self.store.update(lambda tx: tx.state.update(context=context))
            self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        self.assertEqual(self.calls("submit"), [])

    async def test_risk_reference_equity_persisted_once_and_mtm_preserved(self):
        await self.opened()
        self.client.pos[0]["unrealisedPnl"] = "-12.5"
        with patch.object(self.client, "equity", return_value=20000):
            snapshot = await self.reconcile()
        state = await self.store.read()
        self.assertEqual(state["risk_reference_equity"], 10000)
        self.assertEqual(state["risk_reference_equities"]["testnet"], 10000)
        self.assertEqual(snapshot["equity"], 20000)
        self.assertEqual(snapshot["positions"][0]["unrealisedPnl"], "-12.5")

    async def test_partial_entry_expires_cancel_and_verify_keep_filled_position(self):
        await self.opened(qty=1)
        self.now = self.op.expires_at + 1
        await self.reconcile()
        r = await self.record()
        self.assertEqual(r["status"], "OPEN")
        self.assertEqual(r["remaining_qty"], 1)
        self.assertEqual(r["cancel_intent"]["status"], "VERIFIED")
        self.assertEqual(len(self.calls("cancel")), 1)
        await self.reconcile()
        self.assertEqual(len(self.calls("cancel")), 1)

    async def test_cancel_ack_without_terminal_confirmation_blocks(self):
        await self.opened(qty=1)
        self.client.cancel_confirms = False
        self.now = self.op.expires_at + 1
        snapshot = await self.reconcile()
        self.assertTrue(
            any("cancellation unconfirmed" in b for b in snapshot["blockers"])
        )
        self.assertEqual(
            (await self.record())["exchange_order"]["orderStatus"], "PartiallyFilled"
        )
        await self.reconcile()
        self.assertEqual(len(self.calls("cancel")), 1)

    async def test_unfilled_cancelled_entry_is_not_closed_trade(self):
        await self.submit()
        self.now = self.op.expires_at + 1
        await self.reconcile()
        r = await self.record()
        self.assertEqual(r["status"], "CANCELLED")
        self.assertNotIn("net_pnl", r)

    async def test_tp1_fill_recognized_after_price_falls_and_be_only_once(self):
        await self.opened()
        self.client.price = 121
        self.client.reduce_fraction = 0
        await self.reconcile()
        child = (await self.record())["children"]["tp1"]
        self.assertFalse((await self.record())["tp1_done"])
        self.now += 1
        self.client.fill(child["link"], 1, 121)
        self.client.price = 110
        await self.reconcile()
        r = await self.record()
        self.assertTrue(r["tp1_done"])
        self.assertEqual(r["desired_stop"], 100)
        self.assertEqual(len(self.calls("protect")), 1)
        self.manager = ExecutionManager(self.client, self.store, "testnet")
        await self.reconcile()
        self.assertEqual(len(self.calls("reduce")), 1)
        self.assertEqual(len(self.calls("protect")), 1)
        self.assertIn("tp1", (await self.record())["children"])

    async def test_tp1_ambiguous_child_no_duplicate_post(self):
        await self.opened()
        self.client.price = 121
        self.client.reduce_error = TimeoutError()
        self.client.reduce_fraction = 0
        await self.reconcile()
        child = (await self.record())["children"]["tp1"]
        self.client.hidden.add(child["link"])
        self.client.price = 110
        for _ in range(3):
            await self.reconcile()
        self.assertEqual(len(self.calls("reduce")), 1)
        self.assertFalse((await self.record())["tp1_done"])

    async def test_partial_tp1_ioc_uses_verified_residual_new_id(self):
        await self.opened()
        self.client.price = 121
        self.client.reduce_fraction = 0.5
        await self.reconcile()
        self.client.price = 110
        self.client.reduce_fraction = 1
        await self.reconcile()
        await self.reconcile()
        calls = self.calls("reduce")
        self.assertEqual([c[2] for c in calls], [1, 0.5])
        self.assertNotEqual(calls[0][1], calls[1][1])
        self.assertTrue((await self.record())["tp1_done"])
        self.assertEqual((await self.record())["remaining_qty"], 1)

    async def test_single_lot_does_not_liquidate_at_tp1(self):
        self.sizing.update(qty=0.1, risk_cash=1.05, notional=10)
        await self.opened(qty=0.1)
        self.client.price = 121
        await self.reconcile()
        r = await self.record()
        self.assertTrue(r["tp1_single_lot"])
        self.assertEqual(self.calls("reduce"), [])
        self.assertEqual(r["desired_stop"], 100)
        self.assertNotIn("tp1_filled_at", r)

    async def test_be_crossed_after_confirmed_tp1_requests_emergency(self):
        await self.opened()
        self.client.price = 121
        await self.reconcile()
        self.client.price = 99
        await self.reconcile()
        self.assertTrue((await self.record())["tp1_done"])
        self.assertIn("emergency", (await self.record())["children"])
        self.assertEqual(len(self.calls("reduce")), 2)

    async def test_short_tp1_breakeven_correct_side_and_child_buy(self):
        self.op = replace(
            self.op,
            side="Sell",
            setup="2S",
            stop=110,
            target1=80,
            target2=60,
            invalidation=108,
            confirmation=101,
        )
        await self.approve(self.op)
        self.sizing.update(stop=110, target1=80, target2=60)
        self.client.price = 95
        await self.opened()
        self.client.price = 79
        await self.reconcile()
        child = (await self.record())["children"]["tp1"]
        self.assertEqual(self.client.rows[child["link"]]["side"], "Buy")
        self.client.price = 90
        await self.reconcile()
        self.assertTrue((await self.record())["tp1_done"])
        self.assertEqual(self.client.pos[0]["stopLoss"], "100.0")
        self.assertEqual(self.calls("protect")[-1][1:], (100, 60))

    async def test_looser_stop_never_verified_by_nonzero(self):
        await self.opened()
        self.client.pos[0]["stopLoss"] = "80"
        self.client.protect_works = False
        await self.reconcile()
        r = await self.record()
        self.assertIn("emergency", r["children"])
        self.assertTrue(r["protection_pending"])

    async def test_wrong_side_stop_requires_repair(self):
        await self.opened()
        self.client.pos[0]["stopLoss"] = "120"
        await self.reconcile()
        self.assertEqual(self.calls("protect")[-1][1], 90)
        self.assertFalse((await self.record())["protection_pending"])

    async def test_correct_stop_but_wrong_target_repaired_without_widening(self):
        await self.opened()
        self.client.pos[0].update(stopLoss="95", takeProfit="150")
        await self.reconcile()
        self.assertEqual(self.calls("protect")[-1][1:], (95, 140))
        self.assertEqual((await self.record())["desired_stop"], 95)

    async def test_protection_read_failure_triggers_one_emergency_intent(self):
        await self.opened()
        self.client.pos[0]["stopLoss"] = "0"
        self.client.protect_works = False
        self.client.protection_read_error = True
        await self.reconcile()
        self.assertIn("emergency", (await self.record())["children"])
        self.assertEqual(len(self.calls("reduce")), 1)

    async def test_foreign_same_symbol_exit_cannot_be_assigned(self):
        await self.opened()
        self.client.rows["foreign"] = {
            "orderId": "foreign-id",
            "orderLinkId": "foreign",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "positionIdx": 0,
            "qty": "2",
            "leavesQty": "2",
            "cumExecQty": "0",
            "orderStatus": "New",
            "reduceOnly": True,
        }
        self.now += 1
        self.client.fill("foreign", 2, 110)
        await self.reconcile()
        r = await self.record()
        self.assertNotEqual(r["status"], "CLOSED")
        self.assertNotIn("net_pnl", r)
        self.assertFalse(r["ownership_verified"])
        self.assertEqual(self.calls("reduce"), [])

    async def test_same_symbol_size_matching_foreign_entry_is_not_owned(self):
        await self.opened()
        foreign = copy.deepcopy(self.client.fills[0])
        foreign.update(
            execId="foreign", orderId="foreign-id", orderLinkId="", execQty=".1"
        )
        self.client.fills.append(foreign)
        self.client.pos[0]["stopLoss"] = "0"
        await self.reconcile()
        self.assertFalse((await self.record())["ownership_verified"])
        self.assertEqual(self.calls("protect"), [])
        self.assertEqual(self.calls("reduce"), [])

    async def test_all_entry_fills_required_before_adopting_position(self):
        await self.opened()
        self.client.fills = []
        self.client.pos[0]["stopLoss"] = "0"
        # Delete stored audit to simulate missing initial evidence, not an API page overlap.
        await self.store.update(
            lambda tx: tx.state["orders"][self.link].update(executions={})
        )
        await self.reconcile()
        self.assertFalse((await self.record())["ownership_verified"])
        self.assertEqual(self.calls("protect"), [])

    async def test_exchange_order_must_match_persisted_entry_intent(self):
        await self.submit()
        self.client.rows[self.link]["qty"] = "20"
        snapshot = await self.reconcile()
        self.assertTrue(
            any("differs from persisted intent" in s for s in snapshot["blockers"])
        )
        self.assertEqual(self.calls("reduce"), [])

    async def test_hedged_or_foreign_position_cannot_be_reprotected(self):
        await self.opened()
        self.client.pos[0].update(positionIdx=1, stopLoss="0")
        await self.reconcile()
        self.assertFalse((await self.record())["ownership_verified"])
        self.assertEqual(self.calls("protect"), [])
        self.assertEqual(self.calls("reduce"), [])

    async def test_reconciliation_never_exposes_client_exception_messages(self):
        await self.opened()
        with patch.object(
            self.client, "order", side_effect=ValueError("api-secret-sensitive")
        ):
            snapshot = await self.reconcile()
        self.assertNotIn("api-secret-sensitive", str(snapshot))
        self.assertNotIn("api-secret-sensitive", str(await self.record()))

    def anonymous_protection(self, key="anonymous", kind="StopLoss", qty=2):
        row = {
            "orderId": "venue-" + key,
            "orderLinkId": "",
            "symbol": self.op.symbol,
            "side": "Sell" if self.op.side == "Buy" else "Buy",
            "positionIdx": 0,
            "stopOrderType": kind,
            "createType": (
                "CreateByStopLoss" if kind == "StopLoss" else "CreateByTakeProfit"
            ),
            "reduceOnly": True,
            "closeOnTrigger": True,
            "tpslMode": "Full",
            "orderType": "Market",
            "qty": str(qty),
            "leavesQty": str(qty),
            "cumExecQty": "0",
            "orderStatus": "Untriggered",
            "createdTime": str(int(self.now * 1000)),
            "updatedTime": str(int(self.now * 1000)),
        }
        self.client.rows[key] = row
        return row

    async def test_anonymous_full_sl_before_first_reconcile_uses_actual_order_id(self):
        await self.submit()
        self.now += 1
        self.client.fill(self.link, 2)
        row = self.anonymous_protection()
        self.now += 1
        self.client.fill("anonymous", 2, 90)
        self.manager = ExecutionManager(self.client, self.store, "testnet")
        self.assertEqual((await self.reconcile())["blockers"], [])
        order = await self.record()
        self.assertEqual(order["status"], "CLOSED")
        self.assertEqual(order["exit_reason"], "SL")
        self.assertAlmostEqual(order["net_pnl"], -20.04)
        self.assertEqual(order["protective_orders"][row["orderId"]], row)
        self.assertEqual(
            self.calls("order_by_id"), [("order_by_id", "BTCUSDT", row["orderId"])]
        )
        self.assertEqual(self.calls("reduce"), [])
        self.assertEqual(self.calls("protect"), [])

    async def test_anonymous_active_full_protection_not_foreign_then_tp2_is_attributed(
        self,
    ):
        await self.opened()
        row = self.anonymous_protection(kind="TakeProfit")
        self.assertEqual((await self.reconcile())["blockers"], [])
        self.assertIn(row["orderId"], (await self.record())["protective_orders"])
        self.client.fill("anonymous", 2, 140)
        self.assertEqual((await self.reconcile())["blockers"], [])
        order = await self.record()
        self.assertEqual(order["status"], "CLOSED")
        self.assertEqual(order["exit_reason"], "TP2")
        self.assertEqual(len(self.calls("order_by_id")), 2)

    async def test_anonymous_create_type_classifies_sl_even_at_target_price(self):
        await self.opened()
        row = self.anonymous_protection()
        row.pop("stopOrderType")
        self.now += 1
        self.client.fill("anonymous", 2, 140)
        await self.reconcile()
        self.assertEqual((await self.record())["exit_reason"], "SL")

    async def test_anonymous_order_evidence_mismatch_blocks_without_mutating_exposure(
        self,
    ):
        await self.opened()
        self.anonymous_protection()
        self.now += 1
        self.client.fill("anonymous", 2, 90)
        baseline = await self.store.read()
        venue = copy.deepcopy(self.client.rows)
        mutations = [
            {"orderId": "other"},
            {"symbol": "ETHUSDT"},
            {"side": "Buy"},
            {"positionIdx": 1},
            {"orderLinkId": "foreign-client"},
            {"parentOrderLinkId": "foreign-parent"},
            {"reduceOnly": False},
            {"reduceOnly": "true"},
            {"closeOnTrigger": False},
            {"tpslMode": "Partial"},
            {"orderType": "Limit"},
            {"qty": "3"},
            {"cumExecQty": "1"},
            {"leavesQty": "1"},
            {"qty": "nan"},
            {"createdTime": str(int(NOW * 1000) - 1)},
            {"updatedTime": str(int(NOW * 1000))},
            {"updatedTime": str(int((NOW + 1000) * 1000))},
            {"stopOrderType": "UNKNOWN", "createType": "UNKNOWN"},
            {"createType": "CreateByTakeProfit"},
            {"createType": "CreateByUser"},
        ]
        for changes in mutations:
            with self.subTest(changes=changes):
                await self.store.update(
                    lambda tx: tx.state.update(copy.deepcopy(baseline))
                )
                self.client.rows = copy.deepcopy(venue)
                self.client.rows["anonymous"].update(changes)
                self.assertTrue((await self.reconcile())["blockers"])
                order = await self.record()
                self.assertFalse(order["ownership_verified"])
                self.assertNotIn("net_pnl", order)
                self.assertEqual(order.get("protective_orders", {}), {})
        self.assertEqual(self.calls("reduce"), [])
        self.assertEqual(self.calls("protect"), [])

    async def test_anonymous_sl_cannot_hide_foreign_same_side_inventory(self):
        await self.opened()
        self.anonymous_protection()
        self.now += 1
        self.client.fill("anonymous", 2, 90)
        foreign = {
            **self.client.fills[0],
            "execId": "foreign-entry",
            "orderId": "foreign",
            "orderLinkId": "",
            "execQty": "1",
        }
        self.client.fills.append(foreign)
        await self.reconcile()
        self.assertFalse((await self.record())["ownership_verified"])
        self.assertEqual((await self.record()).get("protective_orders", {}), {})
        self.assertNotIn("net_pnl", await self.record())

    async def test_anonymous_exit_requires_actual_closed_size_and_causal_inventory(
        self,
    ):
        await self.opened()
        row = self.anonymous_protection()
        self.now += 1
        fill = self.client.fill("anonymous", 2, 90)
        base = await self.store.read()
        original_fill, original_row = copy.deepcopy(fill), copy.deepcopy(row)
        for mode in ("closedSize", "before_entry", "overlarge_order"):
            with self.subTest(mode=mode):
                await self.store.update(lambda tx: tx.state.update(copy.deepcopy(base)))
                fill.clear()
                fill.update(original_fill)
                row.clear()
                row.update(original_row)
                if mode == "closedSize":
                    fill["closedSize"] = "1"
                elif mode == "before_entry":
                    fill["execTime"] = str(int(NOW * 1000) + 500)
                    row.update(
                        createdTime=str(int(NOW * 1000)), updatedTime=fill["execTime"]
                    )
                else:
                    row.update(qty="3", orderStatus="PartiallyFilledCanceled")
                self.assertTrue((await self.reconcile())["blockers"])
                self.assertFalse((await self.record())["ownership_verified"])
                self.assertNotIn("net_pnl", await self.record())

    async def test_anonymous_order_lookup_delay_retries_reads_only_after_restart(self):
        await self.opened()
        row = self.anonymous_protection()
        self.now += 1
        self.client.fill("anonymous", 2, 90)
        self.client.hidden.add(row["orderId"])
        self.assertTrue((await self.reconcile())["blockers"])
        self.assertNotIn("net_pnl", await self.record())
        self.client.hidden.clear()
        self.manager = ExecutionManager(self.client, self.store, "testnet")
        await self.reconcile()
        self.assertEqual((await self.record())["status"], "CLOSED")
        self.assertEqual(len(self.calls("submit")), 1)
        self.assertEqual(self.calls("reduce"), [])

    async def test_anonymous_protection_needs_persisted_flat_origin_and_sole_intent(
        self,
    ):
        await self.opened()
        self.anonymous_protection()
        self.now += 1
        self.client.fill("anonymous", 2, 90)
        state = await self.store.read()
        await self.store.update(
            lambda tx: tx.state["orders"][self.link].pop("flat_verified_at")
        )
        self.assertTrue((await self.reconcile())["blockers"])
        self.assertFalse((await self.record())["ownership_verified"])
        await self.store.update(lambda tx: tx.state.update(copy.deepcopy(state)))
        other = {**copy.deepcopy(state["orders"][self.link]), "id": "other-intent"}
        await self.store.update(
            lambda tx: tx.state["orders"].update({"other-intent": other})
        )
        self.assertTrue((await self.reconcile())["blockers"])
        self.assertFalse((await self.record())["ownership_verified"])
        self.assertEqual(self.calls("order_by_id"), [])

    async def test_anonymous_partial_sl_and_later_residual_preserve_inventory(self):
        await self.opened()
        self.anonymous_protection()
        self.now += 1
        self.client.fill("anonymous", 1, 90, terminal="PartiallyFilledCanceled")
        await self.reconcile()
        self.assertEqual((await self.record())["status"], "OPEN")
        self.assertEqual((await self.record())["remaining_qty"], 1)
        self.manager = ExecutionManager(self.client, self.store, "testnet")
        self.anonymous_protection(key="residual", qty=1)
        self.now += 1
        self.client.fill("residual", 1, 90)
        await self.reconcile()
        self.assertEqual((await self.record())["status"], "CLOSED")
        self.assertEqual(len((await self.record())["protective_orders"]), 2)
        self.assertEqual(self.calls("reduce"), [])

    async def exit_with(self, kind="StopLoss", create_type=None, price=90):
        protective = {
            "orderId": "attached-exit",
            "orderLinkId": "exit-link",
            "symbol": "BTCUSDT",
            "side": "Sell",
            "positionIdx": 0,
            "parentOrderLinkId": self.link,
            "stopOrderType": kind,
            "createType": create_type,
            "reduceOnly": True,
            "qty": "2",
            "leavesQty": "2",
            "cumExecQty": "0",
            "orderStatus": "Untriggered",
        }
        self.client.external = [protective]
        await self.reconcile()  # Observe explicit parent association before the trigger.
        self.client.rows["exit-link"] = copy.deepcopy(protective)
        self.now += 1
        self.client.fill("exit-link", 2, price)
        self.client.external = []
        return await self.reconcile()

    async def test_exchange_sl_evidence_and_transaction_backed_net_pnl(self):
        await self.opened()
        await self.exit_with()
        r = await self.record()
        self.assertEqual(r["exit_reason"], "SL")
        self.assertEqual(r["status"], "CLOSED")
        self.assertTrue(r["funding_complete"])
        self.assertAlmostEqual(r["net_pnl"], -20.04)
        self.assertEqual(r["funding"], 0)
        self.assertTrue(self.calls("closed_pnl"))
        self.assertTrue(self.calls("transaction_log"))

    async def test_missing_exit_trigger_is_unknown_not_price_inferred(self):
        await self.opened()
        await self.exit_with(kind="UNKNOWN", price=140)
        self.assertEqual((await self.record())["exit_reason"], "UNKNOWN")

    async def test_explicit_manual_exit_is_not_labeled_tp_or_sl(self):
        await self.opened()
        await self.exit_with(kind="", create_type="CreateByUser", price=120)
        self.assertEqual((await self.record())["exit_reason"], "MANUAL")

    async def test_delayed_transaction_log_stays_closing_and_retries_reads(self):
        await self.opened()
        self.client.logs_enabled = False
        await self.exit_with(kind="TakeProfit", price=140)
        r = await self.record()
        self.assertEqual(r["status"], "CLOSING")
        self.assertEqual(r["exit_reason"], "TP2")
        self.assertFalse(r["funding_complete"])
        self.assertNotIn("net_pnl", r)
        self.client.logs_enabled = True
        self.manager = ExecutionManager(self.client, self.store, "testnet")
        await self.reconcile()
        self.assertEqual((await self.record())["status"], "CLOSED")
        self.assertEqual(len(self.calls("submit")), 1)

    async def test_crossed_funding_boundary_requires_actual_signed_settlement(self):
        await self.opened()
        boundary = (int(self.now) // (8 * 3600) + 1) * 8 * 3600
        self.now = boundary + 20
        self.client.logs_enabled = True
        await self.exit_with(kind="TakeProfit", price=140)
        self.assertEqual((await self.record())["status"], "CLOSING")
        self.client.settlements = [
            {
                "id": "funding-1",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "size": "2",
                "category": "linear",
                "currency": "USDT",
                "type": "SETTLEMENT",
                "transactionTime": str(boundary * 1000),
                "funding": "-1.25",
                "cashFlow": "0",
                "fee": "0",
                "change": "-1.25",
            }
        ]
        await self.reconcile()
        r = await self.record()
        self.assertTrue(r["funding_complete"])
        self.assertAlmostEqual(r["net_pnl"], 80 - 0.04 - 1.25)
        self.assertEqual(r["funding"], -1.25)

    async def test_missing_closed_pnl_quantity_blocks_accounting_completion(self):
        await self.opened()
        self.client.pnl_enabled = False
        await self.exit_with()
        self.assertEqual((await self.record())["status"], "CLOSING")
        self.client.pnl_enabled = True
        await self.reconcile()
        self.assertEqual((await self.record())["status"], "CLOSED")

    async def test_long_held_trade_uses_incremental_audit_not_truncated_history(self):
        await self.opened()
        for _ in range(5):
            self.now += 5 * ex.DAY
            await self.reconcile()
        self.assertEqual((await self.record())["status"], "OPEN")
        self.assertEqual(len((await self.record())["entry_fills"]), 1)
        self.assertNotIn("history_gap", await self.record())

    async def test_favorable_four_hour_zone_break_suppresses_twenty_close_timeout(self):
        await self.opened()
        opened = (await self.record())["opened_at"]
        self.now = opened + 21 * ex.DAY
        daily = [
            Candle(t * 1000, 100, 115, 96, 101)
            for t in range(
                int(opened // ex.DAY) * ex.DAY, int(self.now // ex.DAY) * ex.DAY, ex.DAY
            )
        ]
        execution = [
            Candle(t * 1000, 100, 115, 96, 111)
            for t in range(
                int(opened // ex.H4) * ex.H4, int(self.now // ex.H4) * ex.H4, ex.H4
            )
        ]
        await self.manager.manage_structure("BTCUSDT", daily, execution)
        r = await self.record()
        self.assertTrue(r["favorable_4h_close"])
        self.assertEqual(r["daily_bars_held"], 21)
        self.assertNotIn("exit_request", r)
        self.assertNotIn("ten_day_reviewed", r)

    async def test_elapsed_days_without_closed_history_do_not_create_timeout(self):
        await self.opened()
        self.now += 21 * ex.DAY
        await self.manager.manage_structure("BTCUSDT", [], [])
        self.assertNotIn("exit_request", await self.record())
        self.assertNotIn("ten_day_reviewed", await self.record())

    async def test_unrecoverable_history_gap_blocks_without_adoption(self):
        await self.opened()
        self.now += 8 * ex.DAY
        await self.reconcile()
        self.assertTrue((await self.record())["history_gap"])
        self.assertFalse((await self.record())["ownership_verified"])
        self.assertEqual(self.calls("reduce"), [])

    async def test_missing_isolated_equity_is_explicit_entry_blocker(self):
        self.client.equity_error = True
        snapshot = await self.reconcile()
        self.assertIsNone(snapshot["equity"])
        self.assertTrue(
            any("Sizing equity unavailable" in b for b in snapshot["blockers"])
        )
        self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))

    async def test_failed_reconcile_invalidates_previously_fresh_snapshot(self):
        with patch.object(
            self.client, "positions", side_effect=RuntimeError("read failed")
        ):
            with self.assertRaises(RuntimeError):
                await self.manager.reconcile(self.instruments)
        self.assertEqual((await self.store.read())["account"]["as_of"], 0)
        self.assertFalse(await self.manager.submit(self.op, self.sizing, 0))
        self.assertEqual(self.calls("submit"), [])

    async def test_daily_close_invalidation_ignores_wick_and_open_bar(self):
        await self.opened()
        opened = (await self.record())["opened_at"]
        self.now = (int(opened // ex.DAY) + 2) * ex.DAY
        bar = Candle(int((self.now - ex.DAY) * 1000), 100, 110, 91, 95)
        await self.manager.manage_structure("BTCUSDT", [bar], [])
        self.assertNotIn("exit_request", await self.record())
        open_bar = replace(bar, open_time=int(self.now * 1000), close=91)
        await self.manager.manage_structure("BTCUSDT", [bar, open_bar], [])
        self.assertNotIn("exit_request", await self.record())
        self.now += ex.DAY
        await self.manager.manage_structure("BTCUSDT", [bar, open_bar], [])
        self.assertEqual(
            (await self.record())["exit_request"]["action"], "daily_invalidation"
        )
        self.assertEqual(self.calls("reduce"), [])
        await self.reconcile()
        self.assertIn("daily_invalidation", (await self.record())["children"])

    async def test_ten_day_review_once_twenty_day_timeout_preserve_opened_at(self):
        await self.opened()
        opened = (await self.record())["opened_at"]

        def history():
            daily = [
                Candle(t * 1000, 100, 108, 96, 101)
                for t in range(
                    int(opened // ex.DAY) * ex.DAY,
                    int(self.now // ex.DAY) * ex.DAY,
                    ex.DAY,
                )
            ]
            execution = [
                Candle(t * 1000, 100, 108, 96, 101)
                for t in range(
                    int(opened // ex.H4) * ex.H4, int(self.now // ex.H4) * ex.H4, ex.H4
                )
            ]
            return daily, execution

        self.now = opened + 10 * ex.DAY
        await self.manager.manage_structure("BTCUSDT", *history())
        stamp = (await self.record())["ten_day_reviewed"]
        self.manager = ExecutionManager(self.client, self.store, "testnet")
        self.now += ex.DAY
        await self.manager.manage_structure("BTCUSDT", *history())
        self.assertEqual((await self.record())["ten_day_reviewed"], stamp)
        self.now = opened + 20 * ex.DAY
        await self.manager.manage_structure("BTCUSDT", *history())
        r = await self.record()
        self.assertEqual(r["opened_at"], opened)
        self.assertEqual(r["daily_bars_held"], 20)
        self.assertEqual(r["exit_request"]["action"], "timeout")

    async def test_trailing_requires_midpoint_confirmed_later_pivot_and_ratchets(self):
        await self.opened()
        opened = (await self.record())["opened_at"]
        await self.store.update(
            lambda tx: tx.state["orders"][self.link].update(
                tp1_done=True, tp1_filled_at=opened, desired_stop=100
            )
        )
        self.now = (int(self.now // ex.H4) + 1) * ex.H4
        first = Candle(int((self.now - ex.H4) * 1000), 100, 120, 98, 110)
        await self.manager.manage_structure("BTCUSDT", [], [first])
        self.now += ex.H4
        bar = Candle(int((self.now - ex.H4) * 1000), 120, 132, 119, 131)
        old = Pivot(
            "old", "low", 115, int((self.now - ex.H4) * 1000), self.now - 1, 0, 2
        )
        later = Pivot("later", "low", 116, int(self.now * 1000), self.now + ex.H4, 1, 2)
        with patch.object(ex, "confirmed_zigzag", return_value=[old]):
            await self.manager.manage_structure("BTCUSDT", [], [first, bar])
        self.assertEqual((await self.record())["desired_stop"], 100)
        self.now += ex.H4
        next_bar = Candle(int((self.now - ex.H4) * 1000), 131, 133, 116, 130)
        with patch.object(ex, "confirmed_zigzag", return_value=[old, later]):
            await self.manager.manage_structure("BTCUSDT", [], [first, bar, next_bar])
        self.assertAlmostEqual((await self.record())["desired_stop"], 115.8)
        self.assertEqual((await self.record())["trail_pivot_id"], "later")
        lower = replace(later, id="lower", price=110, available_at=self.now + ex.H4)
        self.now += ex.H4
        with patch.object(ex, "confirmed_zigzag", return_value=[old, later, lower]):
            await self.manager.manage_structure("BTCUSDT", [], [bar, next_bar])
        self.assertAlmostEqual((await self.record())["desired_stop"], 115.8)

    async def test_sqlite_restart_preserves_consumed_intent(self):
        with tempfile.TemporaryDirectory(prefix="apex-exec-test-") as tmp:
            store = Store(sqlite_path=tmp + "/execution.sqlite3")
            await store.initialize()
            await store.lease()
            state = await self.store.read()
            await store.update(lambda tx: tx.state.update(state))
            client = FakeBybit(store)
            client.entry_error = TimeoutError()
            manager = ExecutionManager(client, store, "testnet")
            manager.instruments = self.instruments
            await manager.submit(self.op, self.sizing, 0)
            await store.close()
            store = Store(sqlite_path=tmp + "/execution.sqlite3")
            await store.initialize()
            await store.lease()
            try:
                client.store = store
                manager = ExecutionManager(client, store, "testnet")
                manager.instruments = self.instruments
                self.assertFalse(await manager.submit(self.op, self.sizing, 0))
                self.assertEqual(len([c for c in client.calls if c[0] == "submit"]), 1)
            finally:
                await store.close()


if __name__ == "__main__":
    unittest.main()
