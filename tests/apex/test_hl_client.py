"""Hyperliquid client: request weight and rate-limit behaviour (no network)."""

import asyncio

import pytest
from hyperliquid.utils.error import ClientError

from apex_bot import hl_client
from apex_bot.hl_client import COOLDOWN_MAX, COOLDOWN_START, ORDERS_TTL, HLError, HyperliquidClient


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeInfo:
    """Counts the weight-20 frontendOpenOrders calls; everything else is weight 2."""

    def __init__(self):
        self.order_calls = 0
        self.meta_calls = 0
        self.positions = []
        self.listed = {"": ["BTC", "ETH"], "xyz": ["xyz:AMD"]}
        self.name_to_coin = {"BTC": "BTC", "ETH": "ETH", "xyz:AMD": "xyz:AMD"}
        self.fail_next = 0

    def _maybe_429(self):
        if self.fail_next:
            self.fail_next -= 1
            raise ClientError(429, None, "rate limited", {})

    def frontend_open_orders(self, address, dex=""):
        self._maybe_429()
        self.order_calls += 1
        return [{"coin": "BTC", "oid": 1, "sz": "0.1", "origSz": "0.1", "limitPx": "74250", "side": "A"}] if dex == "" else []

    def user_state(self, address, dex=""):
        self._maybe_429()
        rows = [{"position": {"coin": c, "szi": str(s)}} for c, s in self.positions] if dex == "" else []
        return {"marginSummary": {"accountValue": "1000", "totalMarginUsed": "0"}, "withdrawable": "500", "assetPositions": rows}

    def query_user_abstraction_state(self, address):
        return "standard"

    def meta(self, dex=""):
        self.meta_calls += 1
        return {"universe": [{"name": n, "szDecimals": 2, "maxLeverage": 20} for n in self.listed[dex]]}


def make_client():
    client = HyperliquidClient("0x" + "ab" * 20, "0x" + "cd" * 32)
    client._exchange = type("Ex", (), {"bulk_orders": lambda self, w, grouping="na": {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 9}}]}}}, "cancel": lambda self, c, o: {"status": "ok", "response": {"data": {"statuses": ["success"]}}}})()
    client._info = FakeInfo()
    client.dexs = ("", "xyz")
    clock = Clock()
    client._clock = clock
    return client, clock


def test_open_orders_are_reused_until_an_action_a_position_change_or_the_ttl():
    async def run():
        client, clock = make_client()
        info = client._info
        await client.account()
        await client.open_orders()
        assert info.order_calls == 2  # one per dex
        for _ in range(4):  # four 4-second desk ticks
            clock.now += 4
            await client.account()
            orders = await client.open_orders()
        assert info.order_calls == 2 and orders[0]["oid"] == 1
        await client.place([{"coin": "BTC", "is_buy": False, "size": 0.1, "limit_px": 75000.0, "kind": "limit"}])
        await client.open_orders()
        assert info.order_calls == 4  # our own order: fresh snapshot
        info.positions = [("BTC", -0.1)]  # the entry filled
        await client.account()
        await client.open_orders()
        assert info.order_calls == 6
        clock.now += ORDERS_TTL + 0.1
        await client.open_orders()
        assert info.order_calls == 8
        await client.cancel("BTC", 1)
        await client.open_orders()
        assert info.order_calls == 10

    asyncio.run(run())


def test_rate_limit_pauses_reads_instead_of_hammering(monkeypatch):
    monkeypatch.setattr(hl_client, "_backoff", lambda exc, attempt: 0)

    async def run():
        client, clock = make_client()
        info = client._info
        info.fail_next = 1
        with pytest.raises(HLError, match="ClientError 429"):
            await client.open_orders()
        assert info.order_calls == 0
        # Paused: no request leaves at all, and the message says so.
        with pytest.raises(HLError, match="reads paused for 15 s"):
            await client.account()
        clock.now += COOLDOWN_START + 0.1
        info.fail_next = 1  # still limited: the pause doubles
        with pytest.raises(HLError):
            await client.account()
        assert client._cool_for == 2 * COOLDOWN_START
        clock.now += 2 * COOLDOWN_START + 0.1
        await client.account()  # recovered
        orders = await client.open_orders()
        assert orders and client._cool_for == 0.0
        for _ in range(10):
            client._cool_for = client._cool_for * 2 or COOLDOWN_START
        assert min(COOLDOWN_MAX, client._cool_for) == COOLDOWN_MAX

    asyncio.run(run())


def test_exchange_actions_are_not_blocked_by_the_read_pause():
    async def run():
        client, clock = make_client()
        client._cool_until = clock.now + 60
        statuses = await client.place([{"coin": "BTC", "is_buy": True, "size": 0.1, "limit_px": 70000.0, "kind": "limit"}])
        assert statuses[0]["oid"] == 9  # a stop or close still goes out

    asyncio.run(run())


def test_assets_rebuild_the_sdk_only_for_a_new_listing():
    async def run():
        client, clock = make_client()
        rebuilt = []

        async def connect(force=False):
            if force:
                rebuilt.append(1)
                client._info.name_to_coin.update({n: n for names in client._info.listed.values() for n in names})

        client.connect = connect
        assets = await client.assets()
        assert set(assets) == {"BTC", "ETH", "xyz:AMD"} and assets["xyz:AMD"]["dex"] == "xyz"
        assert rebuilt == []  # the hourly refresh no longer rebuilds (~200 weight)
        client._info.listed["xyz"].append("xyz:NEW")
        assets = await client.assets()
        assert "xyz:NEW" in assets and rebuilt == [1]

    asyncio.run(run())
