"""Observer acceptance on disposable CI Postgres; never use DATABASE_URL.

Only TEST_APEX_DATABASE_URL enables this test. Market prices/funding are local
fixtures; the sole connection is to that explicitly configured test database.
Each case owns and drops a random schema, including its write-blocking triggers.
"""

import asyncio
import json
import os
from types import SimpleNamespace
import uuid

import pytest

from apex_bot.shadow_observer import ShadowObserver, TABLE, VERSION
from apex_bot.storage import NotLeader, Store

from .test_shadow_observer import Cache, Client, NOW, consider, opportunity


@pytest.fixture
def postgres_observer_schema():
    dsn = os.getenv("TEST_APEX_DATABASE_URL")
    if not dsn:
        pytest.skip(
            "Disposable TEST_APEX_DATABASE_URL not configured; no PostgreSQL test"
        )

    import psycopg2
    from psycopg2 import sql
    from psycopg2.pool import ThreadedConnectionPool

    schema = "apex_observer_ci_" + uuid.uuid4().hex
    admin = psycopg2.connect(dsn, connect_timeout=5)
    admin.autocommit = True
    stores = []
    try:
        with admin.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

        def make_store():
            store = Store(database_url=dsn)
            store.pool = ThreadedConnectionPool(
                1,
                4,
                dsn,
                connect_timeout=5,
                options=f"-c search_path={schema} -c statement_timeout=5000",
            )
            stores.append(store)
            return store

        def rows():
            with admin.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "SELECT id,payload,trade,status,revision,done,outcome_complete,net_r,error "
                        "FROM {}.{} ORDER BY id"
                    ).format(sql.Identifier(schema), sql.Identifier(TABLE))
                )
                return [
                    dict(
                        id=row[0],
                        payload=row[1],
                        trade=json.loads(row[2]),
                        status=row[3],
                        revision=row[4],
                        done=row[5],
                        outcome_complete=row[6],
                        net_r=row[7],
                        error=row[8],
                    )
                    for row in cur.fetchall()
                ]

        def shared_rows():
            result = {}
            with admin.cursor() as cur:
                for table in ("apex_state", "apex_events", "apex_outbox"):
                    cur.execute(
                        sql.SQL("SELECT xmin::text,* FROM {}.{} ORDER BY 2").format(
                            sql.Identifier(schema),
                            sql.Identifier(table),
                        )
                    )
                    result[table] = cur.fetchall()
            return result

        def block_shared_writes():
            # Postgres, not a mocked Store method, rejects INSERT/UPDATE/DELETE
            # even when the attempted write would leave equivalent JSON behind.
            with admin.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                    CREATE FUNCTION {}.reject_observer_shared_write() RETURNS trigger
                    LANGUAGE plpgsql AS $$
                    BEGIN
                        RAISE EXCEPTION 'Observer attempted shared-ledger write to %', TG_TABLE_NAME;
                    END;
                    $$
                """
                    ).format(sql.Identifier(schema))
                )
                for table in ("apex_state", "apex_events", "apex_outbox"):
                    cur.execute(
                        sql.SQL(
                            """
                        CREATE TRIGGER reject_observer_shared_write
                        BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE ON {}.{}
                        FOR EACH STATEMENT EXECUTE FUNCTION {}.reject_observer_shared_write()
                    """
                        ).format(
                            sql.Identifier(schema),
                            sql.Identifier(table),
                            sql.Identifier(schema),
                        )
                    )

        yield SimpleNamespace(
            admin=admin,
            schema=schema,
            make_store=make_store,
            rows=rows,
            shared_rows=shared_rows,
            block_shared_writes=block_shared_writes,
        )
    finally:
        # Close every pool before dropping only this test's schema. No public
        # schema tables, production URL, exchange clients or Telegram are used.
        for store in stores:
            if store.pool and not store.pool.closed:
                store.pool.closeall()
        try:
            with admin.cursor() as cur:
                cur.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(schema),
                    )
                )
        finally:
            admin.close()


@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_postgres_observer_capture_dedup_restart_save_summary_and_shared_isolation(
    postgres_observer_schema,
    side,
):
    pg = postgres_observer_schema

    async def scenario():
        first, second = pg.make_store(), pg.make_store()
        await first.initialize()
        await second.initialize()
        assert first.sqlite is None and second.sqlite is None
        assert await first.lease(ttl=120)
        assert not await second.lease()
        client = Client()
        observer = ShadowObserver(first, client, Cache())
        follower = ShadowObserver(second, Client(), Cache())
        await observer.initialize()
        await observer.initialize()  # Postgres DDL and indexes are idempotent.
        with pytest.raises(NotLeader):
            await follower.initialize()

        with pg.admin.cursor() as cur:
            cur.execute(
                "SELECT column_name,data_type FROM information_schema.columns "
                "WHERE table_schema=%s AND table_name=%s",
                (pg.schema, TABLE),
            )
            columns = dict(cur.fetchall())
            assert columns["payload"] == columns["trade"] == "text"
            assert columns["net_r"] == "double precision"
            assert columns["revision"] == columns["done"] == "integer"
            cur.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname=%s AND tablename=%s",
                (pg.schema, TABLE),
            )
            assert {row[0] for row in cur.fetchall()} == {
                TABLE + "_pkey",
                TABLE + "_due",
                TABLE + "_symbol",
            }

        def seed(tx):
            tx.state["trades"]["existing-live"] = {"symbol": "TESTUSDT", "qty": 3}
            tx.state["orders"]["existing-order"] = {"status": "OPEN"}
            tx.state["opportunities"]["existing-plan"] = {"state": "READY"}
            tx.event("existing-event", "acceptance", {"keep": True}, "existing notice")

        await first.update(seed)
        before_state = await first.read()
        before_shared = pg.shared_rows()
        pg.block_shared_writes()
        empty = await observer.summary()
        assert empty["total"] == 0 and empty["wr"] is None and empty["mean_r"] is None

        op = opportunity("pg-" + side, side=side)
        results = await asyncio.gather(*(consider(observer, op) for _ in range(8)))
        assert sum(results) == 1  # Real ON CONFLICT(id), not mocked insertion.
        assert not await consider(observer, op, equity=11000)
        # Independent rejected plans may coexist for the same symbol.
        assert await consider(observer, opportunity("second-" + side, side=side))
        initial = pg.rows()
        assert len(initial) == 2
        immutable = {row["id"]: row["payload"] for row in initial}
        for row in initial:
            captured = json.loads(row["payload"])
            assert captured["experiment"] == VERSION
            assert captured["equity_reference"] == 10000
            assert captured["independent_assessment"]["allowed"]
            assert captured["original_price_r"] > 0
            assert row["status"] == "PENDING" and row["revision"] == 0
            assert row["done"] == row["outcome_complete"] == 0
            assert row["net_r"] is None
        assert (await observer.summary())["pending"] == 2
        with pytest.raises(NotLeader):
            await consider(follower, opportunity("fenced", side=side))

        stale_claims = await observer._claim(NOW)
        assert len(stale_claims) == 2
        client.funding_failure = True
        assert (await observer.resolve(NOW + 360))["processed"] == 2
        provisional = await observer.summary()
        assert provisional["closed"] == provisional["incomplete_closed"] == 2
        assert provisional["complete_closed"] == provisional["losses"] == 0
        assert provisional["wr"] is None and provisional["net_r"] == 0
        assert all(row["revision"] == 1 and row["error"] for row in pg.rows())
        assert pg.shared_rows() == before_shared

        # Transfer the real SQL lease to a different Store/pool and recover the
        # saved close cursor. The market client never contacts an exchange.
        await first.close()
        assert await second.lease(ttl=120)
        await follower.initialize()
        assert not await consider(follower, op, equity=12000)
        follower.client.rates = [
            {
                "symbol": "TESTUSDT",
                "fundingRateTimestamp": str((NOW + 270) * 1000),
                "fundingRate": "0.001",
            }
        ]
        assert (await follower.resolve(NOW + 390))["processed"] == 2
        final = pg.rows()
        assert (
            not follower.client.calls
        )  # Recover funding without replaying closed prices.
        for row in final:
            assert row["payload"] == immutable[row["id"]]
            assert row["revision"] == 2 and row["status"] == "CLOSED"
            assert row["done"] == row["outcome_complete"] == 1 and row["error"] is None
            trade = row["trade"]
            assert trade["observer"] == VERSION
            assert trade["funding_complete"] and trade["management_complete"]
            sign = 1 if side == "Buy" else -1
            assert trade["funding"] == pytest.approx(
                -sign * trade["qty"] * trade["entry"] * 0.001
            )
            assert row["net_r"] == pytest.approx(
                trade["net_pnl"] / json.loads(row["payload"])["original_price_r"]
            )

        # Compare-and-swap revision prevents an old claimant from overwriting
        # a persisted completed outcome after lease handoff.
        assert not await follower._save(
            stale_claims[0], stale_claims[0]["trade"], NOW + 420
        )
        assert pg.rows() == final
        summary = await follower.summary()
        assert summary["total"] == summary["closed"] == summary["complete_closed"] == 2
        assert summary["losses"] == 2 and summary["wins"] == 0 and summary["wr"] == 0
        assert summary["unfinished"] == summary["data_issues"] == 0
        assert summary["net_r"] == pytest.approx(sum(row["net_r"] for row in final))
        assert summary["mean_r"] == pytest.approx(summary["net_r"] / 2)
        assert (await follower.resolve(NOW + 420))["processed"] == 0
        assert await follower.summary() == summary
        assert await second.read() == before_state
        assert pg.shared_rows() == before_shared
        await second.close()

    asyncio.run(scenario())
