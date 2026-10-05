"""Run against CI's disposable Postgres, never DATABASE_URL or a production DB."""

import asyncio
import os
import uuid

import pytest

from apex_bot.storage import NotLeader, Store


def test_postgres_lease_transactions_and_durable_outbox():
    dsn = os.getenv("TEST_APEX_DATABASE_URL")
    if not dsn:
        pytest.skip("Disposable PostgreSQL not configured; SQLite tests are separate")
    import psycopg2
    from psycopg2 import sql
    from psycopg2.pool import ThreadedConnectionPool

    schema = "apex_ci_" + uuid.uuid4().hex
    conn = psycopg2.connect(dsn, connect_timeout=5)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

    def store():
        result = Store(database_url=dsn)
        result.pool = ThreadedConnectionPool(
            1, 4, dsn, options=f"-c search_path={schema} -c statement_timeout=5000"
        )
        return result

    async def scenario():
        first, second = store(), store()
        try:
            await first.initialize()
            await second.initialize()
            assert await first.lease()
            assert not await second.lease()
            with pytest.raises(NotLeader):
                await second.update(lambda tx: tx.state.update(forbidden=True))

            def failed(tx):
                tx.state["broken"] = True
                tx.event("rollback", "test", {}, "must not deliver")
                raise ValueError("rollback")

            with pytest.raises(ValueError):
                await first.update(failed)
            assert "broken" not in await first.read()
            assert (await first.outbox_health())["pending"] == 0

            def save(tx):
                tx.state["durable"] = True
                tx.event("once", "test", {}, "durable notification")

            await first.update(save)
            await first.update(save)
            assert (await first.outbox_health())["pending"] == 1
            await first.close()
            first = None
            assert await second.lease()
            assert (await second.read())["durable"]
            pending = await second.pending_notifications()
            assert len(pending) == 1
            await second.notification_result("once", message_id="123")
            assert (await second.outbox_health())["pending"] == 0
        finally:
            if first:
                await first.close()
            await second.close()

    try:
        asyncio.run(scenario())
    finally:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )
        conn.close()
