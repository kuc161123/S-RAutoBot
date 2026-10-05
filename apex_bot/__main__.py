"""Railway entrypoint and credential-free configuration inspection."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time

import aiohttp
from aiohttp import web

from .config import Config, ConfigError
from .storage import Store


async def serve(config):
    from .runtime import Runtime

    store = Store(config.database_url, config.sqlite_path)
    timeout = aiohttp.ClientTimeout(total=50, connect=10)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        runtime = Runtime(config, store, session)
        preflight_ready = False
        app = web.Application(client_max_size=1024)

        async def health(_):
            # Liveness is intentionally separate from leadership: Railway starts
            # replacements before terminating old workers. Never disclose account
            # balances, opportunities, settings, environment or tokens publicly.
            ready = preflight_ready and not runtime.stopping.is_set()
            return web.json_response(
                {"service": "apex", "process": "running" if ready else "starting"},
                status=200 if ready else 503,
            )

        async def ready(_):
            try:
                await store.assert_leader()
                scan = runtime.last_loop.get("scan", 0)
                ready = runtime.started and time.time() - scan < max(
                    180, config.scan_seconds * 3
                )
            except Exception:
                ready = False
            return web.json_response({"ready": ready}, status=200 if ready else 503)

        app.router.add_get("/healthz", health)
        app.router.add_get("/readyz", ready)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", config.port).start()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, runtime.stopping.set)
        try:
            # New worker waits without polling Telegram or trading while the old
            # worker drains. Healthcheck can pass without forcing two leaders.
            await store.initialize()
            existing = await store.read()
            if existing["orders"] and existing.get("execution_venue") != config.mode:
                raise RuntimeError("Existing execution ledger belongs to another mode")
            # Validate read-only dependencies before Railway retires an old
            # deployment. None of these calls polls Telegram or places orders.
            await runtime.refresh_instruments()
            if runtime.telegram:
                await runtime.telegram.initialize(publish_commands=False)
            preflight_ready = True
            while not runtime.stopping.is_set() and not await store.lease():
                try:
                    await asyncio.wait_for(runtime.stopping.wait(), timeout=5)
                except asyncio.TimeoutError:
                    pass
            if not runtime.stopping.is_set():
                await runtime.initialize()
                if runtime.telegram:
                    await runtime.telegram.initialize()
                await runtime.run()
        finally:
            await runner.cleanup()
            if not runtime.worker_tasks:
                await runtime.cache.close()
                try:
                    await store.close()
                except Exception:
                    logging.warning(
                        "Startup cleanup could not release database lease; lease will expire"
                    )


def main():
    parser = argparse.ArgumentParser(description="Apex cloud Elliott Wave system")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration without network calls",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    try:
        config = Config.from_env()
        if args.check:
            print(
                json.dumps(
                    {
                        "mode": config.mode,
                        "universe_mode": config.universe_mode,
                        "target_symbols": (
                            config.universe_size
                            if config.universe_mode == "dynamic"
                            else len(config.symbols)
                        ),
                        "symbols": (
                            len(config.symbols)
                            if config.universe_mode == "static"
                            else "selected from fresh market data at startup"
                        ),
                        "postgres_configured": bool(config.database_url),
                        "telegram_configured": bool(config.telegram_token),
                        "ai_configured": bool(
                            config.openai_key and config.openai_model
                        ),
                        "live_enabled": config.live_enabled,
                    },
                    indent=2,
                )
            )
            return
        asyncio.run(serve(config))
    except KeyboardInterrupt:
        return
    except ConfigError as exc:
        logging.error("Apex configuration: %s", str(exc))
        sys.exit(1)
    except Exception as exc:
        # Do not echo a third-party exception that might contain a secret URL.
        logging.error(
            "Apex startup stopped (%s). Validate required settings with --check.",
            type(exc).__name__,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
