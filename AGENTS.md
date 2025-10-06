# Repository Guidelines

## Project Structure & Module Organization
Core trading logic lives in `live_bot.py`, supported by modular helpers (e.g., `strategy_trend_breakout.py`, `strategy_mean_reversion.py`, `ml_scorer_trend.py`). Infrastructure-facing modules sit alongside in the repo root: exchange wrapper (`broker_bybit.py`), persistence (`candle_storage_postgres.py`, `trade_tracker_postgres.py`), data services (`symbol_data_collector.py`). Configuration files (`config.yaml`, `.env` variables via Railway) and scripts like `start.py` reside at the top level. SQLite fallback artifacts (e.g., `candles.db`) are checked in for local debugging.

## Build, Test, and Development Commands
- `python start.py`: single entry point; enforces one running bot, validates Redis models, and launches `live_bot.py`.
- `python live_bot.py`: bypasses startup guard for local experiments; expect Redis/DB credentials in environment.
- `python enhanced_backtester.py`: run targeted historical simulations after wiring symbol-specific parameters inside the script.
- `python backtester.py` / `python enhanced_backtester.py`: legacy and enhanced backtesting flows for validating strategy changes.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation. Keep functions cohesive and log messages actionable (use uppercase symbol tags as seen in `live_bot.py`). Name new modules with snake_case and align class names with PascalCase (e.g., `TradingBot`, `PhantomTradeTracker`). Persist feature dictionaries using descriptive keys consistent with existing ML pipelines.

## Testing Guidelines
Backtesting scripts double as acceptance tests—run the enhanced backtester against representative symbols before merging strategy or ML updates. Add synthetic data sets under `tests/` if you need deterministic cases; mirror live feature generation to avoid divergence. Capture regression metrics (win rate, PnL) and document deltas in the PR body.

## Commit & Pull Request Guidelines
Write commits in present tense with concise scope (e.g., `Improve phantom tracker persistence`). Reference strategy/ML changes explicitly to facilitate audits. Pull requests should include: purpose summary, testing notes (backtester runs, sandbox trials), and any configuration steps required for Railway. Attach relevant log snippets or screenshots when altering Telegram commands or dashboards.

## Security & Configuration Tips
Store secrets in Railway environment variables (`BYBIT_API_KEY`, `REDIS_URL`, `DATABASE_URL`). Never commit credential files. Redis and Postgres outages should be assumed transient—wrap new network calls with the existing retry patterns and log warnings instead of failing fast.
