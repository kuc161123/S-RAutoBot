"""Explicit configuration; importing this module never contacts a service."""

import math
import os
import re
from dataclasses import dataclass, field


class ConfigError(ValueError):
    """A local validation message containing field names, never supplied values."""


def _numeric(env, name, default, kind=int):
    try:
        return kind(env.get(name, default))
    except (TypeError, ValueError, OverflowError):
        raise ConfigError(
            f"{name} requires a valid {'integer' if kind is int else 'number'}"
        ) from None


def _users(env):
    try:
        return frozenset(
            int(x.strip())
            for x in env.get("TELEGRAM_ALLOWED_USER_IDS", "").split(",")
            if x.strip()
        )
    except (TypeError, ValueError):
        raise ConfigError(
            "TELEGRAM_ALLOWED_USER_IDS needs comma-separated numeric personal user IDs, not the bot username"
        ) from None


def _flag(env, name, default=False):
    value = env.get(name, str(default)).lower()
    if value not in {"true", "false", "1", "0"}:
        raise ConfigError(f"{name} must be true or false")
    return value in {"true", "1"}


def _credential(env, name):
    value = env.get(name, "").strip()
    return "" if value.upper().startswith("REPLACE_WITH_") else value


@dataclass(frozen=True)
class Config:
    mode: str = "shadow"
    database_url: str = field(default="", repr=False)
    redis_url: str = field(default="", repr=False)
    sqlite_path: str = "data/apex.sqlite3"
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LINKUSDT")
    universe_mode: str = "dynamic"
    universe_size: int = 50
    universe_review_seconds: int = 21600
    universe_min_turnover: float = 20000000.0
    universe_max_spread_bps: float = 10.0
    universe_min_depth: float = 25000.0
    telegram_token: str = field(default="", repr=False)
    telegram_chat_id: str = ""
    telegram_username: str = "AutotradingBot222_bot"
    telegram_user_ids: frozenset[int] = frozenset()
    bybit_key: str = field(default="", repr=False)
    bybit_secret: str = field(default="", repr=False)
    bybit_url: str = "https://api.bybit.com"
    openai_key: str = field(default="", repr=False)
    openai_model: str = ""
    ai_daily_calls: int = 60
    ai_max_output_tokens: int = 1800
    live_enabled: bool = False
    migration_verified: bool = False
    shadow_equity: float = 10000.0
    scan_seconds: int = 60
    port: int = 8080
    drive_file_ids: tuple[str, ...] = ()
    drive_credentials_json: str = field(default="", repr=False)
    macro_feed_url: str = ""
    hl_account_address: str = ""
    hl_private_key: str = field(default="", repr=False)
    hl_api_url: str = "https://api.hyperliquid.xyz"
    hl_perp_dexs: tuple[str, ...] = ("xyz",)
    shadow_book_sheet_id: str = ""
    shadow_book_csv_url: str = field(default="", repr=False)
    # auto = Telegram shows only the Hyperliquid desk (engine off) once HL keys are set.
    desk_only: str = "auto"

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        mode = env.get("APEX_MODE", "shadow").lower()
        if mode not in {"shadow", "testnet", "live"}:
            raise ConfigError("APEX_MODE must be shadow, testnet or live")
        symbols = tuple(
            dict.fromkeys(
                s.strip().upper()
                for s in env.get(
                    "APEX_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,LINKUSDT"
                ).split(",")
                if s.strip()
            )
        )
        if (
            not symbols
            or len(symbols) > 80
            or any(not s.isalnum() or not s.endswith("USDT") for s in symbols)
        ):
            raise ConfigError("APEX_SYMBOLS requires 1–80 USDT symbols")
        cfg = cls(
            mode=mode,
            database_url=env.get("DATABASE_URL", ""),
            redis_url=env.get("REDIS_URL", ""),
            sqlite_path=env.get("APEX_SQLITE_PATH", "data/apex.sqlite3"),
            symbols=symbols,
            universe_mode=env.get("APEX_UNIVERSE_MODE", "dynamic").lower(),
            universe_size=_numeric(env, "APEX_UNIVERSE_SIZE", 50),
            universe_min_turnover=_numeric(
                env, "APEX_UNIVERSE_MIN_TURNOVER_USDT", 20000000, float
            ),
            universe_max_spread_bps=_numeric(
                env, "APEX_UNIVERSE_MAX_SPREAD_BPS", 10, float
            ),
            universe_min_depth=_numeric(
                env, "APEX_UNIVERSE_MIN_DEPTH_USDT", 25000, float
            ),
            telegram_token=env.get("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=env.get("TELEGRAM_CHAT_ID", ""),
            telegram_username=env.get(
                "TELEGRAM_BOT_USERNAME", "AutotradingBot222_bot"
            ).lstrip("@"),
            telegram_user_ids=_users(env),
            bybit_key=env.get("BYBIT_API_KEY", ""),
            bybit_secret=env.get("BYBIT_API_SECRET", env.get("BYBIT_SECRET", "")),
            bybit_url=(
                "https://api-testnet.bybit.com"
                if mode == "testnet"
                else "https://api.bybit.com"
            ),
            openai_key=_credential(env, "OPENAI_API_KEY"),
            openai_model=_credential(env, "OPENAI_MODEL"),
            ai_daily_calls=_numeric(env, "APEX_AI_DAILY_CALLS", 60),
            ai_max_output_tokens=_numeric(env, "APEX_AI_MAX_OUTPUT_TOKENS", 1800),
            live_enabled=_flag(env, "APEX_LIVE_ENABLED"),
            migration_verified=_flag(env, "APEX_MIGRATION_VERIFIED"),
            shadow_equity=_numeric(env, "APEX_SHADOW_EQUITY", 10000, float),
            scan_seconds=_numeric(env, "APEX_SCAN_SECONDS", 60),
            port=_numeric(env, "PORT", 8080),
            drive_file_ids=tuple(
                x.strip()
                for x in env.get("APEX_DRIVE_FILE_IDS", "").split(",")
                if x.strip()
            ),
            drive_credentials_json=env.get("APEX_GOOGLE_SERVICE_ACCOUNT_JSON", ""),
            macro_feed_url=env.get("APEX_CONTEXT_FEED_URL", ""),
            hl_account_address=_credential(env, "HL_ACCOUNT_ADDRESS"),
            hl_private_key=_credential(env, "HL_API_PRIVATE_KEY"),
            hl_api_url=env.get("HL_API_URL", "https://api.hyperliquid.xyz").rstrip("/"),
            hl_perp_dexs=tuple(
                x.strip() for x in env.get("HL_PERP_DEXS", "xyz").split(",") if x.strip()
            ),
            shadow_book_sheet_id=env.get("APEX_SHADOW_BOOK_SHEET_ID", "").strip(),
            shadow_book_csv_url=env.get("APEX_SHADOW_BOOK_CSV_URL", "").strip(),
            desk_only=env.get("APEX_DESK_ONLY", "auto").strip().lower() or "auto",
        )
        if cfg.desk_only not in ("auto", "on", "off"):
            raise ConfigError("APEX_DESK_ONLY must be auto, on or off")
        from .shadow_book import valid_csv_url, valid_sheet_id

        if any(not re.fullmatch(r"[a-z0-9]{1,16}", d) for d in cfg.hl_perp_dexs):
            raise ConfigError("HL_PERP_DEXS must be comma-separated lowercase dex names, e.g. xyz")
        if cfg.shadow_book_sheet_id and not valid_sheet_id(cfg.shadow_book_sheet_id):
            raise ConfigError("APEX_SHADOW_BOOK_SHEET_ID must be a Google Sheet ID")
        if cfg.shadow_book_sheet_id and not cfg.drive_credentials_json:
            raise ConfigError(
                "APEX_SHADOW_BOOK_SHEET_ID needs APEX_GOOGLE_SERVICE_ACCOUNT_JSON (or use APEX_SHADOW_BOOK_CSV_URL)"
            )
        if cfg.shadow_book_csv_url and not valid_csv_url(cfg.shadow_book_csv_url):
            raise ConfigError(
                "APEX_SHADOW_BOOK_CSV_URL must be a docs.google.com 'Publish to the web' CSV link"
            )
        from .hl_client import valid_address, valid_private_key

        if bool(cfg.hl_account_address) != bool(cfg.hl_private_key):
            raise ConfigError(
                "Hyperliquid needs both HL_ACCOUNT_ADDRESS and HL_API_PRIVATE_KEY"
            )
        if cfg.hl_account_address and not valid_address(cfg.hl_account_address):
            raise ConfigError("HL_ACCOUNT_ADDRESS must be a 0x-prefixed 40-hex address")
        if cfg.hl_private_key and not valid_private_key(cfg.hl_private_key):
            raise ConfigError("HL_API_PRIVATE_KEY must be a 64-hex API wallet private key")
        if cfg.hl_api_url not in (
            "https://api.hyperliquid.xyz",
            "https://api.hyperliquid-testnet.xyz",
        ):
            raise ConfigError("HL_API_URL must be the Hyperliquid mainnet or testnet API")
        if not math.isfinite(cfg.shadow_equity) or cfg.shadow_equity <= 0:
            raise ConfigError("APEX_SHADOW_EQUITY must be positive and finite")
        if (
            cfg.universe_mode not in {"static", "dynamic"}
            or not 1 <= cfg.universe_size <= 80
        ):
            raise ConfigError("APEX_UNIVERSE_MODE must be static/dynamic and size 1–80")
        if (
            any(
                not math.isfinite(v) or v <= 0
                for v in (
                    cfg.universe_min_turnover,
                    cfg.universe_max_spread_bps,
                    cfg.universe_min_depth,
                )
            )
            or cfg.universe_max_spread_bps > 20
        ):
            raise ConfigError(
                "Universe turnover/depth must be positive; spread must be 0–20 basis points"
            )
        if not 1 <= cfg.port <= 65535 or any(x <= 0 for x in cfg.telegram_user_ids):
            raise ConfigError(
                "PORT or TELEGRAM_ALLOWED_USER_IDS outside supported bounds"
            )
        if not 15 <= cfg.scan_seconds <= 900 or not 1 <= cfg.ai_daily_calls <= 500:
            raise ConfigError("Scan interval or AI budget outside supported bounds")
        if not 256 <= cfg.ai_max_output_tokens <= 8000:
            raise ConfigError("AI output token limit outside supported bounds")
        if cfg.telegram_token and (
            not cfg.telegram_chat_id or not cfg.telegram_user_ids
        ):
            raise ConfigError(
                "Telegram needs TELEGRAM_CHAT_ID and TELEGRAM_ALLOWED_USER_IDS"
            )
        if env.get("RAILWAY_ENVIRONMENT") and not cfg.database_url:
            raise ConfigError(
                "Railway requires DATABASE_URL; ephemeral fallback is disabled"
            )
        if mode != "shadow":
            missing = [
                name
                for name, present in {
                    "DATABASE_URL": bool(cfg.database_url),
                    "Bybit credentials": bool(cfg.bybit_key and cfg.bybit_secret),
                    "Telegram authorization": bool(
                        cfg.telegram_token and cfg.telegram_user_ids
                    ),
                    "OPENAI_API_KEY / OPENAI_MODEL": bool(
                        cfg.openai_key and cfg.openai_model
                    ),
                    "APEX_LIVE_ENABLED": cfg.live_enabled,
                    "APEX_MIGRATION_VERIFIED": cfg.migration_verified,
                }.items()
                if not present
            ]
            if missing:
                raise ConfigError(
                    "Execution prerequisites missing: " + ", ".join(missing)
                )
        if mode == "live":
            from .release import LIVE_APPROVED, REASON

            if not LIVE_APPROVED:
                raise ConfigError(REASON)
        return cfg
