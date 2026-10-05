import pytest

from apex_bot.config import Config


def test_default_shadow_has_no_live_authority():
    cfg = Config.from_env({})
    assert cfg.mode == "shadow" and not cfg.live_enabled
    assert not cfg.telegram_user_ids and not cfg.openai_key
    assert cfg.universe_mode == "dynamic" and cfg.universe_size == 50
    assert cfg.universe_review_seconds == 21600


@pytest.mark.parametrize(
    "key,value",
    [
        ("APEX_UNIVERSE_MODE", "arbitrary"),
        ("APEX_UNIVERSE_SIZE", "0"),
        ("APEX_UNIVERSE_SIZE", "81"),
        ("APEX_UNIVERSE_SIZE", "50.5"),
        ("APEX_UNIVERSE_MIN_TURNOVER_USDT", "nan"),
        ("APEX_UNIVERSE_MIN_DEPTH_USDT", "inf"),
        ("APEX_UNIVERSE_MIN_DEPTH_USDT", "-1"),
        ("APEX_UNIVERSE_MAX_SPREAD_BPS", "21"),
    ],
)
def test_invalid_liquidity_policy_is_rejected(key, value):
    with pytest.raises(ValueError):
        Config.from_env({key: value})


def test_static_universe_requires_explicit_mode_and_keeps_manual_symbols():
    cfg = Config.from_env(
        {"APEX_UNIVERSE_MODE": "static", "APEX_SYMBOLS": "ETHUSDT,BTCUSDT"}
    )
    assert cfg.universe_mode == "static" and cfg.symbols == ("ETHUSDT", "BTCUSDT")


@pytest.mark.parametrize("mode", ["live", "testnet"])
def test_execution_missing_prerequisites_fails_before_network(mode):
    with pytest.raises(ValueError, match="prerequisites"):
        Config.from_env({"APEX_MODE": mode})


def test_cloud_cannot_fall_back_to_ephemeral_sqlite():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        Config.from_env({"RAILWAY_ENVIRONMENT": "production"})


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0"])
def test_shadow_capital_must_be_finite_positive(value):
    with pytest.raises(ValueError):
        Config.from_env({"APEX_SHADOW_EQUITY": value})


def test_auth_is_explicit_and_secrets_not_in_repr():
    env = {"TELEGRAM_BOT_TOKEN": "123:secret", "TELEGRAM_CHAT_ID": "123"}
    with pytest.raises(ValueError, match="TELEGRAM_ALLOWED_USER_IDS"):
        Config.from_env(env)
    env.update(TELEGRAM_ALLOWED_USER_IDS="12, 34", OPENAI_API_KEY="secret-key")
    cfg = Config.from_env(env)
    assert cfg.telegram_user_ids == frozenset({12, 34})
    assert "secret-key" not in repr(cfg) and "123:secret" not in repr(cfg)


def test_generic_connector_telegram_token_is_not_a_bot_credential():
    assert not Config.from_env({"TELEGRAM_TOKEN": "unrelated"}).telegram_token


def test_placeholders_do_not_enable_ai_and_database_url_is_not_printed():
    cfg = Config.from_env(
        {
            "OPENAI_API_KEY": "REPLACE_WITH_OPENAI_API_KEY",
            "OPENAI_MODEL": "REPLACE_WITH_OPENAI_MODEL_ID",
            "DATABASE_URL": "postgres://user:private@localhost/db",
        }
    )
    assert not cfg.openai_key and not cfg.openai_model
    assert "private" not in repr(cfg)


def test_live_release_cannot_be_enabled_by_credentials_and_flags_alone():
    with pytest.raises(ValueError, match="Live release blocked"):
        Config.from_env(
            {
                "APEX_MODE": "live",
                "DATABASE_URL": "postgres://fixture",
                "BYBIT_API_KEY": "fixture",
                "BYBIT_API_SECRET": "fixture",
                "TELEGRAM_BOT_TOKEN": "1:fixture",
                "TELEGRAM_CHAT_ID": "1",
                "TELEGRAM_ALLOWED_USER_IDS": "1",
                "OPENAI_API_KEY": "fixture",
                "OPENAI_MODEL": "fixture",
                "APEX_LIVE_ENABLED": "true",
                "APEX_MIGRATION_VERIFIED": "true",
            }
        )
