from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
from enum import Enum

class MarginMode(str, Enum):
    CROSS = "cross"
    ISOLATED = "isolated"

class PositionMode(str, Enum):
    ONE_WAY = "one_way"
    HEDGE = "hedge"

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class Settings(BaseSettings):
    # Bybit API
    bybit_api_key: str = Field(..., description="Bybit API Key")
    bybit_api_secret: str = Field(..., description="Bybit API Secret")
    bybit_testnet: bool = Field(True, description="Use Bybit testnet")
    bybit_recv_window: int = Field(5000, description="Recv window for API calls")
    
    # Telegram
    telegram_bot_token: str = Field(..., description="Telegram bot token")
    telegram_webhook_url: Optional[str] = Field(None, description="Webhook URL")
    telegram_secret_token: str = Field(..., description="Webhook secret token")
    telegram_allowed_chat_ids: List[int] = Field([], description="Allowed chat IDs")
    
    # Database
    database_url: str = Field(
        "postgresql://user:password@localhost:5432/crypto_bot",
        description="PostgreSQL connection URL"
    )
    redis_url: str = Field("redis://localhost:6379/0", description="Redis URL")
    
    # Trading Parameters
    default_risk_percent: float = Field(1.0, ge=0.1, le=5.0)
    max_concurrent_positions: int = Field(5, ge=1, le=20)
    max_daily_loss_percent: float = Field(5.0, ge=1.0, le=10.0)
    default_leverage: int = Field(10, ge=1, le=125)
    default_margin_mode: MarginMode = Field(MarginMode.ISOLATED)
    default_position_mode: PositionMode = Field(PositionMode.ONE_WAY)
    
    # Trading settings - Top 300 crypto futures on Bybit
    default_symbols: List[str] = Field(default=[
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT",
        "LINKUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT", "ATOMUSDT",
        "XLMUSDT", "ICPUSDT", "FILUSDT", "ETCUSDT", "APTUSDT",
        "ARBUSDT", "OPUSDT", "VETUSDT", "ALGOUSDT", "XTZUSDT",
        "AAVEUSDT", "SANDUSDT", "MANAUSDT", "THETAUSDT", "AXSUSDT",
        "FLOWUSDT", "HBARUSDT", "EGLDUSDT", "GRTUSDT", "SNXUSDT",
        "UNIUSDT", "RUNEUSDT", "INJUSDT", "RNDRUSDT", "IMXUSDT",
        "GALAUSDT", "CHZUSDT", "SUSHIUSDT", "COMPUSDT", "ENJUSDT",
        "KSMUSDT", "ZECUSDT", "BATUSDT", "MKRUSDT", "LDOUSDT",
        "QNTUSDT", "CFXUSDT", "BLURUSDT", "ARUSDT", "WLDUSDT",
        "AGIXUSDT", "FETUSDT", "OCEANUSDT", "MASKUSDT", "GMTUSDT",
        "APEUSDT", "SSVUSDT", "FLRUSDT", "FXSUSDT", "HOOKUSDT",
        "MAGICUSDT", "TUSDT", "ROSEUSDT", "HIGHUSDT", "MINAUSDT",
        "ASTRUSDT", "AGLDUSDT", "PHBUSDT", "GMXUSDT", "STXUSDT",
        "ACHUSDT", "SXPUSDT", "WOOUSDT", "SKLUSDT", "SPELLUSDT",
        "1000PEPEUSDT", "CVXUSDT", "STGUSDT", "PEOPLEUSDT", "CKBUSDT",
        "ENSUSDT", "PERPUSDT", "TRUUSDT", "LQTYUSDT", "USDCUSDT",
        "IDUSDT", "JOEUSDT", "TLMUSDT", "AMBUSDT",
        "RDNTUSDT", "HFTUSDT", "XVSUSDT", "EDUUSDT", "IDEXUSDT",
        "SUIUSDT", "1000LUNCUSDT", "ETHWUSDT",
        "RPLUSDT", "ALICEUSDT", "ANKRUSDT", "BIGTIMEUSDT", "BLZUSDT",
        "BONDUSDT", "CAKEUSDT", "CELUSDT", "CELRUSDT", "CITYUSDT",
        "COTIUSDT", "CRVUSDT", "CTSIUSDT", "CVCUSDT",
        "CYBERUSDT", "DASHUSDT", "DENTUSDT",
        "DUSKUSDT", "DYDXUSDT", "ILVUSDT", "IOSTUSDT", "IOTAUSDT",
        "JASMYUSDT", "KAVAUSDT", "KNCUSDT",
        "LPTUSDT", "LRCUSDT", "LUNA2USDT", "MAVUSDT",
        "MTLUSDT", "NKNUSDT", "NMRUSDT", "NTRNUSDT", "OGUSDT",
        "ONEUSDT", "ONTUSDT", "ORDIUSDT", "OXTUSDT",
        "PENDLEUSDT", "POLYXUSDT", "POWRUSDT", "QTUMUSDT", "RADUSDT",
        "REQUSDT", "RLCUSDT",
        "RSRUSDT", "RVNUSDT", "SFPUSDT", "SLPUSDT",
        "STORJUSDT",
        "SWEATUSDT", "SYSUSDT",
        "TONUSDT", "TRBUSDT", "TRXUSDT", "UMAUSDT",
        "USTCUSDT", "WAVESUSDT",
        "XEMUSDT", "XMRUSDT",
        "YFIUSDT", "ZILUSDT",
        "ZRXUSDT", "1000BONKUSDT", "1000FLOKIUSDT", "1000XECUSDT",
        "API3USDT", "ARKMUSDT",
        "AUDIOUSDT", "BAKEUSDT",
        "BALUSDT", "BANDUSDT", "BELUSDT",
        "BSVUSDT"
    ])
    
    # Multi-timeframe monitoring
    monitored_timeframes: List[str] = Field(
        default=["5", "15", "60", "240"],
        description="Timeframes to monitor (in minutes)"
    )
    default_timeframe: str = Field("15", description="Default timeframe in minutes")
    
    # Supply & Demand Strategy
    sd_min_base_candles: int = Field(3, description="Min candles for base")
    sd_max_base_candles: int = Field(10, description="Max candles for base")
    sd_departure_atr_multiplier: float = Field(2.0, description="ATR multiplier for departure")
    sd_zone_buffer_percent: float = Field(0.2, description="Buffer beyond zone for stops")
    sd_max_zone_touches: int = Field(3, description="Max touches before invalidation")
    sd_zone_max_age_hours: int = Field(168, description="Max age in hours (7 days)")
    sd_min_zone_score: float = Field(40.0, description="Min score to trade zone")
    
    # Risk Management
    use_trailing_stop: bool = Field(True)
    trailing_stop_activation_percent: float = Field(1.0)
    trailing_stop_callback_percent: float = Field(0.5)
    move_stop_to_breakeven_at_tp1: bool = Field(True)
    tp1_risk_ratio: float = Field(1.0)
    tp2_risk_ratio: float = Field(2.0)
    partial_tp1_percent: float = Field(50.0)
    
    # Server
    server_host: str = Field("0.0.0.0")
    server_port: int = Field(8000)
    environment: Environment = Field(Environment.DEVELOPMENT)
    log_level: str = Field("INFO")
    
    # Security
    secret_key: str = Field(..., description="Secret key for encryption")
    enable_demo_mode: bool = Field(True)
    
    # Rate Limiting
    bybit_rest_rate_limit: int = Field(120, description="Requests per minute")
    symbol_scan_interval_seconds: int = Field(5)
    instrument_refresh_interval_hours: int = Field(24)
    
    @validator("telegram_allowed_chat_ids", pre=True)
    def parse_chat_ids(cls, v):
        if isinstance(v, str):
            # Handle comma-separated string
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        elif isinstance(v, int):
            # Handle single integer
            return [v]
        elif isinstance(v, list):
            # Handle list of integers or strings
            result = []
            for item in v:
                if isinstance(item, str):
                    result.append(int(item.strip()))
                else:
                    result.append(int(item))
            return result
        return v if v else []
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()