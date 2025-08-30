"""
Configuration management for the crypto trading bot
"""
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List, Optional, Union
import os
import json

# Only load .env in development
if not os.getenv('RAILWAY_ENVIRONMENT') and not os.path.exists('/.dockerenv'):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not required in production

class Settings(BaseSettings):
    """Application settings"""
    
    # Bybit API Configuration
    bybit_api_key: str = Field(..., env="BYBIT_API_KEY")
    bybit_api_secret: str = Field(..., env="BYBIT_API_SECRET")
    bybit_testnet: bool = Field(True, env="BYBIT_TESTNET")
    
    # Telegram Configuration (Optional)
    telegram_bot_token: Optional[str] = Field(None, env="TELEGRAM_BOT_TOKEN")
    telegram_chat_ids: List[int] = Field(default_factory=list, env="TELEGRAM_CHAT_IDS")
    
    @field_validator('bybit_testnet', mode='before')
    def parse_bool(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            # Remove quotes if present
            v = v.strip('"').strip("'").lower()
            return v in ('true', '1', 'yes', 'on')
        return bool(v)
    
    @field_validator('telegram_chat_ids', mode='before')
    def parse_chat_ids(cls, v):
        if isinstance(v, list):
            return v
        if isinstance(v, int):
            return [v]
        if isinstance(v, str):
            # Try to parse as JSON array first
            if v.startswith('['):
                try:
                    return json.loads(v)
                except:
                    pass
            # Otherwise treat as single ID
            return [int(v.strip())]
        return v if v else []
    
    @field_validator('initial_symbols', mode='before')
    def parse_symbols(cls, v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            # Try to parse as JSON array first
            if v.startswith('['):
                try:
                    return json.loads(v)
                except:
                    pass
            # Otherwise treat as comma-separated list
            return [s.strip() for s in v.split(',') if s.strip()]
        return v if v else []
    
    # Trading Configuration
    initial_symbols: List[str] = Field(
        env="INITIAL_SYMBOLS",
        default=[
            # Top 100 most liquid crypto pairs on Bybit
            # Core majors
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "LINKUSDT",
            "DOTUSDT", "TONUSDT", "TRXUSDT", "NEARUSDT", "UNIUSDT",
            "LTCUSDT", "BCHUSDT", "APTUSDT", "ICPUSDT", "ETCUSDT",
            
            # Layer 2s and scaling
            "ARBUSDT", "OPUSDT", "INJUSDT", "STXUSDT", "MANTAUSDT",
            "IMXUSDT", "SEIUSDT", "SUIUSDT", "CELOUSDT", "ROSEUSDT",
            
            # DeFi tokens
            "MKRUSDT", "AAVEUSDT", "SNXUSDT", "COMPUSDT", "YFIUSDT",
            "CRVUSDT", "LDOUSDT", "1INCHUSDT", "SUSHIUSDT", "GMXUSDT",
            
            # AI and compute
            "RENDERUSDT", "FETUSDT", "AGIXUSDT", "OCEANUSDT", "RNDR3USDT",
            "TAOUSDT", "GRTUSDT", "ARKMUSDT", "AIUSDT", "PHBUSDT",
            
            # Gaming and metaverse
            "SANDUSDT", "AXSUSDT", "MANAUSDT", "GALAUSDT", "ENJUSDT",
            "GMTUSDT", "MAGICUSDT", "APEUSDT", "ILOUSDT", "HIGHUSDT",
            
            # Infrastructure
            "ATOMUSDT", "FILUSDT", "HBARUSDT", "ALGOUSDT", "VETUSDT",
            "THETAUSDT", "FTMUSDT", "XTZUSDT", "EGLDUSDT", "FLOWUSDT",
            
            # Meme coins (with correct Bybit format)
            "SHIBUSDT", "PEPEUSDT", "BONKUSDT", "FLOKIUSDT", "WIFUSDT",
            "MEMEUSDT", "DOGSUSDT", "1000LUNCUSDT", "1000XECUSDT", "SPELLUSDT",
            
            # Others high volume
            "ORDIUSDT", "RUNEUSDT", "CFXUSDT", "QNTUSDT", "CHZUSDT",
            "PENDLEUSDT", "WLDUSDT", "BLURUSDT", "PEOPLEUSDT", "CKBUSDT",
            "LRCUSDT", "ENSUSDT", "DYDXUSDT", "ZRXUSDT", "BATUSDT",
            
            # Additional verified symbols
            "JASMYUSDT", "STMXUSDT", "ACHUSDT", "RSRUSDT", "SXPUSDT",
            "IOTXUSDT", "CYBERUSDT", "NTRNUSDT", "MAVUSDT", "MDTUSDT",
            "MASKUSDT", "GALAUSDT", "C98USDT", "RNDRUSDT", "ARPAUSDT"
        ],
        description="Top 100 liquid trading pairs"
    )
    
    # Risk Management - ALL FROM ENV VARS
    risk_per_trade: float = Field(..., env="RISK_PER_TRADE", ge=0.001, le=1.0, description="Risk per trade")
    max_positions: int = Field(..., env="MAX_POSITIONS", ge=1, le=200, description="Maximum concurrent positions")
    leverage: int = Field(..., env="LEVERAGE", ge=1, le=125, description="Trading leverage")
    
    # Strategy Parameters - FROM ENV VARS WITH DEFAULTS
    rsi_period: int = Field(14, env="RSI_PERIOD", description="RSI period")
    rsi_oversold: float = Field(30.0, env="RSI_OVERSOLD", description="RSI oversold level")
    rsi_overbought: float = Field(70.0, env="RSI_OVERBOUGHT", description="RSI overbought level")
    macd_fast: int = Field(12, env="MACD_FAST", description="MACD fast period")
    macd_slow: int = Field(26, env="MACD_SLOW", description="MACD slow period")
    macd_signal: int = Field(9, env="MACD_SIGNAL", description="MACD signal period")
    
    # System Configuration - FROM ENV VARS
    scan_interval: int = Field(30, env="SCAN_INTERVAL", description="Scan interval in seconds")
    min_volume_24h: float = Field(1000000, env="MIN_VOLUME_24H", description="Minimum 24h volume in USDT")
    startup_delay: int = Field(5, env="STARTUP_DELAY", description="Startup delay in seconds")
    
    # Scalping specific parameters - FROM ENV VARS
    scalp_timeframe: str = Field("5", env="SCALP_TIMEFRAME", description="Timeframe for scalping")
    scalp_profit_target: float = Field(0.003, env="SCALP_PROFIT_TARGET", description="Quick profit target")
    scalp_stop_loss: float = Field(0.002, env="SCALP_STOP_LOSS", description="Tight stop loss")
    min_risk_reward: float = Field(1.2, env="MIN_RISK_REWARD", description="Minimum risk/reward ratio")
    
    # Scalping leverage settings - FROM ENV VARS
    scalp_leverage: int = Field(5, env="SCALP_LEVERAGE", description="Leverage for scalp trades")
    swing_leverage: int = Field(10, env="SWING_LEVERAGE", description="Leverage for swing trades")
    
    # Logging
    log_level: str = Field("INFO", description="Logging level")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    @property
    def telegram_enabled(self) -> bool:
        """Check if Telegram is configured"""
        return bool(self.telegram_bot_token and self.telegram_chat_ids)
    
    @property
    def is_testnet(self) -> bool:
        """Check if using testnet"""
        return self.bybit_testnet

# Create global settings instance
settings = Settings()