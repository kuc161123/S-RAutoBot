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
    bybit_testnet: bool = Field(False, env="BYBIT_TESTNET")  # Default to mainnet
    
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
            # TOP 100 LIQUID PAIRS - More opportunities with managed rate limiting
            # Majors (15)
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT",
            "DOTUSDT", "TONUSDT", "TRXUSDT", "NEARUSDT", "UNIUSDT",
            
            # High volume alts (20)
            "LTCUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
            "ATOMUSDT", "FILUSDT", "LDOUSDT", "STXUSDT", "IMXUSDT",
            "HBARUSDT", "RNDRUSDT", "EGLDUSDT", "THETAUSDT", "AXSUSDT",
            "SANDUSDT", "MANAUSDT", "GALAUSDT", "ICPUSDT", "QNTUSDT",
            
            # Layer 2 & Scaling (10)
            "STRKUSDT", "ZKUSDT", "MANTAUSDT", "BLURUSDT", "ACEUSDT",
            "NFPUSDT", "AIUSDT", "XAIUSDT", "DYMUSDT", "ALTUSDT",
            
            # DeFi tokens (15)
            "AAVEUSDT", "COMPUSDT", "MKRUSDT", "SNXUSDT", "RUNEUSDT",
            "CRVUSDT", "SUSHIUSDT", "1INCHUSDT", "UMAUSDT", "YFIUSDT",
            "BALUSDT", "DYDXUSDT", "GMXUSDT", "RDNTUSDT", "MAGICUSDT",
            
            # Gaming & Metaverse (10)
            "FLOWUSDT", "CHZUSDT", "ENJUSDT", "ALICEUSDT", "SUPERUSDT",
            "TLMUSDT", "ILVUSDT", "ROSEUSDT", "APEUSDT", "GMTUSDT",
            
            # Memes & Trending (15)
            "1000PEPEUSDT", "WIFUSDT", "1000BONKUSDT", "1000FLOKIUSDT",
            "MEMEUSDT", "ORDIUSDT", "SEIUSDT", "SUIUSDT", "TIAUSDT", 
            "JUPUSDT", "PYTHUSDT", "PIXELUSDT", "STEEMUSDT", "JTOUSDT", "BEAMUSDT",
            
            # Infrastructure & Data (10)
            "FETUSDT", "OCEANUSDT", "RSRUSDT", "ARKUSDT", "GLMUSDT",
            "ANKRUSDT", "CELOUSDT", "ONEUSDT", "IOTXUSDT", "KAVAUSDT",
            
            # Additional high liquidity (5)
            "XLMUSDT", "ALGOUSDT", "VETUSDT", "FTMUSDT", "ETCUSDT"
        ],
        description="Top 100 liquid pairs for more trading opportunities"
    )
    
    # Risk Management - ALL FROM ENV VARS
    risk_per_trade: float = Field(..., env="RISK_PER_TRADE", ge=0.001, le=1.0, description="Risk per trade")
    max_positions: int = Field(..., env="MAX_POSITIONS", ge=1, le=200, description="Maximum concurrent positions")
    leverage: int = Field(..., env="LEVERAGE", ge=1, le=125, description="Trading leverage")
    max_position_value_multiplier: float = Field(..., env="MAX_POSITION_VALUE_MULTIPLIER", ge=0.5, le=5.0, description="Max position value as multiple of balance")
    
    # Strategy Parameters - HARDCODED OPTIMAL VALUES
    rsi_period: int = Field(default=14, description="RSI period (standard)")
    rsi_oversold: float = Field(default=35, description="RSI oversold level")
    rsi_overbought: float = Field(default=65, description="RSI overbought level")
    macd_fast: int = Field(default=12, description="MACD fast period")
    macd_slow: int = Field(default=26, description="MACD slow period")
    macd_signal: int = Field(default=9, description="MACD signal period")
    
    # System Configuration - HARDCODED OPTIMAL VALUES
    scan_interval: int = Field(default=60, description="Scan interval in seconds")
    min_volume_24h: float = Field(default=1000000, description="Minimum 24h volume in USDT")
    startup_delay: int = Field(default=5, description="Startup delay in seconds")
    
    # Signal Requirements - HARDCODED OPTIMAL VALUES
    min_risk_reward: float = Field(default=1.0, description="Minimum risk/reward ratio")
    min_confirmations: int = Field(default=1, description="Minimum confirmations for signal")
    
    # Risk/Reward multipliers - HARDCODED OPTIMAL VALUES
    rr_sl_multiplier: float = Field(default=1.5, description="Stop loss ATR multiplier")
    rr_tp_multiplier: float = Field(default=3.0, description="Take profit ATR multiplier")
    
    # Logging - HARDCODED
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Strategy removed - using enhanced aggressive only
    
    # Signal quality control - HARDCODED OPTIMAL VALUES
    min_signal_score: int = Field(default=4, ge=2, le=6, description="Minimum score for signal (2-6, higher = fewer signals)")
    min_volume_multiplier: float = Field(default=1.5, ge=0.5, le=3.0, description="Volume filter (higher = fewer signals)")
    signal_cooldown_minutes: int = Field(default=10, ge=1, le=60, description="Minutes to wait between signals per symbol")
    
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