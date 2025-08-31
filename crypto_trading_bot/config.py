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
            # TOP 30 MOST LIQUID PAIRS ONLY - Prevents rate limiting
            # Majors (10)
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT",
            
            # High volume alts (10)
            "DOTUSDT", "TONUSDT", "TRXUSDT", "NEARUSDT", "UNIUSDT",
            "LTCUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT",
            
            # Popular memes & trending (10)
            "1000PEPEUSDT", "WIFUSDT", "1000BONKUSDT", "1000FLOKIUSDT",
            "MEMEUSDT", "ORDIUSDT", "SEIUSDT", "SUIUSDT", "TIAUSDT", "JUPUSDT"
        ],
        description="Top 30 liquid pairs to prevent rate limiting"
    )
    
    # Risk Management - ALL FROM ENV VARS
    risk_per_trade: float = Field(..., env="RISK_PER_TRADE", ge=0.001, le=1.0, description="Risk per trade")
    max_positions: int = Field(..., env="MAX_POSITIONS", ge=1, le=200, description="Maximum concurrent positions")
    leverage: int = Field(..., env="LEVERAGE", ge=1, le=125, description="Trading leverage")
    max_position_value_multiplier: float = Field(..., env="MAX_POSITION_VALUE_MULTIPLIER", ge=0.5, le=5.0, description="Max position value as multiple of balance")
    
    # Strategy Parameters - ALL REQUIRED FROM ENV VARS
    rsi_period: int = Field(..., env="RSI_PERIOD", description="RSI period")
    rsi_oversold: float = Field(..., env="RSI_OVERSOLD", description="RSI oversold level")
    rsi_overbought: float = Field(..., env="RSI_OVERBOUGHT", description="RSI overbought level")
    macd_fast: int = Field(..., env="MACD_FAST", description="MACD fast period")
    macd_slow: int = Field(..., env="MACD_SLOW", description="MACD slow period")
    macd_signal: int = Field(..., env="MACD_SIGNAL", description="MACD signal period")
    
    # System Configuration - ALL REQUIRED FROM ENV VARS
    scan_interval: int = Field(..., env="SCAN_INTERVAL", description="Scan interval in seconds")
    min_volume_24h: float = Field(..., env="MIN_VOLUME_24H", description="Minimum 24h volume in USDT")
    startup_delay: int = Field(..., env="STARTUP_DELAY", description="Startup delay in seconds")
    
    # Scalping specific parameters - ALL REQUIRED FROM ENV VARS
    scalp_timeframe: str = Field(..., env="SCALP_TIMEFRAME", description="Timeframe for scalping")
    scalp_profit_target: float = Field(..., env="SCALP_PROFIT_TARGET", description="Quick profit target")
    scalp_stop_loss: float = Field(..., env="SCALP_STOP_LOSS", description="Tight stop loss")
    min_risk_reward: float = Field(..., env="MIN_RISK_REWARD", description="Minimum risk/reward ratio")
    min_confirmations: int = Field(..., env="MIN_CONFIRMATIONS", description="Minimum confirmations for signal")
    
    # Risk/Reward multipliers - ALL REQUIRED FROM ENV VARS
    rr_sl_multiplier: float = Field(..., env="RR_SL_MULTIPLIER", description="Stop loss ATR multiplier")
    rr_tp_multiplier: float = Field(..., env="RR_TP_MULTIPLIER", description="Take profit ATR multiplier")
    scalp_rr_sl_multiplier: float = Field(..., env="SCALP_RR_SL_MULTIPLIER", description="Scalp SL multiplier")
    scalp_rr_tp_multiplier: float = Field(..., env="SCALP_RR_TP_MULTIPLIER", description="Scalp TP multiplier")
    
    # Scalping leverage settings - ALL REQUIRED FROM ENV VARS
    scalp_leverage: int = Field(..., env="SCALP_LEVERAGE", description="Leverage for scalp trades")
    swing_leverage: int = Field(..., env="SWING_LEVERAGE", description="Leverage for swing trades")
    
    # Logging - REQUIRED FROM ENV VARS
    log_level: str = Field(..., env="LOG_LEVEL", description="Logging level")
    
    # Strategy selection - REQUIRED FROM ENV VARS
    strategy_type: str = Field(..., env="STRATEGY_TYPE", description="Strategy type: scalping or aggressive")
    
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