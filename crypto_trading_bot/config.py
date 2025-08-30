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
    
    # Trading Configuration
    initial_symbols: List[str] = Field(
        default=[
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "LINKUSDT"
        ],
        description="Start with top 10 symbols"
    )
    
    # Risk Management (adjusted for scalping)
    risk_per_trade: float = Field(0.005, ge=0.001, le=1.0, description="Risk per trade (0.005 = 0.5% for scalping)")
    max_positions: int = Field(15, ge=1, le=200, description="Maximum concurrent positions (more for scalping)")
    leverage: int = Field(5, ge=1, le=125, description="Trading leverage (lower for scalping safety)")
    
    # Strategy Parameters
    rsi_period: int = Field(14, description="RSI period")
    rsi_oversold: float = Field(30.0, description="RSI oversold level")
    rsi_overbought: float = Field(70.0, description="RSI overbought level")
    macd_fast: int = Field(12, description="MACD fast period")
    macd_slow: int = Field(26, description="MACD slow period")
    macd_signal: int = Field(9, description="MACD signal period")
    
    # System Configuration (optimized for scalping)
    scan_interval: int = Field(30, description="Scan interval in seconds (faster for scalping)")
    min_volume_24h: float = Field(5000000, description="Minimum 24h volume in USDT (higher for liquidity)")
    startup_delay: int = Field(5, description="Startup delay in seconds")
    
    # Scalping specific parameters
    scalp_timeframe: str = Field("5", description="Timeframe for scalping (5m candles)")
    scalp_profit_target: float = Field(0.003, description="Quick profit target (0.3%)")
    scalp_stop_loss: float = Field(0.002, description="Tight stop loss (0.2%)")
    min_risk_reward: float = Field(1.2, description="Minimum risk/reward ratio")
    
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