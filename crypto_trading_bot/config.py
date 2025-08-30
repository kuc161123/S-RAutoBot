"""
Configuration management for the crypto trading bot
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # Bybit API Configuration
    bybit_api_key: str = Field(..., env="BYBIT_API_KEY")
    bybit_api_secret: str = Field(..., env="BYBIT_API_SECRET")
    bybit_testnet: bool = Field(True, env="BYBIT_TESTNET")
    
    # Telegram Configuration
    telegram_bot_token: str = Field(..., env="TELEGRAM_BOT_TOKEN")
    telegram_chat_ids: List[int] = Field(default_factory=list, env="TELEGRAM_CHAT_IDS")
    
    # Trading Configuration
    initial_symbols: List[str] = Field(
        default=[
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "LINKUSDT"
        ],
        description="Start with top 10 symbols"
    )
    
    # Risk Management
    risk_per_trade: float = Field(0.01, ge=0.001, le=0.05, description="Risk 1% per trade")
    max_positions: int = Field(10, ge=1, le=50, description="Maximum concurrent positions")
    leverage: int = Field(10, ge=1, le=20, description="Trading leverage")
    
    # Strategy Parameters
    rsi_period: int = Field(14, description="RSI period")
    rsi_oversold: float = Field(30.0, description="RSI oversold level")
    rsi_overbought: float = Field(70.0, description="RSI overbought level")
    macd_fast: int = Field(12, description="MACD fast period")
    macd_slow: int = Field(26, description="MACD slow period")
    macd_signal: int = Field(9, description="MACD signal period")
    
    # System Configuration
    scan_interval: int = Field(60, description="Scan interval in seconds")
    min_volume_24h: float = Field(1000000, description="Minimum 24h volume in USDT")
    startup_delay: int = Field(5, description="Startup delay in seconds")
    
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