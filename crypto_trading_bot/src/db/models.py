from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    JSON, ForeignKey, Index, Enum as SQLEnum, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

class UserRole(enum.Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

class TradeStatus(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    telegram_chat_id = Column(Integer, unique=True, nullable=False)
    username = Column(String(100))
    role = Column(SQLEnum(UserRole), default=UserRole.USER)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Settings
    risk_percent = Column(Float, default=1.0)
    max_concurrent_positions = Column(Integer, default=5)
    max_daily_loss_percent = Column(Float, default=5.0)
    default_leverage = Column(Integer, default=3)
    trading_enabled = Column(Boolean, default=False)
    
    # Relationships
    trades = relationship("Trade", back_populates="user")
    backtest_results = relationship("BacktestResult", back_populates="user")
    
    __table_args__ = (
        Index('idx_telegram_chat_id', telegram_chat_id),
    )

class Symbol(Base):
    __tablename__ = 'symbols'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False)
    base_coin = Column(String(10))
    quote_coin = Column(String(10))
    tick_size = Column(Float)
    qty_step = Column(Float)
    min_notional = Column(Float)
    max_leverage = Column(Integer)
    enabled = Column(Boolean, default=True)
    margin_mode = Column(String(20), default='isolated')
    leverage = Column(Integer)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    trades = relationship("Trade", back_populates="symbol_ref")
    zones = relationship("SupplyDemandZone", back_populates="symbol_ref")
    
    __table_args__ = (
        Index('idx_symbol', symbol),
        Index('idx_enabled', enabled),
    )

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    symbol_id = Column(Integer, ForeignKey('symbols.id'))
    
    # Trade details
    order_id = Column(String(50))
    side = Column(String(10))  # Buy/Sell
    entry_price = Column(Float)
    exit_price = Column(Float)
    stop_loss = Column(Float)
    take_profit_1 = Column(Float)
    take_profit_2 = Column(Float)
    position_size = Column(Float)
    
    # Status
    status = Column(SQLEnum(TradeStatus), default=TradeStatus.OPEN)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime)
    
    # Performance
    pnl = Column(Float, default=0)
    pnl_percent = Column(Float, default=0)
    fees = Column(Float, default=0)
    
    # Strategy info
    zone_type = Column(String(20))  # demand/supply
    zone_score = Column(Float)
    exit_reason = Column(String(50))
    
    # Metadata
    tp1_hit = Column(Boolean, default=False)
    breakeven_moved = Column(Boolean, default=False)
    trailing_activated = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="trades")
    symbol_ref = relationship("Symbol", back_populates="trades")
    
    __table_args__ = (
        Index('idx_user_id', user_id),
        Index('idx_symbol_id', symbol_id),
        Index('idx_status', status),
        Index('idx_entry_time', entry_time),
    )

class SupplyDemandZone(Base):
    __tablename__ = 'supply_demand_zones'
    
    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'))
    
    # Zone details
    zone_type = Column(String(20))  # demand/supply
    upper_bound = Column(Float)
    lower_bound = Column(Float)
    timeframe = Column(String(10))
    
    # Scoring
    score = Column(Float)
    touches = Column(Integer, default=0)
    departure_strength = Column(Float)
    base_candles = Column(Integer)
    volume_ratio = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_touched_at = Column(DateTime)
    invalidated_at = Column(DateTime)
    
    # Relationships
    symbol_ref = relationship("Symbol", back_populates="zones")
    
    __table_args__ = (
        Index('idx_zone_symbol', symbol_id),
        Index('idx_zone_active', is_active),
        Index('idx_zone_type', zone_type),
        Index('idx_zone_created', created_at),
    )

class BacktestResult(Base):
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    # Backtest parameters
    symbol = Column(String(20))
    timeframe = Column(String(10))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    initial_capital = Column(Float)
    
    # Results
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    total_return = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    
    # Detailed metrics (stored as JSON)
    zone_stats = Column(JSON)
    probability_scores = Column(JSON)
    trade_details = Column(JSON)
    equity_curve = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    execution_time_seconds = Column(Float)
    
    # Relationships
    user = relationship("User", back_populates="backtest_results")
    
    __table_args__ = (
        Index('idx_backtest_user', user_id),
        Index('idx_backtest_created', created_at),
    )

class DailyStats(Base):
    __tablename__ = 'daily_stats'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    
    # Trading stats
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    
    # PnL
    gross_pnl = Column(Float, default=0)
    fees = Column(Float, default=0)
    net_pnl = Column(Float, default=0)
    
    # Best/Worst
    best_trade_pnl = Column(Float)
    worst_trade_pnl = Column(Float)
    best_symbol = Column(String(20))
    worst_symbol = Column(String(20))
    
    # Risk metrics
    max_drawdown = Column(Float)
    max_exposure = Column(Float)
    
    __table_args__ = (
        Index('idx_daily_date', date, unique=True),
    )

class ErrorLog(Base):
    __tablename__ = 'error_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    error_type = Column(String(50))
    error_message = Column(Text)
    context = Column(JSON)
    resolved = Column(Boolean, default=False)
    
    __table_args__ = (
        Index('idx_error_timestamp', timestamp),
        Index('idx_error_type', error_type),
        Index('idx_error_resolved', resolved),
    )