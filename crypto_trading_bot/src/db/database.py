from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import asyncpg
import asyncio
from typing import AsyncGenerator, Optional
import structlog

from ..config import settings
from .models import Base

logger = structlog.get_logger(__name__)

# Sync engine for SQLAlchemy ORM with proper pooling
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=20,  # Number of persistent connections
    max_overflow=40,  # Maximum overflow connections
    pool_timeout=30,  # Timeout for getting connection from pool
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_pre_ping=True,  # Test connections before using
    isolation_level="READ_COMMITTED",  # Proper isolation for trading
    echo=False
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Async connection pool
async_pool = None

async def init_db():
    """Initialize database tables"""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Initialize async pool (optional - only if we need async operations)
        try:
            global async_pool
            # asyncpg expects postgresql:// or postgres://, not postgresql+asyncpg://
            # Just use the original database URL
            db_url = settings.database_url
            # Handle Railway's postgres:// URLs
            if db_url.startswith('postgres://'):
                # asyncpg can handle postgres:// directly
                asyncpg_url = db_url
            elif db_url.startswith('postgresql://'):
                # asyncpg can handle postgresql:// directly
                asyncpg_url = db_url
            else:
                # Fallback
                asyncpg_url = db_url
                
            async_pool = await asyncpg.create_pool(
                asyncpg_url,
                min_size=5,
                max_size=20
            )
            logger.info("Async database pool created")
        except Exception as async_error:
            # Log but don't fail - we can work without async pool
            logger.warning(f"Could not create async pool (non-critical): {async_error}")
            async_pool = None
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

async def close_db():
    """Close database connections"""
    global async_pool
    if async_pool:
        await async_pool.close()
        logger.info("Database connections closed")

@contextmanager
def get_db() -> Session:
    """Get database session (sync)"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

async def get_async_db() -> AsyncGenerator:
    """Get async database connection"""
    if async_pool:
        async with async_pool.acquire() as connection:
            async with connection.transaction():
                yield connection
    else:
        # If no async pool, return None (caller should handle)
        yield None

# Database helper functions
class DatabaseManager:
    """Database operations manager"""
    
    @staticmethod
    def create_user(chat_id: int, username: str = None) -> Optional[int]:
        """Create a new user with error handling"""
        from .models import User
        
        try:
            with get_db() as db:
                # Check if user exists
                existing = db.query(User).filter_by(telegram_chat_id=chat_id).first()
                if existing:
                    return existing.id
                
                # Create new user
                user = User(
                    telegram_chat_id=chat_id,
                    username=username
                )
                db.add(user)
                try:
                    db.commit()
                    db.refresh(user)
                except Exception as e:
                    db.rollback()
                    logger.error(f"Failed to commit user creation: {e}")
                    raise
                
                logger.info(f"Created user {user.id} for chat_id {chat_id}")
                return user.id
        except Exception as e:
            logger.error(f"Failed to create user for chat_id {chat_id}: {e}")
            return None
    
    @staticmethod
    def get_user(chat_id: int):
        """Get user by chat ID"""
        from .models import User
        
        with get_db() as db:
            return db.query(User).filter_by(telegram_chat_id=chat_id).first()
    
    @staticmethod
    def update_user_settings(chat_id: int, **kwargs):
        """Update user settings"""
        from .models import User
        
        with get_db() as db:
            user = db.query(User).filter_by(telegram_chat_id=chat_id).first()
            if user:
                for key, value in kwargs.items():
                    if hasattr(user, key):
                        setattr(user, key, value)
                try:
                    db.commit()
                    logger.info(f"Updated settings for user {user.id}")
                except Exception as e:
                    db.rollback()
                    logger.error(f"Failed to update user settings: {e}")
                    raise
    
    @staticmethod
    def log_trade(trade_data: dict) -> int:
        """Log a trade to database (sync version for backward compatibility)"""
        from .models import Trade, User, Symbol
        
        with get_db() as db:
            # Get user (use default user if chat_id not provided)
            chat_id = trade_data.get('chat_id', 0)
            if chat_id:
                user = db.query(User).filter_by(telegram_chat_id=chat_id).first()
            else:
                # Get or create default user for bot trades
                user = db.query(User).filter_by(telegram_chat_id=0).first()
                if not user:
                    user = User(telegram_chat_id=0, username='bot')
                    db.add(user)
                    db.flush()
            
            if not user:
                logger.error(f"User not found for chat_id {chat_id}")
                return None
            
            # Get or create symbol
            symbol = db.query(Symbol).filter_by(
                symbol=trade_data['symbol']
            ).first()
            
            if not symbol:
                symbol = Symbol(symbol=trade_data['symbol'])
                db.add(symbol)
                db.flush()
            
            # Create trade
            trade = Trade(
                user_id=user.id,
                symbol_id=symbol.id,
                order_id=trade_data.get('order_id'),
                side=trade_data['side'],
                entry_price=trade_data['entry_price'],
                stop_loss=trade_data.get('stop_loss'),
                take_profit_1=trade_data.get('take_profit_1'),
                take_profit_2=trade_data.get('take_profit_2'),
                position_size=trade_data.get('position_size'),
                zone_type=trade_data.get('zone_type'),
                zone_score=trade_data.get('zone_score')
            )
            
            db.add(trade)
            try:
                db.commit()
                db.refresh(trade)
                logger.info(f"Logged trade {trade.id} for {trade_data['symbol']}")
                return trade.id
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to log trade: {e}")
                raise
    
    @staticmethod
    async def log_trade_async(trade_data: dict) -> int:
        """Log a trade to database (async version)"""
        return await asyncio.get_event_loop().run_in_executor(
            None, DatabaseManager.log_trade, trade_data
        )
    
    @staticmethod
    def close_trade(trade_id: int, exit_data: dict):
        """Close a trade"""
        from .models import Trade, TradeStatus
        from datetime import datetime
        
        with get_db() as db:
            trade = db.query(Trade).filter_by(id=trade_id).first()
            if trade:
                trade.status = TradeStatus.CLOSED
                trade.exit_price = exit_data['exit_price']
                trade.exit_time = datetime.utcnow()
                trade.pnl = exit_data['pnl']
                trade.pnl_percent = exit_data['pnl_percent']
                trade.fees = exit_data.get('fees', 0)
                trade.exit_reason = exit_data.get('exit_reason', 'manual')
                
                db.commit()
                logger.info(f"Closed trade {trade_id} with PnL {trade.pnl}")
    
    @staticmethod
    def get_open_trades(chat_id: int = None):
        """Get all open trades"""
        from .models import Trade, TradeStatus, User
        
        with get_db() as db:
            query = db.query(Trade).filter_by(status=TradeStatus.OPEN)
            
            if chat_id:
                user = db.query(User).filter_by(telegram_chat_id=chat_id).first()
                if user:
                    query = query.filter_by(user_id=user.id)
            
            return query.all()
    
    @staticmethod
    def save_backtest_result(chat_id: int, result: dict) -> int:
        """Save backtest result"""
        from .models import BacktestResult, User
        
        with get_db() as db:
            user = db.query(User).filter_by(telegram_chat_id=chat_id).first()
            if not user:
                return None
            
            backtest = BacktestResult(
                user_id=user.id,
                symbol=result['symbol'],
                timeframe=result['timeframe'],
                start_date=result.get('start_date'),
                end_date=result.get('end_date'),
                initial_capital=result.get('initial_capital', 10000),
                total_trades=result['total_trades'],
                winning_trades=result.get('winning_trades', 0),
                losing_trades=result.get('losing_trades', 0),
                win_rate=result['win_rate'],
                profit_factor=result.get('profit_factor', 0),
                total_return=result['total_return'],
                max_drawdown=result['max_drawdown'],
                sharpe_ratio=result.get('sharpe_ratio', 0),
                zone_stats=result.get('zone_stats'),
                probability_scores=result.get('probability_scores')
            )
            
            db.add(backtest)
            db.commit()
            db.refresh(backtest)
            
            logger.info(f"Saved backtest result {backtest.id}")
            return backtest.id
    
    @staticmethod
    def save_zone(zone_data: dict):
        """Save supply/demand zone"""
        from .models import SupplyDemandZone, Symbol
        
        with get_db() as db:
            symbol = db.query(Symbol).filter_by(
                symbol=zone_data['symbol']
            ).first()
            
            if not symbol:
                symbol = Symbol(symbol=zone_data['symbol'])
                db.add(symbol)
                db.flush()
            
            zone = SupplyDemandZone(
                symbol_id=symbol.id,
                zone_type=zone_data['zone_type'],
                upper_bound=zone_data['upper_bound'],
                lower_bound=zone_data['lower_bound'],
                timeframe=zone_data['timeframe'],
                score=zone_data['score'],
                departure_strength=zone_data.get('departure_strength', 0),
                base_candles=zone_data.get('base_candles', 0),
                volume_ratio=zone_data.get('volume_ratio', 1)
            )
            
            db.add(zone)
            try:
                db.commit()
                logger.info(f"Saved {zone_data['zone_type']} zone for {zone_data['symbol']}")
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to save zone: {e}")
                raise
    
    @staticmethod
    def log_error(error_type: str, error_message: str, context: dict = None):
        """Log an error to database"""
        from .models import ErrorLog
        
        with get_db() as db:
            error = ErrorLog(
                error_type=error_type,
                error_message=error_message,
                context=context
            )
            db.add(error)
            db.commit()
            
            logger.error(f"Logged error: {error_type} - {error_message}")
    
    @staticmethod
    async def update_daily_stats(stats: dict):
        """Update daily statistics"""
        from .models import DailyStats
        from datetime import datetime, date
        
        with get_db() as db:
            today = date.today()
            daily = db.query(DailyStats).filter(
                DailyStats.date == today
            ).first()
            
            if not daily:
                daily = DailyStats(date=today)
                db.add(daily)
            
            # Update stats
            daily.total_trades = stats.get('trades', 0)
            daily.gross_pnl = stats.get('pnl', 0)
            daily.fees = stats.get('fees', 0)
            daily.net_pnl = daily.gross_pnl - daily.fees
            
            db.commit()
            logger.info(f"Updated daily stats for {today}")
    
    @staticmethod
    def get_closed_trades_today():
        """Get all closed trades from today"""
        from .models import Trade, TradeStatus
        from datetime import datetime, date
        
        with get_db() as db:
            today = date.today()
            return db.query(Trade).filter(
                Trade.status == TradeStatus.CLOSED,
                Trade.exit_time >= today
            ).all()
    
    @staticmethod
    def get_ml_model(model_name: str):
        """Get ML model from database"""
        from .models import MLModel
        
        with get_db() as db:
            return db.query(MLModel).filter_by(model_name=model_name).first()
    
    @staticmethod
    def create_ml_model(**kwargs):
        """Create new ML model record"""
        from .models import MLModel
        
        with get_db() as db:
            model = MLModel(**kwargs)
            db.add(model)
            try:
                db.commit()
                db.refresh(model)
                return model
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to create ML model: {e}")
                raise
    
    @staticmethod
    def update_ml_model(model_name: str, **kwargs):
        """Update existing ML model"""
        from .models import MLModel
        
        with get_db() as db:
            model = db.query(MLModel).filter_by(model_name=model_name).first()
            if model:
                for key, value in kwargs.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                try:
                    db.commit()
                    db.refresh(model)
                    return model
                except Exception as e:
                    db.rollback()
                    logger.error(f"Failed to update ML model: {e}")
                    raise
            return None
    
    @staticmethod
    def list_ml_models():
        """List all ML models"""
        from .models import MLModel
        
        with get_db() as db:
            return db.query(MLModel).order_by(MLModel.updated_at.desc()).all()
    
    @staticmethod
    def delete_ml_model(model_name: str):
        """Delete ML model"""
        from .models import MLModel
        
        with get_db() as db:
            model = db.query(MLModel).filter_by(model_name=model_name).first()
            if model:
                try:
                    db.delete(model)
                    db.commit()
                    return True
                except Exception as e:
                    db.rollback()
                    logger.error(f"Failed to delete ML model: {e}")
                    raise
            return False
    
    @staticmethod
    def get_daily_trades():
        """Get all trades from today"""
        from .models import Trade
        from datetime import datetime, date
        
        with get_db() as db:
            today = date.today()
            return db.query(Trade).filter(
                Trade.entry_time >= today
            ).all()