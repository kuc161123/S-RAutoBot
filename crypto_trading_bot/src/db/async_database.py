"""
Async Database Manager
Provides async wrappers for database operations to prevent blocking the event loop
"""
import asyncio
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import structlog
from .database import DatabaseManager, Session, get_db
from .models import User, Trade, MLModel

logger = structlog.get_logger(__name__)

# Thread pool for blocking database operations
executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="db")


class AsyncDatabaseManager:
    """
    Async wrapper for DatabaseManager
    Runs synchronous database operations in thread pool to avoid blocking event loop
    """
    
    @staticmethod
    async def create_user(chat_id: int, username: str = None) -> Optional[int]:
        """Create user asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            DatabaseManager.create_user,
            chat_id,
            username
        )
    
    @staticmethod
    async def get_user(chat_id: int) -> Optional[User]:
        """Get user asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            DatabaseManager.get_user,
            chat_id
        )
    
    @staticmethod
    async def update_user_settings(chat_id: int, **kwargs) -> bool:
        """Update user settings asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            DatabaseManager.update_user_settings,
            chat_id,
            **kwargs
        )
    
    @staticmethod
    async def create_trade(**kwargs) -> Optional[int]:
        """Create trade asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            DatabaseManager.create_trade,
            **kwargs
        )
    
    @staticmethod
    async def update_trade(trade_id: int, **kwargs) -> bool:
        """Update trade asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            DatabaseManager.update_trade,
            trade_id,
            **kwargs
        )
    
    @staticmethod
    async def get_open_trades(chat_id: int = None) -> List[Trade]:
        """Get open trades asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            DatabaseManager.get_open_trades,
            chat_id
        )
    
    @staticmethod
    async def close_trade(trade_id: int, **kwargs) -> bool:
        """Close trade asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            DatabaseManager.close_trade,
            trade_id,
            **kwargs
        )
    
    @staticmethod
    async def get_ml_model(model_name: str) -> Optional[MLModel]:
        """Get ML model asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            DatabaseManager.get_ml_model,
            model_name
        )
    
    @staticmethod
    async def create_ml_model(**kwargs) -> bool:
        """Create ML model asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            DatabaseManager.create_ml_model,
            **kwargs
        )
    
    @staticmethod
    async def update_ml_model(model_name: str, **kwargs) -> bool:
        """Update ML model asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            DatabaseManager.update_ml_model,
            model_name,
            **kwargs
        )
    
    @staticmethod
    async def delete_ml_model(model_name: str) -> bool:
        """Delete ML model asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            DatabaseManager.delete_ml_model,
            model_name
        )
    
    @staticmethod
    async def list_ml_models() -> List[MLModel]:
        """List ML models asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            DatabaseManager.list_ml_models
        )
    
    @staticmethod
    async def get_trade_by_order_id(order_id: str) -> Optional[Trade]:
        """Get trade by order ID asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            DatabaseManager.get_trade_by_order_id,
            order_id
        )
    
    @staticmethod
    async def get_daily_stats() -> Dict[str, Any]:
        """Get daily stats asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            DatabaseManager.get_daily_stats
        )
    
    @staticmethod
    async def update_daily_stats(stats: dict) -> bool:
        """Update daily stats (already async in original)"""
        return await DatabaseManager.update_daily_stats(stats)
    
    @staticmethod
    async def execute_with_transaction(operations: List[callable]) -> bool:
        """
        Execute multiple operations in a single transaction
        
        Args:
            operations: List of database operations to execute atomically
        
        Returns:
            Success status
        """
        def _execute_transaction():
            session = get_db()
            try:
                for operation in operations:
                    operation(session)
                session.commit()
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"Transaction failed: {e}")
                return False
            finally:
                session.close()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, _execute_transaction)
    
    @staticmethod
    def cleanup():
        """Cleanup thread pool executor"""
        executor.shutdown(wait=True)


# Global async database manager instance
async_db = AsyncDatabaseManager()