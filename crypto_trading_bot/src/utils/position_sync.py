"""
Position Synchronization Utility
Ensures position tracking is synchronized with actual exchange positions
"""
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)

class PositionSynchronizer:
    """
    Synchronizes internal position tracking with actual exchange positions
    Prevents phantom positions and ensures accurate tracking
    """
    
    def __init__(self, bybit_client):
        self.client = bybit_client
        self.last_sync = datetime.now()
        self.sync_interval = 60  # Sync every 60 seconds
        
    async def sync_positions(self, position_manager: Dict) -> Dict[str, Dict]:
        """
        Sync internal position tracking with exchange
        Returns corrected position dictionary
        """
        try:
            # Get actual positions from exchange
            actual_positions = await self.client.get_positions()
            
            # Build dictionary of actual positions
            exchange_positions = {}
            for pos in actual_positions:
                symbol = pos.get('symbol')
                size = float(pos.get('size', 0))
                
                if size > 0:
                    side = pos.get('side', '').lower()
                    exchange_positions[symbol] = {
                        'side': 'long' if side == 'buy' else 'short',
                        'size': size,
                        'avgPrice': float(pos.get('avgPrice', 0)),
                        'unrealisedPnl': float(pos.get('unrealisedPnl', 0))
                    }
            
            # Clear phantom positions (in tracker but not on exchange)
            symbols_to_clear = []
            for symbol in position_manager.keys():
                if symbol not in exchange_positions:
                    logger.warning(f"Clearing phantom position for {symbol} - not found on exchange")
                    symbols_to_clear.append(symbol)
            
            # Clear them
            for symbol in symbols_to_clear:
                del position_manager[symbol]
            
            # Add missing positions (on exchange but not in tracker)
            for symbol, pos_data in exchange_positions.items():
                if symbol not in position_manager:
                    logger.info(f"Adding missing position for {symbol} from exchange")
                    position_manager[symbol] = pos_data
            
            self.last_sync = datetime.now()
            
            logger.info(f"Position sync complete: {len(exchange_positions)} actual positions, "
                       f"cleared {len(symbols_to_clear)} phantom positions")
            
            return position_manager
            
        except Exception as e:
            logger.error(f"Position sync failed: {e}")
            return position_manager
    
    def should_sync(self) -> bool:
        """Check if it's time to sync"""
        return (datetime.now() - self.last_sync).total_seconds() > self.sync_interval
    
    async def clear_all_tracking(self, position_safety_manager) -> None:
        """
        Clear all position tracking and resync from exchange
        Use when starting up or after errors
        """
        try:
            # Get actual positions
            actual_positions = await self.client.get_positions()
            
            # Clear all existing tracking
            position_safety_manager.active_positions.clear()
            position_safety_manager.pending_orders.clear()
            
            # Re-register actual positions
            for pos in actual_positions:
                symbol = pos.get('symbol')
                size = float(pos.get('size', 0))
                
                if size > 0:
                    side = 'Buy' if pos.get('side', '').lower() == 'buy' else 'Sell'
                    position_safety_manager.register_position(symbol, {
                        'side': side,
                        'size': size,
                        'entry_price': float(pos.get('avgPrice', 0))
                    })
            
            logger.info(f"Cleared and resynced position tracking: {len(position_safety_manager.active_positions)} active positions")
            
        except Exception as e:
            logger.error(f"Failed to clear position tracking: {e}")