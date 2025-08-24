"""
Unified Position Manager
Single source of truth for all position tracking across the system
"""
import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, field
import structlog
from collections import defaultdict

logger = structlog.get_logger(__name__)


@dataclass
class UnifiedPosition:
    """Unified position data structure used by all components"""
    symbol: str
    side: str  # Buy/Sell
    size: float
    entry_price: float
    current_price: float
    
    # Stop management
    stop_loss: float = 0
    take_profit: float = 0
    
    # Tracking
    order_id: str = ""
    opened_at: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    # P&L
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    fees_paid: float = 0
    
    # Risk
    risk_amount: float = 0
    position_value: float = 0
    
    # Status
    status: str = "open"  # open, closing, closed
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'order_id': self.order_id,
            'opened_at': self.opened_at.isoformat(),
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'status': self.status
        }


class UnifiedPositionManager:
    """
    Centralized position manager that all components use
    Ensures consistency across UltraIntelligentEngine, OrderManager, and position_safety
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the position manager"""
        if self._initialized:
            return
            
        self.positions: Dict[str, UnifiedPosition] = {}
        self.position_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.pending_orders: Dict[str, Dict] = {}
        self.position_history: List[UnifiedPosition] = []
        
        # Observers that get notified of position changes
        self.observers: Set[callable] = set()
        
        self._initialized = True
        logger.info("Unified Position Manager initialized")
    
    async def register_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        **kwargs
    ) -> bool:
        """
        Register a new position
        
        Args:
            symbol: Trading symbol
            side: Buy or Sell
            size: Position size
            entry_price: Entry price
            **kwargs: Additional position attributes
        
        Returns:
            Success status
        """
        async with self.position_locks[symbol]:
            if symbol in self.positions:
                logger.warning(f"Position already exists for {symbol}")
                return False
            
            position = UnifiedPosition(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                current_price=entry_price,
                position_value=size * entry_price,
                **kwargs
            )
            
            self.positions[symbol] = position
            logger.info(f"Registered position for {symbol}: {side} {size} @ {entry_price}")
            
            # Notify observers
            await self._notify_observers('position_opened', position)
            
            return True
    
    async def update_position(
        self,
        symbol: str,
        **updates
    ) -> bool:
        """
        Update an existing position
        
        Args:
            symbol: Trading symbol
            **updates: Fields to update
        
        Returns:
            Success status
        """
        async with self.position_locks[symbol]:
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return False
            
            position = self.positions[symbol]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(position, key):
                    setattr(position, key, value)
            
            position.last_update = datetime.now()
            
            # Recalculate P&L if price updated
            if 'current_price' in updates:
                self._calculate_pnl(position)
            
            # Notify observers
            await self._notify_observers('position_updated', position)
            
            return True
    
    async def close_position(
        self,
        symbol: str,
        exit_price: float = None,
        reason: str = ""
    ) -> bool:
        """
        Close a position
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for closing
        
        Returns:
            Success status
        """
        async with self.position_locks[symbol]:
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return False
            
            position = self.positions[symbol]
            
            if exit_price:
                position.current_price = exit_price
                self._calculate_pnl(position)
            
            position.status = "closed"
            
            # Move to history
            self.position_history.append(position)
            del self.positions[symbol]
            
            logger.info(
                f"Closed position for {symbol}: "
                f"P&L={position.unrealized_pnl + position.realized_pnl:.2f} "
                f"({reason})"
            )
            
            # Notify observers
            await self._notify_observers('position_closed', position)
            
            return True
    
    def has_position(self, symbol: str) -> bool:
        """Check if position exists for symbol"""
        return symbol in self.positions
    
    def get_position(self, symbol: str) -> Optional[UnifiedPosition]:
        """Get position for symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, UnifiedPosition]:
        """Get all active positions"""
        return self.positions.copy()
    
    def get_position_count(self) -> int:
        """Get number of active positions"""
        return len(self.positions)
    
    async def sync_with_exchange(self, exchange_positions: List[Dict]) -> None:
        """
        Sync positions with exchange
        
        Args:
            exchange_positions: List of positions from exchange
        """
        exchange_symbols = set()
        
        for pos_data in exchange_positions:
            symbol = pos_data.get('symbol')
            size = float(pos_data.get('size', 0))
            
            if size > 0:
                exchange_symbols.add(symbol)
                
                if symbol not in self.positions:
                    # Position exists on exchange but not locally
                    logger.warning(f"Found untracked position on exchange: {symbol}")
                    await self.register_position(
                        symbol=symbol,
                        side=pos_data.get('side'),
                        size=size,
                        entry_price=float(pos_data.get('avgPrice', 0)),
                        stop_loss=float(pos_data.get('stopLoss', 0)),
                        take_profit=float(pos_data.get('takeProfit', 0))
                    )
                else:
                    # Update existing position
                    await self.update_position(
                        symbol=symbol,
                        size=size,
                        current_price=float(pos_data.get('markPrice', 0)),
                        unrealized_pnl=float(pos_data.get('unrealisedPnl', 0))
                    )
        
        # Check for positions that exist locally but not on exchange
        for symbol in list(self.positions.keys()):
            if symbol not in exchange_symbols:
                logger.warning(f"Position {symbol} not found on exchange, removing")
                await self.close_position(symbol, reason="NOT_ON_EXCHANGE")
    
    def register_observer(self, callback: callable) -> None:
        """
        Register an observer for position changes
        
        Args:
            callback: Async function to call on position changes
        """
        self.observers.add(callback)
        logger.debug(f"Registered position observer: {callback.__name__}")
    
    def unregister_observer(self, callback: callable) -> None:
        """
        Unregister an observer
        
        Args:
            callback: Observer to remove
        """
        self.observers.discard(callback)
    
    async def _notify_observers(self, event: str, position: UnifiedPosition) -> None:
        """
        Notify all observers of position change
        
        Args:
            event: Event type (position_opened, position_updated, position_closed)
            position: The position that changed
        """
        for observer in self.observers:
            try:
                if asyncio.iscoroutinefunction(observer):
                    await observer(event, position)
                else:
                    observer(event, position)
            except Exception as e:
                logger.error(f"Error notifying observer {observer.__name__}: {e}")
    
    def _calculate_pnl(self, position: UnifiedPosition) -> None:
        """
        Calculate P&L for a position
        
        Args:
            position: Position to calculate P&L for
        """
        if position.side == "Buy":
            pnl_per_unit = position.current_price - position.entry_price
        else:
            pnl_per_unit = position.entry_price - position.current_price
        
        position.unrealized_pnl = pnl_per_unit * position.size
        position.position_value = position.size * position.current_price
    
    def get_total_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        return sum(pos.position_value for pos in self.positions.values())
    
    def get_total_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_positions_by_side(self, side: str) -> List[UnifiedPosition]:
        """Get all positions for a specific side"""
        return [
            pos for pos in self.positions.values()
            if pos.side == side
        ]
    
    async def emergency_close_all(self, reason: str = "EMERGENCY") -> int:
        """
        Emergency close all positions
        
        Args:
            reason: Reason for emergency close
        
        Returns:
            Number of positions closed
        """
        count = 0
        for symbol in list(self.positions.keys()):
            if await self.close_position(symbol, reason=reason):
                count += 1
        
        logger.warning(f"Emergency closed {count} positions ({reason})")
        return count


# Global singleton instance
unified_position_manager = UnifiedPositionManager()