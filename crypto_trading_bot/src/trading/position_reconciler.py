"""
Position Reconciliation Engine
Ensures position consistency between local state and exchange
Handles partial fills, amendments, and synchronization
"""
import asyncio
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
import hashlib
import json

logger = structlog.get_logger(__name__)

class PositionState(Enum):
    """Position lifecycle states"""
    PENDING = "pending"       # Order placed, not filled
    PARTIAL = "partial"       # Partially filled
    OPEN = "open"            # Fully filled
    CLOSING = "closing"      # Close order placed
    CLOSED = "closed"        # Position closed
    ERROR = "error"          # Sync error

class SyncStatus(Enum):
    """Synchronization status"""
    SYNCED = "synced"
    PENDING = "pending"
    MISMATCH = "mismatch"
    ERROR = "error"

@dataclass
class Position:
    """Enhanced position tracking"""
    symbol: str
    side: str  # Buy/Sell
    size: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # State management
    state: PositionState = PositionState.OPEN
    sync_status: SyncStatus = SyncStatus.SYNCED
    
    # Order tracking
    order_ids: List[str] = field(default_factory=list)
    fill_history: List[Dict] = field(default_factory=list)
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_sync: Optional[datetime] = None
    version: int = 1
    checksum: Optional[str] = None
    
    def calculate_checksum(self) -> str:
        """Calculate position checksum for comparison"""
        data = f"{self.symbol}:{self.side}:{self.size}:{self.entry_price}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'state': self.state.value,
            'sync_status': self.sync_status.value,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'version': self.version
        }

class PositionReconciler:
    """
    Position reconciliation engine with:
    - Periodic synchronization
    - Partial fill handling
    - State machine management
    - Conflict resolution
    - Audit logging
    """
    
    def __init__(self, exchange_client):
        self.exchange_client = exchange_client
        
        # Position tracking
        self.local_positions: Dict[str, Position] = {}
        self.exchange_positions: Dict[str, Dict] = {}
        self.position_locks: Dict[str, asyncio.Lock] = {}
        
        # Order tracking
        self.pending_orders: Dict[str, Dict] = {}
        self.order_to_position: Dict[str, str] = {}
        
        # Synchronization
        self.sync_interval = 30  # seconds
        self.fast_sync_interval = 5  # For pending positions
        self.last_full_sync = None
        self.sync_in_progress = False
        
        # Conflict resolution
        self.conflict_history: List[Dict] = []
        self.max_conflicts = 100
        
        # Statistics
        self.sync_count = 0
        self.mismatch_count = 0
        self.auto_resolved = 0
        self.manual_interventions = 0
        
        # Tasks
        self.sync_task = None
        self.monitor_task = None
        self.running = False
        
    async def start(self):
        """Start reconciliation engine"""
        if not self.running:
            self.running = True
            
            # Initial sync
            await self.full_sync()
            
            # Start background tasks
            self.sync_task = asyncio.create_task(self._sync_loop())
            self.monitor_task = asyncio.create_task(self._monitor_positions())
            
            logger.info("Position reconciler started")
    
    async def stop(self):
        """Stop reconciliation engine"""
        self.running = False
        
        if self.sync_task:
            self.sync_task.cancel()
        if self.monitor_task:
            self.monitor_task.cancel()
            
        logger.info("Position reconciler stopped")
    
    async def _sync_loop(self):
        """Background synchronization loop"""
        while self.running:
            try:
                # Determine sync interval
                has_pending = any(
                    p.state in [PositionState.PENDING, PositionState.PARTIAL]
                    for p in self.local_positions.values()
                )
                
                interval = self.fast_sync_interval if has_pending else self.sync_interval
                
                await asyncio.sleep(interval)
                
                # Perform sync
                await self.full_sync()
                
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(self.sync_interval)
    
    async def full_sync(self):
        """Perform full position synchronization"""
        if self.sync_in_progress:
            logger.warning("Sync already in progress, skipping")
            return
        
        try:
            self.sync_in_progress = True
            self.sync_count += 1
            
            logger.info("Starting full position sync")
            
            # Fetch exchange positions
            exchange_positions = await self.exchange_client.get_positions()
            
            # Convert to dict for easy lookup
            self.exchange_positions = {
                pos['symbol']: pos 
                for pos in exchange_positions 
                if float(pos.get('size', 0)) > 0
            }
            
            # Fetch pending orders
            pending_orders = await self.exchange_client.get_open_orders()
            self.pending_orders = {
                order['orderId']: order 
                for order in pending_orders
            }
            
            # Reconcile each position
            all_symbols = set(self.local_positions.keys()) | set(self.exchange_positions.keys())
            
            for symbol in all_symbols:
                await self._reconcile_position(symbol)
            
            # Clean up closed positions
            await self._cleanup_closed_positions()
            
            self.last_full_sync = datetime.now()
            
            logger.info(f"Full sync complete: {len(self.local_positions)} positions")
            
        except Exception as e:
            logger.error(f"Full sync failed: {e}")
            
        finally:
            self.sync_in_progress = False
    
    async def _reconcile_position(self, symbol: str):
        """Reconcile a single position"""
        
        # Get lock for symbol
        if symbol not in self.position_locks:
            self.position_locks[symbol] = asyncio.Lock()
        
        async with self.position_locks[symbol]:
            local_pos = self.local_positions.get(symbol)
            exchange_pos = self.exchange_positions.get(symbol)
            
            # Case 1: Position exists on exchange but not locally
            if exchange_pos and not local_pos:
                await self._handle_new_exchange_position(symbol, exchange_pos)
            
            # Case 2: Position exists locally but not on exchange
            elif local_pos and not exchange_pos:
                await self._handle_missing_exchange_position(symbol, local_pos)
            
            # Case 3: Position exists both places
            elif local_pos and exchange_pos:
                await self._handle_position_mismatch(symbol, local_pos, exchange_pos)
            
            # Update sync time
            if symbol in self.local_positions:
                self.local_positions[symbol].last_sync = datetime.now()
    
    async def _handle_new_exchange_position(self, symbol: str, exchange_pos: Dict):
        """Handle position that exists on exchange but not locally"""
        
        logger.warning(f"Found untracked position on exchange: {symbol}")
        
        # Create local position from exchange data
        position = Position(
            symbol=symbol,
            side=exchange_pos['side'],
            size=float(exchange_pos['size']),
            entry_price=float(exchange_pos.get('avgPrice', 0)),
            current_price=float(exchange_pos.get('markPrice', 0)),
            unrealized_pnl=float(exchange_pos.get('unrealisedPnl', 0)),
            realized_pnl=float(exchange_pos.get('realisedPnl', 0)),
            state=PositionState.OPEN,
            sync_status=SyncStatus.SYNCED
        )
        
        # Add stop loss/take profit if present
        if exchange_pos.get('stopLoss'):
            position.stop_loss = float(exchange_pos['stopLoss'])
        if exchange_pos.get('takeProfit'):
            position.take_profit = float(exchange_pos['takeProfit'])
        
        self.local_positions[symbol] = position
        self.auto_resolved += 1
        
        logger.info(f"Auto-imported position for {symbol}")
    
    async def _handle_missing_exchange_position(self, symbol: str, local_pos: Position):
        """Handle position that exists locally but not on exchange"""
        
        # Check if position is pending
        if local_pos.state == PositionState.PENDING:
            # Check if order still exists
            for order_id in local_pos.order_ids:
                if order_id in self.pending_orders:
                    # Order still pending, position is valid
                    return
            
            # No pending orders, mark as error
            local_pos.state = PositionState.ERROR
            local_pos.sync_status = SyncStatus.ERROR
            logger.error(f"Position {symbol} has no pending orders")
        
        elif local_pos.state == PositionState.CLOSING:
            # Position is being closed, mark as closed
            local_pos.state = PositionState.CLOSED
            local_pos.sync_status = SyncStatus.SYNCED
            logger.info(f"Position {symbol} closed")
        
        else:
            # Position should exist but doesn't
            logger.error(f"Position {symbol} missing from exchange")
            
            # Record conflict
            self._record_conflict(symbol, 'missing_from_exchange', local_pos.to_dict(), None)
            
            # Mark as error
            local_pos.sync_status = SyncStatus.ERROR
            self.mismatch_count += 1
    
    async def _handle_position_mismatch(self, symbol: str, local_pos: Position, exchange_pos: Dict):
        """Handle position that exists in both but may have differences"""
        
        exchange_size = float(exchange_pos['size'])
        exchange_side = exchange_pos['side']
        exchange_price = float(exchange_pos.get('avgPrice', 0))
        
        # Check for mismatches
        size_match = abs(local_pos.size - exchange_size) < 0.0001
        side_match = local_pos.side == exchange_side
        price_match = abs(local_pos.entry_price - exchange_price) < 0.01
        
        if not (size_match and side_match):
            logger.warning(f"Position mismatch for {symbol}")
            
            # Record conflict
            self._record_conflict(
                symbol,
                'position_mismatch',
                local_pos.to_dict(),
                exchange_pos
            )
            
            # Resolve based on trust policy
            if self._should_trust_exchange():
                # Update local from exchange
                local_pos.size = exchange_size
                local_pos.side = exchange_side
                local_pos.entry_price = exchange_price
                local_pos.sync_status = SyncStatus.SYNCED
                local_pos.version += 1
                
                self.auto_resolved += 1
                logger.info(f"Auto-resolved position mismatch for {symbol} using exchange data")
            else:
                # Mark for manual intervention
                local_pos.sync_status = SyncStatus.MISMATCH
                self.manual_interventions += 1
                logger.error(f"Position mismatch for {symbol} requires manual intervention")
        
        # Always update market data
        local_pos.current_price = float(exchange_pos.get('markPrice', 0))
        local_pos.unrealized_pnl = float(exchange_pos.get('unrealisedPnl', 0))
        local_pos.realized_pnl = float(exchange_pos.get('realisedPnl', 0))
        local_pos.updated_at = datetime.now()
        
        # Update risk parameters
        if exchange_pos.get('stopLoss'):
            local_pos.stop_loss = float(exchange_pos['stopLoss'])
        if exchange_pos.get('takeProfit'):
            local_pos.take_profit = float(exchange_pos['takeProfit'])
    
    async def _cleanup_closed_positions(self):
        """Remove closed positions older than 1 hour"""
        
        cutoff_time = datetime.now() - timedelta(hours=1)
        symbols_to_remove = []
        
        for symbol, position in self.local_positions.items():
            if position.state == PositionState.CLOSED:
                if position.updated_at < cutoff_time:
                    symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            del self.local_positions[symbol]
            logger.info(f"Removed closed position: {symbol}")
    
    async def _monitor_positions(self):
        """Monitor positions for risk and state changes"""
        
        while self.running:
            try:
                for symbol, position in self.local_positions.items():
                    if position.state == PositionState.OPEN:
                        # Check stop loss
                        if position.stop_loss and position.current_price:
                            if position.side == 'Buy' and position.current_price <= position.stop_loss:
                                logger.warning(f"Stop loss triggered for {symbol}")
                            elif position.side == 'Sell' and position.current_price >= position.stop_loss:
                                logger.warning(f"Stop loss triggered for {symbol}")
                        
                        # Check take profit
                        if position.take_profit and position.current_price:
                            if position.side == 'Buy' and position.current_price >= position.take_profit:
                                logger.info(f"Take profit reached for {symbol}")
                            elif position.side == 'Sell' and position.current_price <= position.take_profit:
                                logger.info(f"Take profit reached for {symbol}")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(5)
    
    def _should_trust_exchange(self) -> bool:
        """Determine if exchange data should be trusted over local"""
        # In most cases, exchange is source of truth
        # Could implement more complex logic based on:
        # - Recent order history
        # - Network issues
        # - Timing of updates
        return True
    
    def _record_conflict(self, symbol: str, conflict_type: str, local_data: Dict, exchange_data: Optional[Dict]):
        """Record position conflict for audit"""
        
        conflict = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'type': conflict_type,
            'local_data': local_data,
            'exchange_data': exchange_data
        }
        
        self.conflict_history.append(conflict)
        
        # Limit history size
        if len(self.conflict_history) > self.max_conflicts:
            self.conflict_history = self.conflict_history[-self.max_conflicts:]
        
        self.mismatch_count += 1
    
    # Public API methods
    
    async def open_position(self, symbol: str, side: str, size: float, order_id: str) -> Position:
        """Register a new position"""
        
        if symbol not in self.position_locks:
            self.position_locks[symbol] = asyncio.Lock()
        
        async with self.position_locks[symbol]:
            if symbol in self.local_positions:
                raise ValueError(f"Position already exists for {symbol}")
            
            position = Position(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=0,  # Will be updated on fill
                state=PositionState.PENDING,
                sync_status=SyncStatus.PENDING
            )
            
            position.order_ids.append(order_id)
            self.local_positions[symbol] = position
            self.order_to_position[order_id] = symbol
            
            logger.info(f"Opened position for {symbol}: {side} {size}")
            
            return position
    
    async def update_position_fill(self, order_id: str, fill_data: Dict):
        """Update position with fill information"""
        
        symbol = self.order_to_position.get(order_id)
        if not symbol:
            logger.warning(f"No position found for order {order_id}")
            return
        
        if symbol not in self.position_locks:
            self.position_locks[symbol] = asyncio.Lock()
        
        async with self.position_locks[symbol]:
            position = self.local_positions.get(symbol)
            if not position:
                return
            
            # Update fill history
            position.fill_history.append(fill_data)
            
            # Update position data
            filled_qty = float(fill_data.get('execQty', 0))
            filled_price = float(fill_data.get('execPrice', 0))
            
            if position.entry_price == 0:
                position.entry_price = filled_price
            else:
                # Calculate weighted average
                total_value = (position.size * position.entry_price) + (filled_qty * filled_price)
                position.size += filled_qty
                position.entry_price = total_value / position.size
            
            # Update state
            if fill_data.get('orderStatus') == 'Filled':
                position.state = PositionState.OPEN
                position.sync_status = SyncStatus.SYNCED
            elif fill_data.get('orderStatus') == 'PartiallyFilled':
                position.state = PositionState.PARTIAL
            
            position.updated_at = datetime.now()
            position.version += 1
            
            logger.info(f"Updated position {symbol} with fill: {filled_qty} @ {filled_price}")
    
    async def close_position(self, symbol: str) -> bool:
        """Mark position for closing"""
        
        if symbol not in self.position_locks:
            self.position_locks[symbol] = asyncio.Lock()
        
        async with self.position_locks[symbol]:
            position = self.local_positions.get(symbol)
            if not position:
                return False
            
            position.state = PositionState.CLOSING
            position.updated_at = datetime.now()
            
            logger.info(f"Closing position for {symbol}")
            
            return True
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol"""
        return self.local_positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        return self.local_positions.copy()
    
    def get_open_positions(self) -> Dict[str, Position]:
        """Get only open positions"""
        return {
            symbol: pos 
            for symbol, pos in self.local_positions.items()
            if pos.state == PositionState.OPEN
        }
    
    def get_stats(self) -> Dict:
        """Get reconciliation statistics"""
        return {
            'total_positions': len(self.local_positions),
            'open_positions': len([p for p in self.local_positions.values() if p.state == PositionState.OPEN]),
            'pending_positions': len([p for p in self.local_positions.values() if p.state == PositionState.PENDING]),
            'sync_count': self.sync_count,
            'mismatch_count': self.mismatch_count,
            'auto_resolved': self.auto_resolved,
            'manual_interventions': self.manual_interventions,
            'last_sync': self.last_full_sync.isoformat() if self.last_full_sync else None,
            'conflicts': len(self.conflict_history)
        }