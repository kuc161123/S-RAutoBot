import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import structlog

from ..api.bybit_client import BybitClient
from ..strategy.supply_demand import TradingSignal, Zone
from ..config import settings
from ..utils.rounding import (
    calculate_position_size,
    calculate_breakeven_price,
    round_to_tick,
    round_to_qty_step
)

logger = structlog.get_logger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    PLACED = "placed"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PositionStatus(Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"

@dataclass
class ManagedOrder:
    """Order being managed by the system"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    qty: float
    price: Optional[float]
    status: OrderStatus
    created_at: datetime = field(default_factory=datetime.now)
    filled_qty: float = 0
    avg_fill_price: float = 0
    
@dataclass
class ManagedPosition:
    """Position being managed by the system"""
    symbol: str
    side: str
    entry_price: float
    current_qty: float
    original_qty: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    zone: Zone
    status: PositionStatus = PositionStatus.OPEN
    entry_time: datetime = field(default_factory=datetime.now)
    tp1_hit: bool = False
    tp2_hit: bool = False
    breakeven_moved: bool = False
    trailing_activated: bool = False
    realized_pnl: float = 0
    fees_paid: float = 0

class OrderManager:
    """Manages order lifecycle and position management"""
    
    def __init__(self, bybit_client: BybitClient):
        self.client = bybit_client
        self.active_orders: Dict[str, ManagedOrder] = {}
        self.active_positions: Dict[str, ManagedPosition] = {}
        self.daily_stats = {
            'trades': 0,
            'pnl': 0,
            'fees': 0,
            'max_drawdown': 0,
            'current_exposure': 0
        }
        self.position_locks: Dict[str, asyncio.Lock] = {}
        
    async def execute_signal(self, signal: TradingSignal, account_balance: float) -> Optional[str]:
        """Execute a trading signal"""
        try:
            symbol = signal.zone.symbol
            
            # Check if we already have a position for this symbol
            if symbol in self.active_positions:
                logger.warning(f"Position already exists for {symbol}")
                return None
            
            # Check exposure limits
            if not await self._check_exposure_limits(signal, account_balance):
                logger.warning(f"Exposure limits exceeded for {symbol}")
                return None
            
            # Get or create lock for this symbol
            if symbol not in self.position_locks:
                self.position_locks[symbol] = asyncio.Lock()
            
            async with self.position_locks[symbol]:
                # Double-check position doesn't exist
                if symbol in self.active_positions:
                    return None
                
                # Place the entry order
                order_id = await self._place_entry_order(signal)
                
                if order_id:
                    # Create managed position
                    position = ManagedPosition(
                        symbol=symbol,
                        side=signal.side,
                        entry_price=signal.entry_price,
                        current_qty=signal.position_size,
                        original_qty=signal.position_size,
                        stop_loss=signal.stop_loss,
                        take_profit_1=signal.take_profit_1,
                        take_profit_2=signal.take_profit_2,
                        zone=signal.zone
                    )
                    
                    self.active_positions[symbol] = position
                    logger.info(f"Position opened for {symbol}: {signal.side} {signal.position_size} @ {signal.entry_price}")
                    
                    # Set initial stop loss and take profit
                    await self._set_position_stops(position)
                    
                    return order_id
                    
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return None
    
    async def _check_exposure_limits(self, signal: TradingSignal, account_balance: float) -> bool:
        """Check if we can take this trade based on exposure limits"""
        
        # Check one position per symbol rule (no total position limit)
        if signal.zone.symbol in self.active_positions:
            logger.warning(f"Position already exists for {signal.zone.symbol}")
            return False
        
        # Check daily loss limit
        daily_loss_percent = abs(self.daily_stats['pnl']) / account_balance * 100
        if daily_loss_percent >= settings.max_daily_loss_percent:
            logger.warning(f"Daily loss limit reached: {daily_loss_percent:.2f}%")
            return False
        
        # Check current exposure
        position_value = signal.position_size * signal.entry_price
        total_exposure = self.daily_stats['current_exposure'] + position_value
        max_exposure = account_balance * 0.5  # Max 50% of account
        
        if total_exposure > max_exposure:
            return False
        
        return True
    
    async def _place_entry_order(self, signal: TradingSignal) -> Optional[str]:
        """Place entry order for a signal"""
        try:
            instrument = self.client.get_instrument(signal.zone.symbol)
            if not instrument:
                logger.error(f"Unknown symbol: {signal.zone.symbol}")
                return None
            
            # Determine position index for hedge mode
            position_idx = None
            if settings.default_position_mode == "hedge":
                position_idx = 1 if signal.side == "Buy" else 2
            
            # Place market order for immediate entry
            order_id = await self.client.place_order(
                symbol=signal.zone.symbol,
                side=signal.side,
                qty=signal.position_size,
                order_type="Market",
                position_idx=position_idx
            )
            
            if order_id:
                # Track the order
                order = ManagedOrder(
                    order_id=order_id,
                    symbol=signal.zone.symbol,
                    side=signal.side,
                    order_type="Market",
                    qty=signal.position_size,
                    price=None,
                    status=OrderStatus.PLACED
                )
                self.active_orders[order_id] = order
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing entry order: {e}")
            return None
    
    async def _set_position_stops(self, position: ManagedPosition) -> bool:
        """Set or update stop loss and take profit for a position"""
        try:
            # Determine position index for hedge mode
            position_idx = None
            if settings.default_position_mode == "hedge":
                position_idx = 1 if position.side == "Buy" else 2
            
            # Set trading stops
            success = await self.client.set_trading_stop(
                symbol=position.symbol,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit_1 if not position.tp1_hit else position.take_profit_2,
                position_idx=position_idx
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error setting position stops: {e}")
            return False
    
    async def update_positions(self, market_data: Dict[str, float]) -> None:
        """Update positions based on current market data"""
        for symbol, position in list(self.active_positions.items()):
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol]
            
            # Check for TP1 hit
            if not position.tp1_hit:
                if (position.side == "Buy" and current_price >= position.take_profit_1) or \
                   (position.side == "Sell" and current_price <= position.take_profit_1):
                    await self._handle_tp1_hit(position)
            
            # Check for breakeven move
            if settings.move_stop_to_breakeven_at_tp1 and position.tp1_hit and not position.breakeven_moved:
                await self._move_stop_to_breakeven(position)
            
            # Check for trailing stop activation
            if settings.use_trailing_stop and not position.trailing_activated:
                await self._check_trailing_activation(position, current_price)
    
    async def _handle_tp1_hit(self, position: ManagedPosition) -> None:
        """Handle TP1 being hit - partial close"""
        try:
            # Calculate partial close quantity
            partial_qty = position.current_qty * (settings.partial_tp1_percent / 100)
            partial_qty = round_to_qty_step(
                partial_qty, 
                float(self.client.get_instrument(position.symbol)['qty_step'])
            )
            
            # Place reduce-only order to take partial profit
            close_side = "Sell" if position.side == "Buy" else "Buy"
            
            order_id = await self.client.place_order(
                symbol=position.symbol,
                side=close_side,
                qty=partial_qty,
                order_type="Market",
                reduce_only=True
            )
            
            if order_id:
                position.tp1_hit = True
                position.current_qty -= partial_qty
                
                # Calculate approximate PnL
                pnl = partial_qty * (position.take_profit_1 - position.entry_price)
                if position.side == "Sell":
                    pnl = -pnl
                position.realized_pnl += pnl
                
                logger.info(f"TP1 hit for {position.symbol}: Closed {partial_qty} units, PnL: ${pnl:.2f}")
                
                # Update stops for remaining position
                await self._set_position_stops(position)
                
        except Exception as e:
            logger.error(f"Error handling TP1: {e}")
    
    async def _move_stop_to_breakeven(self, position: ManagedPosition) -> None:
        """Move stop loss to breakeven after TP1"""
        try:
            instrument = self.client.get_instrument(position.symbol)
            
            # Calculate breakeven including fees
            breakeven = calculate_breakeven_price(
                position.entry_price,
                float(instrument['tick_size']),
                fee_percent=0.06  # Taker fee
            )
            
            # Only move if it's better than current stop
            if (position.side == "Buy" and breakeven > position.stop_loss) or \
               (position.side == "Sell" and breakeven < position.stop_loss):
                
                position.stop_loss = breakeven
                position.breakeven_moved = True
                
                # Update on exchange
                await self._set_position_stops(position)
                
                logger.info(f"Moved stop to breakeven for {position.symbol} at {breakeven:.4f}")
                
        except Exception as e:
            logger.error(f"Error moving stop to breakeven: {e}")
    
    async def _check_trailing_activation(self, position: ManagedPosition, current_price: float) -> None:
        """Check if trailing stop should be activated"""
        try:
            # Calculate profit percentage
            if position.side == "Buy":
                profit_percent = (current_price - position.entry_price) / position.entry_price * 100
            else:
                profit_percent = (position.entry_price - current_price) / position.entry_price * 100
            
            # Activate trailing if profit exceeds threshold
            if profit_percent >= settings.trailing_stop_activation_percent:
                
                # Calculate trailing distance in basis points
                trailing_distance = settings.trailing_stop_callback_percent * 100
                
                success = await self.client.set_trading_stop(
                    symbol=position.symbol,
                    trailing_stop=trailing_distance
                )
                
                if success:
                    position.trailing_activated = True
                    logger.info(f"Trailing stop activated for {position.symbol} at {profit_percent:.2f}% profit")
                    
        except Exception as e:
            logger.error(f"Error checking trailing activation: {e}")
    
    async def handle_order_update(self, update: Dict[str, Any]) -> None:
        """Handle order status update from WebSocket"""
        try:
            order_id = update.get('orderId')
            if order_id not in self.active_orders:
                return
            
            order = self.active_orders[order_id]
            status = update.get('orderStatus', '').lower()
            
            # Update order status
            if status == 'filled':
                order.status = OrderStatus.FILLED
                order.filled_qty = float(update.get('cumExecQty', 0))
                order.avg_fill_price = float(update.get('avgPrice', 0))
                
                logger.info(f"Order {order_id} filled: {order.filled_qty} @ {order.avg_fill_price}")
                
            elif status == 'cancelled':
                order.status = OrderStatus.CANCELLED
                logger.info(f"Order {order_id} cancelled")
                
            elif status == 'rejected':
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order {order_id} rejected: {update.get('rejectReason')}")
                
        except Exception as e:
            logger.error(f"Error handling order update: {e}")
    
    async def handle_position_update(self, update: Dict[str, Any]) -> None:
        """Handle position update from WebSocket"""
        try:
            symbol = update.get('symbol')
            if symbol not in self.active_positions:
                return
            
            position = self.active_positions[symbol]
            size = float(update.get('size', 0))
            
            # Check if position closed
            if size == 0:
                position.status = PositionStatus.CLOSED
                
                # Calculate final PnL
                unrealized_pnl = float(update.get('unrealisedPnl', 0))
                position.realized_pnl += unrealized_pnl
                
                # Update daily stats
                self.daily_stats['trades'] += 1
                self.daily_stats['pnl'] += position.realized_pnl
                self.daily_stats['fees'] += position.fees_paid
                
                # Remove from active positions
                del self.active_positions[symbol]
                
                logger.info(f"Position closed for {symbol}: PnL ${position.realized_pnl:.2f}")
                
            else:
                # Update position size
                position.current_qty = size
                
        except Exception as e:
            logger.error(f"Error handling position update: {e}")
    
    async def close_position(self, symbol: str, reason: str = "manual") -> bool:
        """Manually close a position"""
        try:
            if symbol not in self.active_positions:
                logger.warning(f"No position found for {symbol}")
                return False
            
            position = self.active_positions[symbol]
            
            # Place market order to close
            close_side = "Sell" if position.side == "Buy" else "Buy"
            
            order_id = await self.client.place_order(
                symbol=symbol,
                side=close_side,
                qty=position.current_qty,
                order_type="Market",
                reduce_only=True
            )
            
            if order_id:
                position.status = PositionStatus.CLOSING
                logger.info(f"Closing position for {symbol}: {reason}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    async def close_all_positions(self, reason: str = "manual") -> int:
        """Close all open positions"""
        closed = 0
        for symbol in list(self.active_positions.keys()):
            if await self.close_position(symbol, reason):
                closed += 1
        return closed
    
    async def emergency_stop(self) -> None:
        """Emergency stop - close all positions and cancel all orders"""
        logger.warning("EMERGENCY STOP ACTIVATED")
        
        # Cancel all pending orders
        for order_id, order in list(self.active_orders.items()):
            if order.status in [OrderStatus.PENDING, OrderStatus.PLACED]:
                await self.client.cancel_order(order.symbol, order_id)
        
        # Close all positions
        closed = await self.close_all_positions("emergency_stop")
        logger.warning(f"Emergency stop: Closed {closed} positions")
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily trading statistics"""
        return {
            **self.daily_stats,
            'active_positions': len(self.active_positions),
            'active_orders': len([o for o in self.active_orders.values() 
                                if o.status in [OrderStatus.PENDING, OrderStatus.PLACED]])
        }
    
    async def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at UTC midnight)"""
        self.daily_stats = {
            'trades': 0,
            'pnl': 0,
            'fees': 0,
            'max_drawdown': 0,
            'current_exposure': sum(
                p.current_qty * p.entry_price 
                for p in self.active_positions.values()
            )
        }
        logger.info("Daily statistics reset")