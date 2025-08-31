"""
Position management system
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class Position:
    """Position information"""
    symbol: str
    side: str  # BUY or SELL
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    entry_time: datetime = field(default_factory=datetime.now)
    pnl: float = 0.0
    pnl_percent: float = 0.0
    status: str = "OPEN"  # OPEN, CLOSED, PENDING
    trade_id: Optional[str] = None  # ML tracking ID

class PositionManager:
    """Manage trading positions"""
    
    def __init__(self, max_positions: int, risk_per_trade: float, max_position_multiplier: float = 1.0):
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.max_position_multiplier = max_position_multiplier  # Max position value as multiple of balance
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        logger.info(f"Position manager initialized - Max positions: {max_positions}, Risk: {risk_per_trade*100}%")
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in this symbol"""
        return symbol in self.positions
    
    def can_open_position(self, symbol: str) -> bool:
        """Check if we can open a new position"""
        # Check if we already have a position in this symbol
        if symbol in self.positions:
            logger.warning(f"Already have position in {symbol}")
            return False
        
        # Check max positions limit
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Max positions reached ({self.max_positions})")
            return False
        
        return True
    
    def calculate_position_size(self, balance: float, entry_price: float, 
                               stop_loss: float, leverage: int = 10) -> float:
        """Calculate position size based on risk management"""
        try:
            # CRITICAL: Log all inputs for debugging
            logger.info(f"=== POSITION SIZE CALCULATION ===")
            logger.info(f"Inputs: Balance=${balance:.2f}, Entry=${entry_price:.4f}, Stop=${stop_loss:.4f}, Leverage={leverage}x")
            logger.info(f"Config: Risk={self.risk_per_trade*100:.2f}%, Max multiplier={self.max_position_multiplier}")
            
            # Risk amount in USDT (what we're willing to lose)
            risk_amount = balance * self.risk_per_trade
            logger.info(f"Target risk amount: ${risk_amount:.2f}")
            
            # Stop loss distance as percentage
            stop_distance = abs(entry_price - stop_loss) / entry_price
            
            # Minimum stop distance to prevent huge positions (0.15% minimum for scalping)
            min_stop_distance = 0.0015  # Balanced for scalping with proper risk
            if stop_distance < min_stop_distance:
                logger.warning(f"Stop distance {stop_distance*100:.3f}% too small, using minimum {min_stop_distance*100}%")
                stop_distance = min_stop_distance
            
            # Calculate position value based on risk
            # Position value = risk_amount / stop_distance
            position_value = risk_amount / stop_distance
            
            # IMMEDIATE CAP: Position value should never be more than 8x risk
            # For 0.5% risk on $250 = $1.25 risk, max position = $10
            initial_max_position = risk_amount * 8
            if position_value > initial_max_position:
                logger.warning(f"Initial position ${position_value:.2f} exceeds 8x risk ${initial_max_position:.2f}, capping")
                position_value = initial_max_position
            
            # Calculate margin required
            margin_required = position_value / leverage
            
            # IMPORTANT SAFETY CHECKS:
            
            # 1. Never use more margin than available balance
            max_margin = balance * 0.95  # Use max 95% of balance for safety
            if margin_required > max_margin:
                logger.warning(f"Margin ${margin_required:.2f} exceeds max ${max_margin:.2f}, reducing position")
                margin_required = max_margin
                position_value = margin_required * leverage
            
            # 2. Never risk more than intended
            actual_risk = position_value * stop_distance
            if actual_risk > risk_amount * 1.1:  # Allow 10% tolerance
                logger.warning(f"Actual risk ${actual_risk:.2f} exceeds intended ${risk_amount:.2f}, reducing position")
                position_value = risk_amount / stop_distance
                margin_required = position_value / leverage
            
            # 3. Apply maximum position value cap (configurable multiplier)
            max_position_value = balance * self.max_position_multiplier
            if position_value > max_position_value:
                logger.warning(f"Position value ${position_value:.2f} exceeds max ${max_position_value:.2f} ({self.max_position_multiplier}x balance), capping")
                position_value = max_position_value
                margin_required = position_value / leverage
            
            # Calculate final position size in coins
            position_size_in_coins = position_value / entry_price
            
            # 4. HARD CAP: Never let position value exceed a safe maximum
            # STRICT LIMIT: Max position should be 8x the risk amount for safety
            # For $250 balance with 0.5% risk ($1.25), max position = $10
            absolute_max_position_value = risk_amount * 8  # 8x risk amount max
            if position_value > absolute_max_position_value:
                logger.error(f"SAFETY OVERRIDE: Position value ${position_value:.2f} exceeds hard cap ${absolute_max_position_value:.2f}")
                position_value = absolute_max_position_value
                position_size_in_coins = position_value / entry_price
                margin_required = position_value / leverage
            
            # 5. Final safety check on actual risk
            final_risk = position_size_in_coins * abs(entry_price - stop_loss)
            
            # 6. ABSOLUTE MAXIMUM RISK CHECK
            absolute_max_risk = balance * 0.01  # Never risk more than 1% absolute
            if final_risk > absolute_max_risk:
                logger.error(f"RISK TOO HIGH: ${final_risk:.2f} exceeds absolute max ${absolute_max_risk:.2f}")
                # Reduce position to meet risk limit
                safe_position_size = absolute_max_risk / abs(entry_price - stop_loss)
                logger.info(f"Reducing position from {position_size_in_coins:.6f} to {safe_position_size:.6f}")
                position_size_in_coins = safe_position_size
                position_value = position_size_in_coins * entry_price
                final_risk = position_size_in_coins * abs(entry_price - stop_loss)
            
            logger.info(f"=== FINAL POSITION ===")
            logger.info(f"Position value: ${position_value:.2f} (max allowed: ${balance * self.max_position_multiplier:.2f})")
            logger.info(f"Margin required: ${margin_required:.2f}")
            logger.info(f"Position size: {position_size_in_coins:.8f} coins")
            logger.info(f"Actual risk: ${final_risk:.2f} (target: ${risk_amount:.2f})")
            
            # Validate final risk is acceptable
            if final_risk > risk_amount * 2:  # Absolutely never risk more than 2x intended
                logger.error(f"REJECTED: Final risk ${final_risk:.2f} is more than 2x target ${risk_amount:.2f}")
                return 0
            
            # FINAL SANITY CHECK: Position value should never exceed balance
            if position_value > balance:
                logger.error(f"REJECTED: Position value ${position_value:.2f} exceeds balance ${balance:.2f}")
                return 0
            
            return position_size_in_coins
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def add_position(self, symbol: str, side: str, entry_price: float, 
                    size: float, stop_loss: float, take_profit: float, trade_id: Optional[str] = None) -> bool:
        """Add a new position"""
        try:
            if not self.can_open_position(symbol):
                return False
            
            position = Position(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                size=size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trade_id=trade_id
            )
            
            self.positions[symbol] = position
            
            logger.info(f"Position opened: {symbol} {side} @ {entry_price:.4f}, "
                       f"Size: {size:.4f}, SL: {stop_loss:.4f}, TP: {take_profit:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False
    
    def update_position(self, symbol: str, current_price: float):
        """Update position P&L"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.side == "BUY":
            pnl_percent = (current_price - position.entry_price) / position.entry_price * 100
        else:  # SELL
            pnl_percent = (position.entry_price - current_price) / position.entry_price * 100
        
        position.pnl_percent = pnl_percent
        position.pnl = position.size * position.entry_price * pnl_percent / 100
    
    def should_close_position(self, symbol: str, current_price: float) -> Optional[str]:
        """Check if position should be closed"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Check stop loss
        if position.side == "BUY":
            if current_price <= position.stop_loss:
                return "STOP_LOSS"
            if current_price >= position.take_profit:
                return "TAKE_PROFIT"
        else:  # SELL
            if current_price >= position.stop_loss:
                return "STOP_LOSS"
            if current_price <= position.take_profit:
                return "TAKE_PROFIT"
        
        # Trailing stop logic (optional)
        # If profit > 2%, move stop loss to break even
        if position.pnl_percent > 2:
            if position.side == "BUY":
                new_stop = position.entry_price * 1.001  # 0.1% above entry
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    logger.info(f"Trailing stop updated for {symbol}: {new_stop:.4f}")
            else:  # SELL
                new_stop = position.entry_price * 0.999  # 0.1% below entry
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    logger.info(f"Trailing stop updated for {symbol}: {new_stop:.4f}")
        
        return None
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "MANUAL"):
        """Close a position"""
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return
        
        position = self.positions[symbol]
        
        # Final P&L calculation
        if position.side == "BUY":
            pnl_percent = (exit_price - position.entry_price) / position.entry_price * 100
        else:  # SELL
            pnl_percent = (position.entry_price - exit_price) / position.entry_price * 100
        
        position.pnl_percent = pnl_percent
        position.pnl = position.size * position.entry_price * pnl_percent / 100
        position.status = "CLOSED"
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        logger.info(f"Position closed: {symbol} @ {exit_price:.4f} "
                   f"({reason}) - P&L: {position.pnl:.2f} ({position.pnl_percent:.2f}%)")
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return list(self.positions.values())
    
    def get_total_pnl(self) -> float:
        """Get total P&L of open positions"""
        return sum(p.pnl for p in self.positions.values())
    
    def get_statistics(self) -> dict:
        """Get trading statistics"""
        total_trades = len(self.closed_positions)
        
        if total_trades == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'average_pnl': 0
            }
        
        winning_trades = len([p for p in self.closed_positions if p.pnl > 0])
        losing_trades = len([p for p in self.closed_positions if p.pnl <= 0])
        total_pnl = sum(p.pnl for p in self.closed_positions)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades * 100,
            'total_pnl': total_pnl,
            'average_pnl': total_pnl / total_trades,
            'open_positions': len(self.positions)
        }