"""
Risk and Leverage Validation
Ensures all trades use 10x leverage with 1% risk
"""
import structlog
from typing import Dict, Tuple, Optional

logger = structlog.get_logger(__name__)

class RiskValidator:
    """
    Validates that all trading parameters comply with:
    - 10x leverage requirement
    - 1% risk per trade requirement
    """
    
    def __init__(self):
        self.required_leverage = 10
        self.required_risk_percent = 1.0
        self.max_concurrent_positions = 3
        
    def validate_trade_parameters(
        self,
        account_balance: float,
        position_size: float,
        entry_price: float,
        stop_loss: float,
        leverage: int = 10
    ) -> Tuple[bool, str]:
        """
        Validate that trade parameters meet risk requirements
        
        Returns:
            (is_valid, error_message)
        """
        
        # Check leverage
        if leverage != self.required_leverage:
            return False, f"Leverage must be {self.required_leverage}x, got {leverage}x"
        
        # Calculate actual risk
        position_value = position_size * entry_price
        
        # Calculate stop distance
        if entry_price > stop_loss:  # Long position
            stop_distance_percent = (entry_price - stop_loss) / entry_price * 100
        else:  # Short position
            stop_distance_percent = (stop_loss - entry_price) / entry_price * 100
        
        # Calculate risk amount
        risk_amount = position_value * (stop_distance_percent / 100)
        risk_percent = (risk_amount / account_balance) * 100
        
        # Allow 10% tolerance for rounding
        min_risk = self.required_risk_percent * 0.9
        max_risk = self.required_risk_percent * 1.1
        
        if risk_percent < min_risk or risk_percent > max_risk:
            return False, f"Risk must be ~{self.required_risk_percent}%, calculated {risk_percent:.2f}%"
        
        # Check margin requirement
        margin_required = position_value / leverage
        if margin_required > account_balance * 0.95:
            return False, f"Margin required ${margin_required:.2f} exceeds 95% of balance ${account_balance:.2f}"
        
        logger.info(
            f"Trade validated: Risk={risk_percent:.2f}%, "
            f"Leverage={leverage}x, "
            f"Position=${position_value:.2f}, "
            f"Margin=${margin_required:.2f}"
        )
        
        return True, "Valid"
    
    def calculate_position_size_for_risk(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        qty_step: float,
        leverage: int = 10
    ) -> float:
        """
        Calculate exact position size for 1% risk
        """
        # Calculate stop distance
        if entry_price > stop_loss:  # Long
            stop_distance_percent = (entry_price - stop_loss) / entry_price * 100
        else:  # Short
            stop_distance_percent = (stop_loss - entry_price) / entry_price * 100
        
        # Calculate position value for 1% risk
        risk_amount = account_balance * (self.required_risk_percent / 100)
        position_value = risk_amount / (stop_distance_percent / 100)
        
        # Calculate quantity
        qty = position_value / entry_price
        
        # Round to qty step
        from decimal import Decimal, ROUND_DOWN
        step = Decimal(str(qty_step))
        qty_decimal = Decimal(str(qty))
        rounded = (qty_decimal / step).quantize(Decimal('1'), rounding=ROUND_DOWN) * step
        
        return float(rounded)
    
    def get_max_position_value(self, account_balance: float) -> float:
        """
        Get maximum position value with leverage
        """
        # With 10x leverage, max position value is 10x the available margin
        # Use 95% of balance for safety
        return account_balance * 0.95 * self.required_leverage
    
    def validate_total_exposure(
        self,
        account_balance: float,
        open_positions: Dict[str, Dict]
    ) -> Tuple[bool, str]:
        """
        Validate that total exposure doesn't exceed limits
        """
        if len(open_positions) >= self.max_concurrent_positions:
            return False, f"Maximum {self.max_concurrent_positions} concurrent positions allowed"
        
        total_margin = 0
        total_risk = 0
        
        for symbol, pos in open_positions.items():
            position_value = pos.get('size', 0) * pos.get('entry_price', 0)
            margin = position_value / self.required_leverage
            total_margin += margin
            
            # Assume 1% risk per position
            risk = account_balance * 0.01
            total_risk += risk
        
        if total_margin > account_balance * 0.95:
            return False, f"Total margin ${total_margin:.2f} exceeds 95% of balance"
        
        if total_risk > account_balance * 0.03:  # 3% max total risk
            return False, f"Total risk ${total_risk:.2f} exceeds 3% of balance"
        
        return True, "Valid"

# Global instance
risk_validator = RiskValidator()