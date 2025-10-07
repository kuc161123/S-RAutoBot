from position_mgr import RiskConfig, round_step
import logging

logger = logging.getLogger(__name__)

class Sizer:
    def __init__(self, risk:RiskConfig, account_balance:float=None, fee_total_pct:float=0.0, include_fees:bool=False):
        self.risk = risk
        self.account_balance = account_balance
        self.fee_total_pct = float(fee_total_pct or 0.0)
        self.include_fees = bool(include_fees)

    def qty_for(self, entry:float, sl:float, qty_step:float, min_qty:float, min_order_value:float=5.0, ml_score:float=None) -> float:
        R = abs(entry - sl)
        if R <= 0: return 0.0
        
        # Calculate actual risk percentage
        if self.risk.use_ml_dynamic_risk and ml_score is not None:
            # Linear scaling based on ML score
            score_range = self.risk.ml_risk_max_score - self.risk.ml_risk_min_score
            risk_range = self.risk.ml_risk_max_percent - self.risk.ml_risk_min_percent
            
            # Clamp ML score to valid range
            clamped_score = max(self.risk.ml_risk_min_score, 
                              min(self.risk.ml_risk_max_score, ml_score))
            
            # Linear interpolation
            if score_range > 0:
                score_position = (clamped_score - self.risk.ml_risk_min_score) / score_range
                actual_risk_percent = self.risk.ml_risk_min_percent + (score_position * risk_range)
            else:
                actual_risk_percent = self.risk.ml_risk_min_percent
            
            logger.info(f"ML Dynamic Risk: Score={ml_score:.1f} → Risk={actual_risk_percent:.2f}%")
        else:
            actual_risk_percent = self.risk.risk_percent
        
        # Calculate risk amount based on mode
        if self.risk.use_percent_risk and self.account_balance and self.account_balance > 0:
            # Use percentage of account balance
            risk_amount = self.account_balance * (actual_risk_percent / 100.0)
        else:
            # Fall back to fixed USD risk
            risk_amount = self.risk.risk_usd
        
        # Incorporate estimated fees if enabled (approx entry + exit)
        if self.include_fees and self.fee_total_pct > 0.0:
            try:
                expected_exit = sl  # worst-case exit at stop
                fee_per_unit = self.fee_total_pct * max(0.0, (float(entry) + float(expected_exit)))
                denom = R + fee_per_unit
                if denom > 0:
                    raw = risk_amount / denom
                else:
                    raw = risk_amount / R
            except Exception:
                raw = risk_amount / R
        else:
            raw = risk_amount / R
        q = round_step(raw, qty_step)
        
        # Ensure minimum quantity and minimum order value
        if q < min_qty: 
            return 0.0
        
        # Check minimum order value (e.g., 5 USDT for Bybit)
        order_value = q * entry
        if order_value < min_order_value:
            # Adjust quantity to meet minimum order value
            q_min_value = round_step(min_order_value / entry, qty_step)
            if q_min_value < min_qty: # If even min_order_value results in less than min_qty, something is wrong
                logger.warning(f"Calculated min_order_value quantity ({q_min_value}) is less than min_qty ({min_qty}). Returning 0.")
                return 0.0
            logger.info(f"Adjusting quantity from {q} to {q_min_value} to meet minimum order value of {min_order_value} USDT.")
            q = q_min_value
            
        return q
