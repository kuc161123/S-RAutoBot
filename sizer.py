from position_mgr import RiskConfig, round_step

class Sizer:
    def __init__(self, risk:RiskConfig, account_balance:float=None):
        self.risk = risk
        self.account_balance = account_balance

    def qty_for(self, entry:float, sl:float, qty_step:float, min_qty:float) -> float:
        R = abs(entry - sl)
        if R <= 0: return 0.0
        
        # Calculate risk amount based on mode
        if self.risk.use_percent_risk and self.account_balance and self.account_balance > 0:
            # Use percentage of account balance
            risk_amount = self.account_balance * (self.risk.risk_percent / 100.0)
        else:
            # Fall back to fixed USD risk
            risk_amount = self.risk.risk_usd
            
        raw = risk_amount / R
        q = round_step(raw, qty_step)
        if q < min_qty: return 0.0
        return q