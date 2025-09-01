from position_mgr import RiskConfig, round_step

class Sizer:
    def __init__(self, risk:RiskConfig):
        self.risk = risk

    def qty_for(self, entry:float, sl:float, qty_step:float, min_qty:float) -> float:
        R = abs(entry - sl)
        if R <= 0: return 0.0
        raw = self.risk.risk_usd / R
        q = round_step(raw, qty_step)
        if q < min_qty: return 0.0
        return q