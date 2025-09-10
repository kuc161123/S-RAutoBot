from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime

def round_step(x:float, step:float) -> float:
    """Round to nearest step value with proper decimal handling"""
    if step <= 0: return x
    
    # Determine decimal places from step
    import decimal
    step_str = str(step)
    if '.' in step_str:
        decimal_places = len(step_str.split('.')[1].rstrip('0'))
    else:
        decimal_places = 0
    
    # Round to step
    rounded = round(x / step) * step
    
    # Format to avoid floating point issues
    return round(rounded, decimal_places)

@dataclass
class RiskConfig:
    risk_usd:float=50.0  # Fixed USD risk (will be overridden by percentage)
    risk_percent:float=1.0  # Risk as percentage of account (default 1%)
    use_percent_risk:bool=True  # Use percentage-based risk instead of fixed USD
    max_leverage:int=5
    # defaults overridden per symbol via config.yaml.symbol_meta
    qty_step:float=0.001
    min_qty:float=0.001
    # ML dynamic risk fields
    use_ml_dynamic_risk:bool=False  # Enable ML-based dynamic risk
    ml_risk_min_score:float=70.0    # ML score for minimum risk
    ml_risk_max_score:float=100.0   # ML score for maximum risk
    ml_risk_min_percent:float=1.0   # Minimum risk percentage
    ml_risk_max_percent:float=5.0   # Maximum risk percentage

@dataclass
class Position:
    side:str
    qty:float
    entry:float
    sl:float
    tp:float
    entry_time:Optional[datetime] = None

@dataclass
class Book:
    positions:Dict[str,Position]=field(default_factory=dict)