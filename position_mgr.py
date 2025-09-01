from dataclasses import dataclass, field
from typing import Dict

def round_step(x:float, step:float) -> float:
    if step <= 0: return x
    return round(x / step) * step

@dataclass
class RiskConfig:
    risk_usd:float=50.0
    max_leverage:int=5
    # defaults overridden per symbol via config.yaml.symbol_meta
    qty_step:float=0.001
    min_qty:float=0.001

@dataclass
class Position:
    side:str
    qty:float
    entry:float
    sl:float
    tp:float

@dataclass
class Book:
    positions:Dict[str,Position]=field(default_factory=dict)