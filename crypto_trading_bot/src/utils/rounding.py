from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Union

def round_to_tick(price: Union[float, Decimal], tick_size: Union[float, Decimal]) -> float:
    """Round price to the nearest tick size"""
    price = Decimal(str(price))
    tick = Decimal(str(tick_size))
    
    if tick == 0:
        return float(price)
    
    rounded = (price / tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * tick
    return float(rounded)

def round_to_qty_step(qty: Union[float, Decimal], qty_step: Union[float, Decimal]) -> float:
    """Round quantity to the nearest quantity step"""
    qty = Decimal(str(qty))
    step = Decimal(str(qty_step))
    
    if step == 0:
        return float(qty)
    
    rounded = (qty / step).quantize(Decimal('1'), rounding=ROUND_DOWN) * step
    return float(rounded)

def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    stop_distance_percent: float,
    price: float,
    qty_step: float,
    min_notional: float = 0
) -> float:
    """
    Calculate position size based on risk management rules
    
    Args:
        account_balance: Total account balance
        risk_percent: Percentage of account to risk (e.g., 1.0 for 1%)
        stop_distance_percent: Distance to stop loss as percentage
        price: Current price of the asset
        qty_step: Minimum quantity step for the symbol
        min_notional: Minimum notional value for the position
    
    Returns:
        Position size in base currency
    """
    # Calculate risk amount
    risk_amount = account_balance * (risk_percent / 100)
    
    # Calculate position value based on stop distance
    position_value = risk_amount / (stop_distance_percent / 100)
    
    # Calculate quantity
    qty = position_value / price
    
    # Round to qty step
    qty = round_to_qty_step(qty, qty_step)
    
    # Check minimum notional
    if qty * price < min_notional:
        qty = round_to_qty_step(min_notional / price, qty_step)
    
    return qty

def calculate_stop_loss(
    entry_price: float,
    zone_edge: float,
    buffer_percent: float,
    tick_size: float,
    is_long: bool
) -> float:
    """
    Calculate stop loss price with buffer beyond zone edge
    
    Args:
        entry_price: Entry price of the position
        zone_edge: Edge of the supply/demand zone
        buffer_percent: Buffer percentage beyond zone
        tick_size: Minimum price tick for the symbol
        is_long: True for long position, False for short
    
    Returns:
        Stop loss price
    """
    if is_long:
        # For long, stop below demand zone
        stop = zone_edge * (1 - buffer_percent / 100)
    else:
        # For short, stop above supply zone
        stop = zone_edge * (1 + buffer_percent / 100)
    
    return round_to_tick(stop, tick_size)

def calculate_take_profits(
    entry_price: float,
    stop_loss: float,
    tp1_ratio: float,
    tp2_ratio: float,
    tick_size: float,
    is_long: bool
) -> tuple[float, float]:
    """
    Calculate take profit levels based on risk/reward ratios
    
    Args:
        entry_price: Entry price of the position
        stop_loss: Stop loss price
        tp1_ratio: Risk/reward ratio for TP1 (e.g., 1.0 for 1:1)
        tp2_ratio: Risk/reward ratio for TP2 (e.g., 2.0 for 1:2)
        tick_size: Minimum price tick for the symbol
        is_long: True for long position, False for short
    
    Returns:
        Tuple of (TP1 price, TP2 price)
    """
    risk = abs(entry_price - stop_loss)
    
    if is_long:
        tp1 = entry_price + (risk * tp1_ratio)
        tp2 = entry_price + (risk * tp2_ratio)
    else:
        tp1 = entry_price - (risk * tp1_ratio)
        tp2 = entry_price - (risk * tp2_ratio)
    
    return (
        round_to_tick(tp1, tick_size),
        round_to_tick(tp2, tick_size)
    )

def calculate_breakeven_price(
    entry_price: float,
    tick_size: float,
    fee_percent: float = 0.06
) -> float:
    """
    Calculate breakeven price including fees
    
    Args:
        entry_price: Entry price of the position
        tick_size: Minimum price tick for the symbol
        fee_percent: Trading fee percentage (default 0.06% for taker)
    
    Returns:
        Breakeven price
    """
    # Account for entry and exit fees
    total_fee_percent = fee_percent * 2
    breakeven = entry_price * (1 + total_fee_percent / 100)
    
    return round_to_tick(breakeven, tick_size)