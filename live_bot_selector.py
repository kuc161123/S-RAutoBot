"""
Strategy selector for live_bot.py
This file allows easy switching between original and pullback strategies
"""

def get_strategy_module(use_pullback: bool):
    """
    Returns the appropriate strategy module based on configuration
    
    Args:
        use_pullback: If True, use pullback strategy. If False, use original strategy.
    
    Returns:
        Tuple of (detect_signal_function, reset_function)
    """
    if use_pullback:
        from strategy_pullback import detect_signal_pullback, reset_symbol_state
        return detect_signal_pullback, reset_symbol_state
    else:
        from strategy import detect_signal
        # Original strategy doesn't need state reset
        def dummy_reset(symbol: str):
            pass
        return detect_signal, dummy_reset