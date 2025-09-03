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
        # Using ML learning-friendly strategy for better data collection
        from strategy_pullback_ml_learning import get_ml_learning_signals, reset_symbol_state
        
        # Wrapper to return single signal (live_bot expects single signal or None)
        def detect_signal_wrapper(df, settings, symbol):
            signals = get_ml_learning_signals(df, settings, None, symbol)
            return signals[0] if signals else None
        
        return detect_signal_wrapper, reset_symbol_state
    else:
        from strategy import detect_signal
        # Original strategy doesn't need state reset
        def dummy_reset(symbol: str):
            pass
        return detect_signal, dummy_reset