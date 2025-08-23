"""
Safe Testing Configuration for ML Training Phase
Uses minimal risk to gather data and train the model
"""

TESTING_CONFIG = {
    # Position Sizing - ULTRA CONSERVATIVE (Adjusted for many symbols)
    "max_position_size_percent": 0.3,  # 0.3% of account per trade ($0.30 per trade on $100)
    "default_risk_percent": 0.3,  # Risk only 0.3% per trade (lower risk with more symbols)
    "max_concurrent_positions": 3,  # Maximum 3 positions at once (spread risk across symbols)
    "max_daily_loss_percent": 2.0,  # Stop trading if down 2% in a day ($2)
    
    # Leverage - MINIMAL
    "default_leverage": 2,  # Very low leverage for safety
    "max_leverage": 3,  # Never exceed 3x
    
    # Symbol Selection - ALL SYMBOLS for maximum learning
    "use_all_symbols": True,  # Use all available Bybit symbols
    "symbol_rotation": True,  # Rotate through symbols to find best performers
    "max_symbols_per_scan": 20,  # Scan 20 symbols at a time to manage resources
    "prioritize_liquid_symbols": True,  # Focus on high-volume symbols first
    
    # ML Training Parameters
    "min_trades_before_ml": 50,  # Need 50 trades before ML starts predicting
    "min_trades_per_symbol": 10,  # Need 10 trades per symbol for learning
    "paper_trade_first": 20,  # First 20 trades are paper (simulation) only
    
    # Risk Management
    "stop_loss_atr_multiplier": 1.5,  # Tight stop losses
    "take_profit_rr_ratio": 2.0,  # Target 1:2 risk/reward minimum
    "trailing_stop_activation": 1.5,  # Activate trailing stop at 1.5R profit
    "trailing_stop_distance": 0.5,  # Trail by 0.5 ATR
    
    # Time Restrictions
    "avoid_news_hours": True,  # Avoid major news times
    "avoid_weekend_positions": True,  # Close all positions before weekend
    "max_holding_hours": 24,  # Maximum 24 hours per position
    
    # Signal Quality Requirements - STRICT
    "min_zone_score": 75,  # Only trade high-quality zones
    "min_confidence": 70,  # Minimum confidence required
    "require_htf_confluence": True,  # Must have HTF/LTF alignment
    "require_structure_confirmation": True,  # Must have market structure confirmation
    
    # Learning Mode
    "learning_mode": True,  # Enable extensive logging for analysis
    "save_all_signals": True,  # Save all signals for later analysis
    "screenshot_trades": False,  # Don't screenshot (resource intensive)
    
    # Safety Features
    "emergency_stop_loss": 5.0,  # Emergency stop if position down 5%
    "max_slippage_percent": 0.5,  # Cancel if slippage > 0.5%
    "require_volume_confirmation": True,  # Volume must be above average
    "check_correlation": True,  # Avoid correlated positions
}

# Per-Symbol Position Sizing (even more conservative for testing)
POSITION_SIZES = {
    "BTCUSDT": 0.001,  # Minimum BTC position
    "ETHUSDT": 0.01,   # Minimum ETH position  
    "BNBUSDT": 0.1,    # Small BNB position
    "SOLUSDT": 0.5,    # Small SOL position
    "MATICUSDT": 10    # Small MATIC position
}

# ML Training Schedule
ML_TRAINING_SCHEDULE = {
    "initial_phase": {
        "trades": 50,
        "mode": "observation",  # Just collect data
        "risk_percent": 0.5
    },
    "learning_phase": {
        "trades": 100,
        "mode": "cautious",  # Start using ML with low confidence
        "risk_percent": 0.75
    },
    "validation_phase": {
        "trades": 150,
        "mode": "normal",  # Normal ML usage
        "risk_percent": 1.0
    },
    "optimization_phase": {
        "trades": 200,
        "mode": "optimized",  # Full ML with parameter optimization
        "risk_percent": 1.5
    }
}

# Monitoring and Alerts
MONITORING_CONFIG = {
    "log_every_signal": True,
    "log_ml_predictions": True,
    "alert_on_loss": True,
    "daily_summary": True,
    "performance_tracking": {
        "track_by_symbol": True,
        "track_by_pattern": True,
        "track_by_timeframe": True,
        "track_by_hour": True
    }
}