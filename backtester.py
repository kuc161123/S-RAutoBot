"""
Backtesting Engine

This module simulates trading strategies over historical data to generate
a dataset of trade signals and their outcomes (win/loss).
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Callable, Optional

from candle_storage_postgres import CandleStorage
from strategy_pullback import Settings, Signal

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, strategy_func: Callable, strategy_settings: Settings):
        """
        Initializes the Backtester.

        Args:
            strategy_func: The function that detects signals (e.g., get_pullback_signals).
            strategy_settings: The settings for the strategy.
        """
        self.strategy_func = strategy_func
        self.settings = strategy_settings
        self.candle_storage = CandleStorage()

    def run(self, symbol: str, history_df: Optional[pd.DataFrame] = None) -> List[Dict]:
        """
        Runs a backtest for a single symbol.

        Args:
            symbol: The trading symbol to backtest.
            history_df: Optional pre-loaded dataframe of historical data.

        Returns:
            A list of dictionaries, where each dict represents a signal and its outcome.
        """
        if history_df is None:
            logger.info(f"[{symbol}] Loading historical data from database...")
            history_df = self.candle_storage.load_candles(symbol, limit=100000) # Load all available

        if history_df is None or len(history_df) < 200:
            logger.warning(f"[{symbol}] Insufficient historical data to run backtest ({len(history_df) if history_df is not None else 0} candles).")
            return []

        logger.info(f"[{symbol}] Backtesting on {len(history_df)} candles from {history_df.index[0]} to {history_df.index[-1]}...")

        results = []
        # Iterate through the historical data, simulating the live bot
        for i in range(200, len(history_df)):
            # Create a dataframe slice representing the point-in-time data available
            current_df_slice = history_df.iloc[i-200:i]

            # Run the signal detection logic
            # Pass df_1h=None as backtester currently only uses 15m data
            signal = self.strategy_func(current_df_slice, self.settings, df_1h=None, symbol=symbol)

            if signal:
                # A signal was generated. Now, simulate the trade outcome.
                outcome = self._simulate_trade(history_df, i, signal)
                if outcome:
                    results.append({
                        "timestamp": history_df.index[i],
                        "features": signal.meta.get('ml_features', {}),
                        "outcome": outcome
                    })
        
        logger.info(f"[{symbol}] Backtest complete. Found {len(results)} signals.")
        return results

    def _simulate_trade(self, df: pd.DataFrame, entry_index: int, signal: Signal) -> Optional[str]:
        """
        Checks future candles to determine if a trade would have been a win or loss.
        """
        # Look ahead up to 100 candles for an outcome
        for i in range(entry_index + 1, min(entry_index + 100, len(df))):
            future_high = df['high'].iloc[i]
            future_low = df['low'].iloc[i]

            if signal.side == "long":
                # Check for TP hit
                if future_high >= signal.tp:
                    return "win"
                # Check for SL hit
                if future_low <= signal.sl:
                    return "loss"
            elif signal.side == "short":
                # Check for TP hit
                if future_low <= signal.tp:
                    return "win"
                # Check for SL hit
                if future_high >= signal.sl:
                    return "loss"
        
        return None # No outcome within 100 candles
