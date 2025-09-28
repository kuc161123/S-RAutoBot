"""
Offline ML Model Trainer

This script runs the backtester over all historical data to generate a large
training dataset, and then uses that data to pre-train the ML models for
both the Pullback and Mean Reversion strategies.
"""
import logging
import yaml
import numpy as np
import pickle
import redis
import os
from typing import List, Dict

from backtester import Backtester
from strategy_pullback_ml_learning import get_ml_learning_signals, MinimalSettings as PullbackSettings, reset_symbol_state as reset_pullback_state
from strategy_mean_reversion import detect_signal as detect_signal_mean_reversion, Settings as ReversionSettings, reset_symbol_state as reset_mean_reversion_state
from ml_signal_scorer_immediate import get_immediate_scorer
from ml_scorer_mean_reversion import get_mean_reversion_scorer, MLScorerMeanReversion # New import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_pullback_model(training_data: List[Dict]):
    """Trains and saves the pullback model."""
    if not training_data:
        logger.warning("No training data provided for pullback model.")
        return

    logger.info(f"Training Pullback ML model on {len(training_data)} total signals...")

    scorer = get_immediate_scorer(enabled=True) # Ensure scorer is enabled for training
    
    # Load data into scorer's internal storage for training
    scorer.memory_storage = {'trades': training_data, 'phantoms': []} # Use memory storage for offline training
    scorer.completed_trades = len(training_data)
    
    # Force a retrain using the loaded data
    scorer.startup_retrain() # Use startup_retrain which handles full training

    if scorer.is_ml_ready:
        logger.info("✅ Pullback Model Trained and Saved to Redis.")
    else:
        logger.error("❌ Pullback Model training failed.")

def train_mean_reversion_model(training_data: List[Dict]):
    """Trains and saves the mean reversion model."""
    if not training_data:
        logger.warning("No training data provided for mean reversion model.")
        return

    logger.info(f"Training Mean Reversion ML model on {len(training_data)} total signals...")

    scorer = get_mean_reversion_scorer(enabled=True) # Ensure scorer is enabled for training
    
    # Load data into scorer's internal storage for training
    scorer.memory_storage = {'trades': training_data}
    scorer.completed_trades = len(training_data)
    
    # Force a retrain using the loaded data
    scorer._retrain_models() # Call internal retrain method

    if scorer.is_ml_ready:
        logger.info("✅ Mean Reversion Model Trained and Saved to Redis.")
    else:
        logger.error("❌ Mean Reversion Model training failed.")

def main():
    """Main function to run the offline training process."""
    logger.info("--- Starting Offline ML Model Training Pipeline ---")

    # 1. Load Config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    symbols = cfg["trade"]["symbols"]

    # 2. Backtest and Train Pullback Strategy
    logger.info("\n--- Phase 1: Processing Pullback Strategy ---")
    pullback_backtester = Backtester(get_ml_learning_signals, PullbackSettings(), reset_state_func=reset_pullback_state)
    all_pullback_data = []
    for symbol in symbols:
        results = pullback_backtester.run(symbol)
        all_pullback_data.extend(results)
    
    train_pullback_model(all_pullback_data)

    # 3. Backtest and Train Mean Reversion Strategy
    logger.info("\n--- Phase 2: Processing Mean Reversion Strategy ---")
    reversion_backtester = Backtester(detect_signal_mean_reversion, ReversionSettings(), reset_state_func=reset_mean_reversion_state)
    all_reversion_data = []
    for symbol in symbols:
        results = reversion_backtester.run(symbol)
        all_reversion_data.extend(results)

    train_mean_reversion_model(all_reversion_data)

    logger.info("\n--- Offline Training Pipeline Complete ---")

if __name__ == "__main__":
    # Load .env variables for database connection
    from dotenv import load_dotenv
    load_dotenv()
    main()
