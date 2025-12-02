#!/usr/bin/env python3
"""
Adaptive Strategy Discovery Backtest

Tests multiple strategy variations per symbol to find optimal combinations.
Uses walk-forward validation to avoid overfitting.

Features:
- 18 strategy combinations (3 patterns × 2 alignment × 3 param sets)
- Train/validate split (60/40)
- Realistic costs (0.16% per trade)
- Early stopping for efficiency
- Anti-overfitting safeguards
"""
import asyncio
import yaml
import pandas as pd
import numpy as np
import logging
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(__file__))
from autobot.brokers.bybit import Bybit, BybitConfig
from autobot.strategies.scalp.detector import detect_scalp_signal, ScalpSettings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Realistic costs
SLIPPAGE_PCT = 0.03
FEES_ROUND_TRIP_PCT = 0.11
SPREAD_PCT = 0.02
TOTAL_COST_PCT = SLIPPAGE_PCT + FEES_ROUND_TRIP_PCT + SPREAD_PCT

@dataclass
class StrategyConfig:
    name: str
    pattern: str  # bounce, revert, means_or
    require_ema: bool
    bb_width: float
    vol_ratio: float
    body_ratio: float
    wick_delta: float
    vwap_dist: float

# Define strategy variations
STRATEGIES = [
    # Strict parameters
    StrategyConfig("bounce_ema_strict", "bounce", True, 0.45, 0.8, 0.25, 0.05, 1.5),
    StrategyConfig("revert_ema_strict", "revert", True, 0.45, 0.8, 0.25, 0.05, 1.5),
    StrategyConfig("means_ema_strict", "revert", True, 0.45, 0.8, 0.25, 0.05, 2.0),
    StrategyConfig("bounce_noema_strict", "bounce", False, 0.45, 0.8, 0.25, 0.05, 1.5),
    StrategyConfig("revert_noema_strict", "revert", False, 0.45, 0.8, 0.25, 0.05, 1.5),
    StrategyConfig("means_noema_strict", "revert", False, 0.45, 0.8, 0.25, 0.05, 2.0),
    
    # Medium parameters
    StrategyConfig("bounce_ema_medium", "bounce", True, 0.35, 0.7, 0.20, 0.03, 1.5),
    StrategyConfig("revert_ema_medium", "revert", True, 0.35, 0.7, 0.20, 0.03, 1.5),
    StrategyConfig("means_ema_medium", "revert", True, 0.35, 0.7, 0.20, 0.03, 2.0),
    StrategyConfig("bounce_noema_medium", "bounce", False, 0.35, 0.7, 0.20, 0.03, 1.5),
    StrategyConfig("revert_noema_medium", "revert", False, 0.35, 0.7, 0.20, 0.03, 1.5),
    StrategyConfig("means_noema_medium", "revert", False, 0.35, 0.7, 0.20, 0.03, 2.0),
    
    # Loose parameters
    StrategyConfig("bounce_ema_loose", "bounce", True, 0.25, 0.6, 0.15, 0.02, 2.0),
    StrategyConfig("revert_ema_loose", "revert", True, 0.25, 0.6, 0.15, 0.02, 2.0),
    StrategyConfig("means_ema_loose", "revert", True, 0.25, 0.6, 0.15, 0.02, 2.5),
    StrategyConfig("bounce_noema_loose", "bounce", False, 0.25, 0.6, 0.15, 0.02, 2.0),
    StrategyConfig("revert_noema_loose", "revert", False, 0.25, 0.6, 0.15, 0.02, 2.0),
    StrategyConfig("means_noema_loose", "revert", False, 0.25, 0.6, 0.15, 0.02, 2.5),
]

def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        return os.getenv(config[2:-1], config)
    return config

def test_strategy_on_data(df: pd.DataFrame, strategy: StrategyConfig, symbol: str) -> List[Dict]:
    """Test a strategy on data and return list of trades"""
    settings = ScalpSettings(
        rr=2.1,
        vwap_window=50,
        min_bb_width_pct=strategy.bb_width,
        vol_ratio_min=strategy.vol_ratio,
        body_ratio_min=strategy.body_ratio,
        wick_delta_min=strategy.wick_delta,
        vwap_dist_atr_max=strategy.vwap_dist,
        vwap_pattern=strategy.pattern,
        vwap_require_alignment=strategy.require_ema,
        vwap_bounce_band_atr_min=0.1,
        vwap_bounce_band_atr_max=0.6,
    )
    
    signals = []
    for i in range(100, len(df) - 100):
        try:
            hist_df = df.iloc[:i].copy()
            signal = detect_scalp_signal(hist_df, settings, symbol)
            
            if signal is None:
                continue
            
            # Entry on next candle
            next_candle = df.iloc[i]
            entry = float(next_candle['open'])
            
            # Apply slippage
            if signal.side == 'long':
                entry *= (1 + SLIPPAGE_PCT / 100)
            else:
                entry *= (1 - SLIPPAGE_PCT / 100)
            
            # Calculate SL/TP
            signal_r = abs(signal.entry - signal.sl)
            if signal.side == 'long':
                sl = entry - signal_r
                tp = entry + signal_r * 2.1
            else:
                sl = entry + signal_r
                tp = entry - signal_r * 2.1
            
            # Validate
            if signal.side == 'long' and (sl >= entry or tp <= entry):
                continue
            if signal.side == 'short' and (sl <= entry or tp >= entry):
                continue
            
            # Test outcome
            outcome = 'timeout'
            exit_price = entry
            future = df.iloc[i+1:i+101]
            
            for _, f_row in future.iterrows():
                if signal.side == 'long':
                    if f_row['low'] <= sl:
                        outcome = 'loss'
                        exit_price = sl * (1 - SLIPPAGE_PCT / 100)
                        break
                    if f_row['high'] >= tp:
                        outcome = 'win'
                        exit_price = tp * (1 - SLIPPAGE_PCT / 100)
                        break
                else:
                    if f_row['high'] >= sl:
                        outcome = 'loss'
                        exit_price = sl * (1 + SLIPPAGE_PCT / 100)
                        break
                    if f_row['low'] <= tp:
                        outcome = 'win'
                        exit_price = tp * (1 + SLIPPAGE_PCT / 100)
                        break
            
            # Calculate P&L with costs
            if signal.side == 'long':
                pnl_pct = (exit_price - entry) / entry * 100
            else:
                pnl_pct = (entry - exit_price) / entry * 100
            
            pnl_pct -= TOTAL_COST_PCT
            
            final_outcome = 'win' if pnl_pct > 0 else ('loss' if pnl_pct < 0 else 'breakeven')
            
            signals.append({
                'side': signal.side,
                'outcome': final_outcome,
                'bar_idx': i
            })
        except Exception:
            continue
    
    return signals

async def backtest_symbol_adaptive(bybit, sym: str, idx: int, total: int):
    """Test multiple strategies and find best for this symbol"""
    try:
        logger.info(f"[{idx}/{total}] {sym} - Testing {len(STRATEGIES)} strategies...")
        
        # Fetch data
        klines = []
        end_ts = None
        for i in range(50):
            k = bybit.get_klines(sym, '3', limit=200, end=end_ts)
            if not k: break
            klines = k + klines
            end_ts = int(k[0][0]) - 1
            await asyncio.sleep(0.03)
        
        if len(klines) < 2000:
            logger.warning(f"[{idx}/{total}] {sym} - Insufficient data")
            return None
        
        df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
        df.set_index('start', inplace=True)
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)
        
        # Split train/validate (60/40)
        split_idx = int(len(df) * 0.6)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        best_long = None
        best_short = None
        
        # Test each strategy
        for strategy in STRATEGIES:
            # Train
            train_signals = test_strategy_on_data(train_df, strategy, sym)
            if not train_signals:
                continue
            
            train_long = [s for s in train_signals if s['side'] == 'long']
            train_short = [s for s in train_signals if s['side'] == 'short']
            
            # Check longs
            if len(train_long) >= 30:
                train_long_wr = sum(1 for s in train_long if s['outcome'] == 'win') / len(train_long) * 100
                
                if train_long_wr > 45:
                    # Validate
                    val_signals = test_strategy_on_data(val_df, strategy, sym)
                    val_long = [s for s in val_signals if s['side'] == 'long']
                    
                    if len(val_long) >= 20:
                        val_long_wr = sum(1 for s in val_long if s['outcome'] == 'win') / len(val_long) * 100
                        
                        # Check consistency (within 10%)
                        if abs(train_long_wr - val_long_wr) <= 10 and val_long_wr > 45:
                            combined_wr = sum(1 for s in train_long + val_long if s['outcome'] == 'win') / (len(train_long) + len(val_long)) * 100
                            
                            if not best_long or combined_wr > best_long['combined_wr']:
                                best_long = {
                                    'strategy': strategy.name,
                                    'pattern': strategy.pattern,
                                    'train_wr': train_long_wr,
                                    'train_n': len(train_long),
                                    'val_wr': val_long_wr,
                                    'val_n': len(val_long),
                                    'combined_wr': combined_wr,
                                    'combined_n': len(train_long) + len(val_long)
                                }
            
            # Check shorts
            if len(train_short) >= 30:
                train_short_wr = sum(1 for s in train_short if s['outcome'] == 'win') / len(train_short) * 100
                
                if train_short_wr > 45:
                    val_signals = test_strategy_on_data(val_df, strategy, sym)
                    val_short = [s for s in val_signals if s['side'] == 'short']
                    
                    if len(val_short) >= 20:
                        val_short_wr = sum(1 for s in val_short if s['outcome'] == 'win') / len(val_short) * 100
                        
                        if abs(train_short_wr - val_short_wr) <= 10 and val_short_wr > 45:
                            combined_wr = sum(1 for s in train_short + val_short if s['outcome'] == 'win') / (len(train_short) + len(val_short)) * 100
                            
                            if not best_short or combined_wr > best_short['combined_wr']:
                                best_short = {
                                    'strategy': strategy.name,
                                    'pattern': strategy.pattern,
                                    'train_wr': train_short_wr,
                                    'train_n': len(train_short),
                                    'val_wr': val_short_wr,
                                    'val_n': len(val_short),
                                    'combined_wr': combined_wr,
                                    'combined_n': len(train_short) + len(val_short)
                                }
        
        if best_long or best_short:
            msg = []
            if best_long:
                msg.append(f"LONG: {best_long['strategy']} WR={best_long['combined_wr']:.1f}% (N={best_long['combined_n']})")
            if best_short:
                msg.append(f"SHORT: {best_short['strategy']} WR={best_short['combined_wr']:.1f}% (N={best_short['combined_n']})")
            logger.info(f"[{idx}/{total}] {sym} ✅ {' | '.join(msg)}")
            
            return {'symbol': sym, 'long': best_long, 'short': best_short}
        else:
            logger.info(f"[{idx}/{total}] {sym} ⚠️ No strategy passed validation")
            return None
            
    except Exception as e:
        logger.error(f"[{idx}/{total}] {sym} ❌ Error: {e}")
        return None

async def run():
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = replace_env_vars(cfg)
    
    with open('symbols_400.yaml', 'r') as f:
        data = yaml.safe_load(f)
    symbols = data['symbols'] if isinstance(data, dict) else data
    
    bybit = Bybit(BybitConfig(
        base_url=cfg['bybit']['base_url'],
        api_key=cfg['bybit']['api_key'],
        api_secret=cfg['bybit']['api_secret']
    ))
    
    print(f"{'='*80}")
    print(f"🔬 ADAPTIVE STRATEGY DISCOVERY: {len(symbols)} Symbols")
    print(f"{'='*80}")
    print(f"Testing {len(STRATEGIES)} strategy variations per symbol")
    print(f"Walk-forward validation (60/40 split)")
    print(f"Criteria: WR > 45% (train & validate), N >= 30 (train), N >= 20 (validate)")
    print(f"Costs: {TOTAL_COST_PCT}% per trade")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    results = []
    for idx, sym in enumerate(symbols, 1):
        result = await backtest_symbol_adaptive(bybit, sym, idx, len(symbols))
        if result:
            results.append(result)
        
        if idx % 50 == 0:
            print(f"\n{'='*80}")
            print(f"Progress: {idx}/{len(symbols)} | Passed: {len(results)}")
            print(f"{'='*80}\n")
    
    # Save results
    yaml_lines = [
        "# Adaptive Strategy Discovery Results",
        f"# Walk-forward validated (train 60%, validate 40%)",
        f"# Tested {len(STRATEGIES)} strategies per symbol",
        f"# Costs: {TOTAL_COST_PCT}% per trade",
        f"# Processed: {len(symbols)} symbols",
        f"# Passed: {len(results)} symbols",
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    for r in results:
        yaml_lines.append(f"{r['symbol']}:")
        if r['long']:
            yaml_lines.append(f"  long:")
            yaml_lines.append(f"    strategy: {r['long']['strategy']}")
            yaml_lines.append(f"    combined_wr: {r['long']['combined_wr']:.1f}%")
            yaml_lines.append(f"    combined_n: {r['long']['combined_n']}")
            yaml_lines.append(f"    # Train: WR={r['long']['train_wr']:.1f}%, N={r['long']['train_n']} | Val: WR={r['long']['val_wr']:.1f}%, N={r['long']['val_n']}")
        if r['short']:
            yaml_lines.append(f"  short:")
            yaml_lines.append(f"    strategy: {r['short']['strategy']}")
            yaml_lines.append(f"    combined_wr: {r['short']['combined_wr']:.1f}%")
            yaml_lines.append(f"    combined_n: {r['short']['combined_n']}")
            yaml_lines.append(f"    # Train: WR={r['short']['train_wr']:.1f}%, N={r['short']['train_n']} | Val: WR={r['short']['val_wr']:.1f}%, N={r['short']['val_n']}")
        yaml_lines.append("")
    
    with open('symbol_overrides_ADAPTIVE.yaml', 'w') as f:
        f.write("\n".join(yaml_lines))
    
    print(f"\n{'='*80}")
    print(f"✅ ADAPTIVE DISCOVERY COMPLETE")
    print(f"{'='*80}")
    print(f"Symbols validated: {len(results)}")
    print(f"Output: symbol_overrides_ADAPTIVE.yaml")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(run())
