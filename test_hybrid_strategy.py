#!/usr/bin/env python3
"""
Test script for the new Hybrid Smart Money Strategy
Tests all components without actually trading
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_trading_bot.src.strategy.hybrid_smart_money_strategy import HybridSmartMoneyStrategy
from crypto_trading_bot.src.strategy.ml_signal_scorer import ml_signal_scorer
from crypto_trading_bot.src.api.enhanced_bybit_client import EnhancedBybitClient
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

async def test_hybrid_strategy():
    """Test the hybrid strategy with real market data"""
    
    print("=" * 60)
    print("HYBRID SMART MONEY STRATEGY TEST")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing components...")
    strategy = HybridSmartMoneyStrategy()
    client = EnhancedBybitClient()
    
    try:
        # Initialize client
        await client.initialize()
        print("✅ Bybit client initialized")
        
        # Test symbols
        test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        
        print(f"\n2. Testing with symbols: {test_symbols}")
        print("-" * 40)
        
        all_signals = []
        
        for symbol in test_symbols:
            print(f"\n📊 Analyzing {symbol}...")
            
            # Get market data
            df = await client.get_klines(symbol, '15', limit=200)
            
            if df is not None and len(df) > 100:
                print(f"  ✅ Got {len(df)} candles of data")
                
                # Run strategy analysis
                signals = strategy.analyze(symbol, df)
                
                if signals:
                    print(f"  🎯 Generated {len(signals)} raw signals:")
                    for sig in signals:
                        print(f"    - {sig.signal_type.value}: {sig.direction} @ {sig.entry_price:.4f} "
                              f"(confidence: {sig.confidence:.1f}%)")
                    
                    # Score with ML
                    scored_signals = ml_signal_scorer.score_signals(signals, {symbol: df})
                    
                    if scored_signals:
                        print(f"  ✅ {len(scored_signals)} signals passed ML filtering:")
                        for sig in scored_signals:
                            print(f"    - {sig.signal_type.value}: {sig.direction} @ {sig.entry_price:.4f} "
                                  f"(ML enhanced confidence: {sig.confidence:.1f}%)")
                            print(f"      Confluences: {', '.join(sig.confluences[:3])}")
                            print(f"      Risk/Reward: {sig.risk_reward_ratio:.2f}")
                            all_signals.append(sig)
                    else:
                        print(f"  ❌ All signals filtered out by ML scorer")
                else:
                    print(f"  📉 No signals generated for {symbol}")
            else:
                print(f"  ❌ Insufficient data for {symbol}")
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total signals generated: {len(all_signals)}")
        
        if all_signals:
            # Group by signal type
            signal_types = {}
            for sig in all_signals:
                if sig.signal_type.value not in signal_types:
                    signal_types[sig.signal_type.value] = []
                signal_types[sig.signal_type.value].append(sig)
            
            print("\nSignals by type:")
            for sig_type, sigs in signal_types.items():
                avg_confidence = sum(s.confidence for s in sigs) / len(sigs)
                print(f"  - {sig_type}: {len(sigs)} signals (avg confidence: {avg_confidence:.1f}%)")
            
            # Best signals
            all_signals.sort(key=lambda x: x.confidence, reverse=True)
            print("\nTop 3 highest confidence signals:")
            for i, sig in enumerate(all_signals[:3], 1):
                print(f"  {i}. {sig.symbol} {sig.signal_type.value} {sig.direction}")
                print(f"     Entry: {sig.entry_price:.4f}, SL: {sig.stop_loss:.4f}, TP: {sig.take_profit_1:.4f}")
                print(f"     Confidence: {sig.confidence:.1f}%, R:R: {sig.risk_reward_ratio:.2f}")
        else:
            print("\nNo signals generated across all test symbols.")
            print("This could be due to:")
            print("  - Current market conditions not matching any strategy criteria")
            print("  - All signals filtered out by confidence thresholds")
            print("  - Need for parameter tuning")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await client.close()
        print("\n🏁 Test finished")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_hybrid_strategy())