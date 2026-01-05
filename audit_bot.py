#!/usr/bin/env python3
"""
Bot Trade Execution Audit Script
Tests the complete trade flow to diagnose why trades are not executing
"""
import asyncio
import logging
import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, '/Users/lualakol/AutoTrading Bot')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("AUDIT")

async def run_audit():
    logger.info("=" * 60)
    logger.info("BOT TRADE EXECUTION AUDIT")
    logger.info("=" * 60)
    
    # Load config
    with open('/Users/lualakol/AutoTrading Bot/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check risk settings
    risk_config = config.get('risk', {})
    risk_per_trade = risk_config.get('risk_per_trade', 'NOT SET')
    logger.info(f"\n[1] RISK CONFIG CHECK")
    logger.info(f"    risk_per_trade: {risk_per_trade}")
    if isinstance(risk_per_trade, (int, float)):
        logger.info(f"    → {risk_per_trade * 100:.2f}% per trade")
        if risk_per_trade > 0.02:
            logger.warning(f"    ⚠️ RISK IS ABOVE 2%!")
    
    # Check execution enabled
    exec_config = config.get('execution', {})
    exec_enabled = exec_config.get('enabled', False)
    logger.info(f"\n[2] EXECUTION CONFIG")
    logger.info(f"    enabled: {exec_enabled}")
    if not exec_enabled:
        logger.error("    ❌ EXECUTION IS DISABLED! Trades will not be placed.")
        return
    
    # Initialize broker
    logger.info(f"\n[3] BROKER INITIALIZATION")
    try:
        from autobot.brokers.bybit import Bybit, BybitConfig
        
        api_key = os.path.expandvars(config['bybit']['api_key'])
        api_secret = os.path.expandvars(config['bybit']['api_secret'])
        base_url = config['bybit']['base_url']
        
        if not api_key or api_key == "${BYBIT_API_KEY}":
            logger.error("    ❌ API KEY NOT SET!")
            return
        
        cfg = BybitConfig(api_key=api_key, api_secret=api_secret, base_url=base_url)
        broker = Bybit(cfg)
        logger.info("    ✅ Broker initialized")
    except Exception as e:
        logger.error(f"    ❌ Broker init failed: {e}")
        return
    
    # Test get_balance
    logger.info(f"\n[4] BALANCE CHECK")
    try:
        balance = await broker.get_balance()
        logger.info(f"    Balance: ${balance:.2f}")
        if balance < 1:
            logger.error("    ❌ BALANCE TOO LOW!")
    except Exception as e:
        logger.error(f"    ❌ Balance check failed: {e}")
    
    # Test get_positions
    logger.info(f"\n[5] POSITIONS CHECK")
    try:
        positions = await broker.get_positions()
        open_pos = [p for p in positions if float(p.get('size', 0)) > 0]
        logger.info(f"    Total positions returned: {len(positions)}")
        logger.info(f"    Open positions (size > 0): {len(open_pos)}")
        for p in open_pos:
            logger.info(f"      - {p.get('symbol')}: size={p.get('size')}")
    except Exception as e:
        logger.error(f"    ❌ Positions check failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Test leverage fetch for specific symbol
    test_symbol = "SYRUPUSDT"
    logger.info(f"\n[6] LEVERAGE CHECK ({test_symbol})")
    try:
        max_lev = await broker.get_max_leverage(test_symbol)
        logger.info(f"    Max leverage: {max_lev}x")
    except Exception as e:
        logger.error(f"    ❌ Leverage fetch failed: {e}")
    
    # Test instrument info fetch
    logger.info(f"\n[7] INSTRUMENT INFO ({test_symbol})")
    try:
        info = await broker.get_instruments_info(symbol=test_symbol)
        if info and len(info) > 0:
            logger.info(f"    ✅ Got instrument info: {len(info)} items")
            inst = info[0]
            logger.info(f"    Symbol: {inst.get('symbol')}")
            logger.info(f"    Max Lev: {inst.get('leverageFilter', {}).get('maxLeverage')}")
            logger.info(f"    Min Qty: {inst.get('lotSizeFilter', {}).get('minOrderQty')}")
        else:
            logger.error("    ❌ No instrument info returned!")
    except Exception as e:
        logger.error(f"    ❌ Instrument fetch failed: {e}")
    
    # Simulate position sizing
    logger.info(f"\n[8] POSITION SIZING SIMULATION")
    try:
        balance = await broker.get_balance()
        risk_pct = risk_config.get('risk_per_trade', 0.002)
        risk_amount = balance * risk_pct
        
        # Assume ATR-based SL distance (typical for 1H)
        assumed_sl_distance = 0.01  # $0.01 for a cheap coin like SYRUP
        
        qty = risk_amount / assumed_sl_distance
        logger.info(f"    Balance: ${balance:.2f}")
        logger.info(f"    Risk %: {risk_pct * 100:.2f}%")
        logger.info(f"    Risk $: ${risk_amount:.2f}")
        logger.info(f"    Assumed SL Distance: ${assumed_sl_distance}")
        logger.info(f"    Calculated Qty: {qty:.4f}")
        
        if qty < 1:
            logger.warning(f"    ⚠️ QTY < 1! May be rejected by exchange.")
    except Exception as e:
        logger.error(f"    ❌ Sizing simulation failed: {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("AUDIT COMPLETE")
    logger.info("=" * 60)
    
    await broker.close()

if __name__ == "__main__":
    asyncio.run(run_audit())
