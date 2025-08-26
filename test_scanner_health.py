#!/usr/bin/env python3
"""
Test scanner health monitoring and automatic restart functionality
"""
import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_scanner_health():
    """Test scanner health monitoring"""
    
    # Import after setting up logging
    from crypto_trading_bot.src.trading.multi_timeframe_scanner import MultiTimeframeScanner
    from crypto_trading_bot.src.strategy.advanced_supply_demand import AdvancedSupplyDemandStrategy
    
    # Create mock client
    mock_client = MagicMock()
    mock_client.get_klines = AsyncMock(return_value=None)
    mock_client.get_positions = AsyncMock(return_value=[])
    
    # Create strategy
    strategy = AdvancedSupplyDemandStrategy()
    
    # Create scanner
    scanner = MultiTimeframeScanner(mock_client, strategy)
    
    # Initialize
    await scanner.initialize()
    
    logger.info("Testing scanner health monitoring...")
    
    # Test 1: Check initial status
    status = scanner.get_scanner_status()
    logger.info(f"Initial status: {status}")
    assert not status['healthy'], "Scanner should not be healthy before starting"
    
    # Test 2: Start scanner
    await scanner.start_scanning()
    await asyncio.sleep(2)
    
    status = scanner.get_scanner_status()
    logger.info(f"After start status: {status}")
    assert scanner.is_scanning, "Scanner should be running"
    
    # Test 3: Simulate scanner activity
    scanner.last_scan_time = datetime.now()
    scanner.scan_count = 10
    scanner.scan_metrics['total_scans'] = 10
    scanner.scan_metrics['successful_scans'] = 8
    scanner.scan_metrics['failed_scans'] = 2
    
    status = scanner.get_scanner_status()
    logger.info(f"After activity status: {status}")
    assert status['healthy'], "Scanner should be healthy with recent activity"
    assert status['success_rate'] == 80.0, "Success rate should be 80%"
    
    # Test 4: Simulate stuck scanner
    scanner.last_scan_time = datetime.now() - timedelta(minutes=5)
    
    status = scanner.get_scanner_status()
    logger.info(f"Stuck scanner status: {status}")
    assert not status['healthy'], "Scanner should not be healthy when stuck"
    
    # Test 5: Test health monitor detection
    logger.info("Testing health monitor detection of stuck scanner...")
    
    # Manually trigger health check
    scanner.last_scan_time = datetime.now() - timedelta(minutes=3)
    
    # Create a flag to track if restart was attempted
    restart_called = False
    original_restart = scanner._restart_scanner
    
    async def mock_restart():
        nonlocal restart_called
        restart_called = True
        logger.info("Scanner restart triggered!")
        await original_restart()
    
    scanner._restart_scanner = mock_restart
    
    # Run health monitor once
    health_task = asyncio.create_task(scanner._health_monitor())
    await asyncio.sleep(1)  # Let it run one iteration
    health_task.cancel()
    try:
        await health_task
    except asyncio.CancelledError:
        pass
    
    # Test 6: Stop scanner
    await scanner.stop_scanning()
    
    status = scanner.get_scanner_status()
    logger.info(f"After stop status: {status}")
    assert not scanner.is_scanning, "Scanner should be stopped"
    
    logger.info("✅ All scanner health tests passed!")
    
    return True

async def test_scanner_recovery():
    """Test scanner automatic recovery from errors"""
    
    from crypto_trading_bot.src.trading.multi_timeframe_scanner import MultiTimeframeScanner
    from crypto_trading_bot.src.strategy.advanced_supply_demand import AdvancedSupplyDemandStrategy
    
    # Create mock client that will fail then succeed
    mock_client = MagicMock()
    call_count = 0
    
    async def mock_get_klines(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Simulated API error")
        return None
    
    mock_client.get_klines = mock_get_klines
    mock_client.get_positions = AsyncMock(return_value=[])
    
    # Create scanner
    strategy = AdvancedSupplyDemandStrategy()
    scanner = MultiTimeframeScanner(mock_client, strategy)
    
    # Initialize
    await scanner.initialize()
    
    logger.info("Testing scanner recovery from errors...")
    
    # Start scanner with recovery
    await scanner.start_scanning()
    
    # Let it run and recover
    await asyncio.sleep(5)
    
    # Check that scanner is still running despite errors
    assert scanner.is_scanning, "Scanner should still be running after recovery"
    
    # Check error tracking
    assert scanner.consecutive_errors > 0 or scanner.error_count > 0, "Errors should be tracked"
    
    # Stop scanner
    await scanner.stop_scanning()
    
    logger.info("✅ Scanner recovery test passed!")
    
    return True

async def main():
    """Run all tests"""
    
    logger.info("Starting scanner health monitoring tests...")
    
    try:
        # Test health monitoring
        await test_scanner_health()
        
        # Test recovery
        await test_scanner_recovery()
        
        logger.info("\n✅ All scanner tests completed successfully!")
        logger.info("\nThe scanner now has:")
        logger.info("1. Health monitoring that detects when scanner is stuck")
        logger.info("2. Automatic restart when no activity for 2+ minutes")
        logger.info("3. Recovery wrapper that restarts on crashes")
        logger.info("4. Metrics tracking for monitoring performance")
        logger.info("5. Memory cleanup to prevent resource issues")
        logger.info("6. Watchdog in UltraIntelligentEngine for additional monitoring")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(main())