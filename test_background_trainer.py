#!/usr/bin/env python3
"""
Test script for the background initial trainer

This script tests the background trainer functionality without running the full live bot.
"""

import asyncio
import logging
from background_initial_trainer import BackgroundInitialTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_background_trainer():
    """Test the background trainer"""
    
    logger.info("🧪 Testing Background Initial Trainer")
    
    # Create trainer instance
    trainer = BackgroundInitialTrainer()
    
    # Check if training is needed
    needs_training = trainer.should_run_initial_training()
    logger.info(f"Training needed: {needs_training}")
    
    if needs_training:
        logger.info("Starting background training...")
        started = await trainer.start_if_needed()
        
        if started:
            logger.info("Training started successfully!")
            
            # Monitor progress for a bit
            for i in range(5):
                await asyncio.sleep(10)
                status = trainer.get_status()
                logger.info(f"Status update {i+1}: {status}")
                
                if status.get('status') in ['completed', 'error']:
                    break
            
            # Stop trainer
            trainer.stop()
            logger.info("Trainer stopped")
        else:
            logger.info("Training did not start (likely not needed)")
    else:
        logger.info("No training needed - models already exist")

    logger.info("🎉 Test complete!")

if __name__ == "__main__":
    asyncio.run(test_background_trainer())