#!/usr/bin/env python3
"""
Simple startup script for Railway deployment
"""
import os
import sys
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the application with minimal setup"""
    try:
        logger.info("Starting bot in simple mode...")
        
        # Get port from environment
        port = int(os.environ.get("PORT", 8000))
        host = "0.0.0.0"
        
        logger.info(f"Starting server on {host}:{port}")
        
        # Import and run uvicorn
        import uvicorn
        uvicorn.run(
            "src.main:app",
            host=host,
            port=port,
            log_level="info",
            access_log=False,  # Disable access logs to reduce noise
            workers=1
        )
        
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()