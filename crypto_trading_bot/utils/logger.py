"""
Simple logging configuration
"""
import structlog
import logging
import sys
import os
from datetime import datetime

def setup_logger(log_level: str = "INFO"):
    """Configure structured logging for production"""
    
    # Detect if running in production
    is_production = os.getenv('RAILWAY_ENVIRONMENT') or os.path.exists('/.dockerenv')
    
    # Configure standard logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" if is_production else "%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Production processors (no colors, JSON-friendly)
    production_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ]
    
    # Development processors (with colors)
    dev_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(colors=True)
    ]
    
    # Configure structlog
    structlog.configure(
        processors=production_processors if is_production else dev_processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger()
    logger.info("Logger initialized", environment="production" if is_production else "development")
    
    return logger

# Create default logger
logger = setup_logger()