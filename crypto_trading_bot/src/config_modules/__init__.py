# Config package initialization
# This file makes the config directory a Python package

# Import scaling_config from this package
from .scaling_config import scaling_config

# Note: settings is imported from src.config module (src/config.py file)
# To use settings, import it directly: from src.config import settings

__all__ = ['scaling_config']