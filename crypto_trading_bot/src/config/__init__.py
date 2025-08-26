# Config package initialization
# This file makes the config directory a Python package

# Import settings from parent directory
from ..config import settings
from .scaling_config import scaling_config

__all__ = ['settings', 'scaling_config']