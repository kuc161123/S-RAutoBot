"""
Scaling Configuration for Gradual Symbol Increase
==================================================

This configuration manages the gradual scaling of monitored symbols
from testing (20 symbols) to production (all symbols).

Scaling Plan:
- Phase 1: 20 symbols (testing basic functionality)
- Phase 2: 50 symbols (testing moderate load)
- Phase 3: 100 symbols (testing higher load)
- Phase 4: 200 symbols (testing significant load)
- Phase 5: 300 symbols (testing near-production load)
- Phase 6: All symbols (full production)
"""

class ScalingConfig:
    # Current phase (change this to scale up)
    CURRENT_PHASE = 1  # Start with Phase 1 (20 symbols)
    
    # Phase definitions
    PHASES = {
        1: {
            'symbol_count': 20,
            'batch_size': 10,
            'scan_delay': 10,  # seconds between batches
            'description': 'Testing Phase - Basic functionality'
        },
        2: {
            'symbol_count': 50,
            'batch_size': 10,
            'scan_delay': 15,
            'description': 'Extended Testing - Moderate load'
        },
        3: {
            'symbol_count': 100,
            'batch_size': 10,
            'scan_delay': 20,
            'description': 'Load Testing - Higher volume'
        },
        4: {
            'symbol_count': 200,
            'batch_size': 10,
            'scan_delay': 25,
            'description': 'Stress Testing - Significant load'
        },
        5: {
            'symbol_count': 300,
            'batch_size': 10,
            'scan_delay': 30,
            'description': 'Pre-Production - Near full load'
        },
        6: {
            'symbol_count': -1,  # -1 means all available symbols
            'batch_size': 5,
            'scan_delay': 30,
            'description': 'Production - All symbols'
        }
    }
    
    @classmethod
    def get_current_config(cls):
        """Get configuration for current phase"""
        return cls.PHASES[cls.CURRENT_PHASE]
    
    @classmethod
    def get_symbol_count(cls):
        """Get target symbol count for current phase"""
        config = cls.get_current_config()
        return config['symbol_count']
    
    @classmethod
    def get_batch_size(cls):
        """Get batch size for current phase"""
        config = cls.get_current_config()
        return config['batch_size']
    
    @classmethod
    def get_scan_delay(cls):
        """Get scan delay for current phase"""
        config = cls.get_current_config()
        return config['scan_delay']
    
    @classmethod
    def get_description(cls):
        """Get description for current phase"""
        config = cls.get_current_config()
        return config['description']
    
    @classmethod
    def next_phase(cls):
        """Move to next phase"""
        if cls.CURRENT_PHASE < 6:
            cls.CURRENT_PHASE += 1
            return True
        return False
    
    @classmethod
    def set_phase(cls, phase: int):
        """Set specific phase"""
        if 1 <= phase <= 6:
            cls.CURRENT_PHASE = phase
            return True
        return False

# Export config instance
scaling_config = ScalingConfig()