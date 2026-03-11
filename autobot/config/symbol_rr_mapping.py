"""
Symbol Configuration Manager (Multi-Config)
=============================================
Manages per-symbol settings including:
- Enabled/disabled status
- MULTIPLE configs per symbol (long + short)
- Per-config R:R ratio, ATR multiplier, divergence type

Supports both old format (single divergence_type) and new format (configs list).
"""

import yaml
from typing import Dict, Optional, List


class SymbolRRConfig:
    """Manages per-symbol Risk:Reward ratios and divergence types.
    
    Supports multi-config: each symbol can have multiple divergence configs
    (e.g., both HID_BULL and HID_BEAR with different R:R and ATR settings).
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize symbol configuration"""
        self.config_path = config_path
        self.symbols: Dict[str, dict] = {}
        self.load_config()
        
    def load_config(self):
        """Load symbol configuration from config.yaml"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            raw_symbols = config.get('symbols', {})
            self.symbols = {}
            
            # Normalize: convert old format to new multi-config format
            for sym, cfg in raw_symbols.items():
                if 'configs' in cfg:
                    # New multi-config format
                    self.symbols[sym] = cfg
                elif 'divergence_type' in cfg or 'divergence' in cfg:
                    # Old single-config format — wrap in configs list
                    div_type = cfg.get('divergence_type', cfg.get('divergence', 'UNKNOWN'))
                    self.symbols[sym] = {
                        'enabled': cfg.get('enabled', True),
                        'configs': [{
                            'divergence_type': div_type,
                            'rr': cfg.get('rr', 3.0),
                            'atr_mult': cfg.get('atr_mult', 1.0)
                        }]
                    }
                else:
                    self.symbols[sym] = cfg
            
            # Count stats
            total_configs = 0
            div_counts = {}
            for sym, cfg in self.symbols.items():
                if cfg.get('enabled', True):
                    for c in cfg.get('configs', []):
                        dt = c.get('divergence_type', 'UNKNOWN')
                        div_counts[dt] = div_counts.get(dt, 0) + 1
                        total_configs += 1
            
            enabled_count = sum(1 for s in self.symbols.values() if s.get('enabled', True))
            dual_count = sum(1 for s in self.symbols.values() if s.get('enabled', True) and len(s.get('configs', [])) > 1)
            
            print(f"[SymbolConfig] Loaded {len(self.symbols)} symbols ({enabled_count} enabled, {dual_count} dual-side)")
            print(f"[SymbolConfig] Total configs: {total_configs} | By type: {div_counts}")
            
        except Exception as e:
            print(f"[SymbolConfig] Error loading config: {e}")
            self.symbols = {}
    
    def get_allowed_divergence_types(self, symbol: str) -> List[str]:
        """
        Get ALL allowed divergence types for a symbol.
        
        Returns:
            List of divergence codes (e.g., ['REG_BEAR', 'HID_BULL'])
        """
        if symbol not in self.symbols:
            return []
        cfg = self.symbols[symbol]
        if not cfg.get('enabled', True):
            return []
        return [c['divergence_type'] for c in cfg.get('configs', [])]
    
    def get_config_for_divergence(self, symbol: str, divergence_code: str) -> Optional[dict]:
        """
        Get the specific config (RR, ATR mult) for a symbol + divergence type pair.
        
        Args:
            symbol: Trading pair
            divergence_code: e.g., 'HID_BEAR'
            
        Returns:
            Dict with 'rr' and 'atr_mult', or None if not found
        """
        if symbol not in self.symbols:
            return None
        cfg = self.symbols[symbol]
        if not cfg.get('enabled', True):
            return None
        for c in cfg.get('configs', []):
            if c.get('divergence_type') == divergence_code:
                return c
        return None
    
    # === Backward-compatible methods ===
    
    def get_rr_for_symbol(self, symbol: str, divergence_code: str = None) -> Optional[float]:
        """
        Get R:R ratio for a symbol. If divergence_code is provided,
        returns the RR for that specific config. Otherwise returns first config's RR.
        """
        if symbol not in self.symbols:
            return None
        cfg = self.symbols[symbol]
        if not cfg.get('enabled', True):
            return None
        configs = cfg.get('configs', [])
        if not configs:
            return cfg.get('rr')  # Legacy fallback
        
        if divergence_code:
            for c in configs:
                if c.get('divergence_type') == divergence_code:
                    return c.get('rr')
        
        return configs[0].get('rr')
    
    def get_divergence_for_symbol(self, symbol: str) -> Optional[str]:
        """
        Get allowed divergence type(s) for a symbol.
        For multi-config, returns None (use get_allowed_divergence_types instead).
        For single-config, returns the single type.
        """
        if symbol not in self.symbols:
            return None
        cfg = self.symbols[symbol]
        if not cfg.get('enabled', True):
            return None
        configs = cfg.get('configs', [])
        if len(configs) == 1:
            return configs[0].get('divergence_type')
        elif len(configs) > 1:
            return None  # Multi-config — caller should use get_allowed_divergence_types
        return None
    
    def get_symbol_config(self, symbol: str, divergence_code: str = None) -> Optional[dict]:
        """
        Get configuration for a symbol.
        If divergence_code specified, returns that config's settings.
        Otherwise returns first config's settings with backward-compatible keys.
        """
        if symbol not in self.symbols:
            return None
        cfg = self.symbols[symbol]
        configs = cfg.get('configs', [])
        
        if divergence_code:
            for c in configs:
                if c.get('divergence_type') == divergence_code:
                    return c
        
        if configs:
            return configs[0]
        
        return cfg  # Legacy fallback
    
    def is_symbol_enabled(self, symbol: str) -> bool:
        """Check if symbol is enabled for trading"""
        if symbol not in self.symbols:
            return False
        return self.symbols[symbol].get('enabled', True)
    
    def is_divergence_allowed(self, symbol: str, divergence_code: str) -> bool:
        """Check if a specific divergence type is allowed for a symbol"""
        return divergence_code in self.get_allowed_divergence_types(symbol)
    
    def get_enabled_symbols(self) -> List[str]:
        """Get list of all enabled symbols"""
        return [
            symbol for symbol, config in self.symbols.items()
            if config.get('enabled', True)
        ]
    
    def get_symbols_by_divergence(self, divergence_code: str) -> List[str]:
        """Get symbols that have a specific divergence type configured"""
        results = []
        for symbol, config in self.symbols.items():
            if not config.get('enabled', True):
                continue
            for c in config.get('configs', []):
                if c.get('divergence_type') == divergence_code:
                    results.append(symbol)
                    break
        return results
    
    def get_divergence_summary(self) -> Dict[str, int]:
        """Get count of enabled configs by divergence type"""
        counts = {}
        for sym, cfg in self.symbols.items():
            if cfg.get('enabled', True):
                for c in cfg.get('configs', []):
                    div = c.get('divergence_type', 'UNKNOWN')
                    counts[div] = counts.get(div, 0) + 1
        return counts
    
    def get_total_enabled(self) -> int:
        """Get count of enabled symbols"""
        return sum(1 for s in self.symbols.values() if s.get('enabled', True))
    
    def get_total_configs(self) -> int:
        """Get total number of active configs across all enabled symbols"""
        total = 0
        for cfg in self.symbols.values():
            if cfg.get('enabled', True):
                total += len(cfg.get('configs', []))
        return total
    
    def get_rr_summary(self) -> Dict[float, int]:
        """Get count of configs by R:R ratio"""
        counts = {}
        for sym, cfg in self.symbols.items():
            if cfg.get('enabled', True):
                for c in cfg.get('configs', []):
                    rr = c.get('rr', 0)
                    counts[rr] = counts.get(rr, 0) + 1
        return counts

    def get_symbols_by_rr(self, rr: float) -> List[str]:
        """Get symbols with a specific R:R ratio in any of their configs"""
        results = []
        for symbol, config in self.symbols.items():
            if not config.get('enabled', True):
                continue
            for c in config.get('configs', []):
                if c.get('rr') == rr:
                    results.append(symbol)
                    break
        return results
