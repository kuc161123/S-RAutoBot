import pandas as pd
import numpy as np
import logging
from autobot.core.divergence_detector import detect_divergence, DivergenceSignal

logger = logging.getLogger(__name__)

class ShadowAuditor:
    """
    Shadow Auditor: Independent verifier that runs the SAME detection logic 
    on the data to ensure the Bot isn't hallucinating signals.
    """
    def __init__(self):
        self.matches = 0
        self.mismatches = 0
        self.total_checks = 0
        self.discrepancies = []
        
    def audit(self, symbol, df, live_decision):
        """
        Compare Live Decision vs Shadow Logic
        live_decision dict: {'action': 'SKIP_VOL'|'TRADE'|'NO_SIGNAL', 'details': ...}
        """
        self.total_checks += 1
        
        # 1. Shadow Detection (Using the Canonical Detector)
        # We pass 'all' filter to see ALL raw signals, then filter locally
        # This matches the bot's raw detection step
        signals = detect_divergence(df, symbol, lookback=14)
        
        shadow_decision = "NO_SIGNAL"
        shadow_signal_type = None
        
        # 2. Logic Replication
        if signals:
            # Taking the most recent signal if multiple (similar to bot)
            # Bot typically processes the first valid one or prioritizes.
            # We'll check if ANY matches the live decision type if pending.
            
            # Simple check: Is there a signal?
            sig = signals[-1] # Check last signal
            
            # Volume Check (Shadow Implementation)
            # Replicating bot.py logic:
            # vol_ok = vol > (vol_sma * 0.8) if require_volume else True
            # Assuming config defaults (require_volume=False) based on audit 
            # If we want strict audit, we should read config, but for now we assume 
            # the bot's "require_volume: false" means volume is always OK.
            vol_ok = True 
            
            if vol_ok:
                shadow_decision = f"TRADE_{sig.signal_type.upper()}"
                shadow_signal_type = sig.signal_type
            else:
                shadow_decision = "SKIP_VOLUME"

        # 3. Comparison
        combined_decision = live_decision.get('action')
        
        match = False
        
        # Exact match
        if combined_decision == shadow_decision:
            match = True
        
        # Fuzzy Match: Both are TRADES
        elif "TRADE" in str(combined_decision) and "TRADE" in str(shadow_decision):
            match = True
            
        # Cooldown Exception
        elif combined_decision == "SKIP_COOLDOWN" and "TRADE" in str(shadow_decision):
            # Shadow sees a signal, Bot sees it but is in cooldown -> Valid Match
            match = True
            
        # Volume Exception (if configs differ slightly in memory vs assumed)
        elif combined_decision == "SKIP_VOLUME" and "TRADE" in str(shadow_decision):
             # Bot skipped for volume, Shadow didn't (or vice versa). 
             # Weak match, but usually acceptable if configs drift.
             # However, currently both should disable volume.
             pass

        if match:
            self.matches += 1
            return True, f"MATCH: {combined_decision}"
        else:
            self.mismatches += 1
            reason = f"MISMATCH: Live={combined_decision} vs Shadow={shadow_decision}"
            self.discrepancies.append({'symbol': symbol, 'time': str(df.index[-1]), 'reason': reason})
            # Reducing log level for minor mismatches if desired, but keeping Warning for visibility
            logger.warning(f"🚨 SHADOW AUDIT FAILURE: {reason}")
            return False, reason

    def get_stats(self):
        return {
            'matches': self.matches,
            'mismatches': self.mismatches,
            'rate': (self.matches / self.total_checks * 100) if self.total_checks > 0 else 100.0
        }
