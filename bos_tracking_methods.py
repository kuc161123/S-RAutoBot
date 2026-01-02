    def _track_divergence_detected(self):
        """Track a divergence detection"""
        # Reset daily counters if new day
        today = datetime.now().date()
        if self.bos_tracking['last_reset'] != today:
            self.bos_tracking['divergences_detected_today'] = 0
            self.bos_tracking['bos_confirmed_today'] = 0
            self.bos_tracking['last_reset'] = today
        
        self.bos_tracking['divergences_detected_today'] += 1
        self.bos_tracking['divergences_detected_total'] += 1
    
    def _track_bos_confirmed(self):
        """Track a BOS confirmation"""
        # Reset daily counters if new day
        today = datetime.now().date()
        if self.bos_tracking['last_reset'] != today:
            self.bos_tracking['divergences_detected_today'] = 0
            self.bos_tracking['bos_confirmed_today'] = 0
            self.bos_tracking['last_reset'] = today
        
        self.bos_tracking['bos_confirmed_today'] += 1
        self.bos_tracking['bos_confirmed_total'] += 1
