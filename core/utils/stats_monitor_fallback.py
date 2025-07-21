# Disable StatsMonitor Rich integration by disabling all Rich calls
STATS_MONITOR_DISABLED = True

class StatsMonitor:
    """Fallback StatsMonitor when Rich is not available."""
    def __init__(self, *args, **kwargs):
        self.metrics = {}
        
    def log_step(self, *args, **kwargs):
        pass
        
    def show(self):
        print("Stats Monitor (Rich disabled)")
        
    def update(self, *args, **kwargs):
        pass
        
    def log_episode(self, *args, **kwargs):
        pass
        
    def render_ascii_summary(self):
        print("=== Training Summary ===")
        print("Rich UI not available")
        
    def start_live_display(self):
        pass
        
    def stop_live_display(self):
        pass
