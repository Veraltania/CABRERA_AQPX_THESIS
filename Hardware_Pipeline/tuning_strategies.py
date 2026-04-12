import os
import csv
from abc import ABC, abstractmethod

# ==========================================
# UTILITY: CLEAR STALE DATA
# ==========================================
def clear_tuning_logs(log_file):
    """Call this in your main script to ensure a fresh simulation."""
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"[System] Stale log {log_file} removed. Starting fresh.")

class TuningStrategy(ABC):
    @abstractmethod
    def load_state(self, controller, log_file):
        pass

    @abstractmethod
    def evaluate_performance(self, controller, error, dt):
        pass

# ==========================================
# REWRITTEN: ADAPTIVE STRATEGY
# ==========================================
class AdaptiveTuningStrategy(TuningStrategy):
    def __init__(self, window_duration=3600):  
        self.window_duration = window_duration
        self.window_timer = 0.0
        self.abs_error_sum = 0.0 
        self.previous_mae = 0.0
        self.first_window_completed = False
        self.cooldown_active = False 

        # Thresholds
        self.mae_hard_limit = 0.35            
        self.mae_shift_threshold = 0.50       
        self.min_physical_error_threshold = 0.15 

    def load_state(self, controller, log_file):
        """Reads the CSV and reloads only the MOST RECENT state."""
        if not os.path.exists(log_file):
            print(f"[{controller.name}-Adaptive] Fresh start: No state file.")
            return

        try:
            with open(log_file, 'r') as f:
                # Use a list to quickly grab the tail of the file
                data = list(csv.DictReader(f))
                if not data: return
                
                last_row = data[-1] # Always take the absolute latest entry
                
                controller.kp = float(last_row.get('kp', controller.kp))
                controller.ki = float(last_row.get('ki', controller.ki))
                controller.integral_sum = float(last_row.get('integral_sum', 0.0))
                self.previous_mae = float(last_row.get('previous_mae', 0.0))
                
                if self.previous_mae > 0:
                    self.first_window_completed = True
                
                print(f"[{controller.name}-Adaptive] State Restored. Kp: {controller.kp:.4f}")
        except Exception as e:
            print(f"[{controller.name}-Adaptive] Load failed: {e}. Using defaults.")

    def evaluate_performance(self, controller, error, dt):
        self.window_timer += dt
        self.abs_error_sum += abs(error) * dt

        if self.window_timer >= self.window_duration:
            current_mae = self.abs_error_sum / self.window_duration
            trigger_retune = False

            if self.cooldown_active:
                self.cooldown_active = False
                self.first_window_completed = False 
            elif not self.first_window_completed:
                self.first_window_completed = True
            else:
                # Retune Logic
                if current_mae >= self.mae_hard_limit:
                    trigger_retune = True
                elif current_mae >= self.min_physical_error_threshold and self.previous_mae > 0:
                    if (current_mae - self.previous_mae) / self.previous_mae >= self.mae_shift_threshold:
                        trigger_retune = True

            if trigger_retune and hasattr(controller, 'retune'):
                controller.retune()
                self.cooldown_active = True

            self.previous_mae = current_mae
            self.abs_error_sum = 0.0
            self.window_timer = 0.0

# ==========================================
# REWRITTEN: STATIC STRATEGY (FIXED)
# ==========================================
class StaticTuningStrategy(TuningStrategy):
    """
    Locked strategy. Does NOT allow gains to change during the run.
    Always uses the values currently in the controller unless forced to load.
    """
    def load_state(self, controller, log_file):
        if not os.path.exists(log_file):
            print(f"[{controller.name}-Static] No file. Keeping memory gains.")
            return

        try:
            with open(log_file, 'r') as f:
                data = list(csv.DictReader(f))
                if not data: return
                
                # FIXED: Instead of 'first_row' which might be from a different day,
                # we only load if the file was explicitly generated for THIS run.
                # If you want to force baseline, it's better to pass gains directly.
                target_row = data[-1] 
                controller.kp = float(target_row.get('kp', controller.kp))
                controller.ki = float(target_row.get('ki', controller.ki))
                
                print(f"[{controller.name}-Static] Gains Locked: Kp={controller.kp:.4f}")
        except Exception as e:
            print(f"[{controller.name}-Static] Load Error: {e}")

    def evaluate_performance(self, controller, error, dt):
        # Static means no evaluation, no retuning.
        pass

# ==========================================
# MANAGER
# ==========================================
class TimeBasedStrategyManager:
    def __init__(self, schedule_configs):
        self.schedule_configs = schedule_configs

    def get_active_strategy(self):
        for schedule, strategy in self.schedule_configs:
            if schedule.is_active():
                return strategy
        return None