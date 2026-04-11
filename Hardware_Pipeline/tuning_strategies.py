import paho.mqtt.client as mqtt
import time
import sys
import os
import csv
import datetime
from abc import ABC, abstractmethod

class TuningStrategy(ABC):
    @abstractmethod
    def load_state(self, controller, log_file):
        """Defines how the controller loads historical PI parameters."""
        pass

    @abstractmethod
    def evaluate_performance(self, controller, error, dt):
        """Defines if/when the controller should trigger a retune."""
        pass

class AdaptiveTuningStrategy(TuningStrategy):
    def __init__(self, window_duration=3600):  
        self.window_duration = window_duration
        self.window_timer = 0.0
        
        # Pure MAE tracking variables
        self.abs_error_sum = 0.0 
        self.previous_mae = 0.0
        self.first_window_completed = False
        
        # MAE Retuning Thresholds
        self.mae_hard_limit = 0.35            # Absolute limit (e.g., >0.35 mg/L error triggers retune immediately)
        self.mae_shift_threshold = 0.50       # 50% relative shift in MAE compared to the previous window
        self.min_physical_error_threshold = 0.15 # Baseline sanity check: don't retune if MAE < 0.15 mg/L
        
        self.cooldown_active = False 

    def load_state(self, controller, log_file):
        """Reads the CSV file and reloads the last tuning parameters and PI state."""
        if not os.path.exists(log_file):
            print(f"[{controller.name}-Adaptive] No state file found. Starting fresh.")
            return

        try:
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                last_row = None
                for row in reader:
                    last_row = row

                if last_row:
                    controller.integral_sum = float(last_row.get('integral_sum', controller.integral_sum))
                    controller.last_error = float(last_row.get('error', controller.last_error))

                    controller.kp = float(last_row.get('kp', controller.kp))
                    controller.ki = float(last_row.get('ki', controller.ki))
                    controller.foptd_gain = float(last_row.get('foptd_gain', controller.foptd_gain))
                    controller.foptd_tau = float(last_row.get('foptd_tau', controller.foptd_tau))
                    controller.foptd_delay = float(last_row.get('foptd_delay', controller.foptd_delay))

                    # Reload MAE State
                    self.previous_mae = float(last_row.get('previous_mae', 0.0))
                    self.window_timer = float(last_row.get('window_timer', 0.0))

                    if self.previous_mae > 0:
                        self.first_window_completed = True

                    print(f"[{controller.name}-Adaptive] Resumed. Kp: {controller.kp:.3f}. Ki: {controller.ki:.3f}")
        except (ValueError, KeyError) as e:
            print(f"[{controller.name}] Corrupt data in state file ({e}). Maintaining current PI state in memory.")
        except Exception as e:
            print(f"[{controller.name}] Unexpected error reading state file: {e}.")

    def evaluate_performance(self, controller, error, dt):
        """Evaluates the window using strict Mean Absolute Error (MAE) limits."""
        self.window_timer += dt
        
        # Integrate absolute error over time for continuous MAE calculation
        self.abs_error_sum += abs(error) * dt

        if self.window_timer >= self.window_duration:
            current_mae = self.abs_error_sum / self.window_duration
            
            print(f"[{controller.name}-Adaptive] {self.window_duration}-Sec Window Closed.")
            print(f"Mean Absolute Error (MAE) - Current: {current_mae:.3f} mg/L | Previous: {self.previous_mae:.3f} mg/L")

            # 1. Enforce cooldown after a bump test
            if self.cooldown_active:
                print(f"[{controller.name}-Adaptive] Cooldown active. Ignoring this window's data to allow system to stabilize.\n")
                self.cooldown_active = False
                self.first_window_completed = False 
            
            # 2. Establish baseline on startup
            elif not self.first_window_completed:
                print(f"[{controller.name}-Adaptive] Baseline established. Skipping retune evaluation for this window.\n")
                self.first_window_completed = True
            
            # 3. Main Evaluation logic
            else:
                trigger_retune = False
                
                # Check 1: Did we breach the hard physical limit?
                if current_mae >= self.mae_hard_limit:
                    print(f"[{controller.name}-Adaptive] Hard MAE limit breached ({current_mae:.3f} >= {self.mae_hard_limit}). Triggering Retune!\n")
                    trigger_retune = True
                
                # Check 2: Are we above baseline sanity and experiencing a massive relative shift?
                elif current_mae >= self.min_physical_error_threshold and self.previous_mae > 0:
                    percent_change = (current_mae - self.previous_mae) / self.previous_mae
                    if percent_change >= self.mae_shift_threshold:
                        print(f"[{controller.name}-Adaptive] MAE degradation shift of {percent_change * 100:.1f}% detected! Triggering Retune!\n")
                        trigger_retune = True
                else:
                    print(f"[{controller.name}-Adaptive] MAE is within acceptable physical limits. No retune needed.\n")

                # Execute Retune if flagged
                if trigger_retune and hasattr(controller, 'retune'):
                    controller.retune()
                    self.cooldown_active = True

            # Shift windows and reset timers
            self.previous_mae = current_mae
            self.abs_error_sum = 0.0
            self.window_timer = 0.0


class StaticTuningStrategy(TuningStrategy):
    def load_state(self, controller, log_file):
        if not os.path.exists(log_file):
            print(f"[{controller.name}-Static] No state file found. Starting fresh.")
            return

        try:
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                first_row = None
                last_row = None

                for row in reader:
                    if first_row is None:
                        first_row = row
                    last_row = row

                if last_row:
                    controller.integral_sum = float(last_row.get('integral_sum', 0.0))
                    controller.last_error = float(last_row.get('error', 0.0))

                if first_row:
                    controller.kp = float(first_row.get('kp', controller.kp))
                    controller.ki = float(first_row.get('ki', controller.ki))
                    controller.foptd_gain = float(first_row.get('foptd_gain', controller.foptd_gain))
                    controller.foptd_tau = float(first_row.get('foptd_tau', controller.foptd_tau))
                    controller.foptd_delay = float(first_row.get('foptd_delay', controller.foptd_delay))

                    print(f"[{controller.name}-Static] Resumed. Locked to Baseline Kp: {controller.kp:.3f}. Ki: {controller.ki:.3f}")

        except (ValueError, KeyError) as e:
            print(f"[{controller.name}] Corrupt data in state file ({e}). Maintaining current PI state in memory.")
        except Exception as e:
            print(f"[{controller.name}] Unexpected error reading state file: {e}.")

    def evaluate_performance(self, controller, error, dt):
        pass


class TimeBasedStrategyManager:
    def __init__(self, schedule_configs):
        self.schedule_configs = schedule_configs
        self.current_strategy = None

    def get_active_strategy(self):
        for schedule, strategy in self.schedule_configs:
            if schedule.is_active():
                return strategy
        return None