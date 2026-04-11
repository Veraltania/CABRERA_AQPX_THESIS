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

import os
import csv

class AdaptiveTuningStrategy: # Inherits from TuningStrategy
    def __init__(self, window_duration=3600):  
        self.window_duration = window_duration
        self.window_timer = 0.0
        
        # Switched from ITAE to ISE
        self.ise_current_window = 0.0
        self.ise_previous_window = 0.0
        
        # Physical check
        self.abs_error_sum = 0.0 

        self.first_window_completed = False
        
        # FIX: ISE Thresholds. 
        # A constant noise of 0.1 mg/L squared is 0.01. Over 1800s, ISE = ~18.0
        # A constant error of 0.5 mg/L squared is 0.25. Over 1800s, ISE = ~450.0
        self.noise_floor_ise = 100.0  
        self.shift_threshold = 0.50  # 50% shift required to trigger a retune
        
        # Physical sanity check. Do not retune unless average error in window is >= 0.25 mg/L
        self.min_physical_error_threshold = 0.25 
        
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

                    self.ise_current_window = float(last_row.get('ise_current_window', 0.0))
                    self.ise_previous_window = float(last_row.get('ise_previous_window', 0.0))
                    self.window_timer = float(last_row.get('window_timer', 0.0))

                    if self.ise_previous_window > 0:
                        self.first_window_completed = True

                    print(f"[{controller.name}-Adaptive] Resumed. Kp: {controller.kp:.3f}. Ki: {controller.ki:.3f}")
        except (ValueError, KeyError) as e:
            print(f"[{controller.name}] Corrupt data in state file ({e}). Maintaining current PI state in memory.")
        except Exception as e:
            print(f"[{controller.name}] Unexpected error reading state file: {e}.")

    def evaluate_performance(self, controller, error, dt):
        """Evaluates the window using both ISE shifts and physical MAE limits."""
        self.window_timer += dt
        
        # Calculate Absolute Error and Squared Error
        abs_e = abs(error)
        squared_e = error ** 2
        
        # Integrate over time
        self.ise_current_window += squared_e * dt
        self.abs_error_sum += abs_e * dt

        if self.window_timer >= self.window_duration:
            mean_abs_error = self.abs_error_sum / self.window_duration
            
            print(f"[{controller.name}-Adaptive] {self.window_duration}-Sec Window Closed.")
            print(f"Current ISE: {self.ise_current_window:.2f} | Prev ISE: {self.ise_previous_window:.2f}")
            print(f"Mean Absolute Error (MAE): {mean_abs_error:.3f} mg/L")

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
                # Sanity check
                if mean_abs_error >= self.min_physical_error_threshold:
                    
                    if self.ise_previous_window > 0:
                        percent_change = (self.ise_current_window - self.ise_previous_window) / self.ise_previous_window
                        
                        if percent_change >= self.shift_threshold:
                            print(f"[{controller.name}-Adaptive] ISE shift of {percent_change * 100:.1f}% detected with physical MAE of {mean_abs_error:.2f}! Triggering EA retune...\n")
                            if hasattr(controller, 'retune'):
                                controller.retune()
                                self.cooldown_active = True
                                
                    # Fallback: Absolute ISE threshold exceeded
                    if not self.cooldown_active and self.ise_current_window > (self.noise_floor_ise * 2):
                        print(f"[{controller.name}-Adaptive] Sustained massive error emerged (ISE Threshold exceeded)! Triggering EA retune...\n")
                        if hasattr(controller, 'retune'):
                            controller.retune()
                            self.cooldown_active = True
                else:
                    print(f"[{controller.name}-Adaptive] MAE ({mean_abs_error:.3f} mg/L) is within physical limits. No retune needed.\n")

            # Shift windows and reset timers
            self.ise_previous_window = self.ise_current_window
            self.ise_current_window = 0.0
            self.abs_error_sum = 0.0
            self.window_timer = 0.0

class StaticTuningStrategy(TuningStrategy):
    def load_state(self, controller, log_file):
        """Reads the CSV file and loads the integral state from the LAST row,
           but locks the tuning parameters to the FIRST row (baseline)."""
        if not os.path.exists(log_file):
            print(f"[{controller.name}-Static] No state file found. Starting fresh.")
            return

        try:
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                first_row = None
                last_row = None

                for row in reader:
                    # Capture the very first row for our baseline tuning parameters
                    if first_row is None:
                        first_row = row

                    # Continuously update to get the final row for our PI state
                    last_row = row

                if last_row:
                    # 1. Reload PI State to prevent control signal jumps (Bumpless Transfer)
                    controller.integral_sum = float(last_row.get('integral_sum', 0.0))
                    controller.last_error = float(last_row.get('error', 0.0))

                if first_row:
                    # 2. Reload Tuning Parameters from the FIRST row to freeze baseline values
                    controller.kp = float(first_row.get('kp', controller.kp))
                    controller.ki = float(first_row.get('ki', controller.ki))
                    controller.foptd_gain = float(first_row.get('foptd_gain', controller.foptd_gain))
                    controller.foptd_tau = float(first_row.get('foptd_tau', controller.foptd_tau))
                    controller.foptd_delay = float(first_row.get('foptd_delay', controller.foptd_delay))

                    print(
                        f"[{controller.name}-Static] Resumed. Locked to Baseline Kp: {controller.kp:.3f}. Ki: {controller.ki:.3f}")

        except (ValueError, KeyError) as e:
            print(f"[{controller.name}] Corrupt data in state file ({e}). Maintaining current PI state in memory.")
        except Exception as e:
            print(f"[{controller.name}] Unexpected error reading state file: {e}.")

    def evaluate_performance(self, controller, error, dt):
        """Static strategy does not tune, so it does nothing here."""
        pass


class TimeBasedStrategyManager:
    def __init__(self, schedule_configs):
        # Expects a list of tuples: (SchedulePolicy, TuningStrategy)
        self.schedule_configs = schedule_configs
        self.current_strategy = None

    def get_active_strategy(self):
        """Returns the strategy that should be active right now."""
        for schedule, strategy in self.schedule_configs:
            if schedule.is_active():
                return strategy
        return None