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
    def load_state(self, controller, log_file):
        """Reads the CSV file and reloads the LAST tuning parameters and PI state."""
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
                    # 1. Reload PI State
                    controller.integral_sum = float(last_row.get('integral_sum', 0.0))
                    controller.last_error = float(last_row.get('error', 0.0))
                    
                    # 2. Reload Adaptive Parameters (latest EA tuning)
                    controller.kp = float(last_row.get('kp', controller.kp))
                    controller.ki = float(last_row.get('ki', controller.ki))
                    controller.foptd_gain = float(last_row.get('foptd_gain', controller.foptd_gain))
                    controller.foptd_tau = float(last_row.get('foptd_tau', controller.foptd_tau))
                    controller.foptd_delay = float(last_row.get('foptd_delay', controller.foptd_delay))
                    
                    # 3. Reload Watchdog state
                    controller.itae_current_window = float(last_row.get('itae_current_window', 0.0))
                    controller.itae_previous_window = float(last_row.get('itae_previous_window', 0.0))
                    controller.window_timer = float(last_row.get('window_timer', 0.0))
                    
                    print(f"[{controller.name}-Adaptive] Resumed. Kp: {controller.kp:.3f}. Ki: {controller.ki:.3f}")
        except Exception as e:
            print(f"[{controller.name}-Adaptive] Error reading state file: {e}. Starting fresh.")

    def evaluate_performance(self, controller, error, dt):
        """Evaluates the 10-minute tumbling window for a 5% ITAE shift."""
        # ITAE: Time-weight (window_timer) * absolute error * dt
        controller.itae_current_window += controller.window_timer * abs(error) * dt
        controller.window_timer += dt

        if controller.window_timer >= controller.window_duration:
            print(f"[{controller.name}-Adaptive] 10-Min Window Closed. Current ITAE: {controller.itae_current_window:.2f} | Prev ITAE: {controller.itae_previous_window:.2f}")
            
            # Avoid division by zero on the very first 10-minute run
            if controller.itae_previous_window > 0:
                percent_change = abs(controller.itae_current_window - controller.itae_previous_window) / controller.itae_previous_window
                
                if percent_change >= 0.05:
                    print(f"[{controller.name}-Adaptive] ITAE shift of {percent_change*100:.1f}% detected! Triggering EA retune...")
                    controller.trigger_retuning()
            
            # Shift windows and reset timer
            controller.itae_previous_window = controller.itae_current_window
            controller.itae_current_window = 0.0
            controller.window_timer = 0.0

class StaticTuningStrategy(TuningStrategy):
    def load_state(self, controller, log_file):
        """Reads the CSV file. Loads tuning parameters from the FIRST row, but integral state from the LAST row."""
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
                
                # 1. Reload PI State from LAST row (so control signal doesn't jump on restart)
                if last_row:
                    controller.integral_sum = float(last_row.get('integral_sum', 0.0))
                    controller.last_error = float(last_row.get('error', 0.0))

                # 2. Reload Tuning Parameters from FIRST row (enforcing static rules)
                if first_row:
                    controller.kp = float(first_row.get('kp', controller.kp))
                    controller.ki = float(first_row.get('ki', controller.ki))
                    controller.foptd_gain = float(first_row.get('foptd_gain', controller.foptd_gain))
                    controller.foptd_tau = float(first_row.get('foptd_tau', controller.foptd_tau))
                    controller.foptd_delay = float(first_row.get('foptd_delay', controller.foptd_delay))
                    
                    print(f"[{controller.name}-Static] Resumed. Locked to Initial Kp: {controller.kp:.3f}. Ki: {controller.ki:.3f}")
        except Exception as e:
            print(f"[{controller.name}-Static] Error reading state file: {e}. Starting fresh.")

    def evaluate_performance(self, controller, error, dt):
        """Static strategy does not tune, so it does nothing here."""
        pass
