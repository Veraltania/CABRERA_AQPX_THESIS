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
    def __init__(self, window_duration=1800): # monitor for 30 minutes / 1800 seconds
        # Watchdog variables for the tuning strategy
        self.window_duration = window_duration
        self.window_timer = 0.0
        self.itae_current_window = 0.0
        self.itae_previous_window = 0.0

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
                    # Reload PI State
                    controller.integral_sum = float(last_row.get('integral_sum', 0.0))
                    controller.last_error = float(last_row.get('error', 0.0))
                    
                    # Reload Adaptive Parameters (latest EA tuning)
                    controller.kp = float(last_row.get('kp', controller.kp))
                    controller.ki = float(last_row.get('ki', controller.ki))
                    controller.foptd_gain = float(last_row.get('foptd_gain', controller.foptd_gain))
                    controller.foptd_tau = float(last_row.get('foptd_tau', controller.foptd_tau))
                    controller.foptd_delay = float(last_row.get('foptd_delay', controller.foptd_delay))

                    # Reload Watchdog state (Change from controller. to self.)
                    self.itae_current_window = float(last_row.get('itae_current_window', 0.0))
                    self.itae_previous_window = float(last_row.get('itae_previous_window', 0.0))
                    self.window_timer = float(last_row.get('window_timer', 0.0))
                    
                    print(f"[{controller.name}-Adaptive] Resumed. Kp: {controller.kp:.3f}. Ki: {controller.ki:.3f}")
        except Exception as e:
            print(f"[{controller.name}-Adaptive] Error reading state file: {e}. Starting fresh.")

    def evaluate_performance(self, controller, error, dt):
        """Evaluates the window for a 5% ITAE shift."""
        self.itae_current_window += self.window_timer * abs(error) * dt
        self.window_timer += dt

        if self.window_timer >= self.window_duration:
            print(f"[{controller.name}-Adaptive] {self.window_duration}-Min Window Closed. \n")
            print(f"Current ITAE: {self.itae_current_window:.2f} \n")
            print(f"Prev ITAE: {self.itae_previous_window:.2f} \n")
            
            # Avoid division by zero on the very first 30-minute run
            if self.itae_previous_window > 0:
                percent_change = abs(self.itae_current_window - self.itae_previous_window) / controller.itae_previous_window
                
                if percent_change >= 0.05:
                    print(f"[{controller.name}-Adaptive] ITAE shift of {percent_change*100:.1f}% detected! Triggering EA retune...")
                    controller.retune()
            
            # Shift windows and reset timer
            self.itae_previous_window = controller.itae_current_window
            self.itae_current_window = 0.0
            self.window_timer = 0.0


class StaticTuningStrategy(TuningStrategy):
    def load_state(self, controller, log_file):
        """Reads the CSV file and loads BOTH integral state and parameters from the LAST row."""
        if not os.path.exists(log_file):
            print(f"[{controller.name}-Static] No state file found. Starting fresh.")
            return

        try:
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                last_row = None

                for row in reader:
                    last_row = row

                if last_row:
                    # 1. Reload PI State to prevent control signal jumps
                    controller.integral_sum = float(last_row.get('integral_sum', 0.0))
                    controller.last_error = float(last_row.get('error', 0.0))

                    # 2. Reload Tuning Parameters from LAST row to freeze adaptive values
                    controller.kp = float(last_row.get('kp', controller.kp))
                    controller.ki = float(last_row.get('ki', controller.ki))
                    controller.foptd_gain = float(last_row.get('foptd_gain', controller.foptd_gain))
                    controller.foptd_tau = float(last_row.get('foptd_tau', controller.foptd_tau))
                    controller.foptd_delay = float(last_row.get('foptd_delay', controller.foptd_delay))

                    print(
                        f"[{controller.name}-Static] Resumed. Locked to Last Adaptive Kp: {controller.kp:.3f}. Ki: {controller.ki:.3f}")
        except Exception as e:
            print(f"[{controller.name}-Static] Error reading state file: {e}. Starting fresh.")

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