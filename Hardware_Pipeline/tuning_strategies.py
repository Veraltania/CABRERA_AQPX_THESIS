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
    def __init__(self, window_duration=1800):  # monitor for 30 minutes / 1800 seconds
        self.window_duration = window_duration
        self.window_timer = 0.0
        self.itae_current_window = 0.0
        self.itae_previous_window = 0.0

        # FIX: Added flag to prevent immediate retunes on fresh schedule blocks
        self.first_window_completed = False
        self.noise_floor_itae = 25.0  # Minimum ITAE to be considered "stable"
        self.shift_threshold = 0.30  # 30% shift required to trigger a retune

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

                    self.itae_current_window = float(last_row.get('itae_current_window', 0.0))
                    self.itae_previous_window = float(last_row.get('itae_previous_window', 0.0))
                    self.window_timer = float(last_row.get('window_timer', 0.0))

                    # FIX: If we successfully loaded a previous baseline from CSV, we can skip the warmup
                    if self.itae_previous_window > 0:
                        self.first_window_completed = True

                    print(f"[{controller.name}-Adaptive] Resumed. Kp: {controller.kp:.3f}. Ki: {controller.ki:.3f}")
        except (ValueError, KeyError) as e:
            print(f"[{controller.name}] Corrupt data in state file ({e}). Maintaining current PI state in memory.")
        except Exception as e:
            print(f"[{controller.name}] Unexpected error reading state file: {e}.")

    def evaluate_performance(self, controller, error, dt):
        """Evaluates the window for a 5% ITAE shift."""
        self.window_timer += dt
        self.itae_current_window += self.window_timer * abs(error) * dt

        if self.window_timer >= self.window_duration:
            print(f"[{controller.name}-Adaptive] {self.window_duration}-Sec Window Closed.")
            print(f"Current ITAE: {self.itae_current_window:.2f} | Prev ITAE: {self.itae_previous_window:.2f}")

            # FIX: Only evaluate for a retune if we have established a real baseline
            if not self.first_window_completed:
                print(
                    f"[{controller.name}-Adaptive] Baseline established. Skipping retune evaluation for this window.\n")
                self.first_window_completed = True
            else:
                # Require the previous window to be above the noise floor
                if self.itae_previous_window > self.noise_floor_itae:
                    percent_change = abs(
                        self.itae_current_window - self.itae_previous_window) / self.itae_previous_window
                    if percent_change >= self.shift_threshold:
                        print(
                            f"[{controller.name}-Adaptive] ITAE shift of {percent_change * 100:.1f}% detected! Triggering EA retune...\n")
                        if hasattr(controller, 'retune'):
                            controller.retune()

                    # Fallback: If the previous window was near-zero, but the new window spikes severely
                    elif self.itae_current_window > max(100.0, self.noise_floor_itae * 2):

                elif self.itae_current_window > 1.0:
                    print(f"[{controller.name}-Adaptive] Error emerged from ideal state! Triggering EA retune...\n")
                    if hasattr(controller, 'retune'):
                        controller.retune()

            # Shift windows and reset timer.
            self.itae_previous_window = self.itae_current_window
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