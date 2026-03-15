import paho.mqtt.client as mqtt
import time
import sys
import os
import csv
import datetime
from abc import ABC, abstractmethod

class ParameterController(ABC):
    # Injected SchedulePolicy instead of raw start/end times
    def __init__(self, name: str, target: float, strategy: TuningStrategy, schedule: SchedulePolicy, initial_kp, initial_ki, init_gain=1.0, init_tau=1.0, init_delay=0.0):
        self.name = name
        self.target = target
        self.strategy = strategy
        self.schedule = schedule  # The new schedule strategy
        
        # Tuning Parameters
        self.kp = initial_kp
        self.ki = initial_ki
        self.foptd_gain = init_gain
        self.foptd_tau = init_tau
        self.foptd_delay = init_delay
        
        self.log_file = f"{self.name.lower().replace(' ', '_')}_state.csv"
        self.dt = 5.0  
        
        # State Variables
        self.integral_sum = 0.0
        self.last_error = 0.0
        self.max_out = 1.0
        self.min_out = 0.0
        
        # Watchdog Variables
        self.window_duration = 600.0  
        self.window_timer = 0.0
        self.itae_current_window = 0.0
        self.itae_previous_window = 0.0 
        
        # Delegate loading state
        # self.strategy.load_state(self, self.log_file) # Uncomment when using actual files

    def is_active(self) -> bool:
        """Delegates the time check to the injected Schedule Policy."""
        return self.schedule.is_active()

    def _log_current_state(self, current_val, error, pi_output):
        pass # (Your existing logging logic)

    def calculate_pi(self, current_val):
        pass # (Your existing PI math logic)
        return 0.0 # Placeholder

    def update_tuning_parameters(self, new_kp, new_ki, gain, tau, delay):
        self.kp = new_kp
        self.ki = new_ki
        self.foptd_gain = gain
        self.foptd_tau = tau
        self.foptd_delay = delay
        print(f"[{self.name}] Retuned. New Kp: {self.kp:.3f}, Ki: {self.ki:.3f}")

    @abstractmethod
    def trigger_retuning(self): pass

    @abstractmethod
    def process(self, data): pass


class DOController(ParameterController):
    def __init__(self, name: str, strategy: TuningStrategy, schedule: SchedulePolicy):
        super().__init__(name=name, 
                        target=6.0,
                        strategy=strategy,
                        schedule=schedule,
                        initial_kp=0.7,
                        initial_ki=0.001,
                        init_gain=50,
                        init_tau=1.0,
                        init_delay=0.5
                        )

    def process(self, data):
        current_do = data.get('mcp_wq', {}).get('do')
        if current_do is not None:
            # pi_output = self.calculate_pi(current_do)
            # print(f"[{self.name}] DO: {current_do}mg/L | Target: {self.target} | Control Signal: {pi_output:.2f}")
            print(f"[{self.name}] Processing DO...")
        else:
            print(f"[{self.name} Controller] No DO data found in payload.")
            
    def trigger_retuning(self): pass


class TDSController(ParameterController):
    def __init__(self, name: str, strategy: TuningStrategy, schedule: SchedulePolicy):
        super().__init__(name=name, 
                        target=100,
                        strategy=strategy,
                        schedule=schedule,
                        initial_kp=0.7,
                        initial_ki=0.001,
                        init_gain=50,
                        init_tau=1.0,
                        init_delay=0.5
                        )

    def process(self, data):
        current_tds = data.get('mcp_wq', {}).get('tds')
        if current_tds is not None:
            # pi_output = self.calculate_pi(current_tds)
            # print(f"[{self.name}] TDS: {current_tds} | Target: {self.target} | Control Signal: {pi_output:.2f}")
            print(f"[{self.name}] Processing TDS...")
        else:
            print(f"[{self.name} Controller] No TDS data found in payload.")
            
    def trigger_retuning(self): pass