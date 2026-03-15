import paho.mqtt.client as mqtt
import time
import sys
import os
import csv
import datetime
from abc import ABC, abstractmethod
from Hardware_Pipeline.aqpx_controller import AqpxController
from Hardware_Pipeline.aqpx_logger import AqpxLogger

class ParameterController(ABC):
    def __init__(self, name, target, initial_kp, initial_ki, init_gain=1.0, init_tau=1.0, init_delay=0.0):
        self.name = name
        self.target = target
        
        # Tuning Parameters
        self.kp = initial_kp
        self.ki = initial_ki
        self.foptd_gain = init_gain
        self.foptd_tau = init_tau
        self.foptd_delay = init_delay
        
        self.log_file = f"{self.name.lower().replace(' ', '_')}_state.csv"
        self.dt = 5.0  # sensor data arrives every five seconds
        
        # State Variables
        self.integral_sum = 0.0
        self.last_error = 0.0
        self.max_out = 1.0
        self.min_out = 0.0
        
        # Watchdog Variables (10-minute window = 600 seconds)
        self.window_duration = 600.0  
        self.window_timer = 0.0
        self.itae_current_window = 0.0
        self.itae_previous_window = 0.0 
        
        self._load_previous_state()

    def _load_previous_state(self):
        """Reads the CSV file and reloads ONLY the tuning parameters and PI state."""
        if not os.path.exists(self.log_file):
            print(f"[{self.name}] No existing state file found. Starting fresh.")
            return

        try:
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                last_row = None
                for row in reader:
                    last_row = row 
                
                if last_row:
                    # 1. Reload PI State (so it doesn't forget its integral sum)
                    self.integral_sum = float(last_row.get('integral_sum', 0.0))
                    self.last_error = float(last_row.get('error', 0.0))
                    
                    # 2. Reload Adaptive Parameters (so it remembers the EA tuning)
                    self.kp = float(last_row.get('kp', self.kp))
                    self.ki = float(last_row.get('ki', self.ki))
                    self.foptd_gain = float(last_row.get('foptd_gain', self.foptd_gain))
                    self.foptd_tau = float(last_row.get('foptd_tau', self.foptd_tau))
                    self.foptd_delay = float(last_row.get('foptd_delay', self.foptd_delay))
                    
                    print(f"[{self.name}] Resumed. Kp: {self.kp:.3f}. Ki: {self.ki:.3f} | Watchdog timer reset to 0.")
        except Exception as e:
            print(f"[{self.name}] Error reading state file: {e}. Starting fresh.")

    def _log_current_state(self, current_val, error, pi_output):
        """Logs current state, including watchdog variables."""
        file_exists = os.path.exists(self.log_file)
        
        with open(self.log_file, 'a', newline='') as f:
            fieldnames = [
                'timestamp', 'target', 'current_val', 'error', 'integral_sum', 'pi_output', 
                'kp', 'ki', 'foptd_gain', 'foptd_tau', 'foptd_delay', 
                'itae_current_window', 'itae_previous_window', 'window_timer'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'target': self.target,
                'current_val': current_val,
                'error': error,
                'integral_sum': self.integral_sum,
                'pi_output': pi_output,
                'kp': self.kp,
                'ki': self.ki,
                'foptd_gain': self.foptd_gain,
                'foptd_tau': self.foptd_tau,
                'foptd_delay': self.foptd_delay,
                'itae_current_window': self.itae_current_window,
                'itae_previous_window': self.itae_previous_window,
                'window_timer': self.window_timer
            })

    def _check_watchdog(self):
        """Evaluates the 10-minute tumbling window for a 5% ITAE shift."""
        # Check if the 10-minute window has elapsed
        if self.window_timer >= self.window_duration:
            print(f"[{self.name}] 10-Minute Window Closed. Current ITAE: {self.itae_current_window:.2f} | Prev ITAE: {self.itae_previous_window:.2f}")
            
            # Avoid division by zero on the very first 10-minute run
            if self.itae_previous_window > 0:
                # Calculate absolute percentage change
                percent_change = abs(self.itae_current_window - self.itae_previous_window) / self.itae_previous_window
                
                if percent_change >= 0.05:
                    print(f"[{self.name}] ITAE shift of {percent_change*100:.1f}% detected! Triggering EA retune...")
                    self.trigger_retuning()
            
            # Shift windows and reset timer
            self.itae_previous_window = self.itae_current_window
            self.itae_current_window = 0.0
            self.window_timer = 0.0

    def calculate_pi(self, current_val):
        error = self.target - current_val
        
        # ITAE: Time-weight (window_timer) * absolute error * dt
        self.itae_current_window += self.window_timer * abs(error) * self.dt
        self.window_timer += self.dt
        
        self._check_watchdog()
        
        # Standard PI Calculation
        p_term = self.kp * error

        self.integral_sum += (error * self.dt)
        i_term = self.ki * self.integral_sum
        pi_output_unclamped = p_term + i_term

        # anti-windup logic
        # prevent controller from asking for percentages of power higher than 100%
        if(pi_output_unclamped > self.max_out):
            pi_output_unclamped = self.max_out
            if(error < 0):
                self.integral_sum += (error * self.dt)
        # prevent controller from asking for negative percentages of power
        elif(pi_output_unclamped < self.min_out):
            pi_output_unclamped = self.min_out
            if(error > 0):
                self.integral_sum += (error * self.dt)
        else:
            self.integral_sum += (error * self.dt)
            
        pi_output = pi_output_unclamped
            
        self.last_error = error
        self._log_current_state(current_val, error, pi_output)
        
        return pi_output

    def update_tuning_parameters(self, new_kp, new_ki, gain, tau, delay):
        self.kp = new_kp
        self.ki = new_ki
        self.foptd_gain = gain
        self.foptd_tau = tau
        self.foptd_delay = delay
        print(f"[{self.name}] Retuned. New Kp: {self.kp:.3f}, Ki: {self.ki:.3f}")

    @abstractmethod
    def trigger_retuning(self):
        """Child classes MUST implement this to execute their specific EA logic."""
        pass

    @abstractmethod
    def process(self, data, dt, elapsed_time):
        pass

class DOController(ParameterController):
    def __init__(self):
        super().__init__(name="Dissolved-Oxygen", 
                        target=6.0,
                        initial_kp = 0.7,
                        initial_ki = 0.001,
                        init_gain = 50,
                        init_tau = 1.0,
                        init_delay = 0.5
                        )

    def process(self, data):
        # Safely extract DO from the incoming payload
        current_do = data.get('mcp_wq', {}).get('do')
        
        if current_do is not None:
            pi_output = self.calculate_pi(current_do)
            print(f"[{self.name}] DO: {current_do}mg/L | Target: {self.target} | Control Signal: {pi_output:.2f}")
        else:
            print(f"[{self.name} Controller] No DO data found in payload.")

class TDSController(ParameterController):
    def __init__(self):
        super().__init__(name="Total-Dissolved-Solids", 
                        target=100,
                        initial_kp = 0.7,
                        initial_ki = 0.001,
                        init_gain = 50,
                        init_tau = 1.0,
                        init_delay = 0.5
                        )

    def process(self, data):
        current_tds = data.get('mcp_wq', {}).get('tds')
        if current_tds is not None:
            pi_output = self.calculate_pi(current_tds)
            print(f"[{self.name}] TDS: {current_tds} | Target: {self.target} | Control Signal: {pi_output:.2f}")
        else:
            print(f"[{self.name} Controller] No TDS data found in payload.")

class AquaponicsSystem:
    """Routes data to dedicated controllers."""
    
    def __init__(self, broker="localhost", port=1883):
        self.broker = broker
        self.port = port
        
        # Instantiate the specific controllers
        self.do_controller = DOController()
        self.tds_controller = TDSController()
        
        self.logger = AqpxLogger(
            broker=self.broker, 
            port=self.port, 
            on_data_received_callback=self.process_sensor_data
        )

    def process_sensor_data(self, data):
        """
        Triggered by the Logger. Routes the full JSON payload to 
        each specific controller so they can extract what they need.
        """
        print("\n[System] New sensor data received. Routing to controllers...")
        
        # Delegate the processing to the specific control loop classes
        self.do_controller.process(data)
        self.tds_controller.process(data)

    def run(self):
        """Starts the system."""
        print(f"Starting Aquaponics System (Broker: {self.broker}:{self.port})...")
        self.logger.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down system...")
            self.logger.stop()
            sys.exit(0)

if __name__ == '__main__':
    system = AquaponicsSystem(broker="localhost", port=1883)
    system.run()