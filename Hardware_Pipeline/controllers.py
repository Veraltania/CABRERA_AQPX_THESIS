import os
import csv
from datetime import datetime
from abc import ABC, abstractmethod
import threading
import time

# IMPORT THE NEW CUSTOM FITTER
from Transfer_Function_Modeling.closed_loop_fitter import extract_csv_and_fit
from Evolutionary_Algorithm_Testing.de.de_optimizer import DEOptimizer

class ParameterController(ABC):
    def __init__(self, name: str, setpoint: float, strategy_manager, actuator,
                 initial_kp, initial_ki, init_gain=1.0, init_tau=1.0, init_delay=0.0):
        self.name = name
        self.setpoint = setpoint
        self.strategy_manager = strategy_manager
        self.actuator = actuator

        # Tuning Parameters
        self.kp = initial_kp
        self.ki = initial_ki
        self.foptd_gain = init_gain
        self.foptd_tau = init_tau
        self.foptd_delay = init_delay

        self.log_file = f"{self.name.lower().replace(' ', '_')}_state.csv"
        self.last_time = time.time()
        self.dt = 5.0

        # State Variables
        self.integral_sum = 0.0
        self.last_error = 0.0
        self.max_out = 1.0
        self.min_out = 0.0

        # Retuning Variables
        self.is_retuning = False
        self.retuning_file = None
        self.retuning_headers = []
        self.retune_thread_active = False

    def is_active(self) -> bool:
        return self.strategy_manager.get_active_strategy() is not None

    def start_retuning_session(self, target_column):
        self.is_retuning = True
        self.retuning_file = f"{self.name}_retuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        # ADDED 'Duty_Cycle' so the fitter can access the control signal
        self.retuning_headers = ['Date', 'Time', 'StatusCode', target_column, 'Duty_Cycle']
        with open(self.retuning_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.retuning_headers)

    def stop_retuning_session(self):
        self.is_retuning = False

    def _record_retuning_data(self, current_val):
        """Logs real-time data to the CSV while the retune step test is running."""
        if not self.is_retuning: return
        
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        
        row = [date_str, time_str, "200", f"{current_val:.4f}", f"{self.actuator.duty_cycle:.4f}"]
        
        with open(self.retuning_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def retune(self):
        if self.retune_thread_active:
            print(f"[{self.name}] Retune already in progress. Ignoring trigger.")
            return
            
        self.retune_thread_active = True
        t = threading.Thread(target=self._retune_process, daemon=True)
        t.start()

    def _retune_process(self):
        old_setpoint = self.setpoint
        step_size = 0.2 
        new_setpoint = old_setpoint + step_size
        
        print(f"[{self.name}] Hardware Adaptive Retune Triggered!")
        print(f"[{self.name}] Running closed-loop step test to {new_setpoint:.2f}...")
        
        self.start_retuning_session(self.target_column)
        self.setpoint = new_setpoint
        
        # 1. Wait for system to reach the new setpoint (Rise time detection)
        tolerance = abs(step_size) * 0.05
        reached = False
        timeout_seconds = 7200  # 2 hours max wait for hardware to reach setpoint
        elapsed = 0
        
        while elapsed < timeout_seconds:
            if abs(self.last_error) <= tolerance:
                reached = True
                break
            time.sleep(5)
            elapsed += 5
            
        # 2. Wait for stabilization (Tail logging)
        if reached:
            print(f"[{self.name}] Rise time achieved. Logging tail for 15 minutes to stabilize...")
            time.sleep(900)
        else:
            print(f"[{self.name}] Step test timeout reached. Logging current tail...")
            time.sleep(300)
            
        self.stop_retuning_session()
        
        # 3. Extract Plant Parameters & EA Optimization
        print(f"[{self.name}] Step test complete. Extracting closed-loop FOPDT parameters...")
        try:
            ex_K, ex_tau, ex_delay = extract_csv_and_fit(
                csv_file_path=self.retuning_file,
                y_column=self.target_column,
                u_column='Duty_Cycle'
            )
            
            print(f"[{self.name}] Custom Fitter Modeled Plant -> K={ex_K:.4f}, Tau={ex_tau:.2f}, Delay={ex_delay:.2f}")

            safe_delay = max(0.05, ex_delay) 
            safe_tau = max(1.0, ex_tau) 
            
            tf_params = {
                'tf_num': [ex_K], 'tf_den': [safe_tau, 1], 'tf_delay': safe_delay,        
                'tf_n_pade': 2, 'computed_delay': safe_delay,
                'is_reverse_acting': ex_K < 0, 'max_kp': 20.0
            }
            
            de_config = {
                'population_size': 50, 'max_iters': 20, 'patience_limit': 5,
                'mutation': (0.5, 1.0), 'recombination': 0.745, 'strategy': 'best1bin', 'n_rounds': 1
            }
            
            optimizer = DEOptimizer(config=de_config, tf_params=tf_params)
            best_Kp, best_Ki, cost, _, _ = optimizer.optimize_round(round_num=1)
            
            self.update_tuning_parameters(best_Kp, best_Ki, ex_K, safe_tau, safe_delay)
            
        except Exception as e:
            print(f"[{self.name}] Hardware retuning pipeline failed: {e}")
            
        self.setpoint = old_setpoint
        self.retune_thread_active = False

    def update_tuning_parameters(self, new_kp, new_ki, new_gain, new_tau, new_delay):
        print(f"[{self.name}] Applying New Tunings -> Kp: {new_kp:.4f}, Ki: {new_ki:.4f}")
        self.kp = new_kp
        self.ki = new_ki
        self.foptd_gain = new_gain
        self.foptd_tau = new_tau
        self.foptd_delay = new_delay
        self.save_state()

    def save_state(self):
        strategy = self.strategy_manager.get_active_strategy()
        row = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'kp': self.kp, 'ki': self.ki,
            'foptd_gain': self.foptd_gain, 'foptd_tau': self.foptd_tau, 'foptd_delay': self.foptd_delay,
            'integral_sum': self.integral_sum, 'error': self.last_error,
            'ise_current_window': getattr(strategy, 'ise_current_window', 0.0),
            'ise_previous_window': getattr(strategy, 'ise_previous_window', 0.0),
            'window_timer': getattr(strategy, 'window_timer', 0.0)
        }
        file_exists = os.path.exists(self.log_file)
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def calculate_pi(self, current_val):
        current_time = time.time()
        self.dt = current_time - self.last_time
        if self.dt <= 0: self.dt = 1.0 
        self.last_time = current_time

        error = self.setpoint - current_val
        self.last_error = error

        tentative_integral = self.integral_sum + (error * self.dt)
        pi_out = (self.kp * error) + (self.ki * tentative_integral)
        clamped_out = max(self.min_out, min(self.max_out, pi_out))

        if self.min_out < pi_out < self.max_out:
            self.integral_sum = tentative_integral

        return clamped_out

    @abstractmethod
    def process(self, data):
        pass


class DOController(ParameterController):
    def __init__(self, name: str, strategy_manager, actuator):
        super().__init__(name=name, setpoint=2.0, strategy_manager=strategy_manager, actuator=actuator,
                         initial_kp=1.0, initial_ki=0.01, init_gain=1.3, init_tau=1500.0, init_delay=100.0)
        self.target_column = "MCP_WQ_DO"

    def process(self, data):
        current_do = data.get('mcp_wq', {}).get('do')
        if current_do is not None:
            pi_output = self.calculate_pi(current_do)

            # print(f"[{self.name}] DO: {current_do} mg/L")
            # print(f"Target: {self.setpoint} | Control Signal: {pi_output:.2f}\n")

            self.actuator.set_duty_cycle(pi_output)
            
            # Log background data if an adaptive retuning bump test is happening
            if self.is_retuning:
                self._record_retuning_data(current_do)
            
            # Evaluate ISE health
            strategy = self.strategy_manager.get_active_strategy()
            if strategy:
                strategy.evaluate_performance(self, self.last_error, self.dt)
        else:
            print(f"[{self.name} Controller] No DO data found in payload.")
            self.actuator.set_duty_cycle(1.0)


class TDSController(ParameterController):
    def __init__(self, name: str, strategy_manager, actuator):
        super().__init__(name=name, setpoint=200, strategy_manager=strategy_manager, actuator=actuator,
                         initial_kp=0.7, initial_ki=0.001, init_gain=50, init_tau=1.0, init_delay=0.5)
        self.target_column = "MCP_WQ_TDS"

    def process(self, data):
        current_tds = data.get('mcp_wq', {}).get('tds')
        if current_tds is not None:
            pi_output = self.calculate_pi(current_tds)

            # print(f"[{self.name}] TDS: {current_tds} mg/L")
            # print(f"Target: {self.setpoint} | Control Signal: {pi_output:.2f}\n")

            self.actuator.set_duty_cycle(pi_output)
            
            if self.is_retuning:
                self._record_retuning_data(current_tds)

            strategy = self.strategy_manager.get_active_strategy()
            if strategy:
                strategy.evaluate_performance(self, self.last_error, self.dt)
        else:
            print(f"[{self.name} Controller] No TDS data found in payload.")
            self.actuator.set_duty_cycle(1.0)