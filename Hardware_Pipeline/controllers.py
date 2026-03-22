import os
import csv
from datetime import datetime
from abc import ABC, abstractmethod
import threading
import time

from Transfer_Function_Modeling.response_modeler import analyze_response
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
        self.dt = 5.0

        # State Variables
        self.integral_sum = 0.0
        self.last_error = 0.0
        self.max_out = 1.0
        self.min_out = 0.0

        # Retuning variables
        self.is_retuning = False
        self.retuning_file = None
        self.target_column = None
        self.retuning_folder = f"retuning_logs/{self.name.lower().replace(' ', '_')}"
        self.current_process_value = 0.0
        self.current_strategy = self.strategy_manager.get_active_strategy()
        self.retune_thread_active = False

        # Load previous state upon initialization using the strategy
        self._load_previous_state()

    def start_retuning_session(self, target_column: str):
        """Initializes folders and the CSV file for a retuning test."""
        if not os.path.exists(self.retuning_folder):
            os.makedirs(self.retuning_folder)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.retuning_file = os.path.join(self.retuning_folder, f"retune_{timestamp}.csv")
        self.target_column = target_column
        self.is_retuning = True

        # Write header in the format of your uploaded AQPX log
        with open(self.retuning_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Time', 'StatusCode', self.target_column])

        print(f"[{self.name}] Retuning session started. Logging to: {self.retuning_file}")

    def _record_retuning_data(self, current_val):
        """Records a single row to the retuning CSV if active."""
        if not self.is_retuning or not self.retuning_file:
            return

        now = datetime.now()
        date_str = now.strftime("%Y/%m/%d")  # Matches AQPX format
        time_str = now.strftime("%H:%M:%S")

        with open(self.retuning_file, 'a', newline='') as f:
            writer = csv.writer(f)
            # StatusCode is hardcoded to 0 for normal retuning logs
            writer.writerow([date_str, time_str, 0, current_val])

    def stop_retuning_session(self):
        """Ends the data collection."""
        self.is_retuning = False
        print(f"[{self.name}] Retuning session closed. File: {self.retuning_file}")

    def _load_previous_state(self):
        """Delegates state loading to the currently active strategy."""
        if self.current_strategy:
            self.current_strategy.load_state(self, self.log_file)

    def _log_current_state(self, current_val, error, pi_output):
        """Logs current state, including watchdog variables."""
        file_exists = os.path.exists(self.log_file)

        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'setpoint', 'current_val', 'error', 'integral_sum', 'pi_output',
                    'kp', 'ki', 'foptd_gain', 'foptd_tau', 'foptd_delay',
                    'itae_current_window', 'itae_previous_window', 'window_timer'
                ])

                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'setpoint': self.setpoint,
                    'current_val': current_val,
                    'error': error,
                    'integral_sum': self.integral_sum,
                    'pi_output': pi_output,
                    'kp': self.kp,
                    'ki': self.ki,
                    'foptd_gain': self.foptd_gain,
                    'foptd_tau': self.foptd_tau,
                    'foptd_delay': self.foptd_delay,
                    'itae_current_window': getattr(self.current_strategy, 'itae_current_window', 0.0),
                    'itae_previous_window': getattr(self.current_strategy, 'itae_previous_window', 0.0),
                    'window_timer': getattr(self.current_strategy, 'window_timer', 0.0)
                })
        except Exception as e:
            print(f"[{self.name}] Failed to write to log file: {e}")

    def is_active(self) -> bool:
        """Controller is active if the manager returns a valid strategy for the current time."""
        return self.strategy_manager.get_active_strategy() is not None

    def _check_and_update_strategy(self):
        """Checks if the time block changed and swaps the strategy dynamically."""
        new_strategy = self.strategy_manager.get_active_strategy()
        if new_strategy and type(new_strategy) != type(self.current_strategy):
            print(
                f"[{self.name}] Time block shift: Swapping strategy from {type(self.current_strategy).__name__ if self.current_strategy else 'None'} to {type(new_strategy).__name__}")
            self.current_strategy = new_strategy

    def calculate_pi(self, current_val):
        """Clean, simplified calculation using hardcoded intervals."""
        # 1. Check if we need to swap strategies before calculating!
        self._check_and_update_strategy()

        self.current_process_value = current_val
        error = self.setpoint - current_val

        # 2. Delegate performance evaluation to the CURRENT strategy
        if self.current_strategy:
            self.current_strategy.evaluate_performance(self, error, self.dt)

        # Standard PI Calculation
        p_term = self.kp * error

        # calculate tentative integral to determine wind-up
        tentative_integral = self.integral_sum + (error * self.dt)
        i_term = self.ki * tentative_integral
        pi_output = p_term + i_term

        if pi_output > self.max_out:
            pi_output = self.max_out
            # Do NOT update self.integral_sum (Anti-windup)
        elif pi_output < self.min_out:
            pi_output = self.min_out
            # Do NOT update self.integral_sum (Anti-windup)
        else:
            # We are within limits, safe to accumulate the integral
            self.integral_sum = tentative_integral

        self._record_retuning_data(current_val)

        self.last_error = error
        self._log_current_state(current_val, error, pi_output)

        return pi_output

    def update_tuning_parameters(self, new_kp, new_ki, gain, tau, delay):
        """Updates both PI gains and the underlying FOPTD model tracking parameters."""
        self.kp = new_kp
        self.ki = new_ki
        self.foptd_gain = gain
        self.foptd_tau = tau
        self.foptd_delay = delay
        print(f"[{self.name}] Active Tuning Updated -> Kp: {self.kp:.4f}, Ki: {self.ki:.4f}")
        print(
            f"[{self.name}] Active Plant Updated  -> K: {self.foptd_gain:.2f}, Tau: {self.foptd_tau:.2f}, Delay: {self.foptd_delay:.2f}")

    def retune(self):
        # 1. Strict Concurrency Check
        if getattr(self, 'retune_thread_active', False):
            print(f"[{self.name}] Retune already in progress. Ignoring duplicate trigger.")
            return

        self.retune_thread_active = True

        def retune_worker():
            try:
                # 1. Setup the closed-loop setpoint step test
                old_setpoint = self.setpoint
                step_size = old_setpoint * 0.10 if old_setpoint != 0 else 1.0  # 10% step
                new_setpoint = old_setpoint + step_size

                # Start logging if not already started
                if not self.is_retuning:
                    self.start_retuning_session(self.target_column)

                self.setpoint = new_setpoint

                step_start_time = time.time()
                step_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{self.name}] Closed-loop setpoint test started. Setpoint: {old_setpoint} -> {new_setpoint:.2f}")

                # 2. Wait for system to reach the new setpoint (Rise Time detection)
                tolerance = abs(step_size) * 0.05
                rise_time = 0
                max_wait_time = 3600

                while True:
                    current_error = abs(self.current_process_value - new_setpoint)

                    if current_error <= tolerance:
                        rise_time = time.time() - step_start_time
                        break

                    if (time.time() - step_start_time) > max_wait_time:
                        print(f"[{self.name}] Retune timeout: System never reached the new setpoint.")
                        self.setpoint = old_setpoint
                        self.stop_retuning_session()
                        return

                    time.sleep(self.dt)

                print(f"[{self.name}] Rise time: {rise_time:.2f}s. Recording for additional {rise_time * 3:.2f}s.")

                # 3. Record for 3x the detected rise time
                time.sleep(rise_time * 3)

                # 4. Finish and Analyze
                end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.stop_retuning_session()

                # --- Step 4a: Extract FOPTD Parameters ---
                params = analyze_response(
                    file_paths=[self.retuning_file],
                    start_step=step_time_str,
                    end_step=end_time_str,
                    target_column=self.target_column,
                    window_seconds=60,
                    tf_name=f"{self.name}_FOPTD",
                    t_step_time=step_time_str,
                    delta_u=step_size  # Passes the setpoint step_size to your closed-loop modeler
                )

                extracted_gain = params['K']
                extracted_tau = params['tau']
                extracted_delay = params['theta']
                print(
                    f"[{self.name}] Plant modeled! K: {extracted_gain:.2f}, Tau: {extracted_tau:.2f}, Delay: {extracted_delay:.2f}")

                # --- Step 4b: Setup Optimizer Environment ---
                tf_params = {
                    'tf_num': [extracted_gain],
                    'tf_den': [extracted_tau],
                    'computed_delay': extracted_delay,
                    'is_reverse_acting': extracted_gain < 0,
                    'max_kp': 20.0  # Safety fallback bound
                }

                de_config = {
                    'population_size': 100,
                    'max_iters': 30,
                    'patience_limit': 10,
                    'mutation': (0.5, 1.0),
                    'recombination': 0.745,
                    'strategy': 'best1bin',
                    'n_rounds': 1,
                    'output_folder': f"online_tuning_logs/{self.name.lower().replace(' ', '_')}"
                }

                # Run Differential Evolution
                print(f"[{self.name}] Running Differential Evolution to optimize PI gains...")
                optimizer = DEOptimizer(config=de_config, tf_params=tf_params)

                # 1 round of optimization
                best_Kp, best_Ki, cost, iterations_run, _ = optimizer.optimize_round(round_num=1)

                print(f"[{self.name}] Optimization finished in {iterations_run} iterations. ITAE Cost: {cost:.4f}")

                # Apply new parameters
                self.update_tuning_parameters(best_Kp, best_Ki, extracted_gain, extracted_tau, extracted_delay)

            except Exception as e:
                print(f"[{self.name}] Retuning process failed: {e}")

            finally:
                # 5. Restore original setpoint AND release the concurrency lock
                self.setpoint = old_setpoint
                self.retune_thread_active = False
                print(f"[{self.name}] Restored original setpoint to {self.setpoint} and released retuning lock.")

        # Execute as a daemon thread so it doesn't block the main control loop
        thread = threading.Thread(target=retune_worker, daemon=True)
        thread.start()

class DOController(ParameterController):
    def __init__(self, name: str, strategy_manager, actuator):
        super().__init__(name=name,
                         setpoint=6.0,
                         strategy_manager=strategy_manager,
                         actuator=actuator,
                         initial_kp=0.7,
                         initial_ki=0.001,
                         init_gain=50,
                         init_tau=1.0,
                         init_delay=0.5
                         )

        self.target_column = "MCP_WQ_DO"

    def process(self, data):
        current_do = data.get('mcp_wq', {}).get('do')
        if current_do is not None:
            pi_output = self.calculate_pi(current_do)

            print(f"[{self.name}] DO: {current_do}mg/L \n")
            print(f"Target: {self.setpoint} \n")
            print(f"Control Signal: {pi_output:.2f} \n")

            self.actuator.set_duty_cycle(pi_output)

        else:
            print(f"[{self.name} Controller] No DO data found in payload.")
            self.actuator.set_duty_cycle(1.0)  # failsafe ON


class TDSController(ParameterController):
    def __init__(self, name: str, strategy_manager, actuator):
        super().__init__(name=name,
                         setpoint=100,
                         strategy_manager=strategy_manager,
                         actuator=actuator,
                         initial_kp=0.7,
                         initial_ki=0.001,
                         init_gain=50,
                         init_tau=1.0,
                         init_delay=0.5
                         )

        self.target_column = "MCP_WQ_TDS"

    def process(self, data):
        current_tds = data.get('mcp_wq', {}).get('tds')
        if current_tds is not None:
            pi_output = self.calculate_pi(current_tds)

            print(f"[{self.name}] TDS: {current_tds} \n")
            print(f"Setpoint: {self.setpoint} \n")
            print(f"Control Signal: {pi_output:.2f} \n")

            self.actuator.set_duty_cycle(pi_output)
        else:
            print(f"[{self.name} Controller] No TDS data found in payload.")
            self.actuator.set_duty_cycle(1.0)  # failsafe ON