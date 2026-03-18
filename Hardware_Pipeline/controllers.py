import os
import csv
from datetime import datetime  # Fixed import for datetime.now()
from abc import ABC, abstractmethod

from Hardware_Pipeline.tuning_strategies import TuningStrategy
from Hardware_Pipeline.schedule_policies import SchedulePolicy
from Hardware_Pipeline.relay_pwm import TimeProportionalRelay

class ParameterController(ABC):
    def __init__(self, name: str, target: float, strategy: TuningStrategy,
                 schedule: SchedulePolicy, actuator,
                 initial_kp, initial_ki, init_gain=1.0, init_tau=1.0, init_delay=0.0):
        self.name = name
        self.target = target
        self.strategy = strategy
        self.schedule = schedule
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

        # Load previous state upon initialization using the strategy
        self._load_previous_state()

    def _load_previous_state(self):
        """Delegates state loading to the injected strategy."""
        self.strategy.load_state(self, self.log_file)

    def _log_current_state(self, current_val, error, pi_output):
        """Logs current state, including watchdog variables."""
        file_exists = os.path.exists(self.log_file)

        try:
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
                    'itae_current_window': getattr(self.strategy, 'itae_current_window', 0.0),
                    'itae_previous_window': getattr(self.strategy, 'itae_current_window', 0.0),
                    'window_timer': getattr(self.strategy, 'window_timer', 0.0)
                })
        except Exception as e:
            print(f"[{self.name}] Failed to write to log file: {e}")

    def is_active(self) -> bool:
        return self.schedule.is_active()

    def calculate_pi(self, current_val):
        """Clean, simplified calculation using hardcoded 5-second intervals."""
        error = self.target - current_val

        # Delegate performance evaluation and watchdog checks to the strategy
        self.strategy.evaluate_performance(self, error, self.dt)

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
    def tune(self):
        pass

    @abstractmethod
    def process(self, data):
        pass


class DOController(ParameterController):
    def __init__(self, name: str, strategy: TuningStrategy,
                 schedule: SchedulePolicy, relay_pwm: TimeProportionalRelay, actuator):
        super().__init__(name=name,
                         target=6.0,
                         strategy=strategy,
                         schedule=schedule,
                         actuator=actuator,
                         initial_kp=0.7,
                         initial_ki=0.001,
                         init_gain=50,
                         init_tau=1.0,
                         init_delay=0.5
                         )

    def process(self, data):
        current_do = data.get('mcp_wq', {}).get('do')
        if current_do is not None:
            pi_output = self.calculate_pi(current_do)

            print(f"[{self.name}] DO: {current_do}mg/L \n")
            print(f"Target: {self.target} \n")
            print(f"Control Signal: {pi_output:.2f} \n")

            # update the background thread with the new duty cycle
            self.actuator.set_duty_cycle(pi_output)

        else:
            print(f"[{self.name} Controller] No DO data found in payload.")
            self.actuator.set_duty_cycle(1.0) # failsafe ON

    def tune(self):
        pass


class TDSController(ParameterController):
    def __init__(self, name: str, strategy: TuningStrategy, schedule: SchedulePolicy, actuator):
        super().__init__(name=name,
                         target=100,
                         strategy=strategy,
                         schedule=schedule,
                         actuator=actuator,
                         initial_kp=0.7,
                         initial_ki=0.001,
                         init_gain=50,
                         init_tau=1.0,
                         init_delay=0.5
                         )

    def process(self, data):
        current_tds = data.get('mcp_wq', {}).get('tds')
        if current_tds is not None:
            pi_output = self.calculate_pi(current_tds)

            print(f"[{self.name}] TDS: {current_tds} \n")
            print(f"Target: {self.target} \n")
            print(f"Control Signal: {pi_output:.2f} \n")

            # update the background thread with the new duty cycle
            self.actuator.set_duty_cycle(pi_output)
        else:
            print(f"[{self.name} Controller] No TDS data found in payload.")
            self.actuator.set_duty_cycle(1.0) # failsafe ON

    def tune(self):
        pass