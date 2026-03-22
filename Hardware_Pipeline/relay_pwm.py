import threading
import time
from Hardware_Pipeline.aqpx_actuation_orchestrator import AqpxActuationOrchestrator

class TimeProportionalRelay:
    def __init__(self, orchestrator: AqpxActuationOrchestrator, relay_num: str, window_secs: float = 60):
        self.orchestrator = orchestrator
        self.relay_num = relay_num
        self.window_secs = window_secs
        self.duty_cycle = 0.0
        self.running = False
        self.thread = None

        # Track state to prevent spamming the MQTT broker with identical commands
        self.current_state = None

        # How often to evaluate the relay state (in seconds)
        self.update_interval = 1.0

    def set_duty_cycle(self, duty_cycle: float):
        """Updates the setpoint duty cycle (0.0 to 1.0)."""
        # ensure that duty_cycle is not out of bounds
        self.duty_cycle = max(0.0, min(1.0, duty_cycle))

    def start(self):
        """Starts the background PWM thread."""
        self.running = True
        self.thread = threading.Thread(target=self._pwm_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Turn off relay, stop the background PWM thread."""
        self.running = False
        if self.thread:
            self.thread.join()

        # always set the pump / aerator ON as the safe state
        self._set_state('ON')

    def _set_state(self, state: str):
        """Helper method to send a command ONLY if the state has changed."""
        if self.current_state != state:
            self.orchestrator.send_relay_command(self.relay_num, state)
            self.current_state = state

    def _pwm_loop(self):
        # Anchor point for our time window calculations
        t_start = time.time()

        while self.running:
            duty = self.duty_cycle

            # reduce chatter: if the duty cycle is at the extremes, lock the state
            if duty <= 0.1:
                self._set_state('OFF')
            elif duty >= 0.9:
                self._set_state('ON')
            else:
                # Calculate where we currently are in the rotating time window
                elapsed_in_window = (time.time() - t_start) % self.window_secs
                on_time = duty * self.window_secs

                # If we are within the 'ON' portion of the window, turn on. Otherwise, off.
                if elapsed_in_window < on_time:
                    self._set_state('ON')
                else:
                    self._set_state('OFF')

            # Sleep for a short interval so we are highly responsive to duty cycle changes
            time.sleep(self.update_interval)