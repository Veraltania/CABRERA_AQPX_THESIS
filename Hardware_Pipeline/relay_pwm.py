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

    def set_duty_cycle(self, duty_cycle: float):
        """Updates the target duty cycle (0.0 to 1.0)."""

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
        self.orchestrator.send_relay_command(self.relay_num, 'ON')

    def _pwm_loop(self):
        while self.running:
            duty = self.duty_cycle

            # reduce chatter
            # if the duty cycle is extremely low, just turn the pump OFF
            if duty <= 0.1:
                self.orchestrator.send_relay_command(self.relay_num, 'OFF')
                time.sleep(self.window_secs)
                continue
            elif duty >= 0.9:
                self.orchestrator.send_relay_command(self.relay_num, 'ON')
                time.sleep(self.window_secs)
                continue

            on_time = duty * self.window_secs
            off_time = self.window_secs - on_time

            # turn relay ON
            self.orchestrator.send_relay_command(self.relay_num, 'ON')
            time.sleep(on_time)

            # turn relay OFF
            if self.running:
                self.orchestrator.send_relay_command(self.relay_num, 'OFF')
                time.sleep(off_time)


