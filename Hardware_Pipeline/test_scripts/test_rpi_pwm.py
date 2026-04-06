import time
from Hardware_Pipeline.aqpx_actuation_orchestrator import AqpxActuationOrchestrator
from Hardware_Pipeline.relay_pwm import TimeProportionalRelay

def run_pwm_test():
    print("Starting RPi Time-Proportional Relay Test...")

    # 1. Initialize Orchestrator
    # Assumes Mosquitto is running locally on default port 1883
    orchestrator = AqpxActuationOrchestrator(broker="localhost", port=1883)
    orchestrator.connect()

    # 2. Initialize the PWM Relay controller
    # Using relay '1' for the aerator.
    # Overriding the window to 10 seconds for faster testing feedback.
    aerator = TimeProportionalRelay(orchestrator, relay_num='1', window_secs=10.0)
    aerator.start()

    try:
        # Test Case 1: 50% Duty Cycle
        print("\n--- Testing 50% Duty Cycle (5s ON, 5s OFF) ---")
        aerator.set_duty_cycle(0.5)
        time.sleep(25)  # Run for 2.5 windows to observe cycles

        # Test Case 2: 20% Duty Cycle
        print("\n--- Testing 20% Duty Cycle (2s ON, 8s OFF) ---")
        aerator.set_duty_cycle(0.2)
        time.sleep(25)

        # Test Case 3: Extreme Low (Locked OFF)
        print("\n--- Testing 0% Duty Cycle (Locked OFF) ---")
        aerator.set_duty_cycle(0.0)
        time.sleep(15)

        # Test Case 4: Extreme High (Locked ON)
        print("\n--- Testing 100% Duty Cycle (Locked ON) ---")
        aerator.set_duty_cycle(1.0)
        time.sleep(15)

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        print("\nCleaning up...")
        # Note: The stop() method intentionally defaults the aerator to ON for safety
        aerator.stop()
        orchestrator.disconnect()
        print("Done.")


if __name__ == "__main__":
    run_pwm_test()