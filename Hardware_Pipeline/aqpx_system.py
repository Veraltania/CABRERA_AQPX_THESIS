import time
import sys
import datetime

from Hardware_Pipeline.controllers import *
from Hardware_Pipeline.schedule_policies import *
from Hardware_Pipeline.tuning_strategies import *
from Hardware_Pipeline.relay_pwm import *
from Hardware_Pipeline.aqpx_actuation_orchestrator import AqpxActuationOrchestrator
from Hardware_Pipeline.aqpx_logger import AqpxLogger


class AquaponicsSystem:
    def __init__(self, controllers, broker="localhost", port=1883):
        self.broker = broker
        self.port = port
        self.controllers = controllers

        # Initialize the logger and hook up the callback
        self.logger = AqpxLogger(
            broker=self.broker,
            port=self.port,
            on_data_received_callback=self.process_sensor_data
        )

    def process_sensor_data(self, data):
        print(f"\n[System] New sensor data received. Checking active schedules...")

        active_count = 0
        for controller in self.controllers:
            if controller.is_active():
                controller.process(data)
                active_count += 1

        if active_count == 0:
            print("[System] No controllers are scheduled to run at this time.")

    def run(self):
        print(f"Starting Aquaponics System (Broker: {self.broker}:{self.port})...")
        print(f"Registered Controllers: {[c.name for c in self.controllers]}")

        # Start the MQTT logger thread
        self.logger.start()
        print("[System] Listening for incoming MQTT sensor data...")

        # The actual work happens asynchronously in the logger's MQTT callback.
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down system...")
            sys.exit(0)


if __name__ == '__main__':
    orchestrator = AqpxActuationOrchestrator(broker="localhost", port=1883)
    orchestrator.connect()

    do_relay = TimeProportionalRelay(orchestrator=orchestrator, relay_num='1', window_secs=600)
    do_relay.start()

    tds_relay = TimeProportionalRelay(orchestrator=orchestrator, relay_num='2', window_secs=1800)
    tds_relay.start()

    # Define the daily schedules and their paired strategies
    do_schedules = [
        (DailyTimeSchedule(datetime.time(0, 0, 0), datetime.time(5, 59, 59)), AdaptiveTuningStrategy()),
        (DailyTimeSchedule(datetime.time(6, 0, 0), datetime.time(11, 59, 59)), StaticTuningStrategy()),
        (DailyTimeSchedule(datetime.time(12, 0, 0), datetime.time(17, 59, 59)), AdaptiveTuningStrategy()),
        (DailyTimeSchedule(datetime.time(18, 0, 0), datetime.time(23, 59, 59)), StaticTuningStrategy())
    ]

    tds_schedules = [
        (DailyTimeSchedule(datetime.time(0, 0, 0), datetime.time(5, 59, 59)), AdaptiveTuningStrategy()),
        (DailyTimeSchedule(datetime.time(6, 0, 0), datetime.time(11, 59, 59)), StaticTuningStrategy()),
        (DailyTimeSchedule(datetime.time(12, 0, 0), datetime.time(17, 59, 59)), AdaptiveTuningStrategy()),
        (DailyTimeSchedule(datetime.time(18, 0, 0), datetime.time(23, 59, 59)), StaticTuningStrategy())
    ]

    # Create the Strategy Managers
    do_strategy_manager = TimeBasedStrategyManager(do_schedules)
    tds_strategy_manager = TimeBasedStrategyManager(tds_schedules)

    # Instantiate exactly ONE controller per actuator
    do_controller = DOController(
        name="DO-Master-Controller",
        strategy_manager=do_strategy_manager,
        actuator=do_relay
    )

    tds_controller = TDSController(
        name="TDS-Master-Controller",
        strategy_manager=tds_strategy_manager,
        actuator=tds_relay
    )

    # Inject controllers into the system
    system = AquaponicsSystem(
        controllers=[do_controller, tds_controller],
        broker="localhost",
        port=1883
    )

    # Start everything
    system.run()