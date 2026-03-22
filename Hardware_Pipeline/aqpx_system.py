from Hardware_Pipeline.controllers import *
from Hardware_Pipeline.schedule_policies import *
from Hardware_Pipeline.tuning_strategies import *
from Hardware_Pipeline.relay_pwm import *

class AquaponicsSystem:
    def __init__(self, controllers, broker="localhost", port=1883):
        self.broker = broker
        self.port = port
        self.controllers = controllers 

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
        
        # Simulating incoming data
        try:
            while True:
                time.sleep(1)
                mock_data = {'mcp_wq': {'do': 5.8, 'tds': 95}}
                self.process_sensor_data(mock_data)
        except KeyboardInterrupt:
            print("\nShutting down system...")
            sys.exit(0)

if __name__ == '__main__':
    orchestrator = AqpxActuationOrchestrator(broker="localhost", port=1883)
    orchestrator.connect()

    do_relay = TimeProportionalRelay(actuator=orchestrator, relay_num='1', window_secs=600)
    do_relay.start()

    tds_relay = TimeProportionalRelay(actuator=orchestrator, relay_num='2', window_secs=1800)
    tds_relay.start()

    # 1. Define the daily schedules and their paired strategies
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

    # 2. Create the Strategy Managers
    do_strategy_manager = TimeBasedStrategyManager(do_schedules)
    tds_strategy_manager = TimeBasedStrategyManager(tds_schedules)

    # 3. Instantiate exactly ONE controller per actuator
    do_controller = DOController(
        name="DO-Master-Controller",
        setpoint=6.0,
        strategy_manager=do_strategy_manager,
        actuator=do_relay
    )

    tds_controller = TDSController(
        name="TDS-Master-Controller",
        setpoint=220,
        strategy_manager=tds_strategy_manager,
        actuator=tds_relay
    )

    # Inject the streamlined list into the system
    system = AquaponicsSystem(
        controllers=[do_controller, tds_controller],
        broker="localhost",
        port=1883
    )

    system.run()