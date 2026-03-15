import paho.mqtt.client as mqtt
import time
import sys
import os
import csv
import datetime
from abc import ABC, abstractmethod

from Hardware_Pipeline.aqpx_controller import AqpxController
from Hardware_Pipeline.aqpx_logger import AqpxLogger
from Hardware_Pipeline.controllers import *
from Hardware_Pipeline.schedule_policies import *
from Hardware_Pipeline.tuning_strategies import *

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

    do_controller_configs = [
        ("DO-Day-Adaptive", AdaptiveTuningStrategy(),   datetime.time(0, 0, 0),  datetime.time(5, 59, 59)),
        ("DO-Day-Fixed", StaticTuningStrategy(), datetime.time(6, 0, 0),  datetime.time(11, 59, 59)),
        ("DO-Evening-Adaptive",   AdaptiveTuningStrategy(),   datetime.time(12, 0, 0), datetime.time(17, 59, 59)),
        ("DO-Evening-Fixed",  StaticTuningStrategy(),   datetime.time(18, 0, 0), datetime.time(23, 59, 59)),
    ]

    tds_controller_configs = [
        ("TDS-Day-Adaptive", AdaptiveTuningStrategy(),   datetime.time(0, 0, 0),  datetime.time(5, 59, 59)),
        ("TDS-Day-Fixed", StaticTuningStrategy(), datetime.time(6, 0, 0),  datetime.time(11, 59, 59)),
        ("TDS-Evening-Adaptive",   AdaptiveTuningStrategy(),   datetime.time(12, 0, 0), datetime.time(17, 59, 59)),
        ("TDS-Evening-Fixed",  StaticTuningStrategy(),   datetime.time(18, 0, 0), datetime.time(23, 59, 59)),
    ]

    # Generate the controllers dynamically
    active_controllers = []
    for name, strategy, start, end in do_controller_configs:
        controller = DOController(
            name=name,
            strategy=strategy,
            schedule=DailyTimeSchedule(start, end)
        )
        active_controllers.append(controller)

    for name, strategy, start, end in tds_controller_configs:
        controller = TDSController(
            name=name,
            strategy=strategy,
            schedule=DailyTimeSchedule(start, end)
        )
        active_controllers.append(controller)
        
    # Inject the list into the system
    system = AquaponicsSystem(
        controllers=active_controllers, 
        broker="localhost", 
        port=1883
    )
        
    system.run()