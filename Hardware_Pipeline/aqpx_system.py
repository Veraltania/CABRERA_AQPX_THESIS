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
    # ---------------------------------------------------------
    # 1. Define DO Schedule (Daily Recurring Windows)
    # ---------------------------------------------------------
    do_morning_schedule = DailyTimeSchedule(datetime.time(8, 0, 0), datetime.time(10, 0, 0))
    do_midday_schedule = DailyTimeSchedule(datetime.time(10, 0, 1), datetime.time(12, 0, 0))

    do_morning = DOController(
        name="DO-Morning-Static", 
        strategy=StaticTuningStrategy(),
        schedule=do_morning_schedule
    )
    
    do_midday = DOController(
        name="DO-Midday-Adaptive", 
        strategy=AdaptiveTuningStrategy(),
        schedule=do_midday_schedule
    )

    # ---------------------------------------------------------
    # 2. Define TDS Schedule (Multi-Day Continuous Windows)
    # ---------------------------------------------------------
    # Example: Day 1 and 2 static, Day 3 and 4 adaptive
    now = datetime.datetime.now()
    
    # 0 to 48 hours from now
    tds_static_dates = MultiDaySchedule(
        start_datetime=now, 
        end_datetime=now + datetime.timedelta(days=2)
    )
    
    # 48 to 96 hours from now
    tds_adaptive_dates = MultiDaySchedule(
        start_datetime=now + datetime.timedelta(days=2, seconds=1), 
        end_datetime=now + datetime.timedelta(days=4)
    )

    tds_static = TDSController(
        name="TDS-Day1to2-Static", 
        strategy=StaticTuningStrategy(),
        schedule=tds_static_dates
    )
    
    tds_adaptive = TDSController(
        name="TDS-Day3to4-Adaptive", 
        strategy=AdaptiveTuningStrategy(),
        schedule=tds_adaptive_dates
    )
    
    # Group them into a list
    active_controllers = [do_morning, do_midday, tds_static, tds_adaptive]
    
    # Inject the list into the System
    system = AquaponicsSystem(
        controllers=active_controllers, 
        broker="localhost", 
        port=1883
    )
    
    # system.run()