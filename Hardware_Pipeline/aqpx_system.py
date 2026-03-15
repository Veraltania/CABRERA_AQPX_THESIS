import paho.mqtt.client as mqtt
import time
import sys
import csv
import os
import json
from datetime import datetime

# NOTE: Ensure the previously defined AqpxLogger class is included in this file 
# or imported properly before running this snippet.

class AquaponicsSystem:
    """The Middleman: Currently configured in Log-Only mode for testing."""
    
    def __init__(self, broker="localhost", port=1883):
        self.broker = broker
        self.port = port
        
        # We only instantiate the logger for this testing phase
        self.logger = AqpxLogger(
            broker=self.broker, 
            port=self.port, 
            on_data_received_callback=self.process_sensor_data
        )

    def process_sensor_data(self, data):
        """
        Triggered automatically every time the Logger receives new JSON.
        PID logic is removed for testing. We just print to verify data flow.
        """
        print(f"\n[System] New sensor data safely routed to the Middleman!")
        
        # Using .get() to safely grab a few values to prove it's parsing correctly
        current_ph = data.get('mcp_wq', {}).get('ph', 'N/A')
        current_temp = data.get('mcp_wq', {}).get('temp', 'N/A')
        
        print(f"[System - Test Output] pH: {current_ph} | Temp: {current_temp}")
        

    def run(self):
        """Starts the logging system."""
        print(f"Starting Aquaponics System in LOG-ONLY Mode (Broker: {self.broker}:{self.port})...")
        self.logger.start()
        
        try:
            # Keep the main thread alive while the logger runs in the background
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down system...")
            self.logger.stop()
            sys.exit(0)

if __name__ == '__main__':
    system = AquaponicsSystem()
    system.run()