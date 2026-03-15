import time
import sys
from Hardware_Pipeline.aqpx_controller import AqpxController
from Hardware_Pipeline.aqpx_logger import AqpxLogger

class AquaponicsSystem:
    """The Middleman: Currently configured in Log-Only mode for testing."""
    
class AquaponicsSystem:
    """The Middleman: Currently configured in Log-Only mode for testing."""
    
    def __init__(self, broker="localhost", port=1883):
        self.broker = broker
        self.port = port
        
        # 1. ADD A VARIABLE TO HOLD THE LATEST STATE
        self.latest_sensor_data = {} 
        
        self.logger = AqpxLogger(
            broker=self.broker, 
            port=self.port, 
            on_data_received_callback=self.process_sensor_data
        )

    def process_sensor_data(self, data):
        """
        Triggered automatically every time the Logger receives new JSON.
        """
        self.latest_sensor_data = data 
        
        print(f"\n[System] New sensor data safely routed to the Middleman!")
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