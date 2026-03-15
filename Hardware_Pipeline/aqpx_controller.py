import paho.mqtt.client as mqtt

class AqpxController:
    """Handles sending commands to the aquaponics hardware."""
    
    VALID_RELAYS = ['1', '2', '3', '4']
    VALID_STATES = ['ON', 'OFF']

    # Notice how broker and port are now injected via the constructor
    def __init__(self, broker="localhost", port=1883, topic="aquaponics_commands"):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt.Client(client_id="rpi_controller")
        
    def connect(self):
        """Establishes a persistent connection for continuous PID control."""
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            print(f"[Controller] Connected to MQTT Broker at {self.broker}:{self.port}.")
        except Exception as e:
            print(f"[Controller] Connection error: {e}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def send_relay_command(self, relay_num, state):
        """Validates and publishes a relay command."""
        relay_num = str(relay_num)
        state = str(state).upper()

        if relay_num not in self.VALID_RELAYS:
            print(f"Error: Invalid relay '{relay_num}'. Must be one of {self.VALID_RELAYS}.")
            return
        if state not in self.VALID_STATES:
            print(f"Error: Invalid state '{state}'. Must be 'ON' or 'OFF'.")
            return
            
        command_payload = f"RELAY{relay_num}_{state}"
        
        result = self.client.publish(self.topic, command_payload)
        status = result.wait_for_publish()

        if status == mqtt.MQTT_ERR_SUCCESS:
            print(f"[Controller] Sent '{command_payload}' to '{self.topic}'")
        else:
            print(f"[Controller] Failed to send. Status: {status}")