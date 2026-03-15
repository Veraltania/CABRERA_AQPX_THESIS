import paho.mqtt.client as mqtt
import sys
import time

# --- Configuration ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
COMMAND_TOPIC = "aquaponics_commands"

VALID_RELAYS = ['1', '2', '3', '4']
VALID_STATES = ['ON', 'OFF']

def publish_command(command):
    """Connects, publishes a single message, and disconnects."""
    client = mqtt.Client(client_id="rpi_controller")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        
        # We use a short loop to ensure the connection is established before publishing
        client.loop_start()
        time.sleep(0.5) 

        result = client.publish(COMMAND_TOPIC, command)
        
        # Wait for the message to be sent
        status = result.wait_for_publish()

        if status == mqtt.MQTT_ERR_SUCCESS:
            print(f"Successfully sent command '{command}' to topic '{COMMAND_TOPIC}'")
        else:
            print(f"Failed to send command. Status code: {status}")
            
        client.loop_stop()
        client.disconnect()

    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    """Parses command-line arguments and sends the MQTT command."""
    if len(sys.argv) != 3:
        print("Usage: python aqpx_controller.py <relay_number> <state>")
        print("Example: python aqpx_controller.py 1 ON")
        sys.exit(1)

    relay_num = sys.argv[1]
    state = sys.argv[2].upper()

    if relay_num not in VALID_RELAYS:
        print(f"Error: Invalid relay number '{relay_num}'. Must be one of {VALID_RELAYS}.")
        sys.exit(1)

    if state not in VALID_STATES:
        print(f"Error: Invalid state '{state}'. Must be 'ON' or 'OFF'.")
        sys.exit(1)
        
    # Construct the message payload, e.g., "RELAY1_ON"
    command_payload = f"RELAY{relay_num}_{state}"
    
    publish_command(command_payload)

if __name__ == '__main__':
    main()
