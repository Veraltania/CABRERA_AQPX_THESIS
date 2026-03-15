import paho.mqtt.client as mqtt
import csv
import os
import json
from datetime import datetime

# --- Configuration ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
# MODIFIED: The script now listens to the new, structured JSON topic for data logging.
SUMMARY_JSON_TOPIC = "aquaponics/summary_json"
STATUS_TOPIC = "aquaponics/status"

# A constant path for the master file that collates all data
MASTER_CSV_FILE_PATH = "aquaponics_master_log.csv"

# This header defines the column order for the output CSV files.
CSV_HEADER = [
    "Date", "Time", "StatusCode",
    # "Relay1", "Relay2", "Relay3", "Relay4", # Relays are disabled in firmware
    "MCP_WQ_DO", "MCP_WQ_EC", "MCP_WQ_PH", "MCP_WQ_TEMP", "MCP_WQ_TURB", "MCP_WQ_TDS", "MCP_WQ_RSV1", "MCP_WQ_RSV2",
    "ADS_NO3_mA", "ADS_NO2_mA", "ADS_NH3_mA", "ADS_RSV_mA"
]

def write_to_csv(file_path, header, data_row):
    """
    A helper function to write a row of data to a specified CSV file.
    It creates the file and writes the header if it doesn't exist.
    """
    try:
        # Check if file exists to determine if we need to write a header
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write header only if the file is new
            if not file_exists:
                writer.writerow(header)
            writer.writerow(data_row)
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")

def on_connect(client, userdata, flags, rc):
    """Callback function for when the client connects to the broker."""
    if rc == 0:
        print("Connected successfully to MQTT Broker!")
        # MODIFIED: Subscribe to the new JSON summary topic
        client.subscribe(SUMMARY_JSON_TOPIC)
        client.subscribe(STATUS_TOPIC)
        print(f"Subscribed to topic: {SUMMARY_JSON_TOPIC}")
        print(f"Subscribed to topic: {STATUS_TOPIC}")
    else:
        print(f"Failed to connect, return code {rc}\n")

def on_message(client, userdata, msg):
    """Callback function for when a message is received."""
    try:
        payload = msg.payload.decode('utf-8')
        
        # <<< MODIFICATION START: Rewritten logic to handle JSON payload >>>
        if msg.topic == SUMMARY_JSON_TOPIC:
            try:
                data = json.loads(payload)

                # 1. Flatten the JSON data into a single list in the correct order for the CSV
                timestamp_parts = data['timestamp'].split(' ')
                esp32_date = timestamp_parts[0]
                esp32_time = timestamp_parts[1]

                # Extract all values in the precise order of the CSV_HEADER
                # This order must be maintained carefully.
                row = [
                    esp32_date,
                    esp32_time,
                    data['system_status']['code'],
                    # data['relay_status']['relay1'], # Relays are disabled in firmware
                    # data['relay_status']['relay2'],
                    # data['relay_status']['relay3'],
                    # data['relay_status']['relay4'],
                    data['mcp_wq']['do'],
                    data['mcp_wq']['ec'],
                    data['mcp_wq']['ph'],
                    data['mcp_wq']['temp'],
                    data['mcp_wq']['turbidity'],
                    data['mcp_wq']['tds'],
                    data['mcp_wq']['rsv1'],
                    data['mcp_wq']['rsv2'],
                    data['ads_wqnit']['nitrate'],
                    data['ads_wqnit']['nitrite'],
                    data['ads_wqnit']['ammonia'],
                    data['ads_wqnit']['rsv'],
                    # data['ads_pwr']['pump1_rms'],
                    # data['ads_pwr']['pump2_rms'],
                    # data['ads_pwr']['aerator_rms'],
                    # data['ads_pwr']['rsv']
                ]

                # 2. Generate the filename for the current day's log
                today_str = datetime.now().strftime('%Y-%m-%d')
                daily_csv_path = f"AQPX_data_log_{today_str}.csv"
                
                # 3. Write the data to BOTH the daily file and the master file
                write_to_csv(daily_csv_path, CSV_HEADER, row)
                write_to_csv(MASTER_CSV_FILE_PATH, CSV_HEADER, row)
                
                print(f"Logged data to {daily_csv_path} and {MASTER_CSV_FILE_PATH}")

            except json.JSONDecodeError:
                print(f"Warning: Received malformed JSON on {SUMMARY_JSON_TOPIC}. Skipping message.")
            except KeyError as e:
                print(f"Warning: JSON message on {SUMMARY_JSON_TOPIC} is missing an expected key: {e}. Skipping message.")
            except Exception as e:
                print(f"An unexpected error occurred while processing summary: {e}")
        # <<< MODIFICATION END >>>

        elif msg.topic == STATUS_TOPIC:
            print(f"--- STATUS UPDATE --- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: {payload}")
    except Exception as e:
        print(f"Error in on_message callback: {e}")


def main():
    """Main function to setup and run the MQTT client."""
    client = mqtt.Client(client_id="rpi_logger_main")
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("Logger stopped by user.")
        client.disconnect()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

