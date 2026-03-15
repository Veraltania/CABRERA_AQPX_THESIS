import paho.mqtt.client as mqtt
import csv
import os
import json
from datetime import datetime

class AqpxLogger:
    """Handles receiving data, logging to CSV, and triggering the PID loop."""
    
    CSV_HEADER = [
        "Date", "Time", "StatusCode",
        "MCP_WQ_DO", "MCP_WQ_EC", "MCP_WQ_PH", "MCP_WQ_TEMP", "MCP_WQ_TURB", "MCP_WQ_TDS", 
        "MCP_WQ_RSV1", "MCP_WQ_RSV2", "ADS_NO3_mA", "ADS_NO2_mA", "ADS_NH3_mA", "ADS_RSV_mA"
    ]
    MASTER_CSV_FILE_PATH = "aquaponics_master_log.csv"

    # Broker and port are injected here as well
    def __init__(self, broker="localhost", port=1883, 
                 summary_topic="aquaponics/summary_json", 
                 status_topic="aquaponics/status",
                 on_data_received_callback=None):
        
        self.broker = broker
        self.port = port
        self.summary_topic = summary_topic
        self.status_topic = status_topic
        
        self.on_data_received_callback = on_data_received_callback 
        
        self.client = mqtt.Client(client_id="rpi_logger_aqpx-neo")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[Logger] Connected successfully to MQTT Broker at {self.broker}:{self.port}!")
            self.client.subscribe(self.summary_topic)
            self.client.subscribe(self.status_topic)
        else:
            print(f"[Logger] Failed to connect, return code {rc}")

    def _write_to_csv(self, file_path, header, data_row):
        try:
            file_exists = os.path.isfile(file_path)
            with open(file_path, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(data_row)
        except IOError as e:
            print(f"Error writing to {file_path}: {e}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            
            if msg.topic == self.summary_topic:
                self._process_summary_json(payload)
            elif msg.topic == self.status_topic:
                print(f"--- STATUS --- [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]: {payload}")
                
        except Exception as e:
            print(f"Error in on_message callback: {e}")

    def _process_summary_json(self, payload):
        try:
            data = json.loads(payload)

            print(f"[Logger] Received Summary Data: {json.dumps(data, indent=2)}")

            if self.on_data_received_callback:
                self.on_data_received_callback(data)

            timestamp_parts = data['timestamp'].split(' ')
            row = [
                timestamp_parts[0], timestamp_parts[1], 
                data['system_status']['code'],
                data['mcp_wq']['do'], data['mcp_wq']['ec'], data['mcp_wq']['ph'],
                data['mcp_wq']['temp'], data['mcp_wq']['turbidity'], data['mcp_wq']['tds'],
                data['mcp_wq']['rsv1'], data['mcp_wq']['rsv2'],
                data['ads_wqnit']['nitrate'], data['ads_wqnit']['nitrite'],
                data['ads_wqnit']['ammonia'], data['ads_wqnit']['rsv']
            ]

            today_str = datetime.now().strftime('%Y-%m-%d')
            daily_csv_path = f"AQPX_data_log_{today_str}.csv"
            
            self._write_to_csv(daily_csv_path, self.CSV_HEADER, row)
            self._write_to_csv(self.MASTER_CSV_FILE_PATH, self.CSV_HEADER, row)

        except json.JSONDecodeError:
            print("[Logger] Warning: Malformed JSON. Skipping.")
        except KeyError as e:
            print(f"[Logger] Warning: Missing JSON key: {e}. Skipping.")

    def start(self):
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start() 
        except Exception as e:
            print(f"[Logger] Connection error: {e}")

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
