import paho.mqtt.client as mqtt
import time
import sys
import os
import csv
import datetime
from abc import ABC, abstractmethod

class SchedulePolicy(ABC):
    @abstractmethod
    def is_active(self) -> bool:
        """Determines if the controller should be active right now."""
        pass

class DailyTimeSchedule(SchedulePolicy):
    """Evaluates activity based on a daily recurring time window (e.g., 8 AM to 10 AM every day)."""
    def __init__(self, start_time: datetime.time, end_time: datetime.time):
        self.start_time = start_time
        self.end_time = end_time

    def is_active(self) -> bool:
        now = datetime.datetime.now().time()
        
        # Handles standard daytime schedules (e.g., 08:00 to 12:00)
        if self.start_time <= self.end_time:
            return self.start_time <= now <= self.end_time
        # Handles overnight schedules (e.g., 22:00 to 06:00)
        else: 
            return now >= self.start_time or now <= self.end_time

class MultiDaySchedule(SchedulePolicy):
    """Evaluates activity based on specific continuous dates/times (e.g., Jan 1st to Jan 3rd)."""
    def __init__(self, start_datetime: datetime.datetime, end_datetime: datetime.datetime):
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

    def is_active(self) -> bool:
        now = datetime.datetime.now()
        return self.start_datetime <= now <= self.end_datetime
