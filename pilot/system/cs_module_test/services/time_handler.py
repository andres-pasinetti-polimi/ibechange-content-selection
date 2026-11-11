import copy
import datetime
from cs_module.config import USE_REAL_TIME


class TimeHandler:
    def __init__(self, current_time=None):
        self.current_time = current_time
        self.start_time = datetime.datetime.now if USE_REAL_TIME else None

    @property
    def now(self):
        if USE_REAL_TIME:
            return datetime.datetime.now()
        return self.current_time

    def set(self, new_time):
        if not USE_REAL_TIME:
            self.current_time = new_time

    def set_start_time(self, start_time):
        self.start_time = start_time
        if not USE_REAL_TIME:
            self.current_time = copy.deepcopy(start_time)

    def string_to_iso(self, string_time):
        return datetime.datetime.fromisoformat(string_time)
