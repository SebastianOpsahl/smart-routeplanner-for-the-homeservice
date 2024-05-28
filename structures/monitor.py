import psutil

class Monitor:
    '''Class to monitor components during tests'''
    def __init__(self):
        self.process = psutil.Process()
        self.cpu_start = None
        self.cpu_end = None
        self.ram_start = None
        self.ram_end = None

    def start(self):
        self.cpu_start = self.process.cpu_times()
        self.ram_start = psutil.virtual_memory().used

    def end(self):
        self.cpu_end = self.process.cpu_times()
        self.ram_end = psutil.virtual_memory().used

    def get_cpu_usage(self):
        # Calculate total CPU time used (user + system)
        user_time = self.cpu_end.user - self.cpu_start.user
        system_time = self.cpu_end.system - self.cpu_start.system
        return user_time + system_time

    def get_ram_usage(self):
        # Calculate change in RAM usage
        return self.ram_end - self.ram_start

    def get_result(self):
        cpu_usage = self.get_cpu_usage()
        ram_usage = self.get_ram_usage()
        # Formatting to display more decimal places for CPU usage
        return f"CPU Time Used: {cpu_usage:.5f} seconds\n" + \
               f"RAM Used: {ram_usage} bytes\n"

class Observed:
    '''Class to se observed values'''
    def __init__(self, name: str):
        self.name = name
        self.start_value = None
        self.end_value = None

    def set_start(self, x):
        self.start_value = x

    def set_end(self, x):
        self.end_value = x

    def value_used(self):
        if self.start_value is None or self.end_value is None:
            raise ValueError(f"{self.name} usage not recorded")
        return self.end_value - self.start_value

    def result(self):
        return f"{self.name} used: {self.value_used()}"
