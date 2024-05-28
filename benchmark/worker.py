import threading
import time
import psutil


class ResourceMonitorWorker(threading.Thread):

    def __init__(self, refresh_sec: int = 1):
        threading.Thread.__init__(self)
        self.cpu_percent = []
        self.mem_kb = []
        self.refresh_sec = refresh_sec

    def collect(self):
        pass

    def run(self):
        self.alive = True

        while self.alive:
            self.collect()
            time.sleep(self.refresh_sec)

    def stop(self):
        self.alive = False


class SystemMonitor(ResourceMonitorWorker):

    def __init__(self, refresh_sec: int = 1):
        super().__init__(refresh_sec)

    def collect(self):
        self.cpu_percent.append(psutil.cpu_percent(interval=self.refresh_sec))
        mem_stats = psutil.virtual_memory()
        self.mem_kb.append((mem_stats.total - mem_stats.available) / 1024)


class ProcMonitor(ResourceMonitorWorker):

    def __init__(self, pid: int, refresh_sec: int = 1):
        super().__init__(refresh_sec)
        self.proc = psutil.Process(pid=pid)

    def collect(self):
        try:
            self.cpu_percent.append(
                self.proc.cpu_percent(interval=self.refresh_sec) /
                psutil.cpu_count())
            self.mem_kb.append(self.proc.memory_full_info().uss / 1024)
        except psutil.NoSuchProcess:
            # Process dead... Nothing to collect.
            return None
