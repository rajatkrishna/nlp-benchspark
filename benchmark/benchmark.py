import json
import time
import psutil
import multiprocessing as mp
from abc import abstractmethod, ABC
from tabulate import tabulate
from datetime import datetime
from typing import List


class Benchmark(ABC):
    def __init__(self, batch_sizes: List[int] = None, input_lengths: List[int] = None,
                 seq_lengths: List[int] = None, n_iter: int = 1, input_cols: List[str] = None,
                 model_path: str = None, memcpu: bool = False, name: str = "benchmark"):
        self.batch_sizes = batch_sizes if batch_sizes else [4]
        self.input_lengths = input_lengths if input_lengths else [16]
        self.seq_lengths = seq_lengths if seq_lengths else [16]
        self.n_iter = n_iter
        self.input_cols = input_cols if input_cols else ["document"]
        self.model_path = model_path
        self.res_profile = memcpu
        self.name = name

        self.results = dict()
        super().__init__()

    @abstractmethod
    def run(self):
        pass

    def benchmark_time(self, func, args):
        result = {}
        dur_runs = []
        for _ in range(self.n_iter):
            start = time.perf_counter_ns() * 1e-9
            func(args)
            end = time.perf_counter_ns() * 1e-9
            dur_runs.append(end - start)

        return dur_runs

    def profile_res(self, proc: mp.Process) -> dict:
        cpu_percent = []
        mem_usage = []
        mem_percent = []

        proc.start()
        while proc.is_alive():
            cpu_percent.append(psutil.cpu_percent(percpu=False))
            mem_usage.append(psutil.virtual_memory().used / 1024 / 1024)
            mem_percent.append(psutil.virtual_memory().percent)
            time.sleep(0.1)

        proc.join()
        if (proc.exitcode > 0):
            raise ValueError(f"Exited with exit code: {proc.exitcode}")

        result = dict()
        result['cpu_percent'] = cpu_percent
        result['mem_usage'] = mem_usage
        result['mem_percent'] = mem_percent
        result['peak_memory'] = max(mem_usage)
        return result

    def make_results(self):
        res = dict()
        for batch, batch_res in self.results.items():
            for inp, inp_res in batch_res.items():
                for seq, seq_res in inp_res.items():
                    res["Batch"] = res.get("Batch", []) + [batch]
                    res["Input length"] = res.get("Input length", []) + [inp]
                    res["Seq length"] = res.get("Seq length", []) + [seq]
                    if self.res_profile:
                        res["Peak CPU%"] = res.get(
                            "Peak CPU%", []) + [max(seq_res['cpu_percent'])]
                        res["Avg CPU%"] = res.get(
                            "Avg CPU%", []) + [sum(seq_res['cpu_percent']) / len(seq_res['cpu_percent'])]
                        # for i, cpu in enumerate(zip(*seq_res['cpu_percent'])):
                        #     res[f"Avg CPU % (CPU {i})"] = res.get(
                        #         f"Avg CPU % (CPU {i})", []) + [sum(cpu) / len(cpu)]
                        res["Peak Memory (MB)"] = res.get(
                            "Peak Memory (MB)", []) + [seq_res['peak_memory']]
                    res['Duration (s)'] = res.get(
                        "Duration (s)", []) + [seq_res['duration']]

        return res

    def print_results(self):
        print(tabulate(self.make_results(), headers="keys", tablefmt="fancy_grid"))

    def save_results(self, file_name: str = None):
        if file_name is None:
            file_name = f'./{self.name}-{datetime.now().strftime("%m-%d-%Y")}.json'

        with open(file_name, 'w') as file:
            json.dump(self.results, file)
