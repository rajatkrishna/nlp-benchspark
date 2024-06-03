import csv
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from tabulate import tabulate
import benchmark

logging.getLogger(__name__).setLevel(logging.DEBUG)


class BaseBenchmark(ABC):

    def __init__(self,
                 batch_sizes: List[int] = None,
                 input_lengths: List[int] = None,
                 seq_lengths: List[int] = None,
                 n_iter: int = 1,
                 input_cols: List[str] = None,
                 model_path: str = None,
                 memcpu: bool = False,
                 name: str = "spark-benchmark"):
        self.batch_sizes = batch_sizes if batch_sizes else [1]
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
    def run_iter(self, batch_size: int, input_length: int, output_length: int):
        pass

    def measure_resource_usage(self, batch_size: int, input_length: int,
                               output_length: int):
        res_monitor = benchmark.SystemMonitor()
        result = dict()

        res_monitor.start()
        try:
            self.run_iter(batch_size, input_length, output_length)
        except Exception as e:
            print("Resource measurement failed...", e)
            return result
        finally:
            res_monitor.stop()
            res_monitor.join()

        result['cpu_percent'] = res_monitor.cpu_percent
        result['mem_usage'] = res_monitor.mem_kb
        result['peak_memory'] = max(result['mem_usage']) / 1024
        return result

    def run(self):
        failed = 0
        num_runs = 0
        for batch_size in self.batch_sizes:
            if batch_size not in self.results:
                self.results[batch_size] = {}
            for input_length in self.input_lengths:
                if input_length not in self.results[batch_size]:
                    self.results[batch_size][input_length] = {}
                for seq_length in self.seq_lengths:
                    if seq_length in self.results[batch_size][input_length]:
                        logging.info("Already benchmarked, skipping...")
                        continue

                    dur_runs = []
                    for _ in range(self.n_iter):
                        start = time.perf_counter_ns() * 1e-9
                        success = True
                        try:
                            self.run_iter(batch_size, input_length, seq_length)
                        except Exception:
                            success = False
                        end = time.perf_counter_ns() * 1e-9
                        if success:
                            dur_runs.append(end - start)
                        else:
                            failed += 1

                    result = dict()
                    if self.res_profile:
                        result = self.measure_resource_usage(
                            batch_size, input_length, seq_length)

                    if len(dur_runs) > 0:
                        result['duration'] = sum(dur_runs) / len(dur_runs)
                        result['per_duration'] = dur_runs
                    self.results[batch_size][input_length][seq_length] = result
                    num_runs += 1
        print(
            f"Summary: {num_runs} benchmarks ran out of which {failed} failed.."
        )

    def print_results(self):
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
                        res["Avg CPU%"] = res.get("Avg CPU%", []) + [
                            sum(seq_res['cpu_percent']) /
                            len(seq_res['cpu_percent'])
                        ]
                        res["Peak Memory (MB)"] = res.get(
                            "Peak Memory (MB)", []) + [seq_res['peak_memory']]
                    res['Duration (s)'] = res.get("Duration (s)",
                                                  []) + [seq_res['duration']]
        print(tabulate(res, headers="keys", tablefmt="fancy_grid"))

    def save_results(self, file_name: str = None):
        if file_name is None:
            file_name = f'./{self.name}-{datetime.now().strftime("%m-%d-%Y")}'
        with open(f"{file_name}.csv", 'w') as file:
            csvwriter = csv.writer(file)

            headers = [
                'Batch', 'Input Length', 'Output Length', 'Duration (s)'
            ]
            csvwriter.writerow(headers)
            for batch, batch_res in self.results.items():
                for inp, inp_res in batch_res.items():
                    for seq, seq_res in inp_res.items():
                        row = [batch, inp, seq, seq_res['duration']]
                        csvwriter.writerow(row)

        if self.res_profile:
            with open(f"{file_name}-resource-usage.csv", 'w') as file:
                csvwriter = csv.writer(file)
                headers = [
                    'Batch', 'Input Length', 'Output Length', 'CPU Usage (%)',
                    'Memory Usage (MB)'
                ]
                csvwriter.writerow(headers)
                for batch, batch_res in self.results.items():
                    for inp, inp_res in batch_res.items():
                        for seq, seq_res in inp_res.items():
                            for cpu, mem in zip(seq_res['cpu_percent'],
                                                seq_res['mem_usage']):
                                csvwriter.writerow(
                                    [batch, inp, seq, cpu, mem / 1024])
