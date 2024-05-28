import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from tabulate import tabulate

logging.getLogger(__name__).setLevel(logging.DEBUG)


class Benchmark(ABC):

    def __init__(self,
                 batch_sizes: List[int] = None,
                 input_lengths: List[int] = None,
                 seq_lengths: List[int] = None,
                 n_iter: int = 1,
                 input_cols: List[str] = None,
                 model_path: str = None,
                 memcpu: bool = False,
                 name: str = "benchmark"):
        self.batch_sizes = batch_sizes if batch_sizes else [4]
        self.input_lengths = input_lengths if input_lengths else [16]
        self.seq_lengths = seq_lengths if seq_lengths else [16]
        self.n_iter = n_iter
        self.input_cols = input_cols if input_cols else ["document"]
        self.model_path = model_path
        self.res_profile = memcpu
        self.name = name

        self.results = dict()
        self.bm_process = None
        super().__init__()

    @abstractmethod
    def _run_iter(self, batch_size: int, input_length: int,
                  output_length: int):
        pass

    @abstractmethod
    def measure_resource_usage(self, batch_size: int, input_length: int,
                               output_length: int):
        pass

    def run(self):
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
                    pipeline_durs = []
                    for _ in range(self.n_iter):
                        start = time.perf_counter_ns() * 1e-9
                        pipeline_dur = self._run_iter(batch_size, input_length,
                                                      seq_length)
                        end = time.perf_counter_ns() * 1e-9
                        dur_runs.append(end - start)
                        if pipeline_dur is not None:
                            pipeline_durs.append(pipeline_dur)

                    result = dict()
                    if self.res_profile:
                        result = self.measure_resource_usage(
                            batch_size, input_length, seq_length)

                    result['duration'] = sum(dur_runs) / len(dur_runs)
                    result['per_duration'] = dur_runs
                    self.results[batch_size][input_length][seq_length] = result
                    logging.info(
                        f"Took an average of {self.results[batch_size][input_length][seq_length]['duration']:.3f} seconds"
                    )

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
                        res["Avg CPU%"] = res.get("Avg CPU%", []) + [
                            sum(seq_res['cpu_percent']) /
                            len(seq_res['cpu_percent'])
                        ]
                        res["Peak Memory (MB)"] = res.get(
                            "Peak Memory (MB)", []) + [seq_res['peak_memory']]
                    res['Duration (s)'] = res.get("Duration (s)",
                                                  []) + [seq_res['duration']]

        return res

    def print_results(self):
        print(
            tabulate(self.make_results(),
                     headers="keys",
                     tablefmt="fancy_grid"))

    def save_results(self, file_name: str = None):
        if file_name is None:
            file_name = f'./{self.name}-{datetime.now().strftime("%m-%d-%Y")}.json'

        with open(file_name, 'w') as file:
            json.dump(self.results, file)
