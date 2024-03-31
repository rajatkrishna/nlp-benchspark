import time
import subprocess
import logging
from typing import List
from .benchmark import Benchmark


class SparkBenchmark(Benchmark):
    def __init__(self, model_path: str, data_path: str, jar_path: str,
                 batch_sizes: List[int], seq_lengths: List[int],
                 n_iter: int, input_cols: List[str] = None,
                 name: str = "spark_benchmark"):
        super().__init__(batch_sizes, seq_lengths, n_iter, input_cols, model_path, name=name)
        self.data_path = data_path
        self.jar_path = jar_path

    def run(self):
        failed = 0
        for batch_size in self.batch_sizes:
            self.results[batch_size] = {}
            for seq_length in self.seq_lengths:
                result = {}
                dur_runs = []
                cmd = ["spark-submit", "--class",
                       "BertEmbeddingsBenchmark", self.jar_path, self.model_path, self.data_path, str(batch_size), str(seq_length), ",".join(self.input_cols)]
                for _ in range(self.n_iter):
                    start = time.perf_counter_ns() * 1e-9
                    sp_result = subprocess.run(cmd)
                    end = time.perf_counter_ns() * 1e-9
                    if sp_result.returncode != 0:
                        logging.error(
                            f"Benchmarking batch size = {batch_size}, seq length = {seq_length} failed...\n{result}")
                        failed += 1
                        dur_runs.append(0)
                    else:
                        dur_runs.append(end - start)
                result['duration'] = sum(dur_runs) / len(dur_runs)
                result['per_duration'] = dur_runs
                self.results[batch_size][seq_length] = result
