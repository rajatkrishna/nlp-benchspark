import json
from abc import abstractmethod, ABC
from tabulate import tabulate
from datetime import datetime
from typing import List


class Benchmark(ABC):
    def __init__(self, batch_sizes: List[int] = [4],
                 seq_lengths: List[int] = [16], n_iter: int = 1, input_cols: List[str] = None,
                 model_path: str = None, memcpu: bool = False, name: str = "benchmark"):
        self.batch_sizes = batch_sizes
        self.seq_lengths = seq_lengths
        self.n_iter = n_iter
        self.input_cols = input_cols
        self.model_path = model_path
        self.res_profile = memcpu
        self.name = name

        self.results = dict()
        super().__init__()

    @abstractmethod
    def run(self):
        pass

    def make_results(self):
        res = dict()
        for batch, batch_res in self.results.items():
            for seq, seq_res in batch_res.items():
                res["Batch"] = res.get("Batch", []) + [batch]
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
