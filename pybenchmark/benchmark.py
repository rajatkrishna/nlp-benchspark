import json
from tabulate import tabulate
from importlib import import_module
from datetime import datetime
from sparknlp import SparkSession
from sparknlp.base import DocumentAssembler
import time
import psutil
import multiprocessing as mp

from pyspark.ml import Pipeline
from typing import List, Any
from pyspark.sql import DataFrame
from pyspark.sql.functions import substring, col

import logging


class Benchmark:
    def __init__(self, spark: SparkSession,
                 annotator: Any, data: DataFrame, batch_sizes: List[int] = None,
                 seq_lengths: List[int] = None, n_iter: int = 1,
                 input_cols: List[str] = None, seq2seq: bool = False,
                 model_path: str = None, pretrained: str = None,
                 res_profile: bool = True):
        self.spark = spark
        self.annotator = annotator
        self.data = data
        self.batch_sizes = batch_sizes
        self.seq_lengths = seq_lengths
        self.n_iter = n_iter
        self.input_cols = input_cols
        self.seq2seq = seq2seq

        if isinstance(self.annotator, str):
            module_path, class_name = self.annotator.rsplit(".", 1)
            logging.debug(f"Importing {class_name} from {module_path}...")

            module = import_module(module_path)
            cls = getattr(module, class_name)
            if model_path:
                self.annotator = cls.loadSavedModel(model_path, self.spark)
            elif pretrained:
                self.annotator = cls.pretrained(pretrained)
            else:
                raise ValueError(
                    "Either local model path or pretrained model name must be provided.")

        self.results = dict()
        if self.seq2seq:
            self.annotator.setInputCols(["corpus"])
        else:
            self.annotator.setInputCols(self.input_cols)
        self.annotator.setOutputCol("output")
        self.res_profile = res_profile

    def measure(self, pipeline, func):
        result = {}

        dur_runs = []
        for _ in range(self.n_iter):
            start = time.perf_counter_ns() * 1e-9
            func(pipeline)
            end = time.perf_counter_ns() * 1e-9
            dur_runs.append(end - start)

        if self.res_profile:
            proc = mp.Process(target=func, args=(pipeline,))
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

            result['cpu_percent'] = cpu_percent
            result['mem_usage'] = mem_usage
            result['mem_percent'] = mem_percent
            result['peak_memory'] = max(mem_usage)

        result['duration'] = sum(dur_runs) / len(dur_runs)
        result['per_duration'] = dur_runs
        return result

    def run(self):
        for batch_size in self.batch_sizes:
            self.results[batch_size] = {}

            for seq_length in self.seq_lengths:
                logging.info(
                    f"Benchmarking {self.annotator.name}, batch size: {batch_size}, sequence length: {seq_length}...")

                self.annotator = self.annotator \
                    .setBatchSize(batch_size)

                input_data = None
                pipeline = None
                if self.seq2seq:
                    input_data = self.data.withColumn(
                        self.input_cols[0], substring(col(self.input_cols[0]), 0, seq_length))
                    document_assembler = DocumentAssembler().setInputCol(
                        self.input_cols[0]).setOutputCol("corpus")

                    pipeline = Pipeline().setStages(
                        [document_assembler, self.annotator])
                elif hasattr(self.annotator, "setMaxSentenceLength"):
                    input_data = self.data
                    self.annotator = self.annotator.setMaxSentenceLength(
                        seq_length)
                    pipeline = Pipeline().setStages([self.annotator])
                else:
                    raise Exception("Failed to set seq length.")

                pipeline = pipeline.fit(input_data).transform(input_data)

                def benchmark_pipeline(pipeline):
                    pipeline.write.mode("overwrite").parquet(
                        "./tmp_bm")

                self.results[batch_size][seq_length] = self.measure(
                    pipeline, benchmark_pipeline)

                logging.info(
                    f"Took an average of {self.results[batch_size][seq_length]['duration']:.3f} seconds")

    def make_results(self):
        res = dict()
        for k, v in self.results.items():
            if k == "name":
                continue
            batch = k
            batch_res = v
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
            file_name = f'./results-{datetime.now().strftime("%m-%d-%Y")}.json'

        with open(file_name, 'w') as file:
            json.dump(self.results, file)
