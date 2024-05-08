from importlib import import_module
from sparknlp import SparkSession
from sparknlp.base import DocumentAssembler
import multiprocessing as mp
from pyspark.ml import Pipeline
from typing import List, Any
from pyspark.sql import DataFrame
import logging
from .benchmark import Benchmark


class PyBenchmark(Benchmark):
    def __init__(self, spark: SparkSession, annotator: Any, data: DataFrame, pretrained: str = None,
                 batch_sizes: List[int] = None, input_lengths: List[int] = None,
                 seq_lengths: List[int] = None, n_iter: int = 1, input_cols: List[str] = None,
                 model_path: str = None, memcpu: bool = True, name: str = "pybenchmark",
                 use_docassembler: bool = False):
        super().__init__(batch_sizes, input_lengths, seq_lengths,
                         n_iter, input_cols, model_path, memcpu=memcpu, name=name)
        self.spark = spark
        self.data = data
        self.pretrained = pretrained
        self.annotator = annotator
        self.use_docassembler = use_docassembler

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
                self.annotator = cls.pretrained()

        if self.use_docassembler:
            self.annotator.setInputCols("corpus")
        else:
            self.annotator.setInputCols(input_cols)
        self.annotator.setOutputCol("output")

    def run(self):
        for batch_size in self.batch_sizes:
            annotator = self.annotator.setBatchSize(batch_size)
            if batch_size not in self.results:
                self.results[batch_size] = {}
            for input_length in self.input_lengths:
                if hasattr(annotator, "setMaxSentenceLength"):
                    annotator = annotator.setMaxSentenceLength(
                        input_length)
                else:
                    logging.debug("Failed to set max input length.")
                    input_length = -1

                if input_length not in self.results[batch_size]:
                    self.results[batch_size][input_length] = {}
                for seq_length in self.seq_lengths:
                    if hasattr(annotator, "setMaxOutputLength"):
                        annotator = annotator.setMaxOutputLength(
                            seq_length)
                    else:
                        logging.debug("Failed to set max output length.")
                        seq_length = -1

                    logging.info(
                        f"Benchmarking {annotator.name}, batch size: {batch_size}, input length: {input_length}, sequence length: {seq_length}...")

                    if seq_length in self.results[batch_size][input_length]:
                        logging.info("Already benchmarked, skipping...")
                        continue

                    if self.use_docassembler:
                        document_assembler = DocumentAssembler().setInputCol(
                            self.input_cols[0]).setOutputCol("corpus")

                        pipeline = Pipeline().setStages(
                            [document_assembler, annotator])
                    else:
                        pipeline = Pipeline().setStages([annotator])
                    pipeline = pipeline.fit(self.data)
                    pipeline.transform(self.data).collect()

                    def benchmark_pipeline(pipeline):
                        pipeline.transform(self.data).collect()

                    result = {}
                    dur_runs = self.benchmark_time(
                        benchmark_pipeline, pipeline)

                    if self.res_profile:
                        proc = mp.Process(
                            target=benchmark_pipeline, args=(pipeline,))
                        result = self.res_profile(proc)

                    result['duration'] = sum(dur_runs) / len(dur_runs)
                    result['per_duration'] = dur_runs
                    self.results[batch_size][input_length][seq_length] = result
                    logging.info(
                        f"Took an average of {self.results[batch_size][input_length][seq_length]['duration']:.3f} seconds")
