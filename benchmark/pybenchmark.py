import logging
from benchmark.worker import SystemMonitor
import time
from importlib import import_module
from typing import Any, List

from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from sparknlp import SparkSession
from sparknlp.base import DocumentAssembler

from benchmark import Benchmark

logging.getLogger(__name__).setLevel(logging.DEBUG)


class PyBenchmark(Benchmark):

    def __init__(self,
                 spark: SparkSession,
                 annotator: Any,
                 data: DataFrame,
                 pretrained: str = None,
                 batch_sizes: List[int] = None,
                 input_lengths: List[int] = None,
                 seq_lengths: List[int] = None,
                 n_iter: int = 1,
                 input_cols: List[str] = None,
                 model_path: str = None,
                 memcpu: bool = True,
                 name: str = "pybenchmark",
                 use_docassembler: bool = False):
        super().__init__(batch_sizes,
                         input_lengths,
                         seq_lengths,
                         n_iter,
                         input_cols,
                         model_path,
                         memcpu=memcpu,
                         name=name)
        self.spark = spark
        self.data = data
        self.pretrained = pretrained
        self.annotator = annotator
        self.use_docassembler = use_docassembler
        self.model_path = model_path
        self.pretrained = pretrained

    def measure_resource_usage(self, batch_size: int, input_length: int,
                               output_length: int):
        res_monitor = SystemMonitor()

        res_monitor.start()
        self._run_iter(batch_size, input_length, output_length)
        res_monitor.stop()
        res_monitor.join()

        result = dict()
        result['cpu_percent'] = res_monitor.cpu_percent
        result['mem_usage'] = res_monitor.mem_kb
        result['peak_memory'] = max(result['mem_usage'])
        return result

    def _run_iter(self, batch_size: int, input_length: int,
                  output_length: int):
        if isinstance(self.annotator, str):
            module_path, class_name = self.annotator.rsplit(".", 1)
            logging.debug(f"Importing {class_name} from {module_path}...")

            module = import_module(module_path)
            cls = getattr(module, class_name)
            if self.model_path is not None:
                annotator = cls.loadSavedModel(self.model_path, self.spark)
            elif self.pretrained is not None:
                annotator = cls.pretrained(self.pretrained)
            else:
                annotator = cls.pretrained()

        annotator = annotator.setBatchSize(batch_size)

        if hasattr(annotator, "setMaxSentenceLength"):
            annotator = annotator.setMaxSentenceLength(input_length)
        elif hasattr(annotator, "setMaxInputLength"):
            annotator = annotator.setMaxInputLength(input_length)
        else:
            logging.debug("Failed to set max input length.")
            input_length = -1

        if hasattr(annotator, "setMaxOutputLength"):
            annotator = annotator.setMaxOutputLength(output_length)
        else:
            logging.debug("Failed to set max output length.")
            output_length = -1

        if self.use_docassembler:
            document_assembler = DocumentAssembler().setInputCol(
                self.input_cols[0]).setOutputCol("corpus")

            annotator.setInputCols("corpus")
            pipeline = Pipeline().setStages([document_assembler, annotator])
        else:
            annotator.setInputCols(self.input_cols)
            pipeline = Pipeline().setStages([annotator])

        annotator.setOutputCol("output")
        start = time.perf_counter_ns()
        pipeline = pipeline.fit(self.data)
        pipeline.transform(self.data).collect()
        end = time.perf_counter_ns()

        return end - start
