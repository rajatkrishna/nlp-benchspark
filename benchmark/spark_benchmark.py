import logging
import subprocess
from typing import List
from benchmark.worker import ProcMonitor

from benchmark import Benchmark

logging.getLogger(__name__).setLevel(logging.DEBUG)


class SparkBenchmark(Benchmark):

    def __init__(self,
                 model_path: str,
                 data_path: str,
                 jar_path: str,
                 sparknlp_jar: str = None,
                 batch_sizes: List[int] = None,
                 input_lengths: List[int] = None,
                 seq_lengths: List[int] = None,
                 n_iter: int = 1,
                 input_cols: List[str] = None,
                 memcpu: bool = False,
                 name: str = "spark_benchmark"):
        super().__init__(batch_sizes=batch_sizes,
                         input_lengths=input_lengths,
                         seq_lengths=seq_lengths,
                         n_iter=n_iter,
                         input_cols=input_cols,
                         model_path=model_path,
                         memcpu=memcpu,
                         name=name)
        self.data_path = data_path
        self.jar_path = jar_path
        self.sparknlp_jar = sparknlp_jar

    def _run_iter(self, batch_size: int, input_length: int,
                  output_length: int):
        cmd = [
            "spark-submit", "--executor-memory", "15G", "--driver-memory",
            "12G", "--jars", self.sparknlp_jar, "--class",
            "BertEmbeddingsBenchmark", self.jar_path, self.model_path,
            self.data_path,
            str(batch_size),
            str(output_length), ",".join(self.input_cols)
        ]

        subprocess.run(cmd)

    def measure_resource_usage(self, batch_size, input_length, output_length):
        cmd = [
            "spark-submit", "--executor-memory", "15G", "--driver-memory",
            "12G", "--jars", self.sparknlp_jar, "--class",
            "BertEmbeddingsBenchmark", self.jar_path, self.model_path,
            self.data_path,
            str(batch_size),
            str(output_length), ",".join(self.input_cols)
        ]
        proc = subprocess.Popen(cmd)
        res_monitor = ProcMonitor(proc.pid)
        res_monitor.start()
        proc.wait()
        res_monitor.stop()
        res_monitor.join()

        result = dict()
        result['cpu_percent'] = res_monitor.cpu_percent
        result['mem_usage'] = res_monitor.mem_kb
        result['peak_memory'] = max(result['mem_usage'])
        return result
