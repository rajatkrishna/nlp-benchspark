import argparse

import benchmark

import logging
import subprocess
from typing import List

logging.getLogger(__name__).setLevel(logging.DEBUG)


class SparkBenchmark(benchmark.BaseBenchmark):

    def __init__(self,
                 model_path: str,
                 data_path: str,
                 jar_path: str,
                 sparknlp_jar: str = None,
                 classname: str = None,
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
        self.classname = classname

    def run_iter(self, batch_size: int, input_length: int, output_length: int):
        cmd = [
            "spark-submit", "--executor-memory", "15G", "--driver-memory",
            "12G", "--jars", self.sparknlp_jar, "--class", self.classname,
            self.jar_path, self.model_path, self.data_path,
            str(batch_size),
            str(output_length), ",".join(self.input_cols)
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            return False
        return True

    def measure_resource_usage(self, batch_size, input_length, output_length):
        cmd = [
            "spark-submit", "--executor-memory", "15G", "--driver-memory",
            "12G", "--jars", self.sparknlp_jar, "--class", self.classname,
            self.jar_path, self.model_path, self.data_path,
            str(batch_size),
            str(output_length), ",".join(self.input_cols)
        ]
        proc = subprocess.Popen(cmd)
        res_monitor = benchmark.ProcMonitor(proc.pid)
        res_monitor.start()
        proc.wait()
        res_monitor.stop()
        res_monitor.join()

        result = dict()
        result['cpu_percent'] = res_monitor.cpu_percent
        result['mem_usage'] = res_monitor.mem_kb
        result['peak_memory'] = max(result['mem_usage']) / 1024
        return result


def parse_config(args) -> dict:
    assert args.jar_path is not None, "Missing benchmark app jar..."
    config = dict()
    config['jar_path'] = args.jar_path

    config['n_iter'] = args.n_iter
    config['input_cols'] = [s.strip() for s in args.input_cols.split(",")]
    config['memcpu'] = args.resource_usage

    config['batch_sizes'] = [
        int(s.strip()) for s in args.batch_sizes.split(",")
    ] if args.batch_sizes else [1]
    config['input_lengths'] = [
        int(s.strip()) for s in args.input_lengths.split(",")
    ] if args.input_lengths else [16]
    config['seq_lengths'] = [
        int(s.strip()) for s in args.output_lengths.split(",")
    ] if args.output_lengths else [16]

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jar_path',
                        type=str,
                        help='Path to the compiled jar file.')
    parser.add_argument('classname',
                        type=str,
                        help='Benchmark script main class name.')
    parser.add_argument('sparknlp_jar',
                        type=str,
                        help='Path to the spark nlp jar file.')
    parser.add_argument(
        '--batch_sizes',
        type=str,
        help=
        'Batch sizes to benchmark (pass multiple values as a comma-separated list). Default [4].'
    )
    parser.add_argument(
        '--input_lengths',
        type=str,
        help=
        'Input lengths to benchmark (pass multiple values as a comma-separated list). Default [16].'
    )
    parser.add_argument(
        '--output_lengths',
        type=str,
        help=
        'Output sequence lengths to benchmark (pass multiple values as a comma-separated list). Default [16].'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help=
        'Path to the model to import for benchmarking custom pre-trained models.'
    )
    parser.add_argument('--conll',
                        type=str,
                        help='Path to the CONLL formatted data file.')
    parser.add_argument('--input_cols',
                        type=str,
                        help='Input columns to use for benchmarking.',
                        default='document')
    parser.add_argument('--resource_usage',
                        type=bool,
                        help='Measure memory and cpu usage.',
                        default=False)
    parser.add_argument("--n_iter",
                        type=int,
                        help="Number of iterations of each case.",
                        default=1)

    args = parser.parse_args()

    assert args.sparknlp_jar is not None, "Missing Spark NLP jar..."
    benchmark_conf = parse_config(args)
    if "model_path" in args:
        model_path = args.model_path
    else:
        raise ValueError("Missing model path...")

    if "conll" in args:
        benchmark_conf["data_path"] = args.conll
    else:
        raise ValueError("Missing data path...")

    if "classname" in args:
        benchmark_conf["classname"] = args.classname
    else:
        raise ValueError("Missing classname...")

    bm = SparkBenchmark(model_path=model_path,
                        sparknlp_jar=args.sparknlp_jar,
                        **benchmark_conf)
    bm.run()
    bm.print_results()
    bm.save_results()
