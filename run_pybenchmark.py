import argparse

from sparknlp import SparkSession
from sparknlp.training import CoNLL

import benchmark
import logging
from importlib import import_module
from typing import Any, List

from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from sparknlp.base import DocumentAssembler

logging.getLogger(__name__).setLevel(logging.DEBUG)


class PyBenchmark(benchmark.BaseBenchmark):

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

    def run_iter(self, batch_size: int, input_length: int, output_length: int):
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
            print("Failed to set max input length...")

        if hasattr(annotator, "setMaxOutputLength"):
            annotator = annotator.setMaxOutputLength(output_length)
            if hasattr(annotator, "setMinOutputLength"):
                annotator = annotator.setMinOutputLength(output_length)
        else:
            print("Failed to set max output length...")

        if self.use_docassembler:
            document_assembler = DocumentAssembler().setInputCol(
                self.input_cols[0]).setOutputCol("corpus")

            annotator.setInputCols("corpus")
            pipeline = Pipeline().setStages([document_assembler, annotator])
        else:
            annotator.setInputCols(self.input_cols)
            pipeline = Pipeline().setStages([annotator])

        annotator.setOutputCol("output")
        pipeline = pipeline.fit(self.data)
        pipeline.transform(self.data).collect()


def parse_config(args) -> dict:
    assert args.annotator is not None, "Missing annotator..."
    config = dict()
    if args.model_path:
        config['model_path'] = args.model_path
    if args.pretrained:
        config['pretrained'] = args.pretrained

    config['annotator'] = args.annotator
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
    parser.add_argument('annotator',
                        type=str,
                        help='Fully qualified name of the Spark transformer.')
    parser.add_argument('-p',
                        '--prompt',
                        type=str,
                        help="Optional prompt to pass as input dataframe.")
    parser.add_argument(
        '--conll',
        type=str,
        help=
        'Path to the CONLL formatted data file. Either prompt or data file must be provided.'
    )
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
    parser.add_argument('--input_cols',
                        type=str,
                        help='Input columns to use for benchmarking.',
                        default='document')
    parser.add_argument(
        '--model_path',
        type=str,
        help=
        'Path to the model to import for benchmarking with pre-trained models.'
    )
    parser.add_argument("--pretrained",
                        type=str,
                        help='Pre-trained model name to download.')
    parser.add_argument('--sparknlp_jar',
                        type=str,
                        help='Path to the spark nlp jar file.')
    parser.add_argument('--resource_usage',
                        type=bool,
                        help='Measure memory and cpu usage.',
                        default=False)
    parser.add_argument("--n_iter",
                        type=int,
                        help="Number of iterations of each case.",
                        default=1)

    args = parser.parse_args()

    sparknlp_jar_path = args.sparknlp_jar
    session_builder = SparkSession.builder \
        .appName("BenchmarkApp") \
        .master("local[*]")

    if sparknlp_jar_path:
        session_builder = session_builder.config("spark.jars",
                                                 sparknlp_jar_path)
    else:
        session_builder = session_builder.config(
            "spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.3.3")
    spark_session = session_builder.getOrCreate()

    benchmark_conf = parse_config(args)
    if args.prompt:
        data_str = args.prompt
        data = spark_session.createDataFrame([[args.prompt]]).toDF("document")
        bm = PyBenchmark(spark=spark_session,
                         data=data,
                         use_docassembler=True,
                         **benchmark_conf)
    elif args.conll:
        data_path = args.conll
        data = CoNLL(explodeSentences=False).readDataset(
            spark_session, data_path)
        bm = PyBenchmark(spark=spark_session, data=data, **benchmark_conf)
    else:
        raise ValueError("No benchmark data provided...")

    bm.run()
    bm.print_results()
    bm.save_results()
