import argparse
import json
from sparknlp import SparkSession
from sparknlp.training import CoNLL
from benchmark.pybenchmark import PyBenchmark


def parse_config(args) -> dict:
    assert (args.annotator)
    config = dict()
    if args.model_path:
        config['model_path'] = args.model_path
    if args.model_name:
        config['pretrained'] = args.model_name

    config['annotator'] = args.annotator
    config['n_iter'] = args.n_iter
    config['input_cols'] = [s.strip() for s in args.input_cols.split(",")]
    config['memcpu'] = args.memcpu

    config['batch_sizes'] = [int(s.strip()) for s in args.batch_sizes.split(
        ",")] if args.batch_sizes else [4]
    config['input_lengths'] = [int(s.strip()) for s in args.input_lengths.split(
        ",")] if args.input_lengths else [32]
    config['seq_lengths'] = [int(s.strip()) for s in args.output_lengths.split(
        ",")] if args.output_lengths else [32]

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotator', type=str, nargs="?",
                        help='Fully qualified name of the annotator.')
    parser.add_argument('--conll', type=str, nargs="?",
                        help='Path to the CONLL formatted data file.')
    parser.add_argument('-c', '--config', type=str,
                        help='Path to benchmark config.')
    parser.add_argument('-p', '--prompt', type=str,
                        help="Prompt as input dataframe.")
    parser.add_argument('--input_cols', type=str,
                        help='The input columns for the annotator.', default='document')
    parser.add_argument('--model_path', type=str,
                        help='Path to the saved model.')
    parser.add_argument("--model_name", type=str,
                        help='Pre-trained model name.')
    parser.add_argument('--batch_sizes', type=str,
                        help='Batch sizes to benchmark (pass multiple values as a comma-separated list).')
    parser.add_argument('--input_lengths', type=str,
                        help='Input lengths to benchmark (pass multiple values as comma-separated list).')
    parser.add_argument('--output_lengths', type=str,
                        help='Output sequence lengths to benchmark (pass multiple values as comma-separated list).')
    parser.add_argument('--sparknlp_jar', type=str,
                        help='Path to the spark nlp jar file.')
    parser.add_argument('--memcpu', type=bool,
                        help='Measure memory and cpu usage.', default=False)
    parser.add_argument(
        "--n_iter",
        type=int,
        help="Number of iterations of each case.",
        default=1
    )

    args = parser.parse_args()
    sparknlp_jar_path = args.sparknlp_jar
    session_builder = SparkSession.builder \
        .appName("BenchmarkApp") \
        .master("local[*]")

    if sparknlp_jar_path:
        session_builder = session_builder.config("spark.jars", sparknlp_jar_path)
    spark_session = session_builder.getOrCreate()

    if args.config:
        with open(args.config, 'r') as f:
            confs = json.load(f)

        for benchmark_conf in confs['configs']:
            assert ('model_path' in benchmark_conf or 'pretrained' in benchmark_conf)
            assert ('annotator' in benchmark_conf)

            data_path = benchmark_conf['conll']
            data = CoNLL(explodeSentences=False).readDataset(
                spark_session, data_path)
            try:
                bm = PyBenchmark(spark=spark_session,
                                 data=data, **benchmark_conf)
                bm.run()
                bm.save_results()
                bm.print_results()
            except Exception as e:
                print(e)
    else:
        benchmark_conf = parse_config(args)
        if args.prompt:
            data_str = args.prompt
            data = spark_session.createDataFrame(
                [["Hey there"]]).toDF("document")
            bm = PyBenchmark(spark=spark_session, data=data,
                             use_docassembler=True, **benchmark_conf)
        elif args.conll:
            data_path = args.conll
            data = CoNLL(explodeSentences=False).readDataset(
                spark_session, data_path)
            bm = PyBenchmark(spark=spark_session, data=data, **benchmark_conf)
        else:
            raise ValueError("No data provided...")

        bm.run()
        bm.print_results()
        bm.save_results()
