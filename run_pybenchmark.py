import argparse
import json
from sparknlp import SparkSession
from sparknlp.training import CoNLL
from benchmark.pybenchmark import PyBenchmark


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotator', type=str, nargs="?",
                        help='Fully qualified name of the annotator.')
    parser.add_argument('data_path', type=str, nargs="?",
                        help='Path to the data file.')
    parser.add_argument('-c', '--config', type=str,
                        help='Path to benchmark config.')
    parser.add_argument('--input_cols', type=str,
                        help='The input columns for the annotator.', default='document')
    parser.add_argument('--model_path', type=str,
                        help='Path to the saved model.')
    parser.add_argument("--model_name", type=str,
                        help='Pre-trained model name.')
    parser.add_argument('--batch_sizes', type=str,
                        help='Batch sizes to benchmark (pass multiple values as a comma-separated list).')
    parser.add_argument('--seq_lengths', type=str,
                        help='Sequence lengths to benchmark (pass multiple values as comma-separated list).')
    parser.add_argument('--sparknlp_jar', type=str,
                        help='Path to the spark nlp jar file.')
    parser.add_argument('--is_seq2seq', type=bool,
                        help='Seq-2-seq model.', default=False)
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
        .appName("BenchmarkOvApp") \
        .master("local[*]")

    if sparknlp_jar_path:
        session_builder = session_builder.config(
            "spark.jars", sparknlp_jar_path)
    spark_session = session_builder.getOrCreate()

    if args.config:
        with open(args.config, 'r') as f:
            confs = json.load(f)

        for benchmark_conf in confs['configs']:
            model_path = None
            model_name = None
            if "model_path" in benchmark_conf:
                model_path = benchmark_conf['model_path']

            if "model_name" in benchmark_conf:
                model_name = benchmark_conf['model_name']

            assert (model_name or model_path)
            n_iter = benchmark_conf['n_iter'] if 'n_iter' in benchmark_conf else 1
            input_cols = benchmark_conf['input_cols'] if 'input_cols' in benchmark_conf else "document"
            memcpu = benchmark_conf['memcpu'] if 'memcpu' in benchmark_conf else False
            is_seq2seq = benchmark_conf['seq2seq'] if 'seq2seq' in benchmark_conf else False

            annotator_str = benchmark_conf['annotator']
            data_path = benchmark_conf['data_path']
            batch_sizes = benchmark_conf['batch_sizes'] if 'batch_sizes' in benchmark_conf else [
                4, 8, 16]
            seq_lengths = benchmark_conf['seq_lengths'] if 'seq_lengths' in benchmark_conf else [
                32, 64, 128, 256]
            name = benchmark_conf['name']

            data = CoNLL(explodeSentences=False).readDataset(
                spark_session, data_path)
            try:
                bm = PyBenchmark(model_path=model_path, memcpu=memcpu,
                                 spark=spark_session, annotator=annotator_str,
                                 data=data, batch_sizes=batch_sizes,
                                 seq_lengths=seq_lengths, seq2seq=is_seq2seq,
                                 n_iter=n_iter, input_cols=input_cols, pretrained=model_name,
                                 name=name)
                bm.run()

                bm.save_results(f"{annotator_str}-{name}.json")
                bm.print_results()
            except Exception as e:
                print(e)
    else:
        assert (args.model_path or args.model_name)
        assert (args.annotator)

        model_path = args.model_path
        model_name = args.model_name
        n_iter = args.n_iter
        input_cols = [s.strip() for s in args.input_cols.split(",")]
        memcpu = args.memcpu
        is_seq2seq = args.is_seq2seq

        annotator_str = args.annotator
        data_path = args.data_path
        batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(
            ",")] if args.batch_sizes else [4, 8, 16]
        seq_lengths = [int(s.strip()) for s in args.seq_lengths.split(
            ",")] if args.seq_lengths else [32, 64, 128, 256]

        data = CoNLL(explodeSentences=False).readDataset(
            spark_session, data_path)

        bm = PyBenchmark(model_path=model_path, memcpu=memcpu,
                         spark=spark_session, annotator=annotator_str,
                         data=data, batch_sizes=batch_sizes,
                         seq2seq=is_seq2seq, seq_lengths=seq_lengths,
                         n_iter=n_iter, input_cols=input_cols, pretrained=model_name)
        bm.run()
        bm.print_results()
