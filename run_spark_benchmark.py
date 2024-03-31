import argparse
import json
from benchmark.spark_benchmark import SparkBenchmark

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, nargs="?",
                        help='Path to the model file.')
    parser.add_argument('data_path', type=str, nargs="?",
                        help='Path to the data file.')
    parser.add_argument('-c', '--config', type=str,
                        help='Path to benchmark config.')
    parser.add_argument('--input_cols', type=str,
                        help='The input columns for the annotator.', default='document')
    parser.add_argument('--model_path', type=str,
                        help='Path to the saved model.')
    parser.add_argument('--batch_sizes', type=str,
                        help='Batch sizes to benchmark (pass multiple values as a comma-separated list).')
    parser.add_argument('--seq_lengths', type=str,
                        help='Sequence lengths to benchmark (pass multiple values as comma-separated list).')
    parser.add_argument('--jar_path', type=str,
                        help='Path to the jar file.')
    parser.add_argument(
        "--n_iter",
        type=int,
        help="Number of iterations of each case.",
        default=1
    )

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            confs = json.load(f)

        for benchmark_conf in confs['configs']:
            input_cols = benchmark_conf['input_cols'] if 'input_cols' in benchmark_conf else ["document"]
            data_path = benchmark_conf['data_path']
            batch_sizes = benchmark_conf['batch_sizes'] if 'batch_sizes' in benchmark_conf else [
                4, 8, 16]
            seq_lengths = benchmark_conf['seq_lengths'] if 'seq_lengths' in benchmark_conf else [
                32, 64, 128, 256]
            jar_path = benchmark_conf['jar_path']
            model_path = benchmark_conf['model_path']
            n_iter = benchmark_conf['n_iter']
            name = benchmark_conf['name']

            try:
                bm = SparkBenchmark(model_path=model_path, data_path=data_path,
                                    jar_path=jar_path, batch_sizes=batch_sizes, seq_lengths=seq_lengths,
                                    n_iter=n_iter, input_cols=input_cols, name=name)
                bm.run()
                bm.save_results()
                bm.print_results()
            except Exception as e:
                print(e)
    else:
        assert (args.model_path and args.data_path)
        assert (args.jar_path)

        model_path = args.model_path
        data_path = args.data_path
        jar_path = args.jar_path

        n_iter = args.n_iter
        input_cols = [s.strip() for s in args.input_cols.split(",")]
        batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(
            ",")] if args.batch_sizes else [4, 8, 16]
        seq_lengths = [int(s.strip()) for s in args.seq_lengths.split(
            ",")] if args.seq_lengths else [32, 64, 128, 256]

        bm = SparkBenchmark(model_path=model_path, data_path=data_path,
                            jar_path=jar_path, batch_sizes=batch_sizes, seq_lengths=seq_lengths,
                            n_iter=n_iter, input_cols=input_cols)
        bm.run()
        bm.print_results()
