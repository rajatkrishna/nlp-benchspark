import argparse
import json

from benchmark import SparkBenchmark


def parse_config(args) -> dict:
    assert args.jar_path is not None, "Missing benchmark app jar..."
    config = dict()
    config['jar_path'] = args.jar_path

    config['n_iter'] = args.n_iter
    config['input_cols'] = [s.strip() for s in args.input_cols.split(",")]
    config['memcpu'] = args.memcpu

    config['batch_sizes'] = [
        int(s.strip()) for s in args.batch_sizes.split(",")
    ] if args.batch_sizes else [4]
    config['input_lengths'] = [
        int(s.strip()) for s in args.input_lengths.split(",")
    ] if args.input_lengths else [32]
    config['seq_lengths'] = [
        int(s.strip()) for s in args.output_lengths.split(",")
    ] if args.output_lengths else [32]

    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jar_path',
                        type=str,
                        help='Path to the compiled jar file.')
    parser.add_argument('sparknlp_jar',
                        type=str,
                        help='Path to the spark nlp jar file.')
    parser.add_argument(
        '--model_path',
        type=str,
        help=
        'Path to the model to import for benchmarking custom pre-trained models.'
    )
    parser.add_argument('--conll',
                        type=str,
                        help='Path to the CONLL formatted data file.')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        help='Path to benchmark config.')
    parser.add_argument('--input_cols',
                        type=str,
                        help='Input columns to use for benchmarking.',
                        default='document')
    parser.add_argument(
        '--batch_sizes',
        type=str,
        help=
        'Batch sizes to benchmark (pass multiple values as a comma-separated list).'
    )
    parser.add_argument(
        '--input_lengths',
        type=str,
        help=
        'Input lengths to benchmark (pass multiple values as a comma-separated list).'
    )
    parser.add_argument(
        '--output_lengths',
        type=str,
        help=
        'Output sequence lengths to benchmark (pass multiple values as a comma-separated list).'
    )
    parser.add_argument('--memcpu',
                        type=bool,
                        help='Measure memory and cpu usage.',
                        default=False)
    parser.add_argument("--n_iter",
                        type=int,
                        help="Number of iterations of each case.",
                        default=1)

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            confs = json.load(f)

        for benchmark_conf in confs['configs']:
            input_cols = benchmark_conf[
                'input_cols'] if 'input_cols' in benchmark_conf else [
                    "document"
                ]
            data_path = benchmark_conf['data_path']
            batch_sizes = benchmark_conf[
                'batch_sizes'] if 'batch_sizes' in benchmark_conf else [
                    4, 8, 16
                ]
            seq_lengths = benchmark_conf[
                'seq_lengths'] if 'seq_lengths' in benchmark_conf else [
                    32, 64, 128, 256
                ]
            jar_path = benchmark_conf['jar_path']
            model_path = benchmark_conf['model_path']
            n_iter = benchmark_conf['n_iter']
            name = benchmark_conf['name']

            try:
                bm = SparkBenchmark(model_path=model_path,
                                    data_path=data_path,
                                    jar_path=jar_path,
                                    batch_sizes=batch_sizes,
                                    seq_lengths=seq_lengths,
                                    n_iter=n_iter,
                                    input_cols=input_cols,
                                    name=name)
                bm.run()
                bm.save_results()
                bm.print_results()
            except Exception as e:
                print(e)
    else:
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

        bm = SparkBenchmark(model_path=model_path,
                            sparknlp_jar=args.sparknlp_jar,
                            **benchmark_conf)
        bm.run()
        bm.print_results()
