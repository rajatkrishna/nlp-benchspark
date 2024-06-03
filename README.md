# Spark NLP Benchmarks

Lightweight, extensible benchmarking tool for [Spark NLP](https://github.com/JohnSnowLabs/spark-nlp) transformers using `psutil`. The motivation behind this was to execute quick, reproducible benchmarks for LLM inference across different inference frameworks utilized in Spark NLP- Tensorflow, ONNX and OpenVINO.

Benchmarking can be done across batch sizes and input/output sequence lengths (when configurable), using either a custom dataset in the CoNLL format, or a prompt passed with the `--prompt` flag, which is converted into a Spark dataframe. The `run_pybenchmark` script can be used to run benchmarks with PySpark, while the `run_spark_benchmark` script can benchmark jobs submitted with `spark-submit`.

## Instructions

Create a new Python virtual environment and install the dependencies in [requirements.txt](./requirements.txt).

```
python3 -m venv create .env
source .env/bin/activate

pip install -r requirements.txt
```

### PySpark

### Requirements

- JDK 8 or 11
- Python 3.11

Ensure that Spark NLP and PySpark are installed and configured. For a fresh install via PyPI, use the following command with the appropriate versions:

```
pip install spark-nlp==5.3.3 pyspark==3.5.1
```

Use the `run_pybenchmark.py` script to run a quick benchmark with a Spark NLP transformer. You can find a full list of supported Spark NLP transformers [here](https://sparknlp.org/docs/en/annotators#available-transformers).

```
$ python3 run_pybenchmark.py sparknlp.annotator.LLAMA2Transformer 
    -p "Hello World!"
```

You can import a custom pre-trained model into the corresponding Spark transformer using the `--model_path` argument. The raw benchmark results are exported to csv format after benchmarking.

The following is a list of all supported args.

```
usage: run_pybenchmark.py [options] annotator

positional arguments:
  annotator             Fully qualified name of the Spark transformer.

options:
  -h, --help            show this help message and exit
  -p PROMPT, --prompt PROMPT
                        Optional prompt to pass as input dataframe.
  --conll CONLL         Path to the CONLL formatted data file. Either prompt or data file must be provided.
  --batch_sizes BATCH_SIZES
                        Batch sizes to benchmark (pass multiple values as a comma-separated list). Default [4].
  --input_lengths INPUT_LENGTHS
                        Input lengths to benchmark (pass multiple values as a comma-separated list). Default [16].
  --output_lengths OUTPUT_LENGTHS
                        Output sequence lengths to benchmark (pass multiple values as a comma-separated list). Default [16].
  --input_cols INPUT_COLS
                        Input columns to use for benchmarking.
  --model_path MODEL_PATH
                        Path to the model to import for benchmarking with pre-trained models.
  --pretrained PRETRAINED
                        Pre-trained model name to download.
  --sparknlp_jar SPARKNLP_JAR
                        Path to the spark nlp jar file.
  --resource_usage RESOURCE_USAGE
                        Measure memory and cpu usage.
  --n_iter N_ITER       Number of iterations of each case.
```

To benchmark custom pipelines, extend the `benchmark.BaseBenchmark` class and provide the implementation for a single iteration in the `run_iter` method.

```
import benchmark

class CustomBenchmark(benchmark.BaseBenchmark):

  def __init__(self):
    batch_sizes = [2, 4]
    input_lengths = [16, 32]
    output_lengths = [16, 32]
    n_iter = 1
    super().__init__(batch_sizes, input_lengths, output_lengths, n_iter)

  def run_iter(self, batch_size: int, input_length: int, output_length: int):
    # provided impl

bm = CustomBenchmark()
bm.run()
bm.print_results()
```

### Spark-submit

To benchmark an annotator using spark-submit, compile the benchmark script first. Sample scripts can be found [here](./benchmark/sparknlp). 

```
scalac -classpath "$SPARK_HOME/jars/*;" ./benchmark/sparknlp/BertEmbeddingsBenchmark.scala -d benchmark.jar
python3 run_spark_benchmark benchmark.jar BertEmbeddingsBenchmark <path-to-sparknlp-jar>
```

Use the `run_spark_benchmark` script to run.

```
usage: run_spark_benchmark.py [options] jar_path classname sparknlp_jar

positional arguments:
  jar_path              Path to the compiled jar file.
  classname             Benchmark script main class name.
  sparknlp_jar          Path to the spark nlp jar file.

options:
  -h, --help            show this help message and exit
  --batch_sizes BATCH_SIZES
                        Batch sizes to benchmark (pass multiple values as a comma-separated list). Default [4].
  --input_lengths INPUT_LENGTHS
                        Input lengths to benchmark (pass multiple values as a comma-separated list). Default [16].
  --output_lengths OUTPUT_LENGTHS
                        Output sequence lengths to benchmark (pass multiple values as a comma-separated list). Default [16].
  --model_path MODEL_PATH
                        Path to the model to import for benchmarking custom pre-trained models.
  --conll CONLL         Path to the CONLL formatted data file.
  --input_cols INPUT_COLS
                        Input columns to use for benchmarking.
  --resource_usage RESOURCE_USAGE
                        Measure memory and cpu usage.
  --n_iter N_ITER       Number of iterations of each case.
```
