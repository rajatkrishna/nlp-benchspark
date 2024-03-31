## Benchmarks for Spark NLP

Tool for benchmarking NLP inference with [Spark NLP](https://github.com/JohnSnowLabs/spark-nlp).
Built on top of Apache Spark, SparkNLP provides performant and scalable NLP annotations for Python, R and the JVM ecosystem,
using OpenVINO, ONNX Runtime and Tensorflow for efficient and optimized NLP inference at scale.

### Usage (spark-submit)

To benchmark an annotator using spark-submit

```
scalac -classpath "$SPARK_HOME/jars/*;" ./benchmark/sparknlp/BertEmbeddingsBenchmark.scala -d benchmark.jar
python3 run_spark_benchmark.py <model_path> <data_path> --jar_path=benchmark.jar
```

### Usage (pyspark)

Create a new Python virtual environment and install the dependencies in [requirements.txt](./requirements.txt).

```
python3 -m venv create .env
source .env/bin/activate

pip install -r requirements.txt
```

To run benchmarks using a custom dataset, 

```
from benchmark.pybenchmark import PyBenchmark
import sparknlp

spark = SparkSession.builder \
    .appName("BenchmarkApp") \
    .master("local[*]") \
    .getOrCreate()
bert = BertEmbeddings.loadSavedModel('/home/ubuntu/models/bert-large-uncased', spark)
data = CoNLL(exposeSentences=False).readDataSet(spark, '/home/ubuntu/data/conll2003/eng.traina')

batches = [2, 4, 6]
max_seq_lengths = [16, 32, 64]

bm = PyBenchmark(spark, bert, data, batches, max_seq_lengths)
bm.run()
bm.print_results()
```

Use the `run_pybenchmark.py` script to measure the inference performance of a SparkNLP Annotator using a dataset in the CoNLL2003 format. 

```
usage: run_benchmark.py [-h] [-c CONFIG] [--input_cols INPUT_COLS] [--model_path MODEL_PATH] [--model_name MODEL_NAME] [--batch_sizes BATCH_SIZES] [--seq_lengths SEQ_LENGTHS] [--sparknlp_jar SPARKNLP_JAR] [--is_seq2seq IS_SEQ2SEQ] [--memcpu MEMCPU]
                        [--n_iter N_ITER]
                        [annotator] [data_path]

positional arguments:
  annotator             Fully qualified name of the annotator.
  data_path             Path to the data file.

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to benchmark config.
  --input_cols INPUT_COLS
                        The input columns for the annotator.
  --model_path MODEL_PATH
                        Path to the saved model.
  --model_name MODEL_NAME
                        Pre-trained model name.
  --batch_sizes BATCH_SIZES
                        Batch sizes to benchmark (pass multiple values as a comma-separated list).
  --seq_lengths SEQ_LENGTHS
                        Sequence lengths to benchmark (pass multiple values as comma-separated list).
  --sparknlp_jar SPARKNLP_JAR
                        Path to the spark nlp jar file.
  --is_seq2seq IS_SEQ2SEQ
                        Seq-2-seq model.
  --memcpu MEMCPU       Measure memory and cpu usage.
  --n_iter N_ITER       Number of iterations of each case.
```
