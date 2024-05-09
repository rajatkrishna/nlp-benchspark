## Benchmarks for Spark NLP

Python Tool for benchmarking NLP inference with [Spark NLP](https://github.com/JohnSnowLabs/spark-nlp) using `psutil`.
Built on top of Apache Spark, SparkNLP provides performant and scalable NLP annotations for Python, R and the JVM ecosystem, using OpenVINO, ONNX Runtime and Tensorflow for efficient and optimized NLP inference at scale. It offers several tasks including tokenization, NER and summarization. You can find a full list of supported feature [here](https://github.com/JohnSnowLabs/spark-nlp?tab=readme-ov-file#features).

The following are some transformers benchmarked as part of the [OpenVINO-SparkNLP Integration](https://github.com/JohnSnowLabs/spark-nlp/pull/14200). 

### LLama2

- Transformer: LLAMA2Transformer
  
  Model: [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

  <img src="https://github.com/rajatkrishna/nlp-benchspark/assets/61770314/14b63022-5780-4b79-9be9-3a66109dc3fd" width="500" height="300">

  <img src="https://github.com/rajatkrishna/nlp-benchspark/assets/61770314/763b8eee-ac0b-4dd9-b006-df8ee2384dca" width="500" height="300">

## Usage

Create a new Python virtual environment and install the dependencies in [requirements.txt](./requirements.txt).

```
python3 -m venv create .env
source .env/bin/activate

pip install -r requirements.txt
```

### PySpark

### Requirements

- JDK 8 or higher
- Python 3.11

To run with PySpark, first install SparkNLP and PySpark from PyPI with the following command:

```
pip install spark-nlp==5.3.3 pyspark==3.5.1
```

Use the `run_pybenchmark.py` script to benchmark a Spark NLP transformer in pyspark. You can find a full list of supported Spark NLP transformers [here](https://sparknlp.org/docs/en/annotators#available-transformers).

```
$ python3 run_pybenchmark.py sparknlp.annotator.LLAMA2Transformer 
    -p "OpenVINO integration with SparkNLP enables faster inference, easier model export and quantization capabilities."
```

You can also use a dataset in the CoNLL format by passing the path to the data file with the `--conll` argument, use a local SparkNLP jar by passing the `--sparknlp_jar` argument and import a custom exported model into its corresponding Spark transformer using the `--model_path` argument. The following is a list of all supported parameters.

```
usage: run_pybenchmark.py [-h] [-c CONFIG] [-p PROMPT] [--conll CONLL] [--input_cols INPUT_COLS] [--model_path MODEL_PATH]
                          [--pretrained PRETRAINED] [--batch_sizes BATCH_SIZES] [--input_lengths INPUT_LENGTHS]
                          [--output_lengths OUTPUT_LENGTHS] [--sparknlp_jar SPARKNLP_JAR] [--memcpu MEMCPU] [--n_iter N_ITER]
                          [annotator]

positional arguments:
  annotator             Fully qualified name of the Spark transformer.

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to benchmark config.
  -p PROMPT, --prompt PROMPT
                        Prompt to pass as input dataframe.
  --conll CONLL         Path to the CONLL formatted data file.
  --input_cols INPUT_COLS
                        Input columns to use for benchmarking.
  --model_path MODEL_PATH
                        Path to the model to import for benchmarking custom pre-trained models.
  --pretrained PRETRAINED
                        Pre-trained model name to download.
  --batch_sizes BATCH_SIZES
                        Batch sizes to benchmark (pass multiple values as a comma-separated list).
  --input_lengths INPUT_LENGTHS
                        Input lengths to benchmark (pass multiple values as a comma-separated list).
  --output_lengths OUTPUT_LENGTHS
                        Output sequence lengths to benchmark (pass multiple values as a comma-separated list).
  --sparknlp_jar SPARKNLP_JAR
                        Path to the spark nlp jar file.
  --memcpu MEMCPU       Measure memory and cpu usage.
  --n_iter N_ITER       Number of iterations of each case.
```

To run custom benchmarks, 

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


### Usage (spark-submit)

To benchmark an annotator using spark-submit

```
scalac -classpath "$SPARK_HOME/jars/*;" ./benchmark/sparknlp/BertEmbeddingsBenchmark.scala -d benchmark.jar
python3 run_spark_benchmark.py <model_path> <data_path> --jar_path=benchmark.jar
```
