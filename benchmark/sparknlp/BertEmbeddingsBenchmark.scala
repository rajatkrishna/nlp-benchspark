import com.johnsnowlabs.nlp.embeddings.BertEmbeddings
import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

import java.util.logging.{Level, Logger}


object BertEmbeddingsBenchmark {

    private val logger: Logger = Logger.getLogger(this.getClass.toString)

    def main(args: Array[String]): Unit = {
        val modelPath: String = args(0)
        val dataPath: String = args(1)
        val batchSize: Integer = if (args.length > 2) args(2).toInt else 8
        val seqLength: Integer = if (args.length > 3) args(3).toInt else 32
        val inputCols: Array[String] = if (args.length >4) args(4).split(",").map(_.trim()).toArray else Array("sentence", "token")
        val loglevel: String = if (args.length > 5) args(5) else "CONFIG"
        val level: Level = Level.parse(loglevel)

        logger.log(level, s"Benchmarking BertEmbeddings, model path: ${modelPath}")
        logger.log(level, s"Benchmarking BertEmbeddings, data path: ${dataPath}")
        logger.log(level, s"Benchmarking BertEmbeddings, batch size: ${batchSize}")
        logger.log(level, s"Benchmarking BertEmbeddings, sequence length: ${seqLength}")

        logger.log(level, s"Building spark session...")
        val spark = SparkSession
            .builder()
            .appName("Benchmark App")
            .master("local[*]")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate()

        val conll = CoNLL(explodeSentences = false)
        val corpus = conll.readDataset(spark, dataPath)

        val embeddings = BertEmbeddings
            .loadSavedModel(modelPath, spark)
            .setInputCols(inputCols)
            .setOutputCol("embeddings")
            .setMaxSentenceLength(seqLength)
            .setBatchSize(batchSize)

        val pipeline = new Pipeline()
            .setStages(Array(embeddings))

        val result = pipeline.fit(corpus).transform(corpus)

        val start = System.nanoTime()
        result.write.mode("overwrite").parquet("./tmp_bm")
        val dur = (System.nanoTime() - start) / 1e9

        logger.info(s"BertEmbeddings took ${dur} seconds..")
        println(s"$dur seconds")
        spark.close()
    }
}
