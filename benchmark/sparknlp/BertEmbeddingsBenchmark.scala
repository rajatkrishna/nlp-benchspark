import com.johnsnowlabs.nlp.embeddings.BertEmbeddings
import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession



object BertEmbeddingsBenchmark {

    def main(args: Array[String]): Unit = {
        val modelPath: String = args(0)
        val dataPath: String = args(1)
        val batchSize: Integer = if (args.length > 2) args(2).toInt else 8
        val seqLength: Integer = if (args.length > 3) args(3).toInt else 32
        val inputCols: Array[String] = if (args.length >4) args(4).split(",").map(_.trim()).toArray else Array("document", "token")

        val spark = SparkSession
            .builder()
            .appName("Benchmark App")
            .master("local[*]")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate()

        val embeddings = BertEmbeddings
            .loadSavedModel(modelPath, spark)
            .setInputCols(inputCols)
            .setOutputCol("embeddings")
            .setMaxSentenceLength(seqLength)
            .setBatchSize(batchSize)

        val pipeline = new Pipeline()
            .setStages(Array(embeddings))

        val conll = CoNLL(explodeSentences = false)
        val corpus = conll.readDataset(spark, dataPath)

        val result = pipeline.fit(corpus).transform(corpus)

        val start = System.nanoTime()
        result.select("embeddings").show()
        val dur = (System.nanoTime() - start) / 1e9

        spark.close()
    }
}
