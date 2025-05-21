import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler


def train(df):
    label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
    assembler = VectorAssembler(
        inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        outputCol="features",
    )
    classifier = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features")

    pipeline = Pipeline(stages=[label_indexer, assembler, classifier])

    with mlflow.start_run():
        model = pipeline.fit(df)
        mlflow.spark.log_model(model, "random_forest_model")
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("num_rows", df.count())

    return model
