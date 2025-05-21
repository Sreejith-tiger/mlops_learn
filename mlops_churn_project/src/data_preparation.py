from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def load_data():
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv("/databricks-datasets/iris.csv", header=True, inferSchema=True)
    return df


def prepare_data(df):
    return df.withColumnRenamed("species", "label").select(
        "sepal_length", "sepal_width", "petal_length", "petal_width", "label"
    )
