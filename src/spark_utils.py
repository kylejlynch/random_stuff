"""Spark session utilities for the NSGA-II pipeline."""

try:
    from pyspark.sql import SparkSession
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PySpark is required to create a SparkSession. Please install pyspark>=2.0."
    ) from exc


def get_spark_session(app_name: str = "GeneticRuleSearch") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()
