"""Utilities for loading the fraud dataset subset using PySpark."""

from pathlib import Path
from typing import Iterable, Tuple

from pyspark.sql import DataFrame as SparkDataFrame
try:
    from pyspark.sql import types as T
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PySpark is required to load the dataset. Please install pyspark>=2.0."
    ) from exc

DATA_DIR = Path("data/raw/bank-account-fraud-dataset")
BASE_FILE = DATA_DIR / "Base.csv"

LABEL_COLUMN = "fraud_bool"
AMOUNT_COLUMN = "intended_balcon_amount"

# Numeric features that already arrive in a model-ready form (no encoding required).
NUMERIC_FEATURES: Tuple[str, ...] = (
    "income",
    "name_email_similarity",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "bank_months_count",
    "proposed_credit_limit",
    "device_distinct_emails_8w",
    "device_fraud_count",
)


def available_features() -> Iterable[str]:
    """Return the tuple of numeric-only features exposed to the rule engine."""
    return NUMERIC_FEATURES


def _cast_columns(frame: SparkDataFrame, columns: Iterable[str]) -> SparkDataFrame:
    for column in columns:
        frame = frame.withColumn(column, frame[column].cast(T.DoubleType()))
    return frame


def load_training_view(
    spark,
    sample_fraction: float = 0.10,
    random_state: int = 42,
) -> SparkDataFrame:
    """Load a Spark DataFrame slice of the base dataset."""

    if not BASE_FILE.exists():
        raise FileNotFoundError(
            f"Expected dataset at {BASE_FILE}. Run the data download step first."
        )

    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be within (0, 1].")

    columns = list(dict.fromkeys([LABEL_COLUMN, AMOUNT_COLUMN] + list(NUMERIC_FEATURES)))
    frame = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(BASE_FILE))
        .select(*columns)
    )

    numeric_columns = list(dict.fromkeys([AMOUNT_COLUMN] + list(NUMERIC_FEATURES)))
    frame = _cast_columns(frame, numeric_columns)
    frame = frame.withColumn(LABEL_COLUMN, frame[LABEL_COLUMN].cast(T.IntegerType()))

    if sample_fraction < 1:
        frame = frame.sample(False, sample_fraction, random_state)

    return frame.cache()
