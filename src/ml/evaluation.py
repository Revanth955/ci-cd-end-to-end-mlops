from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def add_threshold_prediction(
    predictions: DataFrame,
    threshold: float = 0.5,
) -> DataFrame:
    """
    Generate binary predictions using a configurable probability threshold.

    Parameters
    ----------
    predictions : DataFrame
        DataFrame containing target and probability columns.
    threshold : float
        Probability threshold used to classify the positive class.

    Returns
    -------
    DataFrame
        DataFrame containing a threshold-based prediction column.
    """

    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")

    return predictions.withColumn(
        "prediction_threshold",
        (vector_to_array("probability").getItem(1) >= threshold).cast("double"),
    )


def calculate_binary_metrics(
    predictions: DataFrame,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calculate precision, recall, and F1 for the positive class.

    Parameters
    ----------
    predictions : DataFrame
        Model predictions containing target and probability.
    threshold : float
        Probability threshold for positive-class prediction.

    Returns
    -------
    dict[str, float]
        Precision, recall, and F1 metrics.
    """

    evaluated = add_threshold_prediction(
        predictions,
        threshold=threshold,
    )

    true_positive = evaluated.filter(
        (col("target") == 1) &
        (col("prediction_threshold") == 1)
    ).count()

    false_positive = evaluated.filter(
        (col("target") == 0) &
        (col("prediction_threshold") == 1)
    ).count()

    false_negative = evaluated.filter(
        (col("target") == 1) &
        (col("prediction_threshold") == 0)
    ).count()

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }