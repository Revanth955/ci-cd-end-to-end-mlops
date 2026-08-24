from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame

from src.ml.preprocessing import build_preprocessing_pipeline


def build_logistic_regression_pipeline(
    train_df: DataFrame,
) -> Pipeline:
    """
    Build the preprocessing + Logistic Regression pipeline.

    Preprocessing is fitted only on the training data.
    """

    preprocessing = build_preprocessing_pipeline(
        train_df,
        scale_numeric=True,
    )

    classifier = LogisticRegression(
    featuresCol="features",
    labelCol="target",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    maxIter=20,
    regParam=0.1,
    elasticNetParam=0.0,
)

    return Pipeline(
        stages=[
            *preprocessing.getStages(),
            classifier,
        ]
    )