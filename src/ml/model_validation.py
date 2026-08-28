"""
Model validation utilities.

This module validates model evaluation metrics against the configured
model promotion acceptance criteria.

The module does not:
- train models,
- calculate model metrics,
- register models,
- deploy models.

It only validates metrics that have already been produced and logged.
"""

import mlflow

from src.config.configuration import (
    MIN_RECALL,
    MIN_PRECISION,
    MIN_F1_SCORE,
)
from src.utils.logger import logger


def validate_metrics(metrics: dict) -> dict:
    """
    Validate model metrics against configured acceptance criteria.

    Parameters
    ----------
    metrics : dict
        Model evaluation metrics containing recall, precision, and f1.

    Returns
    -------
    dict
        Detailed validation result containing:
        - overall pass/fail status
        - actual metrics
        - configured minimum criteria
        - individual metric checks
    """

    # Define the metrics required for model validation.
    required_metrics = {
        "recall",
        "precision",
        "f1",
    }

    # Identify any required metrics that were not provided.
    missing_metrics = required_metrics - metrics.keys()

    if missing_metrics:
        raise ValueError(
            "Required metrics missing for model validation: "
            f"{sorted(missing_metrics)}"
        )

    # Read the supplied evaluation metrics.
    recall = metrics["recall"]
    precision = metrics["precision"]
    f1_score = metrics["f1"]

    # Compare each metric against its configured minimum.
    recall_passed = recall >= MIN_RECALL
    precision_passed = precision >= MIN_PRECISION
    f1_passed = f1_score >= MIN_F1_SCORE

    # The candidate passes only when every required criterion passes.
    validation_passed = (
        recall_passed
        and precision_passed
        and f1_passed
    )

    return {
        "passed": validation_passed,
        "metrics": {
            "recall": recall,
            "precision": precision,
            "f1": f1_score,
        },
        "criteria": {
            "min_recall": MIN_RECALL,
            "min_precision": MIN_PRECISION,
            "min_f1_score": MIN_F1_SCORE,
        },
        "checks": {
            "recall": recall_passed,
            "precision": precision_passed,
            "f1": f1_passed,
        },
    }


def validate_model(run_id: str) -> dict:
    """
    Retrieve an MLflow run and validate its logged metrics.

    Parameters
    ----------
    run_id : str
        MLflow run ID containing the evaluation metrics.

    Returns
    -------
    dict
        Detailed model validation result.
    """

    logger.info("Validating MLflow model run: %s", run_id)

    # Retrieve the existing MLflow run.
    # No Spark execution or model training occurs here.
    run = mlflow.get_run(run_id)

    # Validate the metrics recorded by the training pipeline.
    result = validate_metrics(run.data.metrics)

    logger.info(
        "Model validation completed: %s",
        "PASS" if result["passed"] else "FAIL",
    )

    return result