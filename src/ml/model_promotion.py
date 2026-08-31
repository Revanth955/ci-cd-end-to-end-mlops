"""
Model promotion orchestration.

This module coordinates model validation, registration,
and Champion assignment.

The module does not:
- train models,
- calculate evaluation metrics,
- implement validation rules,
- deploy models.

It orchestrates the workflow:
    Validate → Register → Promote if eligible.
"""

import mlflow
from mlflow.exceptions import MlflowException
from src.config.configuration import MIN_F1_IMPROVEMENT, MIN_PRECISION

from src.ml.model_validation import validate_model
from src.ml.model_registration import register_model
from src.utils.logger import logger


def should_register(validation_result: dict) -> bool:
    """
    Determine whether a validated model is eligible for registration.

    Parameters
    ----------
    validation_result : dict
        Detailed result returned by the model validation component.

    Returns
    -------
    bool
        True when the model passes validation; otherwise False.
    """

    return validation_result["passed"]


def should_promote(
    candidate_metrics: dict,
    champion_metrics: dict,
) -> bool:
    """
    Determine whether the candidate should replace the Champion.

    Parameters
    ----------
    candidate_metrics : dict
        Evaluation metrics for the candidate model.

    champion_metrics : dict
        Evaluation metrics for the current Champion.

    Returns
    -------
    bool
        True when the candidate satisfies all promotion criteria.
    """

    candidate_recall = candidate_metrics["recall"]
    candidate_precision = candidate_metrics["precision"]
    candidate_f1 = candidate_metrics["f1"]

    champion_recall = champion_metrics["recall"]
    champion_f1 = champion_metrics["f1"]

    recall_passed = candidate_recall >= champion_recall

    precision_passed = candidate_precision >= MIN_PRECISION

    f1_passed = (
        candidate_f1
        >= champion_f1 + MIN_F1_IMPROVEMENT
    )

    return (
        recall_passed
        and precision_passed
        and f1_passed
    )


def get_champion_version(model_name: str):
    """
    Return the registered model version assigned to the Champion alias.

    Parameters
    ----------
    model_name : str
        Name of the registered model.

    Returns
    -------
    ModelVersion or None
        Current Champion version, or None when no Champion exists.
    """

    client = mlflow.MlflowClient()

    try:
        return client.get_model_version_by_alias(
            model_name,
            "champion",
        )

    except MlflowException as exc:
        if "alias champion not found" in str(exc).lower():
            return None

        raise

def set_champion(model_name: str, version: str) -> None:
    """
    Assign the Champion alias to a registered model version.

    Parameters
    ----------
    model_name : str
        Name of the registered model.

    version : str
        Registered model version to assign as Champion.
    """

    client = mlflow.MlflowClient()

    client.set_registered_model_alias(
        model_name,
        "champion",
        version,
    )


def promote_model(run_id: str, model_name: str):
    """
    Validate a model run, register it if validation passes,
    and promote it when it satisfies the promotion criteria.

    Parameters
    ----------
    run_id : str
        MLflow run ID containing the candidate model and metrics.

    model_name : str
        Name of the registered model.

    Returns
    -------
    object or dict
        Registered MLflow model version when validation passes.
        Validation result when validation fails.
    """

    logger.info(
        "Starting model promotion workflow for run: %s",
        run_id,
    )

    # Validate the candidate model.
    validation_result = validate_model(run_id)

    # Stop the workflow when the candidate fails validation.
    if not should_register(validation_result):
        logger.warning(
            "Model validation failed. Registration skipped for run: %s",
            run_id,
        )

        return validation_result

    # Register the validated model artifact.
    registered_model = register_model(
        run_id=run_id,
        model_name=model_name,
    )

    # Check whether a Champion already exists.
    champion = get_champion_version(model_name)

    # Bootstrap the registry when no Champion exists yet.
    if champion is None:
        set_champion(
            model_name,
            registered_model.version,
        )

        logger.info(
            "No Champion existed. Model %s version %s "
            "is now the Champion.",
            registered_model.name,
            registered_model.version,
        )

    else:
        # Retrieve the MLflow run that produced the Champion.
        champion_run = mlflow.get_run(
            champion.run_id,
        )

        # Retrieve the Champion's evaluation metrics.
        champion_metrics = champion_run.data.metrics

        # Compare the candidate against the existing Champion.
        if should_promote(
            candidate_metrics=validation_result["metrics"],
            champion_metrics=champion_metrics,
        ):
            set_champion(
                model_name,
                registered_model.version,
            )

            logger.info(
                "Candidate model %s version %s "
                "passed promotion criteria and is now the Champion.",
                registered_model.name,
                registered_model.version,
            )

        else:
            logger.info(
                "Candidate model %s version %s "
                "did not pass promotion criteria. "
                "Existing Champion version %s retained.",
                registered_model.name,
                registered_model.version,
                champion.version,
            )

    logger.info(
        "Model promotion workflow completed: "
        "%s version %s",
        registered_model.name,
        registered_model.version,
    )

    return registered_model