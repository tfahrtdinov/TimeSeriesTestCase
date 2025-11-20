import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from enum import StrEnum

import mlflow


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

MLFLOW_TRACKING_URI = f"http://{os.getenv("MLFLOW_HOST")}:{os.getenv("MLFLOW_PORT")}"

logger = logging.getLogger(__name__)


class TargetColumn(StrEnum):
    HEIZWKES = "HeizwKes"
    F_HEIDA_KORR = "F HeiDa korr"


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        logger.info(f"Found existing MLflow experiment with name {experiment.name}")
        return experiment.experiment_id
    else:
        logger.info(f"Creating new MLflow experiment with name {experiment_name}")
        return mlflow.create_experiment(experiment_name)
