import os
import logging
from pathlib import Path
from dotenv import load_dotenv

import mlflow
import optuna
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


# from train.load_data import load_data_df, calculate_features, split_data
from train.utils import TargetColumn, get_or_create_experiment


BASE_DIR = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)
load_dotenv(Path(BASE_DIR / ".env"))

N_SPLITS = 3
VALID_FRAC = 0.3
TEST_FRAC = 0.2


def optuna_objective(
    trial: optuna.trial.Trial,
    features: NDArray[np.float32],
    target: NDArray[np.float32],
) -> float:
    """
    Define the Optuna objective function for XGBoost hyperparameter optimization.

    Args:
        trial: current Optuna trial providing parameter suggestions.
        features: training feature matrix.
        target: target array corresponding to training data.

    Returns:
        float: mean squared error (MSE) for the evaluated trial.
    """

    with mlflow.start_run(run_name=f"trial number {trial.number}", nested=True):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "booster": trial.suggest_categorical("booster", ["gbtree"]),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1, log=True),
        }

        model = xgb.XGBRegressor(n_jobs=-1, **params)
        scores = cross_validate(
            model,
            features,
            target,
            cv=TimeSeriesSplit(
                n_splits=N_SPLITS,
                test_size=int((target.shape[0] * VALID_FRAC) / N_SPLITS),
            ),
            scoring={
                "mse": "neg_mean_squared_error",
                "mae": "neg_mean_absolute_error",
            },
        )
        mse_score = -scores["test_mse"].mean()

        mlflow.log_params(params)
        mlflow.log_metric("rmse", np.sqrt(mse_score))
        mlflow.log_metric("mae", -scores["test_mae"].mean())

    return mse_score


def train_optuna(
    experiment_id: str,
    run_name: str,
    n_trials: int,
    features_train: NDArray[np.float32],
    target_train: NDArray[np.float32],
    features_test: NDArray[np.float32],
    target_test: NDArray[np.float32],
) -> None:
    """
    Run Optuna study to find optimal hyperparameters set and train final XGBoost model.

    Args:
        experiment_id: MLflow experiment identifier.
        run_name: name for the MLflow run.
        n_trials: number of Optuna trials to execute.
        features_train: training feature matrix.
        target_train: training target array.
        features_test: test feature matrix.
        target_test: test target array.

    Returns:
        None
    """

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: optuna_objective(
                trial, features=features_train, target=target_train
            ),
            n_trials=n_trials,
        )

        logger.info(
            f"Best trial: \nScore: {study.best_value} \nParams: {study.best_params}"
        )
        logger.info("Fitting model with best params")

        best_model = xgb.XGBRegressor(**study.best_params)
        best_model.fit(features_train, target_train)

        target_test_pred = best_model.predict(features_test)

        scorers = (root_mean_squared_error, mean_absolute_error, r2_score)
        for score_name, scorer in zip(("rmse", "mae", "r2"), scorers):
            mlflow.log_metric(score_name, scorer(target_test, target_test_pred))

        mlflow.log_params(study.best_params)

        logger.info("Logging best model")
        mlflow.sklearn.log_model(  # type: ignore
            sk_model=best_model,
            name="xgboost-model",
            input_example=features_train,
            registered_model_name="xgboost-model",
        )
        logger.info("Model logged")

        fig, ax = plt.subplots()
        xgb.plot_importance(best_model, importance_type="gain", ax=ax)
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)


def train(
    target_name: TargetColumn, experiment_name: str, run_name: str, n_trials: int
) -> None:
    """
    Full training pipeline for XGBoost with MLflow tracking and Optuna optimization.

    Args:
        target_name: target variable name either 'HeizwKes' or 'F HeiDa korr'
        experiment_name: MLflow experiment name.
        run_name: name for the MLflow run.
        n_trials: number of Optuna trials to perform.

    Raises:
        RuntimeError: if MLFLOW_TRACKING_URI is not defined in the environment.
    """

    logger.info(f"Getting data for {experiment_name}")

    # features_train, features_test, target_train, target_test = split_data(
    #     calculate_features(load_data_df()),
    #     target_col="monetary_value_30",
    #     test_size=TEST_FRAC,
    # )

    logger.info("Setting up MLFlow")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri is None:
        logger.error("MLFLOW_TRACKING_URI required in your environment")
        raise RuntimeError("MLFLOW_TRACKING_URI required in your environment")

    mlflow.set_tracking_uri(mlflow_uri)
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    logger.info(f"Starting optuna optimization with {n_trials} trials")
    # train_optuna(
    #     experiment_id=experiment_id,
    #     run_name=run_name,
    #     n_trials=n_trials,
    #     features_train=features_train,
    #     features_test=features_test,
    #     target_train=target_train,
    #     target_test=target_test,
    # )
    logger.info("Finished optuna optimization")
