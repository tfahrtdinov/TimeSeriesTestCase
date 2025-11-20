import logging
import argparse

from train.xgboost_train import train


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training script for XGBoost with optuna"
    )

    parser.add_argument(
        "--target_name",
        "-t",
        type=str,
        required=True,
        choices=["HeizwKes", "F HeiDa korr"],
        help="Name of target variable",
    )

    parser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        required=True,
        help="Name of the MLflow experiment",
    )

    parser.add_argument(
        "--run_name", "-r", type=str, required=True, help="Name of the MLflow run"
    )

    parser.add_argument(
        "--n_trials",
        "-n",
        type=int,
        default=10,
        help="Number of Optuna trials (default: 10)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Training started with args: {args}")
    train(args.target_name, args.experiment_name, args.run_name, args.n_trials)
    logger.info("Training completed")
