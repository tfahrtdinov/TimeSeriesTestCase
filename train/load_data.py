# import os
import logging
from typing import cast

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from train.utils import TargetColumn


logger = logging.getLogger(__name__)
DATASET = "../data/data_Jan-Jun.csv"


def load_data_df() -> pd.DataFrame:
    """
    Load the data from the dataset

    Return:
        pd.DataFrame with the data
    """
    df = pd.read_csv(DATASET, index_col="TimeStamp")
    df.index = pd.to_datetime(df.index)
    return df


def calculate_features(df: pd.DataFrame, target_col: TargetColumn) -> None:
    pass


def split_data(
    df: pd.DataFrame,
    target_col: TargetColumn,
    test_size: float = 0.2,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """
    Split the dataframe into training and test sets and return NumPy arrays.

    Args:
        df: input dataframe containing features and target column.
        target_col: name of the target column to predict.
        test_size: fraction of data to allocate for the test split.

    Returns:
        feature_train, feature_test, target_train, target_test arrays, each NDArray[np.float32]
    """
    features = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=test_size,  # type: ignore
    )

    return (
        cast(NDArray[np.float32], X_train),
        cast(NDArray[np.float32], X_test),
        cast(NDArray[np.float32], y_train),
        cast(NDArray[np.float32], y_test),
    )
