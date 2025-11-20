from collections import defaultdict
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sktime.transformations.series.summarize import WindowSummarizer
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.lag import Lag
from statsmodels.tsa.stattools import ccf, grangercausalitytests

from train.utils import TargetColumn, ENV_PATH


load_dotenv(ENV_PATH)


def granger_test(
    df: pd.DataFrame,
    target_col: str,
    predictor_col: str,
    maxlag: int = 10,
    corr_threshold: float = 0.8,
    significance_level: float = 0.05,
) -> NDArray[np.float64]:
    """
    Run a filtered Granger causality test between two time series.

    This function first computes the cross-correlation between the target and
    predictor and selects only those lags whose absolute correlation exceeds
    `corr_threshold`. For each selected lag, a Granger causality test is run.

    Args:
        df: DataFrame containing both time series.
        target_col: Column name of the target series (Y).
        predictor_col: Column name of the predictor series (X).
        maxlag: Maximum lag to consider for the cross-correlation and Granger tests.
        corr_threshold: Minimum absolute cross-correlation needed for a lag to be tested.
        significance_level: The p-value threshold for rejecting the null hypothesis.

    Returns:
        ndarray of lags for which the null hypothesis of the Granger test
        is rejected. Rejection means that at least one coefficient for the
        predictor's lagged values is significantly different from zero, implying
        that the predictor Granger-causes the target at that lag.
    """
    ccfs_mask = (
        np.abs(ccf(df[target_col], df[predictor_col], nlags=maxlag + 1)[1:])
        > corr_threshold
    )
    if ccfs_mask.any():
        res_dict = grangercausalitytests(
            df[[target_col, predictor_col]],
            maxlag=np.arange(1, maxlag + 1)[ccfs_mask],
            verbose=False,
        )
        return np.array(
            [
                k
                for k, v in res_dict.items()
                if v[0]["ssr_ftest"][1] < significance_level
            ]
        )
    else:
        return np.array([])


def calculate_significant_lags(
    df: pd.DataFrame, target_col: TargetColumn
) -> dict[tuple[int], list[str]]:
    """
    Compute statistically significant lag indices between the target column and all
    other predictor columns in the given DataFrame using the Granger causality test.

    The function runs a Granger causality test for each non-target column

    Args:
    df : pd.DataFrame
        Input DataFrame containing the target column and all predictor columns.
    target_col : TargetColumn
        The target variable for which lag dependencies should be evaluated.

    Returns:
        A dictionary mapping a tuple of significant lag indices to a list of column
    """
    col_to_lags = {}
    for col in df.drop(
        columns=[TargetColumn.HEIZWKES, TargetColumn.F_HEIDA_KORR]
    ).columns:
        gtest = granger_test(
            df=df,
            target_col=target_col,
            predictor_col=col,
            maxlag=100,
            corr_threshold=0.5,
            significance_level=0.05,
        )
        if len(gtest) != 0:
            col_to_lags[col] = gtest

    groups = defaultdict(list)
    for col, lags in col_to_lags.items():
        groups[tuple(lags.tolist())].append(col)

    return groups


def get_features(
    df: pd.DataFrame, target_col: TargetColumn, windows: list[int]
) -> NDArray[np.float32]:
    """
    Generate feature matrix

    Args:
    df : pd.DataFrame
        Input DataFrame containing the target column and all predictor columns.
    target_col : TargetColumn
        The target variable for which lag dependencies should be evaluated.
    windows : list of int for the WindowSummarizer extractor

    Returns:
        numpy float32 array with feature matrix
    """
    lag_groups = calculate_significant_lags(df, target_col)
    casted_window = [[1, w] for w in windows]

    window_summarizer = WindowSummarizer(
        lag_feature={
            "mean": casted_window,
            "std": casted_window,
            "min": casted_window,
            "max": casted_window,
        },
        target_cols=[
            col
            for col in df.columns.tolist()
            if col != TargetColumn.HEIZWKES and col != TargetColumn.F_HEIDA_KORR
        ],
    )

    datetime_extractor = DateTimeFeatures(
        ts_freq="T", keep_original_columns=False, feature_scope="minimal"
    )

    target_lags = list(range(1, 12 * 5))
    lag_transformers = [
        (str(target_lags), Lag(target_lags, index_out="original"), [str(target_col)])  # type: ignore
    ]
    lag_transformers.extend(
        [
            (str(lags), Lag(list(lags), index_out="original"), columns)  # type: ignore
            for lags, columns in lag_groups.items()
        ]
    )
    lag_transformer = ColumnTransformer(transformers=lag_transformers, remainder="drop")

    feature_union = FeatureUnion(
        [
            ("window", window_summarizer),
            ("datetime_extractor", datetime_extractor),
            ("lags", lag_transformer),
        ]
    )
    return feature_union.fit_transform(df).astype(np.float32)  # type: ignore
