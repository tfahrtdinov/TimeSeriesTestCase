from typing import cast

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from tslearn.metrics import dtw
from statsmodels.tsa.stattools import adfuller, kpss, ccf, grangercausalitytests
from statsmodels.nonparametric.smoothers_lowess import lowess


def stationarity_test(y: pd.Series) -> tuple[float, float]:
    """
    Args:
        y: time series

    Returns:
        Tuple of two p-values:
        (ADF p-value, KPSS p-value)
    """
    return cast(float, adfuller(y)[1]), kpss(y)[1]


def lowess_smooth(y: pd.Series, frac: float = 0.02) -> pd.Series:
    """
    Args:
        y: time series
        frac: between 0 and 1. The fraction of the data used when estimating each y-value.

    Returns:
        LOWESS smoothed y-value
    """
    sm = lowess(y.values, np.arange(len(y)), frac=frac, return_sorted=False)
    return pd.Series(sm, index=y.index, name=f"{y.name}_loess")


def ccf_two_direct(
    target: pd.Series, predictor: pd.Series, nlags: int
) -> NDArray[np.float64]:
    """
    Calculates bidirectional cross-correlation between two time series.

    Args:
        target: The target time series.
        predictor: The predictor time series.
        nlags: Number of lags to compute on each side.
    Returns:
         ccf
    """
    c_xy = ccf(target, predictor, adjusted=True, nlags=nlags)
    c_yx = ccf(predictor, target, adjusted=True, nlags=nlags)[:0:-1]

    return np.concatenate((c_yx, c_xy))


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


def normalize_windows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalizes numpy sliding_window_view
    Args:
        values: sliding_window_view of 1d array
    Returns:
        Normalized sliding_window_view of 1d array
    """
    return (values - values.mean(axis=1, keepdims=True)) / (
        values.std(axis=1, keepdims=True) + 1e-8
    )


def dtw_rolling(x: pd.Series, y: pd.Series, window: int = 100) -> pd.Series:
    """
    Compute DTW distances on rolling, z-normalized windows of two time series.

    Args:
        x: First time series.
        y: Second time series (same length as x).
        window: Rolling window size.

    Returns:
        A Series of DTW distances, one value per window, aligned to the end
        of each window.
    """
    normalized_x = normalize_windows(sliding_window_view(x.values, window))  # type: ignore
    normalized_y = normalize_windows(sliding_window_view(y.values, window))  # type: ignore

    distances = np.empty(normalized_x.shape[0], dtype=np.float64)

    for i, (x_wind, y_wind) in enumerate(zip(normalized_x, normalized_y)):
        distances[i] = dtw(x_wind, y_wind) / window

    return pd.Series(distances, index=x.index[window - 1 :])
