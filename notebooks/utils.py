from typing import cast

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from statsmodels.tsa.stattools import adfuller, kpss, ccf
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
    c_xy = ccf(predictor, target, adjusted=True, nlags=nlags)
    c_yx = ccf(target, predictor, adjusted=True, nlags=nlags)[:0:-1]

    return np.concatenate((c_yx, c_xy))
