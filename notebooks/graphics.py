import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt
from plotly import graph_objects as go

from utils import ccf_two_direct


def ts_plotly_slider(y: pd.Series) -> None:
    """
    Draws Plotly style time-series plot with an x-axis range slider.

    Args:
        y: time series

    Returns:
        None
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, x=y.index, mode="lines", name=y.name))

    fig.update_layout(
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )
    fig.show()


def ts_autocorr_plot(
    y: pd.Series,
    lags: int | ArrayLike | None = None,
    pacf: bool = False,
    fft: bool = False,
) -> None:
    """
    Plot ACF (and optionally PACF) for a time series.

    Args:
        y: time series.
        lags: Lags to compute autocorrelation for.
        pacf: If True, plot both ACF and PACF.
        fft: If True, use FFT-based computation for ACF.

    Returns:
        None
    """
    with plt.style.context("bmh"):
        if pacf:
            plt.figure(figsize=(10, 3))
            layout = (1, 2)
            acf_ax = plt.subplot2grid(layout, (0, 0))
            pacf_ax = plt.subplot2grid(layout, (0, 1))

            plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5, fft=fft, title=f"{y.name} acf")
            plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5, title=f"{y.name} pacf")

            plt.tight_layout()
        else:
            _, ax = plt.subplots(figsize=(10, 3))
            plot_acf(y, lags=lags, alpha=0.5, fft=fft, title=f"{y.name} acf", ax=ax)
        plt.show()


def plot_ccf_two_direct(target: pd.Series, predictor: pd.Series, nlags: int) -> None:
    """
    Plot bidirectional cross-correlation between two time series.

    Args:
       target: The target time series.
       predictor: The predictor time series.
       nlags: Number of lags to compute on each side.
    Returns:
        None
    """
    ccfs = ccf_two_direct(target, predictor, nlags)
    lags = np.arange(-nlags + 1, nlags)

    plt.figure(figsize=(10, 3))
    markerline, stemlines, _ = plt.stem(lags, ccfs)
    plt.setp(stemlines, linewidth=0.7)
    plt.setp(markerline, markersize=3)
    plt.xlabel("lag")
    plt.ylabel("CCF")
    plt.title(f"Bidirectional CCF, {target.name} vs {predictor.name}")
    plt.tight_layout()
    plt.show()


def plot_histograms_plotly(df: pd.DataFrame) -> None:
    """
    Draws Plotly style histogram.

    Args:
        df: dataframe with time series

    Returns:
        None
    """
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Histogram(x=df[col], name=col, opacity=0.5))

    fig.update_layout(
        barmode="overlay",
        title="Histograms of all columns",
        xaxis_title="Value",
        yaxis_title="Frequency",
    )
    fig.show()


def plot_fc(predict, y_true, index: pd.DatetimeIndex, title: str):
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=y_true, x=index, name="True values"))
    fig.add_trace(go.Scatter(y=predict, x=index, name="Predicted values"))
    fig.update_layout(
        title=title,
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )
    fig.show()
