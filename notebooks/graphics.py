import pandas as pd
from numpy.typing import ArrayLike

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt
from plotly import graph_objects as go


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
    with plt.style.context("bmh"), pd.option_context("plotting.backend", "matplotlib"):
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
