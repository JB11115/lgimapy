from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import lgimapy.vis as vis


def find_drawdowns(s, threshold="auto", distance=5, prominence=3):
    """
    Find drawdowns for a given series. For spreads, simply make the
    input series negative.

    Parameters
    ----------
    s: pd.Series
        Timeseries of values to find drawdowns for.
    threshold: int or ``'auto'``, default='auto'
        Threshold for how large drawdown must be in order to be stored.
        If 'auto', use the standard deviation of the series.
    distance: int, default=5
        Number of observations required between peaks.
    prominence: float, default=3
        Required prominence of peaks.

    Returns
    -------
    pd.DataFrame:
        DataFrame with a row for each drawdown which provides
        the respective start and end dates, levels, and drawdown
        magnitude and percent.
    """
    n = distance
    thresh = np.std(s) if threshold == "auto" else threshold
    peaks, _ = find_peaks(s.values, distance=n, prominence=prominence)
    troughs, _ = find_peaks(-s.values, distance=n, prominence=prominence)
    troughs = np.concatenate([troughs, [len(s) - 1]])  # Add last date
    extrema = np.sort(np.concatenate([peaks, troughs]))

    d = defaultdict(list)
    current_peak = s.iloc[0]
    current_peak_t = 0
    drawdown = 0
    for i, t in enumerate(extrema):
        if t in peaks:
            # Check if new peak is higher, if it is
            # re-set current peak to maximum value
            if s.iloc[t] > current_peak:
                current_peak = s.iloc[t]
                current_peak_t = t

        elif t in troughs:
            # Check if next extrema is also a trough, and if it
            # is make current trough the minimum of the two
            try:
                next_t = extrema[i + 1]
            except IndexError:
                pass  # Last date in series, no next extrema
            else:
                if next_t in troughs and s.iloc[next_t] <= s.iloc[t]:
                    continue

            # Check if current drawdown is above threshold, if it
            # is, store the drawdown and reset the current peak.
            drawdown = current_peak - s.iloc[t]
            if drawdown > thresh:
                d["drawdown"].append(drawdown)
                d["pct_drawdown"].append(drawdown / current_peak)
                d["start_level"].append(s.iloc[current_peak_t])
                d["end_level"].append(s.iloc[t])
                d["start"].append(s.index[current_peak_t])
                d["end"].append(s.index[t])
                current_peak = -np.infty

    return pd.DataFrame(d)


def plot_drawdown_timeseries(s, drawdown_df, **kwargs):
    """
    Plot timeseries with drawdown periods shaded.

    Parameters
    ----------
    s: pd.Series

    """
    vis.style()
    s = s.dropna()

    fig, ax = vis.subplots(figsize=(12, 6))
    ts_kwargs = {"ytickfmt": "${x:.0f}", "xtickfmt": "auto", "ax": ax}
    ts_kwargs.update(**kwargs)
    vis.plot_timeseries(s, **ts_kwargs)
    fill = [np.min(s), np.max(s)]
    for start, end in zip(drawdown_df["start"], drawdown_df["end"]):
        ax.fill_betweenx(fill, start, end, color="lightgrey", alpha=0.2)
    plt.show()
