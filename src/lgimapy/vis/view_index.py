import argparse

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from lgimapy.bloomberg import bdh

pd.plotting.register_matplotlib_converters()
plt.style.use("fivethirtyeight")


def main():
    args = parse_args()

    # Subset data to last year.
    start = pd.to_datetime(dt.date.today() - dt.timedelta(days=365))
    df = bdh("LUACOAS", yellow_button="Index", field="PX_BID", start=start)
    y = df["PX_BID"].values

    # Plot index data.
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.plot(df.index, y, c="steelblue")

    # Plot local minimums with labels to identify tights.
    valleys, _ = find_peaks(-y, distance=60, prominence=args.prominence)
    for date in df.index.values[valleys]:
        ax.axvline(date, color="firebrick", ls="--", lw=1)
        label = pd.to_datetime(date).strftime("%m/%d/%Y")
        ax.text(
            date,
            np.max(y),
            label,
            rotation=90,
            color="firebrick",
            horizontalalignment="right",
        )
    ax.set_xlabel("Date")
    ax.set_ylabel("PX_BID")
    fig.autofmt_xdate()
    plt.show()


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prominence", help="Valley Prominence Level")
    parser.set_defaults(prominence=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
