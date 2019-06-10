import argparse

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

pd.plotting.register_matplotlib_converters()
plt.style.use("fivethirtyeight")


def main():
    args = parse_args()

    # Load data.
    # TODO: scrape this data from bloomberg.
    df = pd.read_csv(
        "../data/LUACOAS.csv",
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )

    # Subset data to last year if observing tights.
    if args.tights:
        start = pd.to_datetime(dt.date.today() - dt.timedelta(days=365))
        df = df[df.index >= start]
    y = df["Bid Price"].values

    # Plot index data, if observing tights add potential tights as vlines.
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.plot(df.index, y, c="steelblue")
    if args.tights:
        valleys, _ = find_peaks(-y, distance=60, prominence=args.p)
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
    ax.set_ylabel("Bid Price")
    fig.autofmt_xdate()
    plt.show()


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--tights", action="store_true", help="Show tights"
    )
    parser.add_argument("-p", "--prominence", help="Valley Prominence Level")
    parser.set_defaults(prominence=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
