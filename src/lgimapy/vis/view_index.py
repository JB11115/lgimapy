import argparse

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import lgimapy.vis as vis
from lgimapy.data import Database


def main():
    args = parse_args()

    # Subset data to last year.
    start = (
        pd.to_datetime(dt.date.today() - dt.timedelta(days=365))
        if args.start is None
        else pd.to_datetime(args.start)
    )

    db = Database()
    df = db.load_bbg_data(args.index, "OAS", nan="drop", start=start)
    y = df.values

    # Plot index data.
    vis.style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(df.index, y, c="steelblue")

    # Plot local minimums with labels to identify tights.
    valleys, _ = find_peaks(-y, distance=50, prominence=args.prominence)
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
    ax.set_ylabel(f"{args.index} (bp)")
    fig.autofmt_xdate()
    plt.show()


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", help="Credit Index")
    parser.add_argument("-s", "--start", help="Start Date")
    parser.add_argument(
        "-p", "--prominence", type=int, help="Valley Prominence Level"
    )
    parser.set_defaults(index="US_IG_10+", start=None, prominence=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
