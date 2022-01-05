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
    db = Database()
    start = db.date("1y") if args.start is None else pd.to_datetime(args.start)
    df = db.load_bbg_data(args.index, "OAS", nan="drop", start=start)
    sign = 1 if args.wides else -1
    y = sign * df.values

    # Plot index data.
    vis.style()
    fig, ax = vis.subplots(figsize=(12, 6))
    vis.plot_timeseries(df, color="navy", ax=ax)

    # Plot local minimums with labels to identify tights.
    valleys, _ = find_peaks(y, distance=50, prominence=args.prominence)
    valleys = np.concatenate((valleys, [-1]))
    for date, oas in df.iloc[valleys].items():
        ax.axvline(date, color="firebrick", ls="--", lw=1)
        label = f"{oas:.0f}bp  {pd.to_datetime(date):%m/%d/%Y}"
        vmax, vmin = np.max(df.values), np.min(df.values)
        height = 0.9 * (vmax - vmin) + vmin
        ax.text(
            date,
            height,
            label,
            rotation=90,
            color="firebrick",
            horizontalalignment="right",
        )
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{db.bbg_names(args.index)} OAS (bp)")
    plt.tight_layout()
    plt.show()


class Temp:
    index = "US_IG"
    start = "1/1/2000"
    prominence = 50
    wides = False


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", help="Credit Index")
    parser.add_argument("-s", "--start", help="Start Date")
    parser.add_argument("-w", "--wides", action="store_true", help="Find Wides")
    parser.add_argument(
        "-p", "--prominence", type=int, help="Valley Prominence Level"
    )
    parser.set_defaults(index="US_IG_10+", start=None, prominence=200)
    return parser.parse_args()


# %%
if __name__ == "__main__":
    main()
