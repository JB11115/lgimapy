from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database, Bond
from lgimapy.utils import root

plt.style.use("seaborn-dark-palette")
# %matplotlib qt

# %%
def main():
    # ------------------------------------------------------------------ #
    ticker = "VZ"  # ABIBB C F KMT VZ
    spread_df = read_csv(f"USD-{ticker}-SEN_2019")
    # date = spread_df.index[-1]  # pick any date
    date = pd.to_datetime("7/17/2019")
    # ------------------------------------------------------------------ #

    # Compute yield by adding swap curve to spread curve using
    # linear interpolation of the swap curve.
    swap_df = read_csv("USDswapCurve_2019")
    yield_df = spread_df.loc[date, :] + np.interp(
        spread_df.columns, swap_df.columns, swap_df.loc[date, :]
    )

    db = Database()
    date = spread_df.index[0]
    db.load_market_data(date=date)
    ix = db.build_market_index(ticker=ticker)
    d = defaultdict(list)
    for bond in ix.bonds:
        d["mats"].append(bond.MaturityYears)
        d["ytm"].append(bond.ytm)
        d["oas"].append(bond.OAS)
        d["cusip"].append(bond.cusip)

    bond_df = pd.DataFrame(d)
    bond_df.sort_values("ytm", inplace=True)
    # %%
    spread_df

    plot_yields_oas(ticker, date, d, spread_df, yield_df)
    plt.show()

    # %%
    vis.style(style="seaborn")
    fig, ax = vis.subplots(figsize=[9, 6])
    ax.set_title("VZ", fontsize=18, fontweight="bold")
    ax.set_ylabel("OAS")
    ax.set_xlabel("Maturity")
    axioma_df = spread_df.loc[date]
    axioma_df = axioma_df[axioma_df.index <= 30]
    df = ix.df.copy()
    df = df[df["MaturityYears"] <= 30]
    df = df[~((df["MaturityYears"] > 10) & (df["OAS"] < 100))]
    ax.scatter(
        df["MaturityYears"].values,
        df["OAS"].values,
        s=df["MarketValue"] / 20,
        color="darkgreen",
    )
    ax.plot(axioma_df.index, 10000 * axioma_df.values, color="firebrick", lw=2)
    vis.savefig("axioma_VZ_curve")

    # %%


def read_csv(fid):
    df = pd.read_csv(
        root(f"data/axioma_curves/{fid}.csv"),
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )
    df.columns = [clean_col(c) for c in df.columns]
    return df[[c for c in df.columns if c <= 40]]


def clean_col(col):
    if "Y" in col:
        c = float(col.strip("Y"))
    elif "M" in col:
        c = float(col.strip("M")) / 12
    elif "W" in col:
        c = float(col.strip("W")) / 52
    elif "D" in col:
        c = float(col.strip("D")) / 365
    return round(c, 3)


def plot_yields_oas(ticker, date, d, spread_df, yield_df):
    fig, axes = plt.subplots(2, 1, figsize=[14, 10], sharex=True)
    tick = mtick.StrMethodFormatter("{x:.2%}")
    axes[0].plot(
        d["mats"], d["ytm"], "o", c="darkgreen", label="Market Value YTM", ms=4
    )
    axes[0].plot(
        yield_df,
        "--o",
        c="k",
        label="Axioma Issuer Yield Curve (Swap + Spread)",
        lw=1,
        ms=4,
    )
    axes[0].set_ylabel("Yield")
    axes[0].yaxis.set_major_formatter(tick)
    axes[0].legend()
    axes[1].plot(
        list(spread_df),
        spread_df.loc[date, :],
        "--o",
        c="k",
        label="Axioma Issuer Spread Curve",
        lw=1,
        ms=4,
    )
    ax2 = axes[1].twinx()
    ax2.plot(d["mats"], d["oas"], "o", c="firebrick", label="Market OAS", ms=4)
    ax2.set_ylabel("OAS")
    ax2.grid(False)
    axes[1].set_xlabel("Maturity")
    axes[1].set_ylabel("Yield")
    axes[1].yaxis.set_major_formatter(tick)
    axes[1].legend()

    fig.suptitle(f'{ticker} - {date.strftime("%m/%d/%Y")}', fontsize=20)


# %%

if __name__ == "__main__":
    main()
