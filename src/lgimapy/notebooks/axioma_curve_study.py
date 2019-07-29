from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

from lgimapy.index import IndexBuilder, Bond
from lgimapy.utils import root

plt.style.use("fivethirtyeight")
# %matplotlib qt

# %%
def main():
    # %%
    # ------------------------------------------------------------------ #
    ticker = "VZ"  # ABIBB C F KMT VZ
    df = read_csv(ticker)
    date = df.index[-1]  # pick any date
    # ------------------------------------------------------------------ #

    ixb.load(date=date)
    ix = ixb.build(ticker=ticker)
    bonds = [Bond(bond) for _, bond in ix.df.iterrows()]
    d = defaultdict(list)
    for b in bonds:
        d["mats"].append(b.MaturityYears)
        d["ytm"].append(b.ytm)
        d["oas"].append(b.OAS)

    # %%
    plot_yields_oas(ticker, date, d)
    plt.show()
    # %%


def read_csv(issuer):
    swap = pd.read_csv(
        root(f"data/axioma_curves/USD-{issuer}-SEN_2019.csv"),
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )
    df = pd.read_csv(
        root(f"data/axioma_curves/USD-{issuer}-SEN_2019.csv"),
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )
    df += swap
    df.columns = [clean_col(c) for c in df.columns]
    return df


def clean_col(col):
    if "Y" in col:
        return float(col.strip("Y"))
    elif "M" in col:
        return float(col.strip("M")) / 12


def plot_yields_oas(ticker, date, d):
    fig, axes = plt.subplots(2, 1, figsize=[14, 10])
    tick = mtick.StrMethodFormatter("{x:.2%}")

    axes[0].plot(
        d["mats"], d["ytm"], "o", c="darkgreen", label="Market Value YTM", ms=4
    )
    axes[0].plot(
        list(df),
        df.loc[date, :],
        "--o",
        c="k",
        label="Axioma Curve",
        lw=1,
        ms=4,
    )
    axes[0].set_xlabel("Maturity")
    axes[0].set_ylabel("Yield")
    axes[0].yaxis.set_major_formatter(tick)
    axes[0].legend()
    axes[0].set_xlim((-1, 2 + max(d["mats"])))

    axes[1].plot(
        list(df),
        df.loc[date, :],
        "--o",
        c="k",
        label="Axioma Curve",
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
    axes[1].set_xlim((-1, 2 + max(d["mats"])))

    fig.suptitle(f'{ticker} - {date.strftime("%m/%d/%Y")}', fontsize=20)


if __name__ == "__main__":
    main()
