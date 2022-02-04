from collections import defaultdict

import pandas as pd

import lgimapy.vis as vis
from lgimapy.data import Database
from lgimapy.utils import root

# %%
def main():
    df = get_bm_tsy_pcts("USBGC")
    fid = root("data/Long_GC_BM_Tsy_pct")
    plot_tsy_pcts(df, fid)


def get_bm_tsy_pcts(account):
    """
    Load benchmark of given account's stats and returns
    benchmark treasury weight for the full month of the
    given date and find the difference between them.
    """
    db = Database()
    dates = db.trade_dates(start=db.date("month_start"))
    d = defaultdict(list)
    for date in dates:
        stats = db.load_portfolio(account=account, date=date, universe="stats")
        rets = db.load_portfolio(account=account, date=date)
        d["stats"].append(stats.bm_tsy_pct())
        d["returns"].append(rets.bm_tsy_pct())

    df = pd.DataFrame(d, index=dates)
    df["diff"] = df["stats"] - df["returns"]
    return df


def plot_tsy_pcts(df, fid):
    vis.style()
    fig, axes = vis.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    axes[0].set_title(f"Current Difference: {df['diff'][-1]:.2%}\n")
    vis.plot_multiple_timeseries(
        df[["stats", "returns"]],
        c_list=["navy", "darkgreen"],
        lw=1.8,
        alpha=0.7,
        xtickfmt="auto",
        ytickfmt="{x:.1%}",
        ylabel="BM Treasury %",
        ax=axes[0],
    )
    axes[1].fill_between(df.index, 0, df["diff"], color="grey", alpha=0.7)
    axes[1].set_ylabel("Difference")
    vis.format_xaxis(axes[1], df["diff"], "auto")
    vis.format_yaxis(axes[1], "{x:.1%}")
    vis.set_n_ticks(axes[1], 4)

    vis.savefig(fid)
    vis.close()


if __name__ == "__main__":
    main()
