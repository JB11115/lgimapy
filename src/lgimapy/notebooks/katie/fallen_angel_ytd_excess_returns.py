import pandas as pd

from lgimapy.data import Database
from lgimapy.utils import root

db = Database()
db.load_market_data(start=db.date("ytd"), local=True)

# %%
tdelta = "1m"


def calculate_fallen_angel_cost(tdelta, end_date, db):
    """
    Calculate cost (in excess returns) that fallen angels
    have on the long duration index.
    """
    # Find index excess returns
    ix = db.build_market_index(in_returns_index=True, end=end_date)
    index_cusips = ix.cusips
    ix_xsret = ix.aggregate_excess_returns()

    # Get rating changes DataFrame
    rc_fid = root("data/rating_changes.csv")
    rc_df = pd.read_csv(rc_fid, index_col=0)
    rc_df.dtypes
    rc_df["Date_PREV"] = pd.to_datetime(rc_df["Date_PREV"])
    rc_df["Date_NEW"] = pd.to_datetime(rc_df["Date_NEW"])

    rc_df = rc_df[
        (rc_df["Date_PREV"] > db.date("YTD"))
        & (rc_df["NumericRating_PREV"] <= 10)
        & (rc_df["NumericRating_NEW"] > 10)
    ]

    for __, row in rc_df.iterrows():
        cusip = row["CUSIP"]
        if cusip not in index_cusips:
            continue
        try:
            sell_date = db.date(tdelta, reference_date=row["Date_PREV"])
        except IndexError:
            sell_date = db.date("today")
        ix.df.drop(
            ix.df[
                (ix.df["CUSIP"] == row["CUSIP"]) & (ix.df["Date"] > sell_date)
            ].index,
            inplace=True,
        )

    ix.clear_day_cache()
    ix_no_fa_xsret = ix.aggregate_excess_returns()

    # print(f"Index XSRET: {ix_xsret:.2%}")
    # print(f"No Fallen Angel XSRET: {ix_no_fa_xsret:.2%}")
    # print(f"Fallen Angel Cost: {ix_no_fa_xsret - ix_xsret:.2%}")
    return ix_no_fa_xsret - ix_xsret


# %%
dates = db.trade_dates(start=db.date("ytd"))[::8]
cost = [calculate_fallen_angel_cost("1m", date, db) for date in dates]

cost_s = pd.Series(cost, dates)
cost_s.to_csv("MC_cost_of_fallen_angels.csv")


# %%
from lgimapy import vis

vis.style()

# %%
cost_mc = pd.read_csv(
    "MC_cost_of_fallen_angels.csv",
    index_col=0,
    parse_dates=True,
    infer_datetime_format=True,
).squeeze()
cost_lc = pd.read_csv(
    "LD_cost_of_fallen_angels.csv",
    index_col=0,
    parse_dates=True,
    infer_datetime_format=True,
).squeeze()
vis.plot_multiple_timeseries(
    [cost_lc.rename("Long Credit"), cost_mc.rename("Market Credit")],
    c_list=["steelblue", "darkorange"],
    lw=3,
    alpha=0.7,
    ytickfmt="{x:.1%}",
    xtickfmt="auto",
    ylabel="Cost of Fallen Angels\n(Excess Returns)",
)
vis.show()
vis.savefig("cost_of_fallen_angels_2020")
