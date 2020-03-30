import pandas as pd

from lgimapy.data import Database
from lgimapy.utils import root, to_datetime

# %%


def get_treasury_oad_values(date):
    date = to_datetime(date)
    db = Database()
    db.load_market_data(date=date)
    ix = db.build_market_index(
        sector="TREASURIES", ticker="T", drop_treasuries=False
    )
    dfs = {mat: df for mat, df in ix.df.groupby("OriginalMaturity")}
    on_the_run_cusips = ix.df.groupby("OriginalMaturity").idxmax()["IssueDate"]
    oad_vals = (
        ix.df.loc[on_the_run_cusips.values][["OriginalMaturity", "OAD"]]
        .set_index("OriginalMaturity")
        .squeeze()
        .rename(date)
        .rename_axis("")
    )
    oad_vals.index = [str(ix) for ix in oad_vals.index]
    return oad_vals


def update_treasury_oad_values():
    # Get all dates that should have values.
    db = Database()
    all_dates = db.trade_dates(start=db.date("PORTFOLIO_START"))

    # Find dates to scrape and get previously scraped values.
    fid = root("data/OTR_treasury_OAD_values.parquet")
    try:
        df = pd.read_parquet(fid)
        dates = set(all_dates) - set(df.index.unique())
    except FileNotFoundError:
        dates = all_dates
        df = pd.DataFrame()

    # Scrape dates if there are missing dates.
    if dates:
        computed_df = pd.concat(
            [get_treasury_oad_values(date) for date in dates],
            axis=1,
            sort=False,
        ).T
    else:
        computed_df = pd.DataFrame()

    # Update old file and save.
    df_to_save = pd.concat([df, computed_df], sort=False).sort_index()
    df_to_save.to_parquet(fid)


if __name__ == "__main__":
    update_treasury_oad_values()
