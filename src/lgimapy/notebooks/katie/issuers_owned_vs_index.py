from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.data import Database
from lgimapy.utils import root

# %%


def quarter_to_date(quarter):
    q = int(quarter[0])
    year = int(f"20{quarter[-2:]}")
    if q < 4:
        month = 3 * q + 1
    else:
        month = 1
        year += 1
    date = f"{month}/1/{year}"
    return Database().nearest_date(date, inclusive=False, after=False)


def next_quarter(quarter):
    q = int(quarter[0])
    year = int(f"20{quarter[-2:]}")
    if q < 4:
        q += 1
    else:
        q = 1
        year += 1
    return f"{q}Q{year - 2000:02.0f}"


def update_number_of_issuers_owned(account):
    """Update the number of issuers owned vs the index file."""
    fid = root("data/issuers_owned_vs_index.csv")
    old_df = pd.read_csv(
        fid, index_col=0, parse_dates=True, infer_datetime_format=True
    ).iloc[:-1, :]
    db = Database()
    today = db.date("today")

    d = defaultdict(list)
    last_quarter = old_df["Quarter"].iloc[-1]
    up_to_date = False
    while not up_to_date:
        quarter = next_quarter(last_quarter)
        date = quarter_to_date(quarter)
        if date == today:
            quarter = "Current"
            up_to_date = True
        tickers = db.load_portfolio(account=account, date=date).ticker_df
        d["date"].append(date)
        d["Quarter"].append(quarter)
        d["LGIMA LD"].append(len(tickers[tickers["P_Weight"] > 0]))
        d["BBG LULC"].append(len(tickers[tickers["BM_Weight"] > 0]))
        last_quarter = quarter

    new_df = pd.DataFrame(d).set_index("date").rename_axis(None)
    updated_df = pd.concat((old_df, new_df))
    updated_df.to_csv(fid)
    current = updated_df.iloc[-1, :]
    ratio = current['LGIMA LD'] / current['BBG LULC']
    print(f'LGIMA currently holds {ratio:.0%} of the issuers in the index.')

def update_emphasis_on_liquidity(account):
    """Update the emphasis on bond liquidty file."""
    fid = root("data/emphasis_on_bond_liquidity.csv")
    db = Database()
    acnt = db.load_portfolio(account=account)
    # %%
    df = acnt.df.sort_values("IssueYears")
    df["LGIMA LD"] = np.cumsum(df["P_Weight"])
    df["BBG LULC"] = np.cumsum(df["BM_Weight"])
    df["Years since Issuance"] = df["IssueYears"]
    df = (
        df[["Years since Issuance", "LGIMA LD", "BBG LULC"]]
        .dropna()
        .reset_index(drop=True)
    )
    df.to_csv(fid)
    df_3yr = df[df[df.columns[0]] > 3].iloc[0]
    delta = df_3yr['LGIMA LD'] - df_3yr['BBG LULC']
    print(f'LGIMA is currently {delta:.0%} more liquid than the index.')

if __name__ == '__main__':
    account = "CHLD"
    update_number_of_issuers_owned(account)
    update_emphasis_on_liquidity(account)
