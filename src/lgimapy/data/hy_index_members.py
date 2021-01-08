import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.bloomberg import bds, id_to_isin
from lgimapy.utils import root

# %%
def update_hy_index_members():
    indexes = ["H4UN", "H0A0", "HC1N", "HUC2", "HUC3"]
    for index in indexes:
        update_index_members(index)


def load_index_members(index):
    fid = root(f"data/index_members/{index}.parquet")
    return pd.read_parquet(fid)


def update_index_members(index):
    fid = root(f"data/index_members/{index}.parquet")
    try:
        saved_df = pd.read_parquet(fid)
    except OSError:
        saved_df = pd.DataFrame()
    saved_dates = set(saved_df.index)

    start = {
        "H4UN": "12/31/2012",
        "H0A0": "1/1/1999",
        "HC1N": "12/31/2012",
        "HUC2": "12/31/2012",
        "HUC3": "12/31/2013",
    }
    trade_dates = set(Database().trade_dates(start=start[index]))
    dates_to_scrape = sorted(trade_dates - saved_dates)
    if not dates_to_scrape:
        return

    scraped_list = []
    for date in dates_to_scrape:
        ovrd = {"END_DT": date.strftime("%Y%m%d")}
        col = "Index Member"
        ids = bds(index, "Index", "INDX_MWEIGHT_HIST", ovrd=ovrd)[col]
        isins = set(pd.Series(id_to_isin(ids)).dropna())
        ones = np.ones(len(isins))
        scraped_list.append(pd.Series(ones, index=isins, name=date))

    scraped_df = pd.concat(scraped_list, axis=1).T
    updated_df = (
        pd.concat((saved_df, scraped_df)).fillna(0).astype("int8").sort_index()
    )
    updated_df.to_parquet(fid)


# %%

if __name__ == "__main__":
    update_hy_index_members()
