from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import lgimapy.vis as vis
from lgimapy.bloomberg import bdp
from lgimapy.data import Database
from lgimapy.utils import root

# %%

db = Database()


# date = db.nearest_date("1/1/2003")
# db.load_market_data(date=date, local=True)
# ix = db.build_market_index()
# ix.df.isna().sum()
# ix.df["CollateralType"].value_counts()


cols = [
    "Issuer",
    "MarketOfIssue",
    "CollateralType",
    "Currency",
    "CountryOfRisk",
    "CountryOfDomicile",
]
trade_dates = db.trade_dates(start=db.date("MARKET_START"))
df_list = []
for date in tqdm(trade_dates[::-250]):
    db.load_market_data(date=date, local=True)
    ix = db.build_market_index()
    # if date < pd.to_datetime("11/1/2018"):
    # ix._fill_missing_columns_with_bbg_data()
    df = ix.df[cols].copy()
    df.replace("No Collateral", np.NaN, inplace=True)
    df_list.append(df.isna().sum().rename(date) / len(df))


missing_df = pd.concat(df_list, axis=1, sort=True).T.sort_index()
# missing_df.to_csv('original_missing_data.csv')

# %%
def plot_missing_data(df, fid=None):
    fig, ax = vis.subplots(1, 1, figsize=(8, 8))
    sns.heatmap(
        100 * df,
        cmap="viridis",
        linewidths=0.2,
        # vmax=100,
        vmin=0,
        # annot=True,
        # annot_kws={"fontsize": 7},
        # fmt=".2f",
        cbar=True,
        cbar_kws={"label": "% Missing Data"},
        ax=ax,
    )
    yticks = [d.strftime("%Y") for d in df.index]
    ax.xaxis.tick_top()
    ax.set_xticklabels(df.columns, rotation=45, ha="left", fontsize=8)
    ax.set_yticklabels(yticks, ha="right", fontsize=7, va="center")
    if fid is not None:
        vis.savefig(fid)
    else:
        vis.show()


# plot_missing_data(missing_df, "current_missing_data")
# plot_missing_data(missing_df, "current_missing_data_rescaled")
plot_missing_data(missing_df)

# %%
# Find most recent values of missing data for each bond if it exists.
saved_d = defaultdict(dict)
cusip_set = set()
for date in tqdm(trade_dates[::-10]):
    # Get data for that day.
    db.load_market_data(date=date, local=True)
    ix = db.build_market_index()
    df = ix.df[cols].copy()
    # Store any new cusips.
    cusip_set = cusip_set | set(df.index)
    df.replace("No Collateral", np.NaN, inplace=True)
    for col in cols:
        # Find bonds not seen before.
        s = df[col]
        s_new = s[~s.isna() & ~s.index.isin(saved_d[col])]
        # Store the most recent values of these bonds.
        for cusip, val in s_new.items():
            saved_d[col][cusip] = val

# Find missing cusips for which data is still missing for each column.
missing_d = {}
for col in cols:
    missing_d[col] = cusip_set - set(saved_d[col].keys())


# %%
print(f"Total Cusips: {len(cusip_set):,.0f}")

print("Found more recently in database")
for key, val in saved_d.items():
    print(f"  {key}: {len(val):,.0f}")


print("\n\nStill Missing")
for key, val in missing_d.items():
    print(f"  {key}: {len(val):,.0f}")


# %%

db = Database()
db.load_market_data(local=True)
ix = db.build_market_index()

ix.df["CollateralType"].replace("No Collateral", np.NaN, inplace=True)


tdf = ix.df["CollateralType"].dropna().value_counts()
tdf[tdf > 0]

fid = root("src/lgimapy/notebooks/scraped_bbg_columns.csv")
bbg_df = pd.read_csv(fid, index_col=0)
equiv_ranks = {
    "UNSECURED": ["BONDS", "SR UNSECURED", "NOTES", "COMPANY GUARNT"],
    "SECURED": ["SR SECURED"],
    "1ST MORTGAGE": ["1ST REF MORT", "GENL REF MORT"],
}
for key, val in equiv_ranks.items():
    bbg_df.loc[bbg_df["CollateralType"].isin(val), "CollateralType"] = key
cols = list(bbg_df)


bbg_df["CollateralType"].dropna().value_counts()

cols
from lgimapy.utils import mkdir, dump_json, load_json

json_dir = root("data/bbg_jsons")
mkdir(json_dir)


for col in cols:
    fid = f"bbg_jsons/{col}"
    dump_json(bbg_df[col].dropna().to_dict(), fid)

    d = load_json(fid)
    print(col, len(d))


# %%
still_missing = bbg_df[bbg_df.isna().all(axis=1)]
len(still_missing)
still_missing.to_csv("still_missing.csv")


# %%


still_missing = pd.read_csv("still_missing.csv", index_col=0)
missing_cusips = still_missing.index
len(missing_cusips)
db2 = Database()

# %%
db2.load_market_data(start="1/1/2019", local=True)
ix2 = db2.build_market_index(cusip=missing_cusips)

# %%
# isin_map = {}
for cusip, isin in ix2.df["ISIN"].items():
    if cusip not in isin_map and isin is not None:
        try:
            if np.isnan(isin):
                continue
        except TypeError:
            isin_map[cusip] = isin


len(isin_map)

# %%
still_missing["ISIN"] = still_missing.index.map(isin_map)
still_missing.to_csv("still_missing.csv")

still_missing_cusips

still_missing
