import numpy as np
import pandas as pd

from lgimapy.data import Database
from lgimapy.utils import root

# %%
comp_account = "HIC"
fid = root("data/tmp/IBXXPGUP.csv")


db = Database()
port = db.load_portfolio(account=comp_account)
db.load_market_data(clean=False, preprocess=False, local=False)

# %%
iboxx_df = (
    pd.read_csv(fid, index_col=1, skiprows=7)
    .rename_axis(None)
    # .rename(columns={" Index % ": "BM_Weight"})
    .rename(columns={"IBOX Weight": "BM_Weight"})
)
iboxx_df = iboxx_df[iboxx_df["Ccy"] == "USD"].copy()
iboxx_df["BM_Weight"] = (
    iboxx_df["BM_Weight"].str.strip().replace("-", 0).astype(float)
)

iboxx_df['BM_Weight'] /= iboxx_df['BM_Weight'].sum()
iboxx_df = iboxx_df[iboxx_df['BM_Weight'] > 0].copy()

# %%
len(iboxx_df)
# ix = db.build_market_index(isin=iboxx_df.index)


iboxx_df[iboxx_df.index.isin(ix.isins)]['BM_Weight'].sum()
missing_df = iboxx_df[~iboxx_df.index.isin(db.df['ISIN'])]
ix_df = db.df[~db.df['ISIN'].isin(iboxx_df.index)]
len(missing_df)


missing_df.to_csv('missing_bonds.csv')
