from collections import Counter

import pandas as pd

from lgimapy.data import Database
from lgimapy.utils import root
from lgimapy.bloomberg import bdpdb

db = Database()

db.load_market_data(date="12/31/2003")
ix = db.build_market_index()

ix_df = db.load_market_data(date="12/31/2003", clean=False, ret_df=True)
ix_df["MaturityDate"] = pd.to_datetime(ix_df["MaturityDate"])
ix_df["cusip5"] = [c[:5] for c in ix_df["CUSIP"]]
list(ix_df)
ix_df[ix_df["Ticker"] == "DPIZZA"]
ix_df[ix_df["CUSIP"] == "25754QAF4"]
ix_df[(ix_df["CouponRate"] == 10.75) & (ix_df["MaturityDate"] == "2005-08-01")]


sips = sorted(ix_df["CUSIP"])

help(bdp)
orig_ids_df = bdp(sips, "Corp", "REGISTERED_BOND_ORIGINAL_ID")
orig_ids = orig_ids_df["REGISTERED_BOND_ORIGINAL_ID"].dropna().values

orig_full_cusips = bdp(orig_ids, "Corp", "ID_CUSIP")
orig_cusips = [c[:-1] for c in orig_full_cusips["ID_CUSIP"].dropna().values]

sips[3500:]


cusips = [c[:-1] for c in ix_df["CUSIP"]]


# %%
from lgimapy.utils import root
from collections import Counter
from lgimapy.bloomberg import bdp

df = pd.read_csv(root("data/H0A0/12_31_2003.csv"))
df["Maturity"] = pd.to_datetime(df["Maturity"])
df["cusip"] = [c.replace("'", "") for c in df["Cusip"]]


ice_cusips = sorted([c.replace("'", "") for c in df["Cusip"]])


ix.df.head()


len(cusips)
len(ice_cusips)


missing = []
for c in ice_cusips:
    if c not in cusips:
        missing.append(c)


len(missing)

df = df[df["cusip"].isin(missing)]

ao_vals = []
for ao in df["AmountOutstanding"]:
    try:
        ao_vals.append(float(ao))
    except ValueError:
        ao_vals.append(0)


df["AmountOutstanding"] = ao_vals
df = df[df["AmountOutstanding"] >= 1.5e8].copy()

df = df[~df["cusip"].isin(orig_cusips)]


len(df)

country
df.head()
lvl2 = Counter(df["Sector Level 2"])
lvl2

df.to_csv("missing_BAML_H0A0_cusips.csv")

df.head()
list(df)[:10]
df["CUSIP"]
df.head()
conversions = {}
for _, row in df.iterrows():
    df_same = ix_df[
        (ix_df["cusip5"] == row["cusip"][:5])
        & (ix_df["CouponRate"] == row["Coupon"])
        & (ix_df["MaturityDate"] == row["Maturity"])
    ]
    if len(df_same):
        conversions[row["cusip"]] = list(df_same["CUSIP"])

conversions
len(conversions)
