import numpy as np
import pandas as pd

from lgimapy.utils import root, to_list

# %%

fid = root("data/tmp/SANTAN.csv")
df = pd.read_csv(fid, index_col=11).rename_axis(None)
df = df[df["QuoteCurrency"] == "USD"].copy()
df["Weight"] /= df["Weight"].sum()
df["MV"] = df["Nominal"] * (df["AccruedInterest"] + df["CleanConstituentPrice"])
df["OAD"] = df["OptionAdjustedDuration"].astype(float)
df["BM_OAD"] = df["OAD"] * df["Weight"]
df = df.rename(columns={"L4": "Sector"})

df["MV"].sum() / 1e12
df["BM_OAD"].sum()


# %%

list(df)
df["MV_Weight"] = df["MV"] / df["MV"].sum()
df[["Weight", "MV_Weight"]]

df.to_csv('SANTAN_model_portfolio_bond_level.csv')

# %%
excel = pd.ExcelWriter("SANTAN_model_portfolio.xlsx")
groupby_cols = ["Sector", "IssuerName", "CreditRating"]
agg_cols = ["MV", "BM_OAD"]
for col in groupby_cols:
    gdf = df[agg_cols + to_list(col, dtype=str)].groupby(col).sum()
    for agg_col in agg_cols:
        gdf[f"{agg_col}_pct_of_BM"] = gdf[agg_col] / df[agg_col].sum()
    gdf = gdf.round(3).rename_axis(None).sort_index()
    if col == "Sector":
        sector_gdf = gdf.copy()
    gdf.to_excel(excel, sheet_name=col)

gdf = df[groupby_cols + agg_cols].groupby(groupby_cols[:2]).sum().reset_index()
for agg_col in agg_cols:
    gdf[f"{agg_col}_pct_of_Sector"] = gdf[agg_col] / gdf["Sector"].map(
        sector_gdf[agg_col].to_dict()
    gdf[f"{agg_col}_pct_of_BM"] = gdf[agg_col] / df[agg_col].sum()

gdf = (
    gdf.set_index(["Sector", "IssuerName"])
    .round(3)
    .sort_values(["Sector", "BM_OAD_pct_of_Sector"], ascending=[True, False])
)
gdf.to_excel(excel, sheet_name="Sector-Issuer")



excel.save()
