import pandas as pd

from lgimapy.data import Database

# %%

db = Database()
port = db.load_portfolio(account="P-LD")
top_30_port = port.subset(
    rating=("BBB+", "BBB-"),
    financial_flag=False,
    sector="LOCAL_AUTHORITIES",
    special_rules="~Sector",
)
tickers = top_30_port.ticker_overweights("BM_Weight").index[:30]

cols = ["Weight", "OAD", "DTS"]
df_list = []
for base_col in cols:
    for col_str in ["P_{}", "BM_{}", "{}"]:
        col = col_str.format(base_col)
        name = f"{base_col}_OW" if col_str == "{}" else col
        df_list.append(port.ticker_overweights(col).rename(name))

df = (
    pd.concat(df_list, axis=1)
    .sort_values("BM_Weight", ascending=False)
    .rename_axis(None)
)
df = df[df.index.isin(tickers)]
df.index = [str(ix) for ix in df.index]
df = df.append(df.sum().rename("Total"))
df
# %%
df.to_csv("LC_BBB_Nonfin_Top_30_full.csv")
