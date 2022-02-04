import pandas as pd

from lgimapy.data import Database

# %%
db = Database()
start = db.date("HY_START")
db.load_market_data(start=start)
ix = db.build_market_index(in_H4UN_index=True)
# %%
dates = ix.dates
date = dates[0]
cols = ["BAMLSector", "BAMLTopLevelSector", "Sector"]


def get_sectors(date):
    df = ix.day(date).set_index("ISIN")
    return (
        df["BAMLSector"].dropna().rename(date),
        df["BAMLTopLevelSector"].dropna().rename(date),
    )


def get_diff(prev, curr):
    df = pd.concat((prev_sector, curr_sector), join="inner", axis=1)
    return df[df.iloc[:, 0] != df.iloc[:, 1]]


prev_sector, prev_top_sector = get_sectors(dates[0])

df_list = []
df_top_list = []
for date in dates[1:]:
    curr_sector, curr_top_sector = get_sectors(date)
    sector = get_diff(prev_sector, curr_sector)
    top_sector = get_diff(prev_top_sector, curr_top_sector)
    if len(sector):
        df_list.append(sector)
    if len(top_sector):
        df_top_list.append(top_sector)
    prev_sector = curr_sector.copy()
    prev_top_sector = curr_top_sector.copy()

# %%

for df in df_list:
    if len(df) > 20:
        print(df.head())

len(df_list)
# %%
for df in df_top_listu:
    if len(df) > 20:
        print(df.head())

len(df_top_list)
# %%
db.trade_dates(start="5/25/2020")[:10]

# %%
df_list[0]
df_list[1]
df_list[2]
df_list[-9]
df_list[-8]
df_list[-7]
df_list[-6]
df_list[-5]
