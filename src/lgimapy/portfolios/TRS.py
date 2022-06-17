import numpy as np
import pandas as pd

from lgimapy.bloomberg import bds, fmt_bbg_dt, id_to_isin
from lgimapy.data import Database

# %%


db = Database()
start = "5/16/2022"
end = '5/20/2022'
db.load_market_data(start=start, end=end)

port = db.load_portfolio(account='LIB350', date=start)
port.derivatives_df.iloc[0]['P_Notional']

# %%
dates = db.trade_dates(start=start, end=end)
tret_ts_list = []
for date in dates:
    df = bds(
        "LGIMA1",
        "Index",
        "INDX_MWEIGHT_HIST",
        ovrd={"END_DATE_OVERRIDE": fmt_bbg_dt(date)},
    )
    df.columns = ['BBG_ID', 'Weight']
    df['ISIN'] = id_to_isin(df['BBG_ID'])
    index_weights = dict(zip(df['ISIN'], df['Weight'] / 100))
    ix = db.build_market_index(date=date, isin=index_weights.keys())
    ix.df['P_Weight'] = -ix.df['ISIN'].map(index_weights)
    ix.df['BM_Weight'] = 0

    port_tret = (ix.df["TRet"] * ix.df["P_Weight"]).sum()
    bm_tret =  (ix.df["TRet"] * ix.df["BM_Weight"]).sum()
    tret = port_tret - bm_tret
    tret_ts_list.append(tret)

tret_ts = pd.Series(tret_ts_list, index=dates)
tret_ts

# %%
1e4* (np.prod(1 + tret_ts) - 1)
