from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.data import Database

# %%
db = Database()
port = db.load_portfolio(account='SICHY')
bb_port = port.subset(rating=('B+', 'B-'), yield_to_worst=(0, None))

# %%
def get_yield(df, col='BM_Weight'):
    return (df['YTW'] * df[col]).sum() / df[col].sum()

bb_yield = get_yield(bb_port.df)
bb_yield

bb_bm_pct = bb_port.df['BM_Weight'].sum()
bb_p_pct = bb_port.df['P_Weight'].sum()
bb_bm_pct
bb_p_pct

bb_port_inside = bb_port.subset(yield_to_worst=(0, bb_yield))
bb_port_inside.df['P_Weight'].sum() / bb_p_pct
bb_port_inside.df['BM_Weight'].sum() / bb_bm_pct

len(bb_port.tickers)
df = bb_port.df.copy()
df['BM_Weight'] = df['BM_Weight'].fillna(0)
df[(df['P_Weight'] > 0) & (df['BM_Weight'] == 0)].to_csv('SICHY_owned_BB_ex-index.csv')

# %%

d = defaultdict(list)
sectors = db.HY_sectors()


for sector in sectors:
    kws = db.index_kwargs(sector, source='baml')
    sector_port = bb_port.subset(**kws)
    sector_bm_size = sector_port.df['BM_Weight'].sum()
    sector_p_size = sector_port.df['P_Weight'].sum()
    sector_port_inside = sector_port.subset(yield_to_worst=(0, bb_yield))
    sector_inside_bm_size = sector_port_inside.df['BM_Weight'].sum()
    sector_inside_p_size = sector_port_inside.df['P_Weight'].sum()
    d['Sector'].append(kws['name'])
    d['BM Yield'].append(get_yield(sector_port.df))
    d['Port Yield'].append(get_yield(sector_port.df, col='P_Weight'))
    d['% of BM sector inside avg B yield'].append(sector_inside_bm_size / sector_bm_size)
    d['% of Port sector inside avg B yield'].append(sector_inside_p_size / sector_p_size)
    d['BM Weight'].append(sector_port.df['BM_Weight'].sum())
    d['Port Weight'].append(sector_port.df['P_Weight'].sum())


df = pd.DataFrame(d).set_index('Sector', drop=True).rename_axis(None).sort_values("BM Yield")
df.to_csv('B_sector_yields_SICHY.csv')
df['BM Weight'].sum()
df['Port Weight'].sum()


# %%
d = defaultdict(list)
for sector in sectors:
    kws = db.index_kwargs(sector, source='baml')
    sector_port = bb_port.subset(**kws)
    for ticker in sector_port.tickers:
        ticker_port = sector_port.subset(ticker=ticker)
        d['Sector'].append(kws['name'])
        d['Ticker'].append(ticker)
        d['BM Yield'].append(get_yield(ticker_port.df))
        d['Portfolio Yield'].append(get_yield(ticker_port.df, col='P_Weight'))
        d['BM Weight'].append(ticker_port.df['BM_Weight'].sum())
        d['Port Weight'].append(ticker_port.df['P_Weight'].sum())



df = pd.DataFrame(d).set_index('Sector', drop=True).rename_axis(None)
df.to_csv('B_ticker_yields_SICHY.csv')
df

bb_yield
