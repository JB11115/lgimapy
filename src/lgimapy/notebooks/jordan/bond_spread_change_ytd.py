import pandas as pd

from lgimapy.data import Database

# %%

db = Database()
db.load_market_data(date=db.date('YTD'))
prev_ix = db.build_market_index(in_stats_index=True, maturity=(10, None))
db.load_market_data()
curr_ix = db.build_market_index(isin=prev_ix.isins)

# %%
cols = ['Ticker', 'CouponRate', 'MaturityDate', 'MarketValue']
cols = ['MarketValue']
df = curr_ix.ticker_df[cols].copy()
df['Current_OAS'] = curr_ix.ticker_df['OAS']
df['Prev_YE_OAS'] = prev_ix.ticker_df['OAS']
df.dropna(inplace=True)
df['YTD_chg_OAS'] = df['Current_OAS'] - df['Prev_YE_OAS']
df.sort_values("YTD_chg_OAS", inplace=True)
# %%

ix_chg = curr_ix.OAS().iloc[0] - prev_ix.OAS().iloc[0]
ix_chg

# %%
df_outperform = df[df['YTD_chg_OAS'] <= ix_chg].copy()
pct_outperform = len(df_outperform) / len(df)
pct_mv_outperform = df_outperform['MarketValue'].sum() / df['MarketValue'].sum()
pct_outperform
pct_mv_outperform
