import pandas as pd

from lgimapy.data import Database


db = Database()
start = db.date('MARKET_START')

db.load_market_data(start=start)

rating_kwargs = {
    'BBB': {'in_stats_index': True, 'rating': ('BBB+', 'BBB-')},
    'BB': {'in_hy_stats_index': True, 'rating': ('BB+', 'BB-')},
    'B': {'in_hy_stats_index': True, 'rating': ('B+', 'B-')},
}

default_kwargs = {
    'financial_flag': 0
}

# %%
df_list = []
for rating, rating_kws in rating_kwargs.items():
    kws = {**rating_kws, **default_kwargs}
    ix = db.build_market_index(**kws)
    ix.df = ix.df[~ix.df['CouponRate'].isna()]
    ix.df = ix.df[~ix.df['YieldToWorst'].isna()]
    ix.df = ix.df[~ix.df['MarketValue'].isna()]
    df_list.append(ix.MEAN('CouponRate').rename(f"{rating}_Coupon"))
    df_list.append(ix.MEAN('YieldToWorst').rename(f"{rating}_YTW"))

df = pd.concat(df_list, axis=1).round(3)
df.to_csv('coupon_and_YTW_history_by_rating_bucket.csv')
