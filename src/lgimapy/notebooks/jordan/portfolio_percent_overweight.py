from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database

# %%

account = 'P-LD'
ticker = 'PEMEX'

db = Database()
dates = db.trade_dates(start='1/1/2018')
d = defaultdict(list)
ix = []
for date in tqdm(dates):
    try:
        acnt = db.load_portfolio(account=account, date=date, market_cols=None)
    except ValueError:
        continue

    d['OAD_OW'].append(acnt.ticker_df.loc[ticker, 'OAD_Diff'])
    d['Port_Weight_OW'].append(acnt.ticker_df.loc[ticker, 'Weight_Diff'])
    ix.append(date)

df = pd.DataFrame(d, index=ix)
df = df[df.index != pd.to_datetime('8/31/2018')]
df.to_csv(f'{ticker}_OW.csv')
df
# %%
import lgimapy.vis as vis

vis.style()
vis.plot_multiple_timeseries(df)
vis.show()

db.load_bbg_data('US_IG_10+', 'OAS', start='4/1/2020')
