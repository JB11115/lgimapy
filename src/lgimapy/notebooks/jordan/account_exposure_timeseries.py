from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database

# %%
account = 'BCBSIC'


# %%
db = Database()
dates = db.trade_dates(start='1/1/2020')
d = defaultdict(list)
for date in tqdm(dates):
    acnt = db.load_portfolio(account=account, date=date)
    sub = acnt.subset(rating=('BBB+', 'BBB-'))
    d['OAD Overweight'].append(sub.oad())
    d['MV Overweight'].append(sub.credit_pct() - sub.bm_credit_pct())


df = pd.DataFrame(d, index=dates)
df.to_csv(f'{account}_BBB_exposure.csv')
