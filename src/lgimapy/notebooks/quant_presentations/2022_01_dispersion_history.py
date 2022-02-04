import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from lgimapy import vis
from lgimapy.data import Database

vis.style()
# %%

db = Database()
dates = db.date('MONTH_STARTS')

oas = []
std = []
for date in tqdm(dates):
    db.load_market_data(date=date)
    ix = db.build_market_index(in_stats_index=True)
    oas.append(ix.OAS().iloc[0])
    std.append(np.std(ix.df['OAS']))

# %%
df = pd.DataFrame({'oas': oas, 'std': std}, index=dates)
df = df[df.index >= '1/1/2002']
df = df[df.index != '4/1/2013']
df = df[df.index != '5/1/2013']
df = df[df.index != '6/1/2013']
df = df[df.index != '6/3/2013']


vis.plot_double_y_axis_timeseries(
    df['oas'].rename('US IG Index OAS (bp)'),
    df['std'].rename('US IG Std Dev of Spreads (bp)'),
    lw=3,
    alpha=0.8,
    color_left='k',
    color_right='dodgerblue',
    figsize=(10, 6),
)

vis.savefig('naive_dispersion_history')

# %%
fig, ax= vis.subplots(figsize=(10, 4))
vis.plot_timeseries(df['oas'], title='US IG Index OAS', color='k', ax=ax)
ax.grid(False)
vis.savefig('us_ig_oas')

# %%
db = Database()
db.load_market_data()
ix = db.build_market_index(in_stats_index=True, maturity=(10, None))
oas = ix.df['OAS']


# %%
randn = (np.random.randn(1000000) * np.std(oas)) + 108.217
fig, ax = vis.subplots(figsize=(8, 6))
kwargs = {'alpha': 0.4, 'fill': True, 'ax': ax, 'linewidth': 2}
sns.kdeplot(data=randn, color='k',  label='Normal Distribution', **kwargs)
sns.kdeplot(data=oas, color='dodgerblue', label='US IG Spreads', **kwargs)
ax.legend()
vis.savefig('oas_distribution')
