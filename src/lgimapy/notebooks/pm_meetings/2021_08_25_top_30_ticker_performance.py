import pandas as pd

from lgimapy import vis
from lgimapy.data import Database

vis.style()

# %%
db = Database()
db.load_market_data(start='6/1/2019')

# %%
df_list = []
ratings = ['A', 'BBB']
for rating in ratings:
    for basket in ['', 'EX_']:
        ix_name = f"{rating}_NON_FIN_{basket}TOP_30"
        kwargs = db.index_kwargs(ix_name)
        ix = db.build_market_index(**kwargs)
        ix_corrected = ix.drop_ratings_migrations()
        oas = ix_corrected.get_synthetic_differenced_history("OAS")
        df_list.append(oas.rename(kwargs['name']))

df = pd.concat(df_list, axis=1)
# %%
colors = ['navy', 'navy', 'darkorchid', 'darkorchid']
linestyles = ['-', '--', '-', '--']

fig, ax = vis.subplots()
for col, color, ls in zip(df.columns, colors, linestyles):
    if col.startswith('A'):
        continue
    vis.plot_timeseries(
        df[col],
        color=color,
        ls=ls,
        label=col,
        lw=2,
        ylabel='OAS',
        ax=ax
    )
vis.show()
# %%
w = 40
beta_df = (df.rolling(w).std() / df.rolling(w).mean()).dropna()
beta_ratio_df = pd.DataFrame()
for rating in ratings:
    col = '{} Non-Fin {}Top 30'.format
    beta_ratio_df[rating] = beta_df[col(rating, "")] / beta_df[col(rating, 'ex ')]

fig, ax = vis.subplots()
for col, color in zip(ratings, ['navy', 'skyblue']):
    vis.plot_timeseries(
        beta_ratio_df[col],
        color=color,
        label=col,
        lw=2,
        ylabel='Beta Ratio (Top 30 / Ex Top 30)',
        ax=ax
    )
vis.savefig('top_30_beta_ratio')
