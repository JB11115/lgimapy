import pandas as pd

from lgimapy.data import Database
from lgimapy.utils import Time

# %%

with Time():
    db = Database()
    db.load_market_data(start='1/1/2000', local=True)

# %%
ind = db.build_market_index(**db.index_kwargs('INDUSTRIALS'), rating=('AAA', 'A-'), maturity=(7, 10))
ind_oas = ind.OAS()

utes = db.build_market_index(**db.index_kwargs('UTILITY'), maturity=(7, 10))
utes_oas =  utes.OAS()

banks = db.build_market_index(**db.index_kwargs('SIFI_BANKS'), maturity=(7, 10))
banks_oas = banks.OAS()


df = pd.concat([ind_oas.rename('Industrial_A_rated_OAS'), utes_oas.rename('Utes_OAS'), banks_oas.rename("Banks_OAS")], axis=1)
df.to_csv('spread_ts.csv')

# %%
from lgimapy import vis
vis.style()


vis.plot_multiple_timeseries(df)
vis.savefig('spreads')
vis.show()
