from lgimapy import vis
from lgimapy.data import Database

vis.style()
# %%

db = Database()

year = 2010
start = f'1/1/{year}'
end = f'1/1/{year + 1}'

# Old data.
db.load_market_data(start=start, end=end, local=True)
ix_old = db.build_market_index(in_returns_index=True)
oas_old = ix_old.market_value_weight('OAS').rename('Old Data')

# New data.
db.load_market_data(start=start, end=end)
ix_new = db.build_market_index(in_returns_index=True)
oas_new = ix_new.market_value_weight('OAS').rename('New Data')

# Bloomberg data.
oas_bbg = db.load_bbg_data('US_IG', 'OAS', start=start, end=end).rename('BBG Data')

# %%

vis.plot_multiple_timeseries(
    [oas_old, oas_new, oas_bbg], xtickfmt='auto', alpha=0.5
)
vis.show()
