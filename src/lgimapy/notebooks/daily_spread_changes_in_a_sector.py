from lgimapy.data import Database

# %%
sector = 'LIFE_SUB'

# Find OAS change for all bonds in sector.
db = Database()
db.load_market_data(start=db.date('yesterday'))
ix = db.build_market_index(**db.index_kwargs(sector))
oas_df = ix.get_value_history("OAS").T
oas_df['diff'] = oas_df.iloc[:, -1] - oas_df.iloc[:, 0]

# Combine with today's spread and other useful columns.
cols = ['Ticker', 'Issuer', 'CouponRate', 'MaturityDate', 'OAS', 'OAS_1D_Change',]
ix_today = ix.subset(date=db.date('today'))
ix_today.df['OAS_1D_Change'] = oas_df['diff'].round(0)
df = ix_today.df.dropna(subset=['OAS_1D_Change']).rename_axis(None)
df['OAS'] = df['OAS'].round(0)
df = df[cols].sort_values('OAS_1D_Change', ascending=False)

# %%
df
