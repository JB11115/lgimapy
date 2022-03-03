from lgimapy.data import Database
from lgimapy.portfolios import AttributionIndex

# %%

db = Database()
print(help(AttributionIndex))

account = 'P-LD'
# %%

# Load data for a specified period (by default uses most current data)
attr = AttributionIndex(account=account, start=db.date('YTD'))

print(attr)

# %%

# Common attribution analysis

print(attr.total())

# %%
print(attr.tickers())
print(attr.tickers(['BAC', 'JPM', 'MS', 'GS']))
print(attr.tickers(['BAC', 'JPM', 'MS', 'GS'], start='1/10/2022', end='2/1/2022'))

# %%
print(attr.sectors())
print(attr.sectors(['ENERGY', 'BANKS'], start=db.date('1W')))

# %%
print(attr.market_segments())

# %%

# An AttributeIndex also as an :attr:`ix` component which allows
# you to slice and dice any way you would a regular :class:`Index`
# and perform any :class:`Index` methods.

# For example, say you want to see attribution of 30yr BAC bonds YTD
bac_attr_ix = attr.ix.subset(ticker='BAC', maturity=(28, 32))
print(bac_attr_ix.SUM('PnL'))

# %%
# Or maybe you want to see the CUSIP level attribution of these
# since yesterday.
bac_1D_attr_ix = attr.ix.subset(ticker='BAC', maturity=(28, 32), start=db.date('today'))
cols = [
    "Ticker",
    "CouponRate",
    "MaturityDate",
    "OAS",
    "PnL",
]
print(bac_1D_attr_ix.df[cols])

# %%

# Suppose you only want to see the on the run A-rated integrated spreads,
# and see thier spread change since last week.

attr = AttributionIndex(account, start=db.date('1W'))
otr_integrated_attr_ix = attr.ix.subset(**db.index_kwargs('INTEGRATED', rating=(None, 'A-'))).subset_on_the_runs()
# Add spread changes from last week.
otr_integrated_attr_ix.add_change("OAS", '1W', db)

cols = [
    "Ticker",
    "Issuer",
    "MaturityDate",
    "CouponRate",
    "OAS",
    "OAS_1W",
    "OAS_abs_Change_1W",
    "OAS_pct_Change_1W",
    "PnL",
]
print(otr_integrated_attr_ix.df[cols].sort_values('OAS_abs_Change_1W'))
