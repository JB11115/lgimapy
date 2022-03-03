from lgimapy.data import Database
from lgimapy.portfolios import AttributionIndex

# %%

db = Database()
print(help(AttributionIndex))

account = "P-LD"
# %%

# Load data for a specified period (by default uses most current data)
attr = AttributionIndex(account=account, start=db.date("YTD"))

print(attr)

# %%

# Common attribution analysis

print(attr.total())
print(attr.total(start=db.date("1D")))


# %%
print(attr.tickers())
print(attr.tickers(["BAC", "JPM", "MS", "GS"]))
print(
    attr.tickers(["BAC", "JPM", "MS", "GS"], start="1/10/2022", end="2/1/2022")
)

# You can also get a DataFrame of the best/worst tickers
print(attr.best_worst_df(attr.tickers(), n=6, prec=2))

# %%
print(attr.sectors())
print(attr.sectors(["ENERGY", "BANKS"], start=db.date("1W")))

# You can similarly look at best/worst of sectors (or any pd.Series)
print(attr.best_worst_df(attr.sectors(), n=3))

# %%
print(attr.market_segments())

# %%

# An AttributeIndex also as an :attr:`ix` component which allows
# you to slice and dice any way you would a regular :class:`Index`
# and perform any :class:`Index` methods.

# For example, say you want to see daily attribution of 30yr BAC bonds YTD
bac_attr_ix = attr.ix.subset(ticker="BAC", maturity=(28, 32))
print(bac_attr_ix.SUM("PnL"))  # Sum the PnL of the bonds daily

# %%
# And if you want to see just the best and worst bonds over the full period.
bac_attr_ix.add_bond_description()
bac_attr_ix

# %%
# Or maybe you want to see the CUSIP level attribution of these
# since yesterday.
bac_1D_attr_ix = attr.ix.subset(
    ticker="BAC", maturity=(28, 32), start=db.date("today")
)
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

attr = AttributionIndex(account, start=db.date("1W"))
otr_integrated_attr_ix = attr.ix.subset(
    **db.index_kwargs("INTEGRATED", rating=(None, "A-"))
).subset_on_the_runs()
# Add spread changes from last week.
change_cols = otr_integrated_attr_ix.add_change("OAS", "1W", db)
otr_integrated_attr_ix.add_bond_description()
cols = [
    "BondDescription",
    "OAS",
    *change_cols,
    "PnL",
]
print(otr_integrated_attr_ix.df[cols].sort_values("OAS_abs_Change_1W"))


# or just look at the best and worst performing bonds and thier PnL
# which can be accomplished easily since we can pass any pd.Series
# into AttributionIndex.best_worst_df() method
print(
    AttributionIndex.best_worst_df(
        otr_integrated_attr_ix.df.set_index("BondDescription")["PnL"]
    )
)
