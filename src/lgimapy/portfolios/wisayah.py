from lgimapy.data import Database
from lgimapy.portfolios import AttributionIndex


account = "SALD"
db = Database()
attr = AttributionIndex(account, start=db.date("WTD"))

print(f"\n\nTotal: {attr.total():.1f}")

print("\n\nTickers")
print(attr.best_worst_df(attr.tickers()))

print("\n\nSectors")
print(attr.best_worst_df(attr.sectors()))

print("\n\nMarket Segments")
print(attr.best_worst_df(attr.market_segments()))
