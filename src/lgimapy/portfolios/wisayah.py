from lgimapy.portfolios import AttributionIndex

account = 'SALD'
attr = AttributionIndex(account, start=db.date("WTD"))

print('\n\nTickers')
print(attr.best_worst_df(attr.tickers()))

print('\n\nSectors')
print(attr.best_worst_df(attr.sectors()))

print('\n\nMarket Segments')
print(attr.best_worst_df(attr.market_segments()))
