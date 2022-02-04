import pandas as pd

from lgimapy.data import Database

# %%
db = Database()

strategies = [
    "US Long Credit",
    "US Credit",
]

for strategy in strategies:
    port = db.load_portfolio(strategy=strategy)
    utes = port.subset(**db.index_kwargs('UTILITY'))
    df = utes.issuer_overweights().rename("OAD OW").to_frame().rename_axis(None)
    df.to_csv(f"{port.fid}_Utility_Issuer_Overweights.csv")
