import pandas as pd

from lgimapy.data import Database

d = {}
markets = ["US", "EUR", "GBP"]
for market in markets:
    db = Database(market=market)
    db.load_market_data()
    if market == "US":
        energy_ix = db.build_market_index(
            **db.index_kwargs(
                "ENERGY",
                in_any_index=True,
                unused_constraints=["in_stats_index", "OAS"],
            )
        )
        energy_issuers = energy_ix.issuers
    else:
        energy_df = db.df[db.df["SectorLevel3"] == "OIL_AND_GAS"]
        energy_issuers = sorted(energy_df["Issuer"].dropna().unique())
    d[market] = energy_issuers


df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
df.to_csv("Energy_Issuers.csv")
