from lgimapy.data import Database

db = Database()
db.load_market_data()

ixs = {
    'BBB': db.build_market_index(
        in_stats_index=True,
        rating=("BBB+", "BBB-"),
        dirty_price=(None, 80),
        maturity=(10, None),
    ),
    "BB": db.build_market_index(
        in_H4UN_index=True,
        rating=("BB+", "BB-"),
        dirty_price=(None, 85),
        maturity=(8, None),
    ),
}


# %%

def bond_descriptions(df):
    return list(
        (
            "*  "
            + df["Ticker"].astype(str)
            + " "
            + df["CouponRate"].apply(lambda x: f"{x:.2f}")
            + " "
            + df["MaturityDate"].apply(lambda x: f"`{x:%y}").astype(str)
            + "    ISIN: "
            + df["ISIN"].astype(str)
        ).values
    )

for rating, ix in ixs.items():
    print(f"{rating}:")
    for bond in bond_descriptions(ix.df.sort_values("MaturityYears")):
        print(bond)
    print('\n')
