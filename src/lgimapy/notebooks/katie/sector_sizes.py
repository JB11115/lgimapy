from datetime import datetime as dt

import pandas as pd

from lgimapy.data import Database
from lgimapy.utils import mkdir, root


def main():
    filename = f"{dt.today().strftime('%Y-%m-%d')}_Energy"
    sectors = [
        "INDEPENDENT",
        "REFINING",
        "OIL_FIELD_SERVICES",
        "INTEGRATED",
        "MIDSTREAM",
    ]
    save_largest_tickers_in_sector(sectors, filename)


def save_largest_tickers_in_sector(sector_list, fid):
    """
    Save a .csv of largest tickers in a list of strategies,
    broken out in total, bonds with 0-10 years until maturity,
    and bonds with 10+ years until maturity.
    """
    csv_dir = root("reports/sectors")
    mkdir(csv_dir)

    # Collect data.
    db = Database()
    db.load_market_data(local=True)
    ix_d = {}
    ix_d["Total"] = db.build_market_index(sector=sector_list, rating="IG")
    ix_d["0-10 yr"] = ix_d["Total"].subset(maturity=(None, 10))
    ix_d["10+ yr"] = ix_d["Total"].subset(maturity=(10, None))

    # Calculate total market value for each ticker.
    market_val_list = []
    for name, ix in ix_d.items():
        ticker_df = (
            ix.df.groupby(["Ticker"], observed=True)
            .sum()
            .reset_index()
            .set_index("Ticker")
        )
        market_val_list.append(
            ticker_df["MarketValue"].rename(f"Market Value ($M) {name}")
        )

    # Sort and save.
    df = (
        pd.concat(market_val_list, axis=1, sort=False)
        .sort_values("Market Value ($M) Total", ascending=False)
        .round(0)
    )
    df.to_csv(csv_dir / fid)
