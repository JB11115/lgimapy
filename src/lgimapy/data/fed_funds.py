import pandas as pd

from lgimapy.bloomberg import bdh
from lgimapy.data import Database
from lgimapy.utils import root

import matplotlib.pyplot as plt


def update_fed_funds():
    """Update fed funds data to current date in database."""
    # Scrape fed funds data.
    dates = Database().load_trade_dates()
    df = bdh("FDTRMID", "Index", fields="PX_MID", start=dates[0])

    # Include all possible dates, filling scraped dates forward.
    fed_funds_df = pd.DataFrame(index=pd.date_range(df.index[0], df.index[-1]))
    fed_funds_df.loc[df.index, "PX_MID"] = df["PX_MID"].values
    fed_funds_df.fillna(method="ffill", inplace=True)

    # Save.
    fid = root("data/fed_funds.csv")
    fed_funds_df.to_csv(fid)


if __name__ == "__main__":
    update_fed_funds()
