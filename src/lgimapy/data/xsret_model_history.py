from collections import defaultdict

import pandas as pd
import psutil
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.models import XSRETPerformance
from lgimapy.utils import root, restart_program


# %%
def main():
    fid = root("data/xsret_model_history.parquet")
    saved_df, last_saved_date = read_saved_data(fid)
    saved_df.index[-1].strftime("%#m/%d/%Y")
    db = Database()
    start_date = db.date("2m", last_saved_date)
    try:
        end_date = db.date("+4m", last_saved_date)
    except IndexError:
        end_date = None
    db.load_market_data(start=start_date, end=end_date)
    model_maturities = {
        "Long Credit": (10, None),
        "5-10 yr Credit": (5, 10),
    }
    dates = db.trade_dates(exclusive_start=last_saved_date)
    if not dates:
        return  # already up to date

    d = defaultdict(list)
    mod = XSRETPerformance(db)
    for date in tqdm(dates):
        for name, maturity in model_maturities.items():
            mod.train(forecast="1m", date=date, maturity=maturity)
            d[f"MAE_{name}"].append(mod.MAE())
            d[f"MAD_{name}"].append(mod.MAD())

    new_df = pd.DataFrame(d, index=dates)
    updated_df = pd.concat((saved_df, new_df), axis=0, sort=True)
    updated_df.to_parquet(fid)


def read_saved_data(fid):
    try:
        df = pd.read_parquet(fid)
    except (FileNotFoundError, OSError):
        df = pd.DataFrame()
        last_date = pd.to_datetime("1/1/2005")
    else:
        # Get last scraped date and remove blank row.
        last_date = df.index[-1]
    return df, last_date


# %%

if __name__ == "__main__":
    main()

# %%
