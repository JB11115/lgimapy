import gc

import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.utils import to_list, root, mkdir

# %%


def load_data(fid):
    try:
        return pd.read_parquet(fid)
    except (FileNotFoundError, OSError):
        return pd.DataFrame()


def update_strategy_overweights():
    # strategies = ["US Credit", "US Long Credit"]
    strategies = ["US Long Credit"]
    # Find tickers to scrape.
    db = Database()
    fid_dir = root("data/strategy_overweights")
    mkdir(fid_dir)
    for date in tqdm(db.trade_dates(start=db.date("PORTFOLIO_START"))[::-1]):
        print(date)
        for strategy in strategies:
            # See if data exists for current date and strategy.
            # If it does move on, else compute results and append.
            strat_fid = strategy.replace(" ", "_")
            fid = f"{strat_fid}_tickers.parquet"
            old_df = load_data(fid_dir / fid)
            if date in old_df.index:
                continue

            # Get data for current day and append to file.
            strat = db.load_portfolio(strategy=strategy, date=date)
            date_df = strat.ticker_overweights().rename(date).to_frame().T
            updated_df = pd.concat([old_df, date_df], sort=False).sort_index()
            updated_df.to_parquet(fid_dir / fid)

            # Garbage collect to reduce memory issues.
            del strat
            gc.collect()


if __name__ == "__main__":
    update_strategy_overweights()
