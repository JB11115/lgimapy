import warnings

import joblib
import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.utils import root, mkdir, load_json, to_list

# %%


def main():
    # override_from_date = Database().date("portfolio_start")
    override_from_date = None
    override_to_date = None
    # specific_strategies = "US Credit Plus"
    specific_strategies = None
    update_portfolio_history(
        override_from_date, override_to_date, specific_strategies
    )


def update_portfolio_history(
    override_from_date=None, override_to_date=None, specific_strategies=None
):
    data_dir = root("data/portfolios/history")
    mkdir(data_dir / "Strategy")
    mkdir(data_dir / "Account")
    fid = data_dir / "completed_dates.parquet"

    Database().update_portfolio_account_data()
    for date in tqdm(
        get_dates_to_update(fid, override_from_date, override_to_date)
    ):
        update_date(date, fid, override_from_date, specific_strategies)


def get_dates_to_update(fid, override_from_date, override_to_date):
    db = Database()
    # Use override date if provided.
    if override_from_date is not None:
        return db.trade_dates(start=override_from_date, end=override_to_date)

    # Check if any dates have been scraped, otherwise use all dates.
    all_dates = db.trade_dates(start=db.date("PORTFOLIO_START"))
    try:
        scraped_dates = pd.read_parquet(fid)
    except FileNotFoundError:
        return all_dates

    # Otherwise use remaining dates.
    last_date = scraped_dates["date"].iloc[-1]
    return db.trade_dates(exclusive_start=last_date)


def update_strategy(date, strategy, override_from_date):
    db = Database()
    if override_from_date is None:
        # See if date has already been scraped for this strategy.
        # Skip the strategy if it has.
        strategy_fid = db.fid_safe_str(strategy)
        fid = root(f"data/portfolios/history/Strategy/{strategy_fid}.parquet")
        try:
            scraped_dates = pd.read_parquet(fid)
        except FileNotFoundError:
            pass
        else:
            if date in scraped_dates.index:
                return
    try:
        port = db.load_portfolio(strategy=strategy, date=date)
    except ValueError:
        # No data.
        return

    port.drop_empty_accounts()
    if not len(port.df):
        return
    try:
        port.save_stored_properties()
    except Exception as e:
        print(port)
        raise e


def update_date(date, fid, override_from_date, specific_strategies):
    all_strategies = set(load_json("strategy_accounts"))
    ignored_strategies = {
        "LDI No Benchmark",
        "Custom Management No Benchmark",
        "Custom Management No Benchmark - MDLZSTRP",
        "Custom BNM - MBRF",
        "Absolute Return - LIB150",
        "Absolute Return LIB350",
        "US High Yield",
        "Global Corporate IBoxx USD",
        "US Strips 15+ Yr",
        "US Strips 20+ Yr",
        "US Treasury 9+ Yr Custom Weighted",
        "US Treasury Long",
        "OLIN CUSTOM STRIPS - OPEB",
        "OLIN CUSTOM STRIPS - Pension",
        "RBS BNMs",
        "Global Agg USD Securitized Passive",
        "Global Corporate IBoxx USD",
        "BNM - US Long A+ Credit",
        "80% US A or Better LC/20% US BBB LC",
        "80% US Credit/20% 7-10 yr Treasury",
        "BNM - US Long A+ Credit",
        "Intermediate TIPS",
        "US Government: Intermediate",
        "US Government: Long",
        "BNM - ICE BofA US Non-Financial Index",
    }
    large_strategies = {
        "US Long Credit",
        "US Credit",
        "US Long Government/Credit",
    }
    strategies = list(all_strategies - ignored_strategies - large_strategies)
    for strategy in large_strategies:
        strategies.insert(0, strategy)

    if specific_strategies is not None:
        strategies = to_list(specific_strategies, dtype=str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        joblib.Parallel(n_jobs=6)(
            joblib.delayed(update_strategy)(date, strategy, override_from_date)
            for strategy in strategies
        )
        # for strategy in strategies:
        #     update_strategy(date, strategy, override_from_date)

    # Update scraped dates.
    try:
        scraped_dates = pd.read_parquet(fid)
    except FileNotFoundError:
        # Start date file
        pd.DataFrame({"date": [date]}).to_parquet(fid)
    else:
        date_list = sorted(list(set(scraped_dates["date"]) | set([date])))
        pd.DataFrame({"date": date_list}).to_parquet(fid)


# %%
if __name__ == "__main__":
    main()
