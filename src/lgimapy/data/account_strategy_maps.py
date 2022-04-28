import warnings
from collections import defaultdict

import pandas as pd

from lgimapy.data import Database
from lgimapy.utils import dump_json, mkdir

# %%


def update_account_strategy_maps():
    db = Database()
    account_strategy_map_list = get_account_strategy_map_list(db)
    update_map_files(account_strategy_map_list, db)


def get_account_strategy_map_list(db):
    account_dfs = get_account_dfs(db)
    account_strategy_map_list = []
    for df in account_dfs.values():
        account_strategy_map_list = update_map_list_for_single_account(
            df, account_strategy_map_list
        )
    return account_strategy_map_list


def get_account_dfs(db):
    db = Database()
    sql = f"""\
        SELECT\
            BloombergId,\
            DateOpen,\
            DateClose,\
            s.[Name] as PMStrategy\

        FROM [LGIMADatamart].[dbo].[DimAccount] a (NOLOCK)\
        INNER JOIN LGIMADatamart.dbo.DimStrategy s (NOLOCK)\
            ON a.StrategyKey = s.StrategyKey\
        WHERE DateEnd='9999-12-31'
        ORDER BY DateOpen
        """
    sql_df = db.query_datamart(sql)

    return {account: df for account, df in sql_df.groupby("BloombergId")}


def to_datetime(date):
    return None if date is None else pd.to_datetime(date)


def update_map_list_for_single_account(df, master_list):
    account_strategy_d = {}
    first_row = True
    for _, row in df.iterrows():
        open = to_datetime(row["DateOpen"])
        close = to_datetime(row["DateClose"])
        account = row["BloombergId"]
        strategy = row["PMStrategy"]

        if first_row:
            account_strategy_d["account"] = account
            account_strategy_d["strategy"] = strategy
            account_strategy_d["open"] = open
            prev_strategy = strategy  # avoid error when checking for change
            first_row = False

        if strategy != prev_strategy:
            # Close the previous account/strategy map with
            # previous close date. Add it to the master list.
            # min(prev_close, open) is used due to database input errors
            account_strategy_d["close"] = min(prev_close, open)
            master_list.append(account_strategy_d.copy())

            # Start a new account/strategy map.
            account_strategy_d = {}
            account_strategy_d["account"] = account
            account_strategy_d["strategy"] = strategy
            account_strategy_d["open"] = open

        # Store values for next iteration.
        prev_close = close
        prev_strategy = strategy

    # The loop finished. Add the last close date to the current
    # map and add the account/strategy to the master list.
    account_strategy_d["close"] = close
    master_list.append(account_strategy_d.copy())

    return master_list


def update_map_files(account_strategy_map_list, db):
    date_account_strategy_map = defaultdict(dict)
    date_strategy_account_map = defaultdict(lambda: defaultdict(list))

    for d in account_strategy_map_list:
        dates = db.trade_dates(start=d["open"], end=d["close"])
        account = d["account"]
        strategy = d["strategy"]
        for date in dates:
            date_account_strategy_map[date][account] = strategy
            date_strategy_account_map[date][strategy].append(account)

    mkdir(db.local("portfolios/account_strategy_maps"))
    mkdir(db.local("portfolios/strategy_account_maps"))
    for date, d in date_account_strategy_map.items():
        fid = db.local(f"portfolios/account_strategy_maps/{date:%Y-%m-%d}.json")
        if not fid.exists():
            dump_json(d, full_fid=fid)

    for date, d in date_strategy_account_map.items():
        fid = db.local(f"portfolios/strategy_account_maps/{date:%Y-%m-%d}.json")
        if not fid.exists():
            dump_json(d, full_fid=fid)
