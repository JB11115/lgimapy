import pandas as pd

from lgimapy.data import Database, Portfolio

# %%
def get_account_dispersion(strategy, date=None):
    db = Database()
    if date is None:
        date = db.date("today")
    accounts = db._account_strategy_map(date).keys()
    df_list = []
    for account in accounts:
        port = db.load_portfolio(date, account=account, empty=True)
        df_list.append(port.stored_properties_history_df.loc[date])

    return pd.DataFrame(df_list, index=accounts)


# %%
def main():
    # %%
    strategy = "US Long Credit"
    date = None
    df = get_account_dispersion(strategy)
    strat = db.load_portfolio(date, strateg=strategy, empty=True)
    df.to_csv(f"{strat.fid}_account_dispersion.csv")
