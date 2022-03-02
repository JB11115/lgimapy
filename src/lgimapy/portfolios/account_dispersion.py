import pandas as pd

from lgimapy.data import Database, Portfolio

# %%
def get_account_dispersion(strategy, date=None):
    db = Database()
    if date is None:
        date = db.date("today")
    accounts = db._strategy_to_accounts[strategy]
    df_list = []
    for account in accounts:
        port = Portfolio(date, name=account, class_name="Account")
        df_list.append(port.stored_properties_history_df.loc[date])

    return pd.DataFrame(df_list, index=accounts)


# %%
def main():
    # %%
    strategy = "US Long Credit"
    date = None
    df = get_account_dispersion(strategy)
    strat = Portfolio(date, name=strategy, class_name="Strategy")
    df.to_csv(f"{strat.fid}_account_dispersion.csv")
