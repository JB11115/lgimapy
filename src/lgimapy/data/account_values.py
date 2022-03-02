from lgimapy.data import Database


def update_account_market_values():
    """
    Update account values for each strategy.
    """
    db = Database()
    df = db.account_values()
    df.to_parquet(db.local("account_values.parquet"))


if __name__ == "__main__":
    update_account_market_values()
