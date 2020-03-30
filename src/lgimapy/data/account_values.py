from lgimapy.data import Database
from lgimapy.utils import root


def update_account_market_values():
    """
    Update account values for each strategy.
    """
    df = Database().account_values()
    df.to_parquet(root('data/account_values.parquet'))
