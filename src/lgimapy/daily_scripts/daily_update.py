from time import sleep

from lgimapy.daily_scripts import (
    build_movers_sheets,
    build_long_credit_snapshot,
)
from lgimapy.data import (
    Database,
    update_bloomberg_data,
    update_feathers,
    update_fed_funds,
    update_trade_dates,
    update_dealer_inventory,
)
from lgimapy.models import update_treasury_curve_dates


def main():
    """
    Run all daily scripts for updating database and creating reports.
    """
    trade_dates = Database().load_trade_dates()
    check_datamart_quality(trade_dates)
    print()
    update_fed_funds(trade_dates)
    print("Updated Fed Funds\n")
    update_treasury_curve_dates()

    update_feathers()
    print()
    update_trade_dates(trade_dates)
    print()
    update_bloomberg_data()
    print("Updated Bloomberg Data\n")
    update_dealer_inventory()
    print("Updated Dealer Inventory\n")
    build_movers_sheets()
    print("Index Mover Report Complete\n")
    build_long_credit_snapshot()
    print("Long Credit Snaphsot Complete\n")


def check_datamart_quality(dates):
    """
    Verify DataMart quality by ensuring a minimum of 17,000
    bonds are available for current day. If less than 17,000
    are available, wait one minute and retry until more
    than 17,000 are availble and the number has stabalized.

    Parameters
    ----------
    dates: List[datetime].
        List of all trade dates available in DataMart.
    """
    db = Database()
    date = dates[-1]
    print(date.strftime("%m/%d/%Y"))
    db.load_market_data(date=date)
    ix = db.build_market_index()

    # Make sure data is fully populated.
    n_bonds = len(ix.df)
    change = 0
    while True:
        if n_bonds > 17_000 and change == 0:
            print("Datamart Quality Confirmed")
            break
        else:
            print(f"Currently only {n_bonds} populated, waiting...")
            sleep(30)
            db.load_market_data(date=date)
            ix = db.build_market_index()
            n_bonds_prev = n_bonds
            n_bonds = len(ix.df)
            change = n_bonds - n_bonds_prev


if __name__ == "__main__":
    main()
