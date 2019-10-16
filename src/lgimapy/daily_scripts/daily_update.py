from time import sleep

from lgimapy.daily_scripts import build_movers_sheets
from lgimapy.data import (
    Database,
    update_feathers,
    update_fed_funds,
    update_trade_dates,
)
from lgimapy.models import update_treasury_curve_dates


def main():
    """
    Run all daily scripts for updating database and creating reports.
    """
    check_datamart_quality()
    print()
    update_fed_funds()
    print("Updated Fed Funds\n")
    update_treasury_curve_dates()

    update_feathers()
    print()
    update_trade_dates()
    print()
    build_movers_sheets()
    print("Index Mover Report Complete\n")


def check_datamart_quality():
    db = Database()
    date = db.load_trade_dates()[-1]
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
            sleep(60)
            db.load_market_data()
            ix = db.build_market_index()
            n_bonds_prev = n_bonds
            n_bonds = len(ix.df)
            change = n_bonds - n_bonds_prev


if __name__ == "__main__":
    main()
