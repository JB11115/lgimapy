from time import sleep

from lgimapy.daily_scripts import (
    build_issuer_change_report,
    make_credit_snapshots,
    update_credit_snapshots,
)
from lgimapy.data import (
    Database,
    update_bloomberg_data,
    update_market_data_feathers,
    update_account_market_values,
    update_treasury_oad_values,
    update_fed_funds,
    update_trade_dates,
    update_dealer_inventory,
    update_lgima_sectors,
    update_nonfin_spreads,
    update_rating_changes,
    update_strategy_overweights,
    update_hy_index_members,
    save_bond_wishlist,
)
from lgimapy.models.treasury_curve import update_treasury_curve_dates

# %%


def main():
    """
    Run all daily scripts for updating database and creating reports.
    """
    trade_dates = Database().load_trade_dates()
    check_datamart_quality(trade_dates)
    print()
    update_fed_funds(trade_dates)
    print("Updated Fed Funds\n")
    update_trade_dates()
    print()
    update_hy_index_members()
    print("Updated HY Index Flags\n")
    update_treasury_curve_dates()

    update_market_data_feathers()
    print()
    update_lgima_sectors()
    print("Updated LGIMA sectors\n")
    update_treasury_oad_values()
    print("Updated On-the-run Treasury OADs\n")
    update_account_market_values()
    print("Updated Account Market Values\n")
    update_bloomberg_data()
    print("Updated Bloomberg Data\n")
    update_dealer_inventory()
    print("Updated Dealer Inventory\n")
    update_rating_changes()
    print("Updated Rating Changes\n")
    build_issuer_change_report()
    print("Issuer Change Report Complete\n")
    make_credit_snapshots()
    update_credit_snapshots()
    print("Credit Snaphshots Complete\n")
    update_nonfin_spreads()
    print("Updated Nonfin Sub-Rating Spreads\n")
    save_bond_wishlist()
    # update_strategy_overweights()
    # print("Updated Strategy Overweights\n")


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
    db.load_market_data(date=date, local=False)
    len(db.df)
    ix = db.build_market_index()

    # Make sure data is fully populated.
    n_bonds = len(ix.df)
    change = 0
    while True:
        if n_bonds > 12_000 and change == 0:
            print("Datamart Quality Confirmed")
            break
        else:
            print(f"Currently only {n_bonds:,.0f} populated, waiting...")
            sleep(30)
            db.load_market_data(date=date, local=False)
            ix = db.build_market_index()
            n_bonds_prev = n_bonds
            n_bonds = len(ix.df)
            change = n_bonds - n_bonds_prev


if __name__ == "__main__":
    main()
