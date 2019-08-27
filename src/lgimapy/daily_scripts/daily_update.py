from lgimapy.daily_scripts import build_movers_sheets
from lgimapy.data import update_feathers, update_fed_funds, update_trade_dates


def main():
    """
    Run all daily scripts for updating database and creating reports.
    """
    update_trade_dates()
    print()
    build_movers_sheets()
    print()
    update_feathers()
    print()
    update_fed_funds()
    print("Updated Fed Funds\n")


if __name__ == "__main__":
    main()
