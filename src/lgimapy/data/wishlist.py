import pandas as pd

from lgimapy.bloomberg import bdp
from lgimapy.data import Database
from lgimapy.utils import root

# %%
def save_bond_wishlist():
    """
    Save bonds to be added to small IG.

    Currently consists of bonds from any BAML HY index
    that are missing from the database.
    """
    fid = root("data/bond_wishlist.xlsx")
    # Get all ISINs currently in database.
    db = Database()
    db.load_market_data()
    database_isins = set(db.df["ISIN"])

    # Get all HY index ISINs.
    isin_wishlist = set()
    indexes = ["H4UN", "H0A0", "HC1N", "HUC2", "HUC3"]
    for index in indexes:
        # Load flag data and subset to correct dates.
        index_fid = root(f"data/index_members/{index}.parquet")
        flag_df = pd.read_parquet(index_fid)
        current_isins = flag_df.iloc[-1, :]
        current_isins = current_isins[current_isins == 1]
        isin_wishlist.update(current_isins.index)

    # Find missing ISINs, convert to CUSIPs, and save.
    bad_isins = set(["US87612BBP67"])
    missing_isins_list = list(isin_wishlist - database_isins - bad_isins)
    missing_isins = pd.Series(missing_isins_list).to_frame()
    missing_isins.to_excel(fid)
    print(f"Updated Bond Wishlist: {len(missing_isins)} bonds")


# %%
if __name__ == "__main__":
    save_bond_wishlist()
