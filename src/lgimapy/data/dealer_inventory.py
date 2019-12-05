import numpy as np
import pandas as pd

from lgimapy.bloomberg import bdh
from lgimapy.utils import root


def update_dealer_inventory():
    """
    Update dealer inventory for IG and HY bonds.
    """
    # Get old data for all of HY.
    df_hy_old = bdh("PDPPCBIG", "Index", fields="PX_LAST", start="1/1/1998")
    # Get new subsetted data for HY and sum for total.
    hy_tickers = ["PDPPBMWQ", "PDPPIAXE", "PDPPHFNH", "PDPPXUAE"]
    df_hy_new = bdh(hy_tickers, "Index", fields="PX_LAST", start="1/1/1998")
    df_hy_new["PX_LAST"] = np.sum(df_hy_new, axis=1)
    # Combine new and old HY data.
    df_hy = pd.concat([df_hy_old, df_hy_new], join="inner", sort=True)
    df_hy.columns = ["HY"]

    # Get data for IG.
    ig_tickers = ["PDPPC13-", "PDPPC13+", "PDPPOTBU", "PDPPAILU", "PDPPOIAN"]
    df_ig = bdh(ig_tickers, "Index", fields="PX_LAST", start="1/1/1998")
    # Remove overlapping subsets in peiord with overlap.
    df_ig.loc[~df_ig["PDPPC13+"].isna(), "PDPPOTBU"] = 0
    df_ig["IG"] = np.sum(df_ig, axis=1)
    # Fix erroneous point in data.
    df_ig.loc[pd.to_datetime("5/8/2019"), "IG"] = np.abs(
        df_ig.loc[pd.to_datetime("5/8/2019"), "IG"]
    )

    # Combine old and new data and save.
    df = pd.concat([df_hy, df_ig["IG"]], axis=1, sort=True) / 1e3

    fid = root("data/dealer_inventory.csv")
    df.to_csv(fid)


if __name__ == "__main__":
    update_dealer_inventory()
