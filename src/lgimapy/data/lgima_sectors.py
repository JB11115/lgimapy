import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database, BondBasket
from lgimapy.utils import dump_json, load_json, mkdir, root

# %%
def save_lgima_sectors(date):
    """
    Save .json mapping file of CUSIPs to LGIMA sector
    for each CUSIP on giving date.
    """

    db = Database()
    df = db.load_market_data(date=(date), clean=False, ret_df=True)
    basket = BondBasket(df)

    industrial_sectors = {
        "CHEMICALS",
        "METALS_AND_MINING",
        "CAPITAL_GOODS",
        "CABLE_SATELLITE",
        "MEDIA_ENTERTAINMENT",
        "WIRELINES_WIRELESS",
        "AUTOMOTIVE",
        "RETAILERS",
        "FOOD_AND_BEVERAGE",
        "HEALTHCARE_EX_MANAGED_CARE",
        "MANAGED_CARE",
        "PHARMACEUTICALS",
        "TOBACCO",
        "INDEPENDENT",
        "INTEGRATED",
        "OIL_FIELD_SERVICES",
        "REFINING",
        "MIDSTREAM",
        "ENVIRONMENTAL_IND_OTHER",
        "TECHNOLOGY",
        "TRANSPORTATION",
    }

    financial_sectors = {
        "SIFI_BANKS_SR",
        "SIFI_BANKS_SUB",
        "US_REGIONAL_BANKS",
        "YANKEE_BANKS",
        "BROKERAGE_ASSETMANAGERS_EXCHANGES",
        "FINANCE_COMPANIES",
        "LIFE",
        "P&C",
        "REITS",
    }

    utility_sectors = {"UTILITY"}

    non_corp_sectors = {
        "OWNED_NO_GUARANTEE",
        "GOVERNMENT_GUARANTEE",
        "HOSPITALS",
        "MUNIS",
        "SOVEREIGN",
        "SUPRANATIONAL",
        "UNIVERSITY",
    }

    all_sectors = (
        industrial_sectors | financial_sectors | utility_sectors | non_corp_sectors
    )

    # Make a map of each cusip to respective LGIMA sector.
    indexes = load_json("indexes")
    unused_constraints = {"in_stats_index", "maturity"}
    lgima_sector_map = {}
    lgiam_top_level_sector_map = {}
    for sector_key in all_sectors:
        kwargs = {
            k: v for k, v in indexes[sector_key].items() if k not in unused_constraints
        }
        sector = kwargs["name"]
        cusips = basket.subset(**kwargs).cusips
        for cusip in cusips:
            lgima_sector_map[cusip] = sector
            if sector_key in industrial_sectors:
                lgiam_top_level_sector_map[cusip] = "Industrials"
            elif sector_key in financial_sectors:
                lgiam_top_level_sector_map[cusip] = "Financials"
            elif sector_key in utility_sectors:
                lgiam_top_level_sector_map[cusip] = "Utilities"
            else:
                lgiam_top_level_sector_map[cusip] = "Non-Corp"

    # Save sector maps.
    sector_file_dir = root("data/lgima_sector_maps")
    mkdir(sector_file_dir)
    fid = f"{sector_file_dir.stem}/{date.strftime('%Y-%m-%d')}"
    dump_json(lgima_sector_map, fid)

    top_level_sector_file_dir = root("data/lgima_top_level_sector_maps")
    mkdir(top_level_sector_file_dir)
    fid = f"{top_level_sector_file_dir.stem}/{date.strftime('%Y-%m-%d')}"
    dump_json(lgiam_top_level_sector_map, fid)


def update_lgima_sectors():
    # Find all currently scraped dates.
    db = Database()
    start = db.date("PORTFOLIO_START")
    dir_names = ["lgima_sector_maps", "lgima_top_level_sector_maps"]
    for dir_name in dir_names:
        fids = root(f"data/{dir_name}").glob("*.json")
        scraped_dates = set(pd.to_datetime([fid.stem for fid in fids]))
        dates_to_scrape = set(db.trade_dates(start=start)) - scraped_dates
        pbar = len(dates_to_scrape) > 10
        if pbar:
            print("Updating LGIMA Sectors...")
        for date in tqdm(dates_to_scrape, disable=(not pbar)):
            save_lgima_sectors(date)


if __name__ == "__main__":
    update_lgima_sectors()
