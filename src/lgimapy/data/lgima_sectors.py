import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database, BondBasket
from lgimapy.utils import dump_json, mkdir, root

# %%


def save_lgima_sectors(date):
    """
    Save .json mapping file of CUSIPs to LGIMA sector
    for each CUSIP on giving date.
    """
    # %%
    db = Database()
    df = db.load_market_data(date=date, clean=False, ret_df=True, local=False)
    df = db._update_utility_business_structure(df)
    basket = BondBasket(df)

    industrial_sectors = [
        "CHEMICALS",
        "METALS_AND_MINING",
        "PACKAGING",
        "PAPER",
        "CAPITAL_GOODS",
        "CABLE_SATELLITE",
        "MEDIA_ENTERTAINMENT",
        "WIRELINES_WIRELESS",
        "CONSUMER_CYCLICAL_SERVICES",
        "AUTOMOTIVE",
        "RETAILERS",
        "HOSPITALITY",
        "CONSUMER_NON_CYCLICAL",
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
    ]

    financial_sectors = [
        "SIFI_BANKS_SR",
        "SIFI_BANKS_SUB",
        "SIFI_BANKS_PREF",
        "SIFI_BANKS_NONSTANDARD",
        "US_REGIONAL_BANKS",
        "YANKEE_BANKS",
        "BROKERAGE_ASSETMANAGERS_EXCHANGES",
        "FINANCE_COMPANIES",
        "FINANCIAL_OTHER",
        "LIFE",
        "LIFE_SR",
        "LIFE_SUB",
        "P_AND_C",
        "REITS",
    ]

    utility_sectors = [
        "UTILITY_NONSTANDARD",
        "UTILITY_OPCO",
        "UTILITY_HOLDCO",
    ]

    non_corp_sectors = [
        "OWNED_NO_GUARANTEE",
        "GOVERNMENT_GUARANTEE",
        "HOSPITALS",
        "MUNIS",
        "SOVEREIGN",
        "SUPRANATIONAL",
        "UNIVERSITY",
        "PFANDBRIEFE",
        "AGENCY_CMBS",
        "NON_AGENCY_CMBS",
        "ABS",
    ]
    securitized_sectors = [
        "AGENCY_CMBS",
        "NON_AGENCY_CMBS",
        "ABS",
        "AGENCY_MBS",
        "MORTGAGES",
    ]
    treasury_sectors = ["TREASURIES"]
    swap_sectors = ["SWAPS"]
    loan_sectors = ["LOANS"]

    all_sectors = (
        industrial_sectors
        + financial_sectors
        + utility_sectors
        + non_corp_sectors
        + securitized_sectors
        + treasury_sectors
        + swap_sectors
        + loan_sectors
    )

    # Make a map of each cusip to respective LGIMA sector.
    lgima_sector_map = {}
    lgiam_top_level_sector_map = {}
    for sector in all_sectors:
        kwargs = db.index_kwargs(
            sector, unused_constraints=["in_stats_index", "maturity", "OAS"]
        )
        cusips = basket.subset(**kwargs).cusips
        for cusip in cusips:
            lgima_sector_map[cusip] = kwargs["name"]
            if sector in industrial_sectors:
                lgiam_top_level_sector_map[cusip] = "Industrials"
            elif sector in financial_sectors:
                lgiam_top_level_sector_map[cusip] = "Financials"
            elif sector in utility_sectors:
                lgiam_top_level_sector_map[cusip] = "Utilities"
            elif sector in securitized_sectors:
                lgiam_top_level_sector_map[cusip] = "Securitized"
            elif sector in treasury_sectors:
                lgiam_top_level_sector_map[cusip] = "Treasuries"
            else:
                lgiam_top_level_sector_map[cusip] = "Non-Corp"

    # df_missing = df[~df["CUSIP"].isin(lgima_sector_map)].reset_index(drop=True)
    # print(len(df_missing))
    # df_missing[["Ticker", "Issuer", "Sector", "Subsector"]]
    # s = df_missing["Sector"].value_counts()
    # s[s > 0].head(25)

    # %%

    # %%
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
