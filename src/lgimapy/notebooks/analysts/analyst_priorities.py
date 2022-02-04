from datetime import datetime as dt
from itertools import chain

from lgimapy.notebooks import (
    plot_strategy_ticker_overweights,
    save_largest_tickers_in_sector,
)
from lgimapy.utils import load_json

# %%
def main():
    analyst = "My Nguyn"
    sector_d = analyst_sectors[analyst]

    today = dt.today().strftime("%Y-%m-%d")
    analyst_sectors = load_json("analyst_sectors")
    for analyst, sector_d in analyst_sectors.items():
        print(analyst)
        # Make overweight heatmaps.
        # for sector_name, sectors in sector_d.items():
        #     ow_fid = f"{analyst.replace(' ', '_')}_{sector_name}_Overweights"
        #     plot_strategy_ticker_overweights(sectors, ow_fid)

        # Make market value .csv's.
        all_sectors = list(chain(*sector_d.values()))
        mv_fid = f"{analyst.replace(' ', '_')}_Ticker_Market_Values.csv"
        save_largest_tickers_in_sector(all_sectors, mv_fid)

        for sector in all_sectors:
            mv_fid = f"{analyst.replace(' ', '_')}_{sector}_Market_Values.csv"
            save_largest_tickers_in_sector(sector, mv_fid)


if __name__ == "__main__":
    main()
