import argparse
import joblib

from datetime import date

from lgimapy.data import Database
from lgimapy.latex import merge_pdfs
from lgimapy.utils import root

from cover_page import update_cover_page
from rating_performance import update_rating_performance
from decile_report import build_decile_report
from sector_comp import update_sector_models
from valuations_issuer_mv_weighted import update_valuations
from sector_dispersion import update_sector_dispersion
from dispersion import update_HY_spread_dispersion
from default_rates import update_default_rate_pdf
from fallen_angels_rising_stars import update_fallen_angels_rising_stars
from cost_of_downgrade import update_cost_of_downgrade
from maturity_walls import update_maturity_walls
from cross_asset_metrics import update_sharpe_ratios_and_correlations

# %%


def main():
    args = parse_args()
    db = Database()
    db.load_market_data(start=db.date("5y"))

    funcs = {
        update_cover_page: ("HY_Cover_Page", db),
        update_rating_performance: "Rating_Performance",
        build_decile_report: "HY_spreads_yields_returns",
        update_sector_models: ("Sector_Valuations", db),
        update_valuations: ("HY_Valuations", db),
        update_sector_dispersion: "HY_Sector_Dispersion",
        update_HY_spread_dispersion: "HY_Spread_Dispersion",
        update_default_rate_pdf: "Default_Rates",
        update_fallen_angels_rising_stars: "Potential_Fallen_Angels_Rising_Stars",
        update_maturity_walls: "Maturity_Walls",
        update_cost_of_downgrade: ("Cost_of_Downgrade", db),
        update_sharpe_ratios_and_correlations: "Correlations_and_SRs",
    }
    fids_to_merge = run_functions(funcs, run=args.norun)

    # Merge into one valuation pack.
    today = date.today().strftime("%Y-%m-%d")
    fid = f"HY_Valuation_Pack_{today}"
    path = root("reports/HY")
    merge_pdfs(fid, fids_to_merge, path=path, keep_bookmarks=True)


def run_functions(funcs, run=True):
    """
    Run functions and return list of pages to merge.

    Parameters
    ----------
    funcs: Dict[Callable: str or Tuple[str, *]]:
        Dictionary where keys are functions to be run, and
        values are the arguments for the respective functions.
        Arguments must begin with the filename for the output
        PDF.

    Returns
    -------
    fids_to_merge: List[str]
        List of individual PDF filenames to merge together.
    """
    fids_to_merge = []
    for func, args in funcs.items():
        if isinstance(args, str):
            if run:
                func(args)
            page_fid = args
        else:
            if run:
                func(*args)
            page_fid = args[0]
        fids_to_merge.append(page_fid)

    return fids_to_merge


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nr", "--norun", action="store_false", help="Just update PDF"
    )
    return parser.parse_args()


# %%
if __name__ == "__main__":
    main()
