import os
import string
from collections import defaultdict
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy.bloomberg import get_call_schedule
from lgimapy.data import Database
from lgimapy.utils import rename_excel_columns, S_drive, to_list


# %%


class HYAnalystReport:
    def __init__(self, fid=None, df=None):
        self._fid = fid
        if fid is not None:
            self._df = self._read_excel() if df is None else df

    def _read_excel(self):
        df = pd.read_excel(self._fid, sheet_name="Main")
        return (
            rename_excel_columns(df)
            .replace(" - ", np.nan)
            .dropna(axis=1, how="all")
        )

    @cached_property
    def _issuer(self):
        return self._fid.parent.stem

    def _loc(self, loc, offset=(0, 0)):
        return self._df.iloc[loc[0] + offset[0], loc[1] + offset[1]]

    @cached_property
    def _date(self):
        return pd.to_datetime(os.path.getmtime(self._fid), unit="s").floor("D")

    def _get_idx_locs(self, matches, col_max=np.inf, row_max=np.inf):
        locs = []
        matches = to_list(matches, dtype=str)
        n_rows = min(self._df.shape[0], row_max + 1)
        n_cols = min(self._df.shape[1], col_max + 1)
        for i, j in np.ndindex((n_rows, n_cols)):
            for match in matches:
                if self._loc((i, j)) == match:
                    locs.append((i, j))
                    break
        return locs

    @cached_property
    def _ESG_verdict(self):
        locs = self._get_idx_locs("ESG Verdict")
        return str(self._loc(locs[0], offset=(0, 1))) if locs else np.nan

    @cached_property
    def _industry(self):
        industry_locs = [
            "Industry Analysis",
            "Industry",
            "Industry:",
            "INDUSTRY",
        ]
        locs = self._get_idx_locs(industry_locs, col_max=20, row_max=15)
        if locs:
            offsets = [(0, 1), (0, 2)]
            for offset in offsets:
                val = self._loc(locs[0], offset)
                if pd.isna(val):
                    continue
                else:
                    return val

        return np.nan

    @cached_property
    def _analyst(self):
        analysts = [
            "David Huang",
            "Frédéric Jourdren",
            "John Colan",
            "John Ryan",
            "Jarrett Brotzman",
            "John Harzich",
            "Jordan Hollander",
            "Kok Onn Yong",
            "Mike Rolnick",
            "Peter Duff",
            "Umesh Bhandary",
            "James Odemuyiwa",
            "Jonny Constable",
        ]
        locs = self._get_idx_locs(analysts, row_max=14, col_max=10)
        return self._loc(locs[0]) if locs else np.nan

    @cached_property
    def _country_of_risk(self):
        countries = [
            "North America",
            "Western Europe",
            "Eastern Europe",
            "Peripheral Europe",
            "Asia / Pacific - EM",
            "Asia / Pacific - DM",
            "Latin America",
            "Global",
            "Russia / CIS",
            "MEA",
            "United States",
            "Italy",
            "United Kingdom",
            "France",
            "Germany",
            "Angola",
            "Argentina",
            "Australia",
            "Austria",
            "Azerbaijan",
            "Bahamas",
            "Bahrain",
            "Bangladesh",
            "Barbados",
            "Belarus",
            "Belgium",
            "Bermuda",
            "Brazil",
            "Bulgaria",
            "Canada",
            "Cayman Islands",
            "Chile",
            "China",
            "Colombia",
            "Costa Rica",
            "Croatia",
            "Cyprus",
            "Czech Republic",
            "Democratic Republic of Congo",
            "Denmark",
            "Dominican Republic",
            "Egypt",
            "El Salvador",
            "Finland",
            "France",
            "Georgia",
            "Germany",
            "Ghana",
            "Greece",
            "Guatemala",
            "Hong Kong",
            "Hungary",
            "Iceland",
            "India",
            "Indonesia",
            "Ireland",
            "Israel",
            "Italy",
            "Jamaica",
            "Japan",
            "Jersey",
            "Jordan",
            "Kazakhstan",
            "Kuwait",
            "Luxembourg",
            "Macau",
            "Malaysia",
            "Mauritius",
            "Mexico",
            "Mongolia",
            "Netherlands",
            "New Zealand",
            "Nigeria",
            "Norway",
            "Oman",
            "Panama",
            "Paraguay",
            "Peru",
            "Poland",
            "Portugal",
            "Puerto Rico",
            "Qatar",
            "Romania",
            "Russia",
            "Saudi Arabia",
            "Serbia",
            "Singapore",
            "Slovenia",
            "South Africa",
            "South Korea",
            "Spain",
            "Sweden",
            "Switzerland",
        ]
        locs = self._get_idx_locs(countries, row_max=14, col_max=10)
        return self._loc(locs[0]) if locs else np.nan

    def _seniority(self, box_corner_loc):
        offsets = [(0, 4), (0, 5)]
        for offset in offsets:
            val = self._loc(box_corner_loc, offset)
            if pd.isna(val):
                continue
            else:
                return val

        return np.nan  # No seniority found

    def _workout_date(self, box_corner_loc):
        return pd.to_datetime(
            self._loc(box_corner_loc, offset=(2, 4)), errors="coerce"
        )

    @cached_property
    def _scenario_box_locs(self):
        downside_locs = self._get_idx_locs("Downside")
        base_locs = self._get_idx_locs("Base")
        upside_locs = self._get_idx_locs("Upside")
        poss_box_locs = []
        for d_loc in downside_locs:
            poss_b_loc = (d_loc[0], d_loc[1] + 1)
            poss_u_loc = (d_loc[0], d_loc[1] + 2)
            poss_isin_loc = (d_loc[0] - 2, d_loc[1] - 1)
            if poss_b_loc in base_locs and poss_u_loc in upside_locs:
                poss_box_locs.append(poss_isin_loc)
        real_box_locs = []
        for loc in poss_box_locs:
            poss_isin = self._loc(loc)
            if isinstance(poss_isin, str) and len(poss_isin) == 12:
                real_box_locs.append(loc)
        return real_box_locs

    def _print_locs(self, key, **kwargs):
        locs = self._get_idx_locs(key, **kwargs)
        locs
        for loc in locs:
            print(loc, self._loc(loc))

    def scrape_template(self):
        d = defaultdict(list)
        for loc in self._scenario_box_locs:
            d["Date"].append(self._date)
            d["Issuer"].append(self._issuer)
            d["ISIN"].append(self._loc(loc).upper())
            d["Analyst"].append(self._analyst)
            d["AnalystCapitalStructure"].append(self._seniority(loc))
            d["AnalystIndustry"].append(self._industry)
            d["AnalystCountryOfRisk"].append(self._country_of_risk)
            d["AnalystESGVerdict"].append(self._ESG_verdict)
            d["AnalystWorkoutDate"].append(self._workout_date(loc))
            d["IndexType"].append(self._loc(loc, offset=(1, 1)))
            d["DownsideRating"].append(self._loc(loc, offset=(3, 1)))
            d["DownsideProbability"].append(self._loc(loc, offset=(4, 1)))
            d["BaseRating"].append(self._loc(loc, offset=(3, 2)))
            d["BaseProbability"].append(self._loc(loc, offset=(4, 2)))
            d["UpsideRating"].append(self._loc(loc, offset=(3, 3)))
            d["UpsideProbability"].append(self._loc(loc, offset=(4, 3)))
            d["fid"].append(str(self._fid))

        return pd.DataFrame(d)

    @property
    def _stored_data_fid(self):
        return Database().local("HY/analyst_reports.parquet")

    @property
    def _stored_data(self):
        try:
            return pd.read_parquet(self._stored_data_fid)
        except FileNotFoundError:
            return pd.DataFrame()

    def save_template(self):
        new_df = self.scrape_template()
        if len(new_df):
            stored_df = self._stored_data
            updated_df = (
                pd.concat((stored_df, new_df))
                .drop_duplicates(keep="last", ignore_index=True)
                .sort_values("Date")
            )
            updated_df.to_parquet(self._stored_data_fid)


# Database().display_all_rows()
# Database().display_all_columns()
#
# parent = S_drive("FrontOffice/Bonds/High Yield/credits")
# box_corner_loc = self._scenario_box_locs[0]
# issuer_fid_d = {
#     "F": ("Ford Motor-FMC-FCE Bank/Ford Motor Co Long Dated USD (ESG).xlsm"),
#     "C": ("Caleres/Caleres Template.xlsm"),
#     "S": ("Square/Square Template.xlsm"),
#     "99": ("99"),
#     "AB": ("ABC Supply/Abengoa/Abengoa default base case.xlsm"),
#     "bad": ("_ Archive/Credit Curve Calculator v2.12 (GBP BBBs).xlsx"),
# }
# issuer = "AB"
# fid = parent / issuer_fid_d[issuer]
# # df = HYAnalystReport(fid)._df.copy()
# self = HYAnalystReport(fid, df)
# self.scrape_template()
# pd.set_option("display.max_colwidth", 255)
#
# self = HYAnalystReport(
#     Path(
#         "/mnt/s/FrontOffice/Bonds/High Yield/credits/Nexstar/Nexstar template.xlsm"
#     )
# )
# self._get_idx_locs('ESG')
# self._df
# # %%
# # self.save_template()
# data_df = HYAnalystReport()._stored_data
# data_df.tail(20)
# data_df[data_df['AnalystESGVerdict'].isna()].tail()
# %%


def update_analyst_ratings():
    parent = S_drive("FrontOffice/Bonds/High Yield/credits")
    issuer_dirs = sorted(os.listdir(parent))
    skip_until = "27"
    if skip_until is not None:
        skip_until_idx = issuer_dirs.index(skip_until)
    else:
        skip_until_idx = 0

    for issuer_dir in issuer_dirs[skip_until_idx:]:
        try:
            fids_and_subdirs = os.listdir(parent / issuer_dir)
        except NotADirectoryError:
            continue

        for fid_or_subdir in fids_and_subdirs:
            if fid_or_subdir.endswith(".xlsm"):
                # Excel file, possible report
                fid = fid_or_subdir
                if fid.startswith("~"):
                    # Temp file.
                    continue
                update_fid(parent / issuer_dir / fid)

            if "." not in fid_or_subdir:
                # Sub-Directory
                subdir = fid_or_subdir
                try:
                    all_fids = os.listdir(parent / issuer_dir / subdir)
                except NotADirectoryError:
                    continue
                excel_fids = [
                    f
                    for f in all_fids
                    if f.endswith(".xlsm") and not f.startswith("~")
                ]
                for fid in excel_fids:
                    update_fid(parent / issuer_dir / subdir / fid)
        print(f"{issuer_dir} Complete")


def update_fid(fid):
    top_dir = fid.parent.parent.stem
    next_dir = fid.parent.stem
    if top_dir != "credits":
        full_fid = f"{top_dir}/{next_dir}/{fid.stem}"
    else:
        full_fid = f"{next_dir}/{fid.stem}"

    try:
        report = HYAnalystReport(fid)
    except Exception as e:
        print(f"{full_fid}\n\t Unable to open File\n\t {e}")
        return
    try:
        report.save_template()
    except Exception as e:
        print(f"{full_fid}\n\t Problem saving\n\t {e}")
        return


def scrape_call_schedules():
    isins = list(reversed(HYAnalystReport()._stored_data["ISIN"]))
    failed_isins = []
    for isin in tqdm(isins):
        try:
            get_call_schedule(isin)
        except FileNotFoundError:
            failed_isins.append(isin)
            continue


if __name__ == "__main__":
    # update_analyst_ratings()
    scrape_call_schedules()


# %%
