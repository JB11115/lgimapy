from collections import Counter, defaultdict
from datetime import timedelta
from itertools import chain

import numpy as np
import pandas as pd

from lgimapy.bloomberg import bdh
from lgimapy.utils import load_json, mkdir, root, to_list


def update_bloomberg_data(print_updates=False):
    """Update all Bloomberg timeseries data."""
    bbg_scraper = BBGTimeseriesScraper()
    for field in sorted(bbg_scraper.fields):
        if print_updates:
            print(field)
        bbg_scraper.create_data_file(field)


class BBGTimeseriesScraper:
    def __init__(self):
        self.start = pd.to_datetime("1/1/1990")  # Start date for all files
        self.fid = root("data/bloomberg_timeseries")
        self.load_bbg_codes()
        mkdir(self.fid)

    def load_bbg_codes(self):
        """
        Read in all saved timeseries codes and find
        the secuirites stored for each field.
        """
        self.bbg_ts_codes = load_json("bloomberg_timeseries_codes")
        self.securities = {}
        fields = list(
            set(
                chain(*[fields.keys() for fields in self.bbg_ts_codes.values()])
            )
        )
        non_fields = {"NAME", "ERRONEOUS_DATA"}
        self.fields = [field for field in fields if field not in non_fields]

        for field in self.fields:
            self.securities[field] = [
                security
                for security, security_fields in self.bbg_ts_codes.items()
                if field in security_fields
            ]

    def create_data_file(self, field):
        """
        Scrape and save data in .csv file for single field.

        Parameters
        ----------
        field: str, ``{'OAS', 'YTW', 'TRET', 'XSRET', etc.}``
            Field to scrape data for.
        """
        self._check_for_duplicates(field)
        self._field = field

        # Load previously scraped data.
        fid = self.fid / f"{field}.csv"
        try:
            old_df = pd.read_csv(
                fid, index_col=0, parse_dates=True, infer_datetime_format=True
            )
        except FileNotFoundError:
            # Create full file from scrath.
            old_df = pd.DataFrame()

        # Scrape any new securities from the original start date
        # and combine with the old data.
        new_securities = [
            security
            for security in self.securities[field]
            if security not in old_df.columns
        ]

        if new_securities:
            old_df = pd.concat(
                [old_df, self.scrape_data(new_securities, self.start)],
                join="outer",
                axis=1,
                sort=True,
            )

        # Update current file with new data. Start with the last
        # day all securities had data, re-scraping for that day to
        # ensure closing values are stored not intra-day values.
        try:
            start = old_df.dropna().index[-1]
        except IndexError:
            start = self.start

        old_df = old_df[old_df.index < start]
        new_df = self.scrape_data(old_df.columns, start - timedelta(20))
        new_df = new_df[new_df.index >= start]
        updated_df = pd.concat([old_df, new_df], axis=0, sort=True)

        # Remove erroneous data.
        for security in updated_df.columns:
            try:
                bad_dates = pd.to_datetime(
                    to_list(
                        self.bbg_ts_codes[security]["ERRONEOUS_DATA"][field],
                        dtype=str,
                    )
                )
            except KeyError:
                continue
            else:
                for date in bad_dates:
                    updated_df.loc[date, security] = np.nan

        # Save the updated data.
        updated_df.to_csv(fid)

    def scrape_data(self, securities, start):
        """
        Scrape data from Bloomberg.

        Parameters
        ----------
        securities: List[str].
            List of security names for Bloomberg code .json file.
        start: datetime
            Start date for scrape.

        Returns
        -------
        df: pd.DataFrame
            Bloomberg timeseries data for given parameters
            through current values.
        """
        # securities = new_securities
        # Group codes by Bloomberg field.
        bbg_codes = {
            security: self.bbg_ts_codes[security][self._field]
            for security in securities
        }
        field_codes = defaultdict(list)
        col_names = {}  # store column names for created DataFrame
        for security, code in bbg_codes.items():
            if "|" in code:
                code, bbg_field = code.split("|")
            else:
                bbg_field = "PX_LAST"
            field_codes[bbg_field].append(code)
            col_names[code] = security

        # Scrape data for each field and combine.
        df_list = []
        try:
            for bbg_field, securities in field_codes.items():
                df_temp = bdh(
                    securities, yellow_keys="", fields=bbg_field, start=start
                )
                if len(securities) == 1:
                    # Column name will be bbg field instead of bbg code,
                    # so make it the security name.
                    df_temp.columns = to_list(securities, dtype=str)
                df_list.append(df_temp)
        except KeyError:
            # Identify which security is causing the error.
            for bbg_field, securities in field_codes.items():
                for security in securities:
                    try:
                        bdh(security, "", fields=bbg_field, start="12/1/2019")
                    except KeyError:
                        raise ValueError(
                            f"{security} - {bbg_field} cannot be scraped."
                        )

        df = pd.concat(df_list, axis=1, sort=True).rename(columns=col_names)
        return self._fix_orders_of_magnitude(df)

    def _check_for_duplicates(self, field):
        """
        Check for duplicate entries in the Bloomberg codes file
        and Return errors for all duplicates.

        Parameters
        ----------
        field: str, ``{'OAS', 'YTW', 'TRET', 'XSRET', etc.}``
            Field to scrape data for.
        """
        securities = [
            security
            for security, codes in self.bbg_ts_codes.items()
            if field in codes
        ]
        bbg_codes = {
            security: self.bbg_ts_codes[security][field]
            for security in securities
        }
        code_count = Counter(bbg_codes.values())
        dupes = [code for code, count in code_count.items() if count > 1]
        for dupe in dupes:
            dupe_securities = [
                security
                for security in bbg_codes.keys()
                if self.bbg_ts_codes[security][field] == dupe
            ]
            raise ValueError(
                f"Duplicate BBG code found: {dupe}, field={field}\n"
                f"\t\tsecurities: {dupe_securities}"
            )

    def _fix_orders_of_magnitude(self, df):
        """
        Correct securities for which Bloomberg does
        not return consistent orders of magnitude.

        Parameters
        ----------
        df: pd.DataFrame
            Scraped DataFrame.

        Returns
        -------
        df: pd.DataFrame
            DataFrame corrected for orders of magnitude.
        """
        if self._field == "OAS":
            exceptions = {"EM_CORP"}
            multiply_cols = [
                col
                for col in df.columns
                if "Index" in self.bbg_ts_codes[col]["OAS"]
                and col not in exceptions
            ]
            df[multiply_cols] = df[multiply_cols].multiply(100)
        elif self._field == "YTW":
            exceptions = {"UST_10Y_RY", "UST_5Y5Y_BE"}
            div_cols = [
                col
                for col in df.columns
                if ("UST" in col and Counter(col.lower())["y"] > 1)
                and col not in exceptions
            ]
            df[div_cols] = df[div_cols].divide(100)
            df /= 100
        return df


# %%
if __name__ == "__main__":
    update_bloomberg_data(print_updates=True)
