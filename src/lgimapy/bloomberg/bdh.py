import warnings

import numpy as np
import pandas as pd
import rpy2
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

importr("Rblpapi")  # R package


def bdh(security, yellow_button, field, start, end=None):
    """
    Bloomberg data history scrape.

    Parameters
    ----------
    security: str
        Security name or cusip.
    yellow_button: {'Govt', 'Corp', 'Equity', 'Index', 'Curncy', etc.}.
        Bloomberg yellow button for specified security.
    field: str
        Bloomberg field to collect history.
    start: datetime object
        Start date for scrape.
    end: datetime object, default=None
        End date for scrape, if None most recent date is used.

    Returns
    -------
    df: pd.DataFrame
        DataFrame with datetime index and specified field as column.
    """
    # Conver inputs to R variables.
    r_security = f"{security} {yellow_button}"
    r_field = field.upper()
    r_start = pd.to_datetime(start).strftime("%Y-%m-%d")
    r_end = "NULL" if end is None else pd.to_datetime(end).strftime("%Y-%m-%d")

    # Scrape bloomberg bdh in R.
    # Note that datetime dtypes don't convert between R and Python,
    # so dates are formated to integers in R and reformatted
    # to dates in Python.
    r_bdh = r(
        r"""
        function(security, field, start, end) {
            con = blpConnect()
            end = if(end == 'NULL') NULL else as.Date(end)
            df = bdh(
                securities=security,
                fields=field,
                start.date=as.Date(start),
                end.date=end,
                )
            df$date = as.integer(gsub("-", "", df$date))
            return(df)
            }
        """
    )
    warnings.simplefilter(action="ignore", category=FutureWarning)
    pandas2ri.activate()
    df = pandas2ri.ri2py(r_bdh(r_security, r_field, r_start, r_end))
    warnings.simplefilter(action="default", category=FutureWarning)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df.set_index("date", inplace=True)
    return df
