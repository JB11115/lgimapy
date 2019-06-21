import warnings

import numpy as np
import pandas as pd

# R specific packages are installed only if needed
# within each core function.


def bdh(security, yellow_key, field, start, end=None):
    """
    Retrieve Bloomberg Data History (BDH) query results.

    Parameters
    ----------
    security: str
        Security name or cusip.
    yellow_key: {'Govt', 'Corp', 'Equity', 'Index', etc.}.
        Bloomberg yellow key for specified security.
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
    # R specific packages are installed when function is called.
    from rpy2.robjects import pandas2ri, r
    from rpy2.robjects.packages import importr

    importr("Rblpapi")  # R package

    # Conver inputs to R variables.
    r_security = f"{security} {yellow_key}"
    r_field = field.upper()
    r_start = pd.to_datetime(start).strftime("%Y-%m-%d")
    r_end = "NULL" if end is None else pd.to_datetime(end).strftime("%Y-%m-%d")

    # Scrape Bloomberg bdh in R.
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


def bdp(securities, yellow_key, field, ovrd=None, date=False):
    """
    Retrieve Bloomberg Data Set (BDS) query results.

    Parameters
    ----------
    security: str
        Security name or cusip.
    yellow_key: ``{'Govt', 'Corp', 'Equity', 'Index', etc.}``.
        Bloomberg yellow key for specified security.
    field: str
        Bloomberg field to collect history.
    ovrd: dict, default=None
        Bloomberg data overrides formated as
        ``{ovrd_key: ovrd_val}``.
    date: bool, default=False
        If True, convert returned column into datetime
        int format int(%Y%m%d).

    Returns
    -------
    [n x 1] np.array
        1 dimensional array of queried values.
    """
    # R specific packages are installed when function is called.
    from rpy2.robjects import pandas2ri, r, StrVector
    from rpy2.robjects.packages import importr

    importr("Rblpapi")  # R package

    # Conver inputs to R variables.
    if isinstance(securities, str):
        securities = [securities]  # convert to list
    r_securities = StrVector([f"{sec} {yellow_key}" for sec in securities])
    r_field = field.upper()
    if ovrd is None:
        r_ovrd_keys, r_ovrd_vals = "NULL", "NULL"
    else:
        r_ovrd_keys = StrVector(list(ovrd.keys()))
        r_ovrd_vals = StrVector(list(ovrd.values()))
    r_date = "TRUE" if date else "FALSE"

    # Scrape Bloomberg bdp in R.
    r_bdp = r(
        r"""
        function(securities, field, ovrd_keys, ovrd_vals, date) {
            con = blpConnect()
            if (ovrd_keys == 'NULL') {
                ovrd = NULL
            } else {
                ovrd = setNames(ovrd_vals, ovrd_keys)
            }
            df = bdp(
                securities=securities,
                fields=field,
                overrides=ovrd,
                )
            if (date == 'TRUE') {
                col = as.integer(gsub("-", "", df[, 1]))
            } else {
                col = df[, 1]
            }
            return(col)
        }
        """
    )
    return np.asarray(
        r_bdp(r_securities, r_field, r_ovrd_keys, r_ovrd_vals, r_date)
    )


def bds(security, yellow_key, field, column=1, date=False):
    """
    Retrieve Bloomberg Data Set (BDS) query results.

    Parameters
    ----------
    security: str
        Security name or cusip.
    yellow_key: {'Govt', 'Corp', 'Equity', 'Index', etc.}.
        Bloomberg yellow key for specified security.
    field: str
        Bloomberg field to collect history.
    column: int, default=1
        Column iloc of R DataFrame to return.
    date: bool, default=False
        If True, convert returned column into datetime
        int format int(%Y%m%d).

    Returns
    -------
    [n x 1] np.array
        1 dimensional array of queried values.
    """
    # R specific packages are installed when function is called.
    from rpy2.robjects import pandas2ri, r
    from rpy2.robjects.packages import importr

    importr("Rblpapi")  # R package

    # Conver inputs to R variables.
    r_security = f"{security} {yellow_key}"
    r_field = field.upper()
    r_date = "TRUE" if date else "FALSE"

    # Scrape Bloomberg bds in R.
    r_bds = r(
        r"""
        function(security, field, column, date) {
            con = blpConnect()
            df = bds(
                securities=security,
                fields=field,
                )
            if (date == 'TRUE') {
                col = as.integer(gsub("-", "", df[, column]))
            } else {
                col = df[, column]
            }
            return(col)
        }
        """
    )

    return np.asarray(r_bds(r_security, r_field, column, r_date))


def main():
    # %%

    print(bdp("C","Equity","PX_LAST"))
    # %%


if __name__ == "__main__":
    main()