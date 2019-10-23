import pandas as pd

from lgimapy.bloomberg import bdh
from lgimapy.utils import load_json, tolist


def bloomberg_index_history(
    index, start, end=None, field=None, bloomberg_field="PX_LAST"
):
    """
    Get bloomberg index history for specified index.

    Parameters
    ----------
    index: str
        Specified index.
    start: datetime or str
        Start date for history.
    end: datetime or str
        Inlcusive end date for history.
    field: str, default=None
        Field for fixed income indexes.
    bloomberg_field: str, default="PX_LAST"
        Bloomberg field for index.

    Returns
    -------
    pd.DataFrame
        DataFrame of index history.
    """
    index_map = load_json("bloomberg_indexes")
    indexes = tolist(index, dtype=str)

    field_map = {"TR": "TRUU", "TRET": "TRUU", "XSRET": "ER", "YTW": "YW"}
    special_cases = {
        "LX01ER": "UISYMI5E",
        "LX07ER": "UISYMH5E",
        "CEMBTOOAS": "CEMBTOBS",
        "CEMBTOTRUU": "CEMBTOTR",
        "CEMBTOYW": "CEMBTOYI",
    }

    df_list = []
    for ix in indexes:
        # Get security and yellow key from json file and append
        # field if necessary. Fix special cases.
        security, yellow_key = index_map[ix]
        if field is not None:
            security += field_map.get(field.upper(), field.upper())
        security = special_cases.get(security, security).upper()
        # Perform BDH call for each index.
        df = bdh(security, yellow_key, bloomberg_field, start, end)
        df.columns = [ix]
        dfs.append(df)

    return pd.concat(df_list, join="outer", sort=False, axis=1)
