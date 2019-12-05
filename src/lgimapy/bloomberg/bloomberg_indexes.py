import pandas as pd

from lgimapy.bloomberg import bdh
from lgimapy.utils import load_json, to_list


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
    indexes = to_list(index, dtype=str)

    field_map = {
        "TR": "TRUU",
        "TRET": "TRUU",
        "XSRET": "ER",
        "YTW": "YW",
        "PR": "PRC",
        "PRICE": "PRC",
    }
    security_special_cases = {
        "LX01OAS": "CDX IG CDSI GEN 5Y",
        "LX07OAS": "CDX HY CDSI GEN 5Y SPRD",
        "LX07PRC": "CDX HY CDSI GEN 5Y PRC",
        "LX10OAS": "CDX EM CDSI GEN 5Y SPRD",
        "LX10PRC": "CDX EM CDSI GEN 5Y PRC",
        "LX01ER": "UISYMI5E",
        "LX07ER": "UISYMH5E",
        "CEMBTOOAS": "CEMBTOBS",
        "CEMBTOTRUU": "CEMBTOTR",
        "CEMBTOYW": "CEMBTOYI",
    }
    yellow_key_special_cases = {
        "LX01OAS": "Corp",
        "LX07OAS": "Corp",
        "LX07PRC": "Corp",
        "LX10OAS": "Corp",
        "LX10PRC": "Corp",
    }

    df_list = []
    for ix in indexes:
        # Get security and yellow key from json file and append
        # field if necessary. Fix special cases.
        security, yellow_key = index_map[ix]
        if field is not None:
            security += field_map.get(field.upper(), field.upper())
        yellow_key = yellow_key_special_cases.get(security, yellow_key)
        security = security_special_cases.get(security, security).upper()

        # Perform BDH call for each index.
        try:
            df = bdh(security, yellow_key, bloomberg_field, start, end)
        except KeyError:
            continue
        else:
            df.columns = [ix]
            df_list.append(df)

    return pd.concat(df_list, join="outer", sort=False, axis=1)
