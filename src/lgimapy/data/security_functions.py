from functools import lru_cache

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, union_categoricals

from lgimapy.stats import mode
from lgimapy.utils import load_json, to_set, to_list

# %%


def clean_dtypes(df):
    """pd.DataFrame: Convert dtypes columns to proper dtype."""
    reverse_dtype_dict = {
        "float32": [
            "CouponRate",
            "CleanPrice",
            "OAD",
            "OAS",
            "OASD",
            "ZSpread",
            "DTS",
            "DTS_Libor",
            "OAS_1W",
            "OAS_1M",
            "OAS_3M",
            "OAS_6M",
            "OAS_12M",
            "LiquidityCostScore",
            "AccruedInterest",
            "AmountOutstanding",
            "LQA",
            "KRD06mo",
            "KRD02yr",
            "KRD05yr",
            "KRD10yr",
            "KRD20yr",
            "KRD30yr",
            "YieldToWorst",
            "YieldToMat",
            "ModDurationToWorst",
            "ModDurationToMat",
            "ModDurtoMat",
            "ModDurtoWorst",
            "NumericRating",
            "MaturityYears",
            "TRet",
            "XSRet",
            "MTDXSRet",
            "MTDLiborXSRet",
            "IssueYears",
            "DirtyPrice",
            "MarketValue",
            "PrevMarketValue",
            "Quantity",
            "AccountWeight",
            "BenchmarkWeight",
            "AnalystRating",
            "MLFI_OAS",
            "MLFI_YieldtoMat",
            "MLFI_YieldtoWorst",
        ],
        "int8": [
            "OriginalMaturity",
            "BMTreasury",
            "FinancialFlag",
            "Eligibility144AFlag",
            "AnyIndexFlag",
            "USCreditReturnsFlag",
            "USCreditStatisticsFlag",
            "USAggReturnsFlag",
            "USAggStatisticsFlag",
            "USHYReturnsFlag",
            "USHYStatisticsFlag",
            "MLHYFlag",
            "H0A0Flag",
            "H4UNFlag",
            "HC1NFlag",
            "HUC2Flag",
            "HUC3Flag",
        ],
        "category": [
            "CUSIP",
            "ISIN",
            "Ticker",
            "Issuer",
            "RiskEntity",
            "RatingRiskBucket",
            "Sector",
            "Subsector",
            "BAMLTopLevelSector",
            "BAMLSector",
            "SectorLevel1",
            "SectorLevel2",
            "SectorLevel3",
            "SectorLevel4",
            "SectorLevel5",
            "SectorLevel6",
            "CompositeRating",
            "MoodyRating",
            "SPRating",
            "FitchRating",
            "CollateralType",
            "CouponType",
            "CallType",
            "Currency",
            "CountryOfRisk",
            "CountryOfDomicile",
            "MarketOfIssue",
            "Account",
        ],
    }
    # Build col:dtype dict and apply to input DataFrame.
    df_columns = set(df.columns)
    dtype_dict = {}
    for dtype, col_names in reverse_dtype_dict.items():
        for col in col_names:
            if col in df_columns:
                dtype_dict[col] = dtype
            else:
                continue
    return df.astype(dtype_dict)


def convert_sectors_to_fin_flags(sectors):
    """
    Convert sectors to flag indicating if they
    are non-financial (0), financial (1), or other (2).

    Parameters
    ----------
    sectors: pd.Series
        Sectors

    Returns
    -------
    fin_flags: nd.array
        Array of values indicating if sectors are non-financial (0),
        financial (1), or other (Treasuries, Sovs, Govt owned).
    """

    financials = {
        "P_AND_C",
        "LIFE",
        "LIFE_SR",
        "LIFE_SUB",
        "APARTMENT_REITS",
        "BANKING",
        "BROKERAGE_ASSETMANAGERS_EXCHANGES",
        "RETAIL_REITS",
        "HEALTHCARE_REITS",
        "OTHER_REITS",
        "FINANCIAL_OTHER",
        "FINANCE_COMPANIES",
        "OFFICE_REITS",
        "SIFI_BANKS_SR",
        "SIFI_BANKS_SUB",
        "YANKEE_BANKS",
        "REITS",
        "US_REGIONAL_BANKS",
        # Merrill Lynch Sectors
        "BANKS",
        "GENERAL_FINANCIAL",
        "GUARANTEED_FINANCIALS",
        "LIFE_INSURANCE",
        "NONLIFE_INSURANCE",
        "PUBLIC_BANKS",
        "REAL_ESTATE_INVESTMENT_AND_SERVICES",
        "REAL_ESTATE_INVESTMENT_TRUSTS",
        "REGIONS",
    }
    other = {
        "TREASURIES",
        "LOCAL_AUTHORITIES",
        "SOVEREIGN",
        "SUPRANATIONAL",
        "INDUSTRIAL_OTHER",
        "GOVERNMENT_GUARANTEE",
        "OWNED_NO_GUARANTEE",
        "HOSPITALS",
        "MUNIS",
        "UNIVERSITY",
        "UTILITY",
        "UTILITY_OPCO",
        "UTILITY_HOLDCO",
        "UTILITY_OTHER",
        "NATURAL_GAS",
        "ELECTRIC",
        # Merrill Lynch Sectors
        "AGENCIES",
        "AUSTRALIA_COVERED",
        "AUSTRIA_COVERED",
        "BELGIUM_COVERED",
        "CANADA_COVERED",
        "CZECH_REPUBLIC_COVERED",
        "DENMARK_COVERED",
        "ELECTRICITY",
        "ESTONIA_COVERED",
        "FINLAND_COVERED",
        "FRANCE_COVERED_LEGAL",
        "FRANCE_COVERED_SFH",
        "FRANCE_COVERED_STRUCTURED",
        "GAS_WATER_AND_MULTIUTILITIES",
        "GREECE_COVERED",
        "HYPOTHEKENPFANDBRIEFE",
        "IRELAND_COVERED",
        "ITALY_COVERED",
        "JAPAN_COVERED",
        "LUXEMBOURG_COVERED",
        "NETHERLANDS_COVERED",
        "NEW_ZEALAND_COVERED",
        "NORWAY_COVERED",
        "OEFFENTLICHE_PFANDBRIEFE",
        "OTHER_COLLATERALIZED",
        "OTHER_PFANDBRIEFE",
        "OTHER_SOVEREIGNS",
        "POLAND_COVERED",
        "POOLED_CEDULAS",
        "PORTUGAL_COVERED",
        "SECURITIZED",
        "SINGAPORE_COVERED",
        "SINGLE_CEDULAS",
        "SLOVAKIA_COVERED",
        "SOUTH_KOREA_COVERED",
        "SOVEREIGNS",
        "SUPRANATIONALS",
        "SWEDEN_COVERED",
        "SWITZERLAND_COVERED",
        "UK_COVERED",
        "US_COVERED",
    }
    fin_flags = np.zeros(len(sectors))
    fin_flags[sectors.isin(financials)] = 1
    fin_flags[sectors.isin(other)] = 2
    return fin_flags


def standardize_cusips(df):
    """
    Standardize CUSIPs, converting cusips which changed name
    to most recent cusip value for full history.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of selected index cusips.

    Returns
    -------
    df: pd.DataFrame
        DataFrame with all CUSIPs updated to current values.
    """
    dates = sorted(list(set(df["Date"])), reverse=True)
    df.set_index("CUSIP", inplace=True)

    # Build CUSIP map of CUSIPs which change.
    # TODO: add Issuer back
    multi_ix = [  # index for identifying CUSIP changes
        "Ticker",
        "CouponRate",
        "MaturityDate",
        "IssueDate",
    ]
    cusip_map = {}
    for i in range(len(dates) - 1):
        df_today = df[df["Date"] == dates[i]].copy()
        df_yesterday = df[df["Date"] == dates[i + 1]].copy()

        # Find CUSIPs that were dropped from yesterday to today
        # and CUSIPs that were newly added today.
        df_dropped = df_yesterday[
            ~df_yesterday.index.isin(df_today.index)
        ].copy()
        df_new = df_today[~df_today.index.isin(df_yesterday.index)].copy()

        # Store CUSIPs, and set index to change identifier.
        df_dropped["CUSIP_old"] = df_dropped.index
        df_new["CUSIP_new"] = df_new.index
        df_dropped.set_index(multi_ix, inplace=True)
        df_new.set_index(multi_ix, inplace=True)

        # Store instances where CUSIPs change in cusip_map dict.
        df_new[df_new.index.isin(df_dropped.index)]
        df_change = df_new[["CUSIP_new"]].join(
            df_dropped[["CUSIP_old"]], how="inner"
        )
        for _, row in df_change.iterrows():
            cusip_map[row["CUSIP_old"]] = row["CUSIP_new"]

    # Update CUSIP map to account for CUSIPs which change multiple times.
    rev_CUSIPs = list(cusip_map.keys())[::-1]
    for i, key in enumerate(rev_CUSIPs):
        for k in rev_CUSIPs[i:]:
            if cusip_map[k] == key:
                cusip_map[k] = cusip_map[key]

    # Map old CUSIPs to new CUSIPs and reset index.
    df.index = [cusip_map.get(ix, ix) for ix in df.index]
    df["CUSIP"] = df.index.astype("category")
    df.reset_index(inplace=True, drop=True)
    return df


def spread_diff(df1, df2):
    """
    Calculate spread difference for cusips from single
    date :attr:`Index.df`s df1 to df2.

    Parameters
    ----------
    df1: :attr:`Index.df`
        Older :attr:`Index.df`.
    df2: :attr:`Index.df`
        Newer :attr:`Index.df`.

    Returns
    -------
    df: pd.DataFrame
        DataFrame with OAS difference computed.
    """
    # Standardize cusips.
    combined_df = standardize_cusips(concat_index_dfs([df1, df2]))
    df1 = combined_df[combined_df["Date"] == df1["Date"].iloc[0]].copy()
    df2 = combined_df[combined_df["Date"] == df2["Date"].iloc[0]].copy()
    df1.set_index("CUSIP", inplace=True)

    # Add OAS column from df1 to df2, only keeping CUSIPS with values
    # in both DataFrames.
    df = df2.join(df1[["OAS"]], on=["CUSIP"], how="inner", rsuffix="_old")
    df["OAS_change"] = df["OAS"] - df["OAS_old"]
    df["OAS_pct_change"] = df["OAS"] / df["OAS_old"] - 1
    return df


def concat_index_dfs(dfs, join="outer"):
    """
    Append two :attr:`Index.df`s.

    Parameters
    ----------
    dfs: List[:attr:`Index.df`].
        List of DataFrames to append together.
    join: {'inner', 'outer'}, default='outer'
        How to handle indexes on other axis(es).

    Returns
    -------
    df: pd.DataFrame
        Combined DataFrame of all specified dfs.
    """
    # Unionize categorical indexes.
    for col, dtype in dfs[0].dtypes.iteritems():
        if isinstance(dtype, CategoricalDtype):
            uc = union_categoricals([df[col] for df in dfs])
            for df in dfs:
                df[col] = pd.Categorical(df[col], categories=uc.categories)

    # Combine all DataFrames.
    df = pd.concat(dfs, join=join, ignore_index=True, sort=False)
    if isinstance(dfs[0].index.dtype, CategoricalDtype):
        # Reset index to cusips if previous index was cusips.
        df.set_index("CUSIP", inplace=True, drop=False)
    return df.drop_duplicates(subset=["CUSIP", "Date"])


def new_issue_mask(df):
    """
    Create mask for input DataFrame which returns
    a boolean mask indicating which bonds are
    currently in the month they were originally issued.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame from :attr:`Index.df`.

    Returns
    -------
    List[bool].
        Boolean mask for whether current date is in
        orinal issue month.
    """
    return [
        (idt.month, idt.year) == (dt.month, dt.year)
        for idt, dt in zip(df["IssueDate"], df["Date"])
    ]


@lru_cache(maxsize=None)
def index_kwargs_dict(source):
    source = source.lower()
    allowable_sources = {"bloomberg", "iboxx", "baml", "bloomberg_js"}
    if source in allowable_sources:
        return load_json(f"index_kwargs/{source}")
    else:
        raise ValueError(
            f"'{source}' not in allowable sources. "
            f"Please select one of {allowable_sources}."
        )


def index_kwargs(key, unused_constraints=None, source=None, **kwargs):
    """
    Index keyword arguments for saved indexes,
    with ability to override/add/remove arguments.

    Parameters
    ----------
    key: str
        Key of stored index in `indexes.json`.
    unused_constraints: str or List[str], optional
        Constraintes to remove from kwargs list if present.
    source: ``{"bloomberg", "iboxx", "baml"}``, optional
        Source for index kwargs. Defaults based on current
        :attr:`market`.
    kwargs:
        Keyword arguments to override or add to index.

    Returns
    -------
    dict:
        Keyword arguments and respective constraints
        for specified index.
    """
    source = "bloomberg" if source is None else source
    try:
        d = index_kwargs_dict(source)[key].copy()
    except KeyError:
        raise KeyError(f"{key} is not a stored Index.")

    if unused_constraints is not None:
        unused_cons = to_set(unused_constraints, dtype=str)
        d = {k: v for k, v in d.items() if k not in unused_cons}

    d.update(**kwargs)
    return d


def groupby(df, cols):
    """
    Group basket of bonds together by seletected features.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame basket of bonds.
    cols: str or List[str].
        Column(s) in `df` to group by.

    Returns
    -------
    pd.DataFrame
        Grouped DataFrame.
    """
    df_to_group = df.copy()
    if cols == "risk entity":
        groupby_cols = ["Ticker", "Issuer"]
    else:
        groupby_cols = to_list(cols, dtype=str)

    # Collect columns that should be market value weighted
    # and create a mv weighted column for each that is present.
    mv_weight_cols = [
        "OAS",
        "OASD",
        "OAD",
        "DTS",
        "YieldToWorst",
        "DirtyPrice",
        "CleanPrice",
        "PX_Adj_OAS",
        "NumericRating",
        "AnalystRating",
    ]
    mv_agg_rules = {}
    df_cols = set(df)
    any_mv_cols_present = len(set(mv_weight_cols) & df_cols)
    if any_mv_cols_present and "MarketValue" in df_cols:
        for col in mv_weight_cols:
            if col not in df_cols:
                continue
            df_to_group[f"MV_{col}"] = df["MarketValue"] * df[col]
            mv_agg_rules[f"MV_{col}"] = np.sum

    # Collect columns that should be market value weighted using
    # previous day's market value and create a mv weighted column
    # for each column that is present.
    prev_mv_weight_cols = ["TRet", "XSRet"]
    prev_mv_agg_rules = {}
    any_prev_mv_cols_present = len(set(prev_mv_weight_cols) & df_cols)
    if any_prev_mv_cols_present and "PrevMarketValue" in df_cols:
        for col in prev_mv_weight_cols:
            if col not in df_cols:
                continue
            df_to_group[f"PMV_{col}"] = df["PrevMarketValue"] * df[col]
            prev_mv_agg_rules[f"PMV_{col}"] = np.sum

    # Combine aggregation rules for all columns.
    agg_rules = {
        "Ticker": mode,
        "Issuer": mode,
        "Sector": mode,
        "Subsector": mode,
        "LGIMASector": mode,
        "BAMLSector": mode,
        "BAMLTopLevelSector": mode,
        "OAD_Diff": np.sum,
        "P_OAD": np.sum,
        "BM_OAD": np.sum,
        "DTS_Diff": np.sum,
        "P_DTS": np.sum,
        "BM_DTS": np.sum,
        "AmountOutstanding": np.sum,
        "MarketValue": np.sum,
        "PrevMarketValue": np.sum,
        "P_Notional": np.sum,
        "P_MarketValue": np.sum,
        "BM_MarketValue": np.sum,
        "P_Weight": np.sum,
        "BM_Weight": np.sum,
        "Weight_Diff": np.sum,
        "OASD_Diff": np.sum,
        "OAS_Diff": np.sum,
        "DTS_Contrib": np.sum,
        **mv_agg_rules,
        **prev_mv_agg_rules,
    }
    # Apply aggregation of present columns.
    agg_cols = {
        col: rule
        for col, rule in agg_rules.items()
        if col in df_to_group.columns and col not in groupby_cols
    }
    gdf = df_to_group.groupby(groupby_cols, observed=True).aggregate(agg_cols)

    # Clean market value weighted columns by dividing by total
    # market value and renaming back to original name.
    col_names = {}
    for col in mv_agg_rules.keys():
        gdf[col] = gdf[col] / gdf["MarketValue"]
        col_names[col] = col[3:]
    for col in prev_mv_agg_rules.keys():
        gdf[col] = gdf[col] / gdf["PrevMarketValue"]
        col_names[col] = col[4:]

    return gdf.rename(columns=col_names)
