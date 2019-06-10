from collections import defaultdict

import datetime as dt
import numpy as np
import pandas as pd

from index_functions import IndexBuilder, spread_diff, standardize_cusips
from utilities import nearest


def main():
    """
    Make 8 data.csv files to show OAS evolution over past week, month,
    and since tights, for 10 year and 30 year bonds. Data is saved to
    `X:/Jason/Projects/Index Analysis/` repository.
    """

    tights_date = "10/3/2018"
    month_date = "5/1/2019"
    fid = "X:/Jason/Projects/Index Analysis/data_test{}.csv"

    # Make dict of dates to analyze, ensuring they are on dates with data.
    ixb = IndexBuilder()
    dates = {
        "yesterday": pd.to_datetime(dt.date.today() - dt.timedelta(days=1)),
        "week": pd.to_datetime(dt.date.today() - dt.timedelta(days=7)),
        # 'month': pd.to_datetime(dt.date.today() - dt.timedelta(days=30)),
        "month": pd.to_datetime(month_date),
        "tights": pd.to_datetime(tights_date),
    }

    dates = {k: ixb.nearest_date(v) for k, v in dates.items()}
    # for k, v in dates.items():
    #     print(k, v)

    # Load index on dates from SQL database, then find changes.
    df_names = ["week", "month", "tights"]
    raw_dfs = {
        k: ixb.load(start=v, end=v, dev=True, ret_df=True)
        for k, v in dates.items()
    }
    index_chg_dfs = {
        n: spread_diff(raw_dfs[n], raw_dfs["yesterday"]) for n in df_names
    }

    # Subset DataFrames to 10 and 30 year maturities, and recent issues.
    # Then add necesarry columns and sort for saving.
    sorted_cols = [
        "Ticker Name Rank",
        "Issuer",
        "Ticker",
        "CollateralType",
        "Sector",
        "MoodyRating",
        "SPRating",
        "FitchRating",
        "AmountOutstanding",
        "OAS",
        "LiquidityCostScore",
        "NumericRating",
        "OAS_old",
        "OAS_change",
        "OAS_pct_change",
        "IsFinancial",
        "IsNonFinancial",
        "BloombergSector",
    ]

    mat_ranges = [(8.25, 11), (25, 32)]
    index_dfs = {}
    for name in df_names:
        index_dfs[name] = {}
        for mat, mat_range, max_issue in zip([10, 30], mat_ranges, [2, 5]):
            df = ixb.build(
                data=index_chg_dfs[name],
                rating="IG",
                amount_outstanding=(None, None),
                municipals=True,
                maturity=(mat_range[0], mat_range[1]),
                issue_years=(None, max_issue),
            ).df
            df["Ticker Name Rank"] = (
                df["Ticker"] + " " + df["Issuer"] + " " + df["CollateralType"]
            )
            df["IsFinancial"] = (df["FinancialFlag"] == "financial").astype(int)
            df["IsNonFinancial"] = (
                df["FinancialFlag"] == "non-financial"
            ).astype(int)
            df["BloombergSector"] = df["Subsector"]
            index_dfs[name][mat] = aggregate_issuers(df)[sorted_cols]

    # Save tables to `data.csv` files.
    index_dfs["tights"][30].to_csv(fid.format(1))
    index_dfs["week"][30].to_csv(fid.format(2))
    index_dfs["month"][30].to_csv(fid.format(3))
    build_weighted_average_table(30, index_dfs).to_csv(fid.format(4))
    index_dfs["tights"][10].to_csv(fid.format(5))
    index_dfs["week"][10].to_csv(fid.format(6))
    index_dfs["month"][10].to_csv(fid.format(7))
    build_weighted_average_table(10, index_dfs).to_csv(fid.format(8))


def aggregate_issuers(df):
    """
    Aggregate bonds of same issuer, ticker, and collateral type using
    a weighted average of their OAS, LiquidityCostScore, NumericRating,
    and OAS_old, then recalculate OAS change and pct change.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of bonds to aggregate.

    Returns
    -------
    agg_df: pd.DataFrame
        Aggregated DataFrame.
    """

    def weighted_average(x, col):
        """Weighted average calcultion for a given column."""
        val = np.sum(
            x[col] * x["AmountOutstanding"] * x["DirtyPrice"]
        ) / np.sum(x["AmountOutstanding"] * x["DirtyPrice"])
        return val

    def my_agg(x):
        """Weighed average aggregation of same ticker/issuer/rank."""
        agg = {col: x[col].iloc[0] for col in list(x)}  # first value
        agg["NumericRating"] = weighted_average(x, "NumericRating")
        agg["LiquidityCostScore"] = weighted_average(x, "LiquidityCostScore")
        agg["OAS"] = weighted_average(x, "OAS")
        agg["OAS_old"] = weighted_average(x, "OAS_old")
        return pd.Series(agg, index=agg.keys())

    agg_df = df.groupby(["Ticker Name Rank"]).apply(my_agg)
    agg_df["LiquidityCostScore"].replace(0.0, np.NaN, inplace=True)
    agg_df["OAS_change"] = agg_df["OAS"] - agg_df["OAS_old"]
    agg_df["OAS_pct_change"] = agg_df["OAS"] / agg_df["OAS_old"] - 1
    agg_df = agg_df.sort_index().reset_index(drop=True)
    return agg_df


def build_weighted_average_table(maturity, df_dict):
    """
    Create weighted average of OAS absolute and percent change
    from weekly, monthly, and tights for specified maturity.

    Parameters
    ----------
    maturity: int
        Maturity value.
    df_dict: Dict[str: Dict[int: pd.DataFrame]].
        `index_dfs` nested dictionary by time and maturity.

    Returns
    -------
    table: pd.DataFrame
        Weighted average table.
    """
    headers = []
    table_d = defaultdict(list)
    for df_name in ["tights", "month", "week"]:
        df_period = df_dict[df_name][maturity]
        for sector in ["IsFinancial", "IsNonFinancial"]:
            df = df_period[df_period[sector] == 1].copy()
            for col in ["OAS_pct_change", "OAS_change", "OAS"]:
                if col == "OAS" and df_name != "week":
                    continue
                headers.append(f"{sector[2:]}_{col}_from_{df_name}_{maturity}")
                table_d[headers[-1]].append(
                    np.sum(df[col] * df["AmountOutstanding"])
                    / np.sum(df["AmountOutstanding"])
                )
    # Rearrange header order to proper position for data files.
    headers.insert(-1, headers.pop(-4))
    table = pd.DataFrame(table_d)
    table = table[headers]  # arrange columns
    return table


if __name__ == "__main__":
    main()
