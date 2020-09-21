"""
Clean .csv files form Bloomberg download of BAML ICE Index data
and combine them with previously saved data.
"""

import pandas as pd

from lgimapy.bloomberg import bdp
from lgimapy.utils import root

# %%


def main():
    # Load old data.
    report_spread_fid = root("data/HY/report_spreads.csv")
    report_yield_fid = root("data/HY/report_yields.csv")
    old_spread_df = pd.read_csv(
        report_spread_fid,
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )
    old_yield_df = pd.read_csv(
        report_yield_fid,
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )
    # Get formatting map from downloaded data to report data.
    col_map = {col.split("*")[0]: col for col in old_spread_df.columns}

    # Load new spread data.
    spread_fid0 = root("data/HY/oas_temp0.csv")
    spread_fid1 = root("data/HY/oas_temp1.csv")
    spread_df = pd.concat(
        (read_csv(spread_fid0), read_csv(spread_fid1)), axis=1, sort=True
    )

    # Calculate difference columns for spreads:
    difference_columns = {
        "US*B-BB": ("HUC2", "HC1N"),
        "US nf*BB-BBB": ("HC1N", "C4NF"),
        "EU*BB-BBB": ("HE1M", "EN40"),
        "US-EU nf*BBB": ("C4NF", "EN40"),
        "US-EU*BB": ("HC1N", "HE1M"),
        "US-EU*B": ("HUC2", "HE20"),
        "EM-US*BB": ("EM3B", "HC1N"),
        "EM-US*B": ("EM6B", "HUC2"),
    }
    for difference_col, (minuend, subtrahend) in difference_columns.items():
        spread_df[difference_col] = spread_df[minuend] - spread_df[subtrahend]

    spread_df.columns = [col_map.get(col, col) for col in spread_df.columns]

    # Load new yield data.
    yield_fid = root("data/HY/ytw_temp.csv")
    yield_df = read_csv(yield_fid)
    yield_df.columns = [col_map.get(col, col) for col in yield_df.columns]

    # Combine and save updated data.
    updated_spread_df = combine_dfs(old_spread_df, spread_df)
    updated_yield_df = combine_dfs(old_yield_df, yield_df)

    updated_spread_df.to_csv(report_spread_fid)
    updated_yield_df.to_csv(report_yield_fid)

    update_total_return_data()


def read_csv(fid):
    raw_df = pd.read_csv(fid, usecols=(0, 1), header=None)

    # Find split points where new index starts and ends.
    key = raw_df.iloc[1, 1]
    ilocs = list(raw_df[raw_df[1] == key].index)
    ilocs.append(len(raw_df))

    # Reformate each index to its own column.
    df_list = []
    for i, iloc in enumerate(ilocs[1:]):
        df = raw_df.iloc[ilocs[i] - 1 : iloc - 1].dropna()
        name = df.iloc[0, 1]
        df = df.iloc[2:]
        df.set_index(0, inplace=True)
        df.index.name = None
        df.columns = [name]
        df.index = pd.to_datetime(df.index, errors="coerce")
        df[name] = pd.to_numeric(df[name])
        df_list.append(df)

    # Combine columns and save.
    return pd.concat(df_list, axis=1, sort=True)


def combine_dfs(old_df, new_df):
    old_df_prev = old_df[~old_df.index.isin(new_df.index)].copy()
    return pd.concat((old_df_prev, new_df), sort=True)


def update_total_return_data():
    """Scrape total return data from Bloomberg and save file."""
    tret_fid = root("data/HY/report_total_returns.csv")
    index_names = {
        "H0A0": "US HY",
        "HC1N": "US BB",
        "HUC2": "US B",
        "HUC3": "US CCC",
        "HE00": "EU HY",
        "HE1M": "EU BB",
        "HE20": "EU B",
        "EMUH": "EM HY",
        "EM3B": "EM BB",
        "EM6B": "EM B",
        "C4NF": "US BBB",
        "ER40": "EU BBB",
        "EM2B": "EM BBB",
        "SPX": "S&P 500",
        "SX5E": "\\EUR stoxx50",
        "MXEF": "EM MSCI",
    }
    periods = ["1WK", "1MO", "3MO", "6MO", "YTD", "1YR", "3YR", "5YR"]
    df_list = []
    for period in periods:
        field = f"LAST_CLOSE_TRR_{period}"
        df_period = bdp(index_names.keys(), "Index", field).squeeze()
        # Un-annualize the 3 and 5 yr total returns.
        if period == "3YR":
            df_period = 100 * ((1 + df_period / 100) ** 3 - 1)
        elif period == "5YR":
            df_period = 100 * ((1 + df_period / 100) ** 5 - 1)
        df_list.append(df_period.rename(period))
    df = pd.concat(df_list, axis=1)

    # Add index names as first column.
    df["Index"] = pd.Series(df.index).map(index_names).values
    df = df[["Index"] + periods]
    df.to_csv(tret_fid)


# %%
if __name__ == "__main__":
    main()
