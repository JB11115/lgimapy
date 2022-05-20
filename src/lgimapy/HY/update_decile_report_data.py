"""
Clean .csv files form Bloomberg download of BAML ICE Index data
and combine them with previously saved data.
"""

import pandas as pd

from lgimapy.bloomberg import bdp
from lgimapy.data import Database

# %%


def update_decile_report_data():
    # Load old data.
    historical_fields = ["spreads", "yields", "prices"]
    data_dir = Database().local("HY/decile_report")
    report_fids = {f: data_dir / f"{f}.csv" for f in historical_fields}
    report_dfs = {f: read_report_csv(fid) for f, fid in report_fids.items()}

    temp_dfs = {}
    for field in historical_fields:
        if field == "spreads":
            fids = [data_dir / f"_temp_spreads_{i}.csv" for i in range(2)]
            temp_dfs[field] = pd.concat(
                (read_temp_csv(fid) for fid in fids),
                axis=1,
                sort=True,
            )
        else:
            fid = data_dir / f"_temp_{field}.csv"
            temp_dfs[field] = read_temp_csv(fid)

    temp_dfs["prices"] = temp_dfs["prices"][
        temp_dfs["prices"].index >= pd.to_datetime("1/1/1997")
    ]

    # Get formatting map from downloaded data to report data.
    col_map = {col.split("*")[0]: col for col in report_dfs["spreads"].columns}

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

    for field, report_df in report_dfs.items():
        temp_df = temp_dfs[field].copy()
        if field == "spreads":
            for col, (minuend, subtrahend) in difference_columns.items():
                temp_df[col] = temp_df[minuend] - temp_df[subtrahend]

        temp_df.columns = [col_map.get(col, col) for col in temp_df.columns]
        updated_df = combine_dfs(report_df, temp_df)
        updated_df.to_csv(report_fids[field])

    update_total_return_data(data_dir)
    update_hedged_total_return_data(data_dir)
    update_excess_return_data(data_dir)


def read_report_csv(fid):
    try:
        df = pd.read_csv(
            fid,
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
        )
        df = df[df.index <= pd.to_datetime("4/5/2022")]
        return df.dropna(how="all", axis=1)
    except FileNotFoundError:
        return pd.DataFrame()


def read_temp_csv(fid):
    try:
        raw_df = pd.read_csv(fid, usecols=(0, 1), header=None)
    except FileNotFoundError:
        return pd.DataFrame()

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


def update_total_return_data(data_dir):
    """Scrape total return data from Bloomberg and save file."""
    tret_fid = data_dir / "total_returns.csv"
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


def update_hedged_total_return_data(data_dir):
    """Scrape total return data from Bloomberg and save file."""
    db = Database()
    tret_fid = data_dir / "total_returns_hedged.csv"
    index_names = {
        "US_HY": "LF98",
        "US_BB": "BCBA",
        "US_B": "BCBH",
        "US_CCC": "HUC3",
        "EU_HY": "LP01",
        "EU_BB": "LP07",
        "EU_B": "LHYB",
        "EM_HY": "BEBG",
        "EM_BB": "-",
        "EM_B": "-",
        "US_BBB": "LDB1",
        "EU_BBB": "LECB",
        "EM_BBB": "-",
        "SP500": "SPX",
        "EURO_STOXX_50": "SX5E",
        "MSCI_EM": "MXEF",
    }
    stock_labels = {
        "SP500": "S&P 500",
        "EURO_STOXX_50": "\\EUR stoxx50",
        "MSCI_EM": "EM MSCI",
    }
    index_labels = {
        k: stock_labels.get(k, k.replace("_", " ")) for k in index_names.keys()
    }
    indexes = [ix for ix, name in index_names.items() if name != "-"]

    periods = ["1WK", "1MO", "3MO", "6MO", "YTD", "1YR", "3YR", "5YR"]
    df_list = []
    for period in periods:
        tret = db.load_bbg_data(
            indexes, "TRET_HEDGED_USD", start=db.date(period), aggregate=True
        )
        df_list.append(tret.rename(period))
    df = 100 * pd.concat(df_list, axis=1).reindex(index_names.keys())

    # Add index names as first column.
    df["Index"] = pd.Series(df.index).map(index_labels).values
    df.index = pd.Series(df.index).map(index_names).values
    df = df[["Index"] + periods]
    df.to_csv(tret_fid)


def update_excess_return_data(data_dir):
    """Scrape total return data from Bloomberg and save file."""
    db = Database()
    xsret_fid = data_dir / "excess_returns.csv"
    index_names = {
        "US_HY": "LF98",
        "US_BB": "BCBA",
        "US_B": "BCBH",
        "US_CCC": "HUC3",
        "EU_HY": "LP01",
        "EU_BB": "LP07",
        "EU_B": "LHYB",
        "EM_HY": "BEBG",
        "EM_BB": "I05040",
        "EM_B": "I05039",
        "US_BBB": "LDB1",
        "EU_BBB": "LECB",
        "EM_BBB": "I12881",
        "SP500": "SPX",
        "EURO_STOXX_50": "SX5E",
        "MSCI_EM": "-",
    }
    stock_labels = {
        "SP500": "S&P 500",
        "EURO_STOXX_50": "\\EUR stoxx50",
        "MSCI_EM": "EM MSCI",
    }
    index_labels = {
        k: stock_labels.get(k, k.replace("_", " ")) for k in index_names.keys()
    }
    indexes = [ix for ix, name in index_names.items() if name != "-"]

    periods = ["1WK", "1MO", "3MO", "6MO", "YTD", "1YR", "3YR", "5YR"]
    df_list = []
    for period in periods:
        xsret = db.load_bbg_data(
            indexes, "XSRET", start=db.date(period), aggregate=True
        )
        df_list.append(xsret.rename(period))
    df = 100 * pd.concat(df_list, axis=1).reindex(index_names.keys())

    # Add index names as first column.
    df["Index"] = pd.Series(df.index).map(index_labels).values
    df.index = pd.Series(df.index).map(index_names).values
    df = df[["Index"] + periods]
    df.to_csv(xsret_fid)


# %%
if __name__ == "__main__":
    main()
