from collections import defaultdict

import pandas as pd

from lgimapy.data import Database, concat_index_dfs, Index
from lgimapy.latex import Document
from lgimapy.utils import replace_multiple

# %%

db = Database()

# %%
prev_label = "4Q18"
today = db.date("today")
prev_start = pd.to_datetime("10/1/2018")
prev_end = pd.to_datetime("1/1/2019")
maturity_range = (25, 32)

# %%
def latex_fmt(text):
    repl = {
        "$": "\$",
        "&": "\&",
    }
    return replace_multiple(text, repl)


def fmt_sector_ticker_df(df):
    df = df.reset_index().rename(columns={"index": "Ticker"})
    df = df[["Sector", "Ticker", "OAS", "OAS_prev_wides", "OAS_change"]]
    df.rename(columns=col_fmt, inplace=True)
    vals = df["Sector"]
    prev = vals[0]
    unique_vals = [prev]
    for val in vals[1:]:
        if val == prev:
            unique_vals.append(" ")
        else:
            unique_vals.append(val)
            prev = val
    df["Sector"] = unique_vals
    return df


def get_prev_wides(start, end, mat_range):
    prev_dates = db.trade_dates(start, end)[::5]
    df_list = []
    for date in prev_dates:
        db.load_market_data(date=date)
        ix = db.build_market_index(
            in_stats_index=True, maturity=mat_range
        ).subset_on_the_runs()
        df_list.append(ix.df)

    prev_ix = Index(concat_index_dfs(df_list))

    prev_oas_wides = (
        prev_ix.get_value_history("OAS").max().dropna().rename("OAS_prev_wides")
    )
    ticker_map = prev_ix.df["Ticker"].to_dict()
    df = prev_oas_wides.to_frame()
    df["Ticker"] = df.index.map(ticker_map)
    return df.groupby("Ticker").max().rename_axis(None).squeeze()


def get_current_ix(mat_range):
    db.load_market_data()
    return db.build_market_index(
        in_stats_index=True, maturity=mat_range
    ).subset_on_the_runs()


def get_spread_change_index(start, end, mat_range):
    prev_wides = get_prev_wides(start, end, mat_range)
    curr_ix = get_current_ix(mat_range)
    curr_ix.df["OAS_prev_wides"] = curr_ix.df["Ticker"].map(prev_wides)
    curr_ix.df.dropna(axis=0, subset=["OAS_prev_wides"], inplace=True)
    curr_ix.df["OAS_change"] = curr_ix.df["OAS"] - curr_ix.df["OAS_prev_wides"]
    return curr_ix


# %%

ix = get_spread_change_index(prev_start, prev_end, maturity_range)

# %%
sectors
skipped_sectors = [
    "ENERGY",
    "UTILITY",
    "BANKS",
    "BASICS",
    "FINANCIALS",
    "LIFE",
    "CONSUMER_NON_CYCLICAL",
    "TRANSPORTATION",
    "COMMUNICATIONS",
]
sectors = [s for s in db.IG_sectors() if s not in skipped_sectors]

ratings = {"A-Rated": (None, "A-"), "BBB-Rated": ("BBB+", "BBB-")}
dfs_d = defaultdict(dict)
for rating, rating_kws in ratings.items():
    rating_ix = ix.subset(rating=rating_kws)

    # Get change in sector spreads.
    sector_d = defaultdict(list)
    sector_ticker_df_list = []
    for sector in sectors:
        sector_ix = rating_ix.subset(**db.index_kwargs(sector))
        if sector_ix.is_empty():
            continue
        sector_d["Sector"].append(sector_ix.name)
        sector_d["OAS"].append(sector_ix.MEAN("OAS").iloc[0])
        sector_d[f"OAS_prev_wides"].append(
            sector_ix.MEAN("OAS_prev_wides").iloc[0]
        )
        sector_ticker_df_i = (
            sector_ix.df.set_index("Ticker")[
                ["OAS", "OAS_prev_wides", "OAS_change"]
            ]
            .sort_values("OAS_change", ascending=False)
            .rename_axis(None)
        )
        sector_ticker_df_i["Sector"] = sector_ix.name
        sector_ticker_df_list.append(sector_ticker_df_i)

    sector_df = pd.DataFrame(sector_d).set_index("Sector").rename_axis(None)
    sector_df["OAS_change"] = sector_df["OAS"] - sector_df["OAS_prev_wides"]
    sector_df.sort_values("OAS_change", ascending=False, inplace=True)
    dfs_d[rating]["Sector"] = sector_df

    # Combine sectors and tickers together sorted by sector first, then ticker.
    sector_ticker_df = pd.concat(sector_ticker_df_list)
    sort_order = {sector: i for i, sector in enumerate(sector_df.index)}
    sector_ticker_df["sort"] = sector_ticker_df["Sector"].map(sort_order)
    dfs_d[rating]["Sector-Ticker"] = sector_ticker_df.sort_values(
        ["sort", "OAS_change"], ascending=[True, False]
    ).drop("sort", axis=1)

    # Get change in ticker spreads.
    dfs_d[rating]["Ticker"] = (
        rating_ix.df.set_index("Ticker")[
            ["OAS", "OAS_prev_wides", "OAS_change"]
        ]
        .sort_values("OAS_change", ascending=False)
        .rename_axis(None)
    )


# %%
# Save data to excel
for rating, rating_d in dfs_d.items():
    for key, df in rating_d.items():
        df.to_csv(f"{rating}_{key}_OAS_change_{prev_label}.csv")


# %%
# dfs_d["BBB-Rated"]["Ticker"]
# dfs_d["BBB-Rated"]["Sector-Ticker"]
# dfs_d["BBB-Rated"]["Sector"]
# dfs_d["A-Rated"]["Ticker"]
# dfs_d["A-Rated"]["Sector"]


# %%
n_tickers = 10
n_sectors = 10
doc = Document(f"30y_spread_change_since_{prev_label}", path="reports/PM_meetings/2022_02")
doc.add_preamble(table_caption_justification="c")

col_fmt = {
    "OAS": f"OAS*{today:%-m/%d/%Y}",
    "OAS_prev_wides": f"OAS Wides*{prev_label}",
    "OAS_change": "$\Delta$OAS",
}


for rating, rating_dfs in dfs_d.items():
    doc.add_section(rating)
    sector_edit, ticker_edit = doc.add_subfigures(n=2, valign="t")
    with doc.start_edit(ticker_edit):
        df = rating_dfs["Ticker"]
        df = pd.concat((df.iloc[:n_tickers], df.iloc[-n_tickers:]))
        df.rename(columns=col_fmt, inplace=True)
        doc.add_table(
            df,
            midrule_locs=latex_fmt(df.index[n_tickers]),
            multi_row_header=True,
            prec={col: "0f" for col in df.columns},
            caption="Tickers with Largest Moves",
            adjust=True,
        )
    with doc.start_edit(sector_edit):
        sector_df = rating_dfs["Sector"]
        df = pd.concat(
            (sector_df.iloc[:n_sectors], sector_df.iloc[-n_sectors:])
        )
        df.rename(columns=col_fmt, inplace=True)
        doc.add_table(
            df,
            midrule_locs=latex_fmt(df.index[n_sectors]),
            multi_row_header=True,
            prec={col: "0f" for col in df.columns},
            caption="Sectors with Largest Moves",
            adjust=True,
        )

    doc.add_pagebreak()

    st_df = rating_dfs["Sector-Ticker"]
    st_df_wide = st_df[st_df["Sector"].isin(sector_df.index[:n_sectors])].copy()
    st_df_tight = st_df[
        st_df["Sector"].isin(sector_df.index[-n_sectors:])
    ].copy()
    st_df_tight["sort"] = range(len(st_df_tight))
    st_df_tight = st_df_tight.sort_values("sort", ascending=False).drop(
        "sort", axis=1
    )

    wide_edit, tight_edit = doc.add_subfigures(n=2, valign="t")
    with doc.start_edit(wide_edit):
        df = fmt_sector_ticker_df(st_df_wide)
        doc.add_table(
            df,
            midrule_locs=list(df[df["Sector"] != " "].index[1:]),
            multi_row_header=True,
            prec={col: "0f" for col in df.columns if "OAS" in col},
            adjust=True,
            hide_index=True,
            caption="Ticker Breakdown for Widest Moving Sectors",
        )
    with doc.start_edit(tight_edit):
        df = fmt_sector_ticker_df(st_df_tight)
        doc.add_table(
            df,
            midrule_locs=list(df[df["Sector"] != " "].index[1:]),
            multi_row_header=True,
            prec={col: "0f" for col in df.columns if "OAS" in col},
            adjust=True,
            hide_index=True,
            caption="Ticker Breakdown for Tightest Moving Sectors",
        )

    doc.add_pagebreak()
doc.save()


# %%
