import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database, groupby
from lgimapy.latex import Document

vis.style()

# %%
account = "P-LD"
old_rep = account
db = Database()
full_acnt = db.load_portfolio(account=account, universe="stats")
full_acnt_2020 = db.load_portfolio(
    account=account, universe="stats", date="12/31/2020"
)
full_acnt_2019 = db.load_portfolio(
    account=account, universe="stats", date="12/31/2019"
)
sectors = [
    "CHEMICALS",
    "CAPITAL_GOODS",
    "METALS_AND_MINING",
    "COMMUNICATIONS",
    "AUTOMOTIVE",
    "RETAILERS",
    "CONSUMER_CYCLICAL_EX_AUTOS_RETAILERS",
    "HEALTHCARE_PHARMA",
    "FOOD_AND_BEVERAGE",
    "CONSUMER_NON_CYCLICAL_OTHER",
    "ENERGY",
    "TECHNOLOGY",
    "TRANSPORTATION",
    "BANKS",
    "OTHER_FIN",
    "INSURANCE",
    "REITS",
    "UTILITY",
    "SOVEREIGN",
    "NON_CORP_OTHER",
    "OTHER_INDUSTRIAL",
]

dates = {"today": db.date("today"), "2020": "12/31/2020", "2019": "12/31/2019"}
df_list = []
for name, date in dates.items():
    db.load_market_data(date=date)
    ix = db.build_market_index(maturity=(10, None))
    df_list.append(ix.ticker_df["OAS"].rename(name))

oas_df = pd.concat(df_list, axis=1)
oas_df["oas_2020"] = oas_df["today"] - oas_df["2020"]
oas_df["oas_2019"] = oas_df["today"] - oas_df["2019"]


# %%
ratings = {"A": ("AAA", "A-"), "BBB": ("BBB+", "BBB-")}
acnts = {
    rating: full_acnt.subset(rating=rating_kws)
    for rating, rating_kws in ratings.items()
}
acnts_2020 = {
    rating: full_acnt_2020.subset(rating=rating_kws)
    for rating, rating_kws in ratings.items()
}
acnts_2019 = {
    rating: full_acnt_2019.subset(rating=rating_kws)
    for rating, rating_kws in ratings.items()
}
dfs = {}
dts_totals = {}
top_30_tickers = {}
for rating, acnt in acnts.items():
    df = (
        acnt.df.groupby("Ticker", observed=True)
        .sum()
        .sort_values("BM_DTS", ascending=False)
    )
    dts_sorted = (
        df["BM_DTS"].sort_values(ascending=False).reset_index(drop=True)
    )
    dfs[rating] = df
    dts_totals[rating] = dts_sorted[dts_sorted > 0]
    top_30_tickers[rating] = list(df.index[:30].values)

# %%


def format_table(df, oas_df, cols):
    # Dont duplicate sectors names
    vals = list(df["Sector"])
    prev = vals[0]
    unique_vals = [prev]
    for val in vals[1:]:
        if val == prev:
            unique_vals.append(" ")
        else:
            unique_vals.append(val)
            prev = val
    df["Sector"] = unique_vals
    for year in [2019, 2020]:
        col = f"oas_{year}"
        df[col] = df["Ticker"].map(oas_df[col].to_dict())

    col_order = ["Sector", "Ticker", "OAS", "oas_2020", "oas_2019", *cols[1:-1]]
    return (
        df[col_order]
        .rename(
            columns={
                "OAS": "OAS",
                "oas_2020": "$\\Delta$OAS*YE 2020",
                "oas_2019": "$\\Delta$OAS*YE 2019",
                "P_Weight": "Port*MV %",
                "P_DTS": "Port*DTS",
                "BM_DTS": "BM*DTS",
                "DTS_Diff": "DTS*OW",
                "P_OAD": "Port*OAD",
                "BM_OAD": "BM*OAD",
                "OAD_Diff": "OAD*OW",
            }
        )
        .reset_index(drop=True)
    )


def sort_table(df):
    sector_gdf = (
        df[["Sector", "BM_DTS"]]
        .groupby("Sector")
        .sum()
        .squeeze()
        .sort_values(ascending=False)
    )
    sorter_index = {sector: i for i, sector in enumerate(sector_gdf.index)}
    df["sort"] = df["Sector"].map(sorter_index)
    df.sort_values(["sort", "BM_DTS"], ascending=[True, False], inplace=True)
    return df.drop("sort", axis=1)


def calculate_total(df, name, cols):
    total_df = df.sum()[cols].to_frame().T
    total_df["OAS"] = (df["OAS"] * df["MarketValue"]).sum() / df[
        "MarketValue"
    ].sum()
    total_df["Ticker"] = np.nan
    total_df["Sector"] = name
    return total_df


# %%
# fig, ax = vis.subplots()
#
# for rating in ["BBB", "A"]:
#     dts = dts_totals[rating]
#     ax.bar(
#         dts.index,
#         dts.values,
#         color=colors[rating],
#         width=1,
#         alpha=0.8,
#         label=rating,
#     )
# ax.legend()
# ax.set_ylabel("DTS")
# ax.set_xlim((-1, 101))
# vis.savefig("Issuer_BM_DTS")

# %%
large_cap_tables = {}
small_ow_tables = {}
small_uw_tables = {}

cols = [
    "OAS",
    "P_Weight",
    "BM_DTS",
    "P_DTS",
    "DTS_Diff",
    "BM_OAD",
    "P_OAD",
    "OAD_Diff",
    "MarketValue",
]
ignored_isins = set()

for rating, acnt in acnts.items():
    large_cap_df_list = []
    ignored_tickers = set(top_30_tickers[rating])
    # Large cap table.
    for sector in sectors:
        kwargs = db.index_kwargs(sector, unused_constraints="in_stats_index")
        sector_acnt = acnt.subset(**kwargs)
        df = sector_acnt.ticker_df
        df = df[df.index.isin(top_30_tickers[rating])].sort_values(
            "BM_DTS", ascending=False
        )
        if not len(df):
            continue
        df = df[cols].reset_index()
        df["Sector"] = kwargs["name"]
        large_cap_df_list.append(df)

    large_cap_df = sort_table(pd.concat(large_cap_df_list))
    hosp_kwargs = db.index_kwargs(
        "HOSPITALS", unused_constraints="in_stats_index"
    )
    hospitals_acnt = acnt.subset(**hosp_kwargs)
    ignored_tickers |= set(hospitals_acnt.df["Ticker"])

    total = calculate_total(large_cap_df, "Total", cols)
    total_2020 = calculate_total(
        acnts_2020[rating].subset(ticker=top_30_tickers[rating]).df,
        "Total",
        cols,
    )
    total_2019 = calculate_total(
        acnts_2019[rating].subset(ticker=top_30_tickers[rating]).df,
        "Total",
        cols,
    )
    large_cap_df = large_cap_df.append(total)

    non_fin_total = calculate_total(
        acnt.subset(
            ticker=ignored_tickers,
            special_rules="~Ticker",
            financial_flag=0,
        ).df,
        "Non-Fin ex Top 30",
        cols,
    )
    non_fin_2020 = calculate_total(
        acnts_2020[rating]
        .subset(
            ticker=ignored_tickers,
            special_rules="~Ticker",
            financial_flag=0,
        )
        .df,
        "Non-Fin ex Top 30",
        cols,
    )
    non_fin_2019 = calculate_total(
        acnts_2019[rating]
        .subset(
            ticker=ignored_tickers,
            special_rules="~Ticker",
            financial_flag=0,
        )
        .df,
        "Non-Fin ex Top 30",
        cols,
    )
    large_cap_df = large_cap_df.append(non_fin_total)

    fin_total = calculate_total(
        acnt.subset(
            ticker=ignored_tickers,
            special_rules="~Ticker",
            financial_flag=1,
        ).df,
        "Fin ex Top 30",
        cols,
    )
    fin_2020 = calculate_total(
        acnts_2020[rating]
        .subset(
            ticker=ignored_tickers,
            special_rules="~Ticker",
            financial_flag=1,
        )
        .df,
        "Fin ex Top 30",
        cols,
    )
    fin_2019 = calculate_total(
        acnts_2019[rating]
        .subset(
            ticker=ignored_tickers,
            special_rules="~Ticker",
            financial_flag=1,
        )
        .df,
        "Fin ex Top 30",
        cols,
    )
    large_cap_df = large_cap_df.append(fin_total)

    non_corp_total = calculate_total(
        acnt.subset(
            ticker=ignored_tickers,
            special_rules="~Ticker",
            financial_flag=2,
        ).df,
        "Non-Corp ex Top 30",
        cols,
    )
    non_corp_2020 = calculate_total(
        acnts_2020[rating]
        .subset(
            ticker=ignored_tickers,
            special_rules="~Ticker",
            financial_flag=2,
        )
        .df,
        "Non-Corp ex Top 30",
        cols,
    )
    non_corp_2019 = calculate_total(
        acnts_2019[rating]
        .subset(
            ticker=ignored_tickers,
            special_rules="~Ticker",
            financial_flag=2,
        )
        .df,
        "Non-Corp ex Top 30",
        cols,
    )
    large_cap_df = large_cap_df.append(non_corp_total)

    hospitals_total = calculate_total(
        hospitals_acnt.df,
        "Hospitals",
        cols,
    )
    hospitals_2020 = calculate_total(
        acnts_2020[rating].subset(**hosp_kwargs).df,
        "Hospitals",
        cols,
    )
    hospitals_2019 = calculate_total(
        acnts_2019[rating].subset(**hosp_kwargs).df,
        "Hospitals",
        cols,
    )

    large_cap_df = large_cap_df.append(hospitals_total)
    large_cap_table = format_table(large_cap_df, oas_df, cols)
    col_20 = "$\Delta$OAS*YE 2020"
    col_19 = "$\Delta$OAS*YE 2019"
    large_cap_table.loc[large_cap_table["Sector"] == "Total", col_20] = (
        total["OAS"] - total_2020["OAS"]
    ).squeeze()
    large_cap_table.loc[large_cap_table["Sector"] == "Total", col_19] = (
        total["OAS"] - total_2019["OAS"]
    ).squeeze()
    large_cap_table.loc[
        large_cap_table["Sector"] == "Non-Fin ex Top 30", col_20
    ] = (non_fin_total["OAS"] - non_fin_2020["OAS"]).squeeze()
    large_cap_table.loc[
        large_cap_table["Sector"] == "Non-Fin ex Top 30", col_19
    ] = (non_fin_total["OAS"] - non_fin_2019["OAS"]).squeeze()
    large_cap_table.loc[
        large_cap_table["Sector"] == "Fin ex Top 30", col_20
    ] = (fin_total["OAS"] - fin_2020["OAS"]).squeeze()
    large_cap_table.loc[
        large_cap_table["Sector"] == "Fin ex Top 30", col_19
    ] = (fin_total["OAS"] - fin_2019["OAS"]).squeeze()
    large_cap_table.loc[
        large_cap_table["Sector"] == "Non-Corp ex Top 30", col_20
    ] = (non_corp_total["OAS"] - non_corp_2020["OAS"]).squeeze()
    large_cap_table.loc[
        large_cap_table["Sector"] == "Non-Corp ex Top 30", col_19
    ] = (non_corp_total["OAS"] - non_corp_2019["OAS"]).squeeze()
    large_cap_table.loc[large_cap_table["Sector"] == "Hospitals", col_20] = (
        hospitals_total["OAS"] - hospitals_2020["OAS"]
    ).squeeze()
    large_cap_table.loc[large_cap_table["Sector"] == "Hospitals", col_19] = (
        hospitals_total["OAS"] - hospitals_2019["OAS"]
    ).squeeze()
    large_cap_tables[rating] = large_cap_table

    # Small OW table.
    small_ow_df_list = []
    for sector in sectors:
        kwargs = db.index_kwargs(sector, unused_constraints="in_stats_index")
        sector_acnt = acnt.subset(**kwargs)
        df = sector_acnt.ticker_df
        df = df[~df.index.isin(ignored_tickers)].sort_values(
            "BM_DTS", ascending=False
        )
        df = df[cols].reset_index()
        df = df[(df["DTS_Diff"] >= 0) & (df["DTS_Diff"] <= 6)].copy()
        if not len(df):
            continue
        df["Sector"] = kwargs["name"]
        small_ow_df_list.append(df)
    small_ow_df = sort_table(pd.concat(small_ow_df_list))
    total = calculate_total(small_ow_df, "Total", cols)
    small_ow_df = small_ow_df.append(total)
    small_ow_tickers = set(small_ow_df["Ticker"].dropna())
    total_2020 = calculate_total(
        acnts_2020[rating].subset(ticker=small_ow_tickers).df, "total", cols
    )
    total_2019 = calculate_total(
        acnts_2019[rating].subset(ticker=small_ow_tickers).df, "total", cols
    )
    small_ow_table = format_table(small_ow_df, oas_df, cols)
    small_ow_table.loc[small_ow_table["Sector"] == "Total", col_20] = (
        total["OAS"] - total_2020["OAS"]
    ).squeeze()
    small_ow_table.loc[small_ow_table["Sector"] == "Total", col_19] = (
        total["OAS"] - total_2019["OAS"]
    ).squeeze()
    small_ow_tables[rating] = small_ow_table

    # Small UW table.
    small_uw_df_list = []
    for sector in sectors:
        kwargs = db.index_kwargs(sector, unused_constraints="in_stats_index")
        sector_acnt = acnt.subset(**kwargs)
        df = sector_acnt.ticker_df
        df = df[~df.index.isin(ignored_tickers)].sort_values(
            "BM_DTS", ascending=False
        )
        df = df[cols].reset_index()
        df = df[
            (df["DTS_Diff"] <= 0)
            & (df["DTS_Diff"] >= -10)
            & (df["P_Weight"] > 0)
        ].copy()
        if not len(df):
            continue
        df["Sector"] = kwargs["name"]
        small_uw_df_list.append(df)
    small_uw_df = sort_table(pd.concat(small_uw_df_list))
    total = calculate_total(small_uw_df, "Total", cols)
    small_uw_df = small_uw_df.append(total)
    small_uw_tickers = set(small_uw_df["Ticker"].dropna())
    total_2020 = calculate_total(
        acnts_2020[rating].subset(ticker=small_uw_tickers).df, "total", cols
    )
    total_2019 = calculate_total(
        acnts_2019[rating].subset(ticker=small_uw_tickers).df, "total", cols
    )
    small_uw_table = format_table(small_uw_df, oas_df, cols)
    small_uw_table.loc[small_uw_table["Sector"] == "Total", col_20] = (
        total["OAS"] - total_2020["OAS"]
    ).squeeze()
    small_uw_table.loc[small_uw_table["Sector"] == "Total", col_19] = (
        total["OAS"] - total_2019["OAS"]
    ).squeeze()
    small_uw_tables[rating] = small_uw_table


# large_cap_tables[rating]
# small_ow_tables[rating]
# small_uw_tables[rating]


# %%
doc = Document(
    f"{account}_Investable_Universe",
    path=f"reports/portfolio_trades/{db.date('today'):%Y-%m-%d}",
)
sides = 1.5
doc.add_preamble(
    margin={
        "paperheight": 32,
        "left": sides,
        "right": sides,
        "top": 0.5,
        "bottom": 0.2,
    },
    table_caption_justification="c",
    header=doc.header(
        left="P-LD Investable Universe",
        right=f"EOD {db.date('today').strftime('%B %#d, %Y')}",
    ),
    footer=doc.footer(logo="LG_umbrella"),
)
for rating, table in large_cap_tables.items():
    midrule_locs = table[table["Sector"] != " "].index[1:-4]
    total_loc = table[table["Sector"] == "Total"].index
    prec = {}
    for col in table.columns:
        if "DTS" in col:
            prec[col] = "1f"
        elif "OAD" in col:
            prec[col] = "2f"
        elif "OAS" in col:
            prec[col] = "0f"
        elif "MV" in col:
            prec[col] = "2%"

    doc.add_table(
        table,
        caption=f"{rating}-Rated Top 30 Tickers by BM DTS \\%",
        col_fmt="lll|rrrr|rrr|rrr",
        font_size="footnotesize",
        row_font={tuple(total_loc): "\\bfseries"},
        midrule_locs=midrule_locs,
        prec=prec,
        adjust=True,
        hide_index=True,
        multi_row_header=True,
    )
    doc.add_pagebreak()

for rating, table in small_ow_tables.items():
    midrule_locs = table[table["Sector"] != " "].index[1:]
    total_loc = table[table["Sector"] == "Total"].index
    doc.add_table(
        table,
        caption=f"{rating}-Rated Small Overweights",
        col_fmt="lll|rrrr|rrr|rrr",
        font_size="footnotesize",
        row_font={tuple(total_loc): "\\bfseries"},
        midrule_locs=midrule_locs,
        prec=prec,
        adjust=True,
        hide_index=True,
        multi_row_header=True,
    )
    doc.add_pagebreak()

for rating, table in small_uw_tables.items():
    midrule_locs = table[table["Sector"] != " "].index[1:]
    total_loc = table[table["Sector"] == "Total"].index
    doc.add_table(
        table,
        caption=f"{rating}-Rated Small Underweights",
        col_fmt="lll|rrrr|rrr|rrr",
        font_size="footnotesize",
        row_font={tuple(total_loc): "\\bfseries"},
        midrule_locs=midrule_locs,
        prec=prec,
        hide_index=True,
        adjust=True,
        multi_row_header=True,
    )
    doc.add_pagebreak()

doc.save()
