from collections import defaultdict
from inspect import cleandoc

import numpy as np
import pandas as pd
import seaborn as sns

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import nearest_date, root

# %%


def update_market_review(fid):
    db = Database()
    doc = Document(fid, path="valuation_pack", fig_dir=True, load_tex=True)

    # Market review.
    market_review_dfs = calculate_market_review_tables()
    doc.start_edit("market_review_equities_commodities")
    notes = cleandoc(
        """
        \\tiny
        *Equity sectors are GICS sectors applied to the S\\&P 500.\\\\
        *Factor ratios are quotients of respective price-to-book ratios.
        """
    )
    doc.add_table(
        market_review_dfs["equities"],
        adjust=True,
        table_notes=notes,
        caption="\\large Equities \\normalfont - Price (\$)",
        col_fmt="lrclrr",
        midrule_locs=[db.bbg_names("SP500_DISC"), "Big / Small"],
        prec={
            "$\\Delta$ 1M (%)": "1%",
            "$\\Delta$ YTD (%)": "1%",
            "1Y %tile": "0f",
        },
        div_bar_col=["$\\Delta$ 1M (%)", "$\\Delta$ YTD (%)"],
        div_bar_kws={"cmin": "firebrick", "cmax": "steelblue"},
        col_style={"1Y %tile": "\pctbar"},
    )

    doc.add_text("\\vskip1em")
    doc.add_table(
        market_review_dfs["commodities"],
        adjust=True,
        caption="\\large Commodities \\normalfont - Price (\$)",
        col_fmt="lrclrr",
        midrule_locs=db.bbg_names("OIL"),
        prec={
            "$\\Delta$ 1M (%)": "1%",
            "$\\Delta$ YTD (%)": "1%",
            "1Y %tile": "0f",
        },
        div_bar_col=["$\\Delta$ 1M (%)", "$\\Delta$ YTD (%)"],
        div_bar_kws={"cmin": "firebrick", "cmax": "steelblue"},
        col_style={"1Y %tile": "\pctbar"},
    )
    doc.end_edit()
    doc.start_edit("market_review_credit_rates")
    notes = cleandoc(
        """
        \\tiny
        *Indexes are Bloomberg Barclays indexes with spreads
        expressed to respective local government securities.
        """
    )
    doc.add_table(
        market_review_dfs["credit"],
        adjust=True,
        caption="\\large Credit \\normalfont - Spread (bp)",
        table_notes=notes,
        col_fmt="lrclrr",
        midrule_locs=db.bbg_names(["EU_IG", "EM_SOV", "CDX_IG"]),
        prec={
            "$\\Delta$ 1M (%)": "1%",
            "$\\Delta$ YTD (%)": "1%",
            "5Y %tile": "0f",
        },
        div_bar_col=["$\\Delta$ 1M (%)", "$\\Delta$ YTD (%)"],
        div_bar_kws={"cmin": "steelblue", "cmax": "firebrick"},
        col_style={"5Y %tile": "\pctbar"},
    )

    doc.add_text("\\vskip1em")
    doc.add_table(
        market_review_dfs["rates"],
        adjust=True,
        caption="\\large Rates \\normalfont - Yield (\%)",
        col_fmt="lrclrr",
        midrule_locs=db.bbg_names(["BUND_10Y", "JGB_10Y", "UK_10Y"]),
        prec={
            "$\\Delta$ 1M (bp)": "0f",
            "$\\Delta$ YTD (bp)": "0f",
            "5Y %tile": "0f",
        },
        div_bar_col=["$\\Delta$ 1M (bp)", "$\\Delta$ YTD (bp)"],
        div_bar_kws={"cmin": "steelblue", "cmax": "firebrick"},
        col_style={"5Y %tile": "\pctbar"},
    )
    doc.end_edit()

    # Market Returns.
    return_dfs = calculate_credit_return_tables()
    doc.start_edit("credit_total_returns")
    doc.add_table(
        return_dfs["tret"],
        adjust=True,
        caption="\\large Total Returns",
        col_fmt="lrr",
        prec={"$\\Delta$ 1M (%)": "2%", "$\\Delta$ YTD (%)": "2%"},
        div_bar_col=["$\\Delta$ 1M (%)", "$\\Delta$ YTD (%)"],
        div_bar_kws={"cmin": "firebrick", "cmax": "steelblue"},
    )
    doc.end_edit()
    doc.start_edit("credit_excess_returns")
    doc.add_table(
        return_dfs["xsret"],
        adjust=True,
        caption="\\large Excess Returns",
        col_fmt="lrr",
        prec={"$\\Delta$ 1M (bp)": "0f", "$\\Delta$ YTD (bp)": "0f"},
        div_bar_col=["$\\Delta$ 1M (bp)", "$\\Delta$ YTD (bp)"],
        div_bar_kws={"cmin": "firebrick", "cmax": "steelblue"},
    )
    doc.end_edit()
    doc.save_tex()

    # Cross asset plot
    update_cross_asset_trets()


def calculate_market_review_tables():
    db = Database()
    sec = secuirites()
    dfs = {}

    # Equities.
    df_sec = db.load_bbg_data(sec["equities"], "price", start=db.date("2y"))
    eq_names = {col: db.bbg_names(col) for col in df_sec.columns}

    # Build ratio columns.
    df_ratios = db.load_bbg_data(
        sec["equity_ratios"], "pb_ratio", start=db.date("2y")
    )
    df_ratios["Big / Small"] = df_ratios.eval("SP500 / RUSSELL_2000")
    df_ratios["Growth / Value"] = df_ratios.eval("SP500_GROW / SP500_VALU")
    df_ratios["Momentum / S&P 500"] = df_ratios.eval("SP500_MOM / SP500")
    df_ratios["High Vol / Low Vol"] = df_ratios.eval("SP500_HIGHV / SP500_LOWV")
    df_ratios = df_ratios[
        [col for col in df_ratios.columns if col not in sec["equity_ratios"]]
    ]
    df_sec = pd.concat([df_sec, df_ratios], sort=True, axis=1)

    d = defaultdict(list)
    for col in df_sec.columns:
        s = df_sec[col].dropna()
        d["Name"].append(eq_names.get(col, col))
        if col in eq_names:
            d["Last"].append(f"{s[-1]:,.0f}")
            d["1Y Range"].append(f"({np.min(s):.0f}, {np.max(s):.0f})")
        else:
            d["Last"].append(f"{s[-1]:,.2f}")
            d["1Y Range"].append(f"({np.min(s):.2f}, {np.max(s):.2f})")
        last_year = s[
            s.index >= nearest_date(db.date("1Y"), s.index, after=False)
        ]
        d["1Y %tile"].append(100 * last_year.rank(pct=True)[-1])
        for date in ["1M", "YTD"]:
            start_date = nearest_date(db.date(date), s.index, after=False)
            start_val = s.loc[start_date]
            delta = (s[-1] - start_val) / start_val
            d[f"$\\Delta$ {date} (%)"].append(delta)

    df = pd.DataFrame(d).set_index("Name", drop=True)
    del df.index.name
    dfs["equities"] = df.copy()

    # Commodities.
    df_sec = db.load_bbg_data(sec["commodities"], "price", start=db.date("2Y"))
    d = defaultdict(list)
    for col in df_sec.columns:
        s = df_sec[col].dropna()
        d["Name"].append(db.bbg_names(col))
        d["Last"].append(f"{s[-1]:,.0f}")
        d["1Y Range"].append(f"({np.min(s):.0f}, {np.max(s):.0f})")
        last_year = s[
            s.index >= nearest_date(db.date("1Y"), s.index, after=False)
        ]
        d["1Y %tile"].append(100 * last_year.rank(pct=True)[-1])
        for date in ["1M", "YTD"]:
            start_date = nearest_date(db.date(date), s.index, after=False)
            start_val = s.loc[start_date]
            delta = (s[-1] - start_val) / start_val
            d[f"$\\Delta$ {date} (%)"].append(delta)

    df = pd.DataFrame(d).set_index("Name", drop=True)
    del df.index.name
    dfs["commodities"] = df.copy()

    # Credit.
    df_sec = db.load_bbg_data(sec["credit"], "OAS", start=db.date("6Y"))
    d = defaultdict(list)
    for col in df_sec.columns:
        s = df_sec[col].dropna()
        d["Name"].append(db.bbg_names(col))
        d["Last"].append(f"{s[-1]:,.0f}")
        d["5Y Range"].append(f"({np.min(s):.0f}, {np.max(s):.0f})")
        last_year = s[
            s.index >= nearest_date(db.date("5Y"), s.index, after=False)
        ]
        d["5Y %tile"].append(100 * last_year.rank(pct=True)[-1])
        for date in ["1M", "YTD"]:
            start_date = nearest_date(db.date(date), s.index, after=False)
            start_val = s.loc[start_date]
            delta = (s[-1] - start_val) / start_val
            d[f"$\\Delta$ {date} (%)"].append(delta)

    df = pd.DataFrame(d).set_index("Name", drop=True)
    del df.index.name
    dfs["credit"] = df.copy()

    # Rates.
    df_sec = db.load_bbg_data(sec["rates"], "YTW", start=db.date("6Y"))
    d = defaultdict(list)
    for col in df_sec.columns:
        s = df_sec[col].dropna()
        d["Name"].append(db.bbg_names(col))
        d["Last"].append(f"{s[-1]:.2%}")
        d["5Y Range"].append(f"({np.min(s):.2%}, {np.max(s):.2%})")
        last_year = s[
            s.index >= nearest_date(db.date("5Y"), s.index, after=False)
        ]
        d["5Y %tile"].append(100 * last_year.rank(pct=True)[-1])
        for date in ["1M", "YTD"]:
            start_date = nearest_date(db.date(date), s.index, after=False)
            start_val = s.loc[start_date]
            delta = (s[-1] - start_val) * 1e4
            d[f"$\\Delta$ {date} (bp)"].append(delta)

    df = pd.DataFrame(d).set_index("Name", drop=True)
    del df.index.name
    dfs["rates"] = df.copy()

    return dfs


def calculate_credit_return_tables():
    db = Database()
    sec = secuirites()
    dfs = {}

    # Load data.
    no_xsret_secs = ["CDX_EM", "EM_CORP"]
    xsret_sec = [sec for sec in sec["credit"] if sec not in no_xsret_secs]
    tret_df = db.load_bbg_data(sec["credit"], "tret", start=db.date("2Y"))
    xsret_df = db.load_bbg_data(xsret_sec, "xsret", start=db.date("2Y"))

    # Total returns.
    d = defaultdict(list)
    for col in tret_df.columns:
        s = tret_df[col].dropna()
        d["Name"].append(db.bbg_names(col))
        for date in ["1M", "YTD"]:
            start_date = nearest_date(db.date(date), s.index, after=False)
            start_val = s.loc[start_date]
            delta = (s[-1] - start_val) / start_val
            d[f"$\\Delta$ {date} (%)"].append(delta)

    df = pd.DataFrame(d).set_index("Name", drop=True)
    del df.index.name
    dfs["tret"] = df.sort_values("$\\Delta$ 1M (%)", ascending=False)

    # Excess returns.
    d = defaultdict(list)
    derivatives = ["CDX_HY", "CDX_IG", "ITRAXX_MAIN", "ITRAXX_XOVER"]
    for col in xsret_df.columns:
        d["Name"].append(db.bbg_names(col))
        for date in ["1M", "YTD"]:
            start_date = nearest_date(db.date(date), s.index, after=False)
            derivative_flag = col in derivatives
            xsret = aggregate_excess_returns(
                xsret_df[col],
                tret_df[col],
                start=start_date,
                derivative=derivative_flag,
            )
            d[f"$\\Delta$ {date} (bp)"].append(xsret)

    df = pd.DataFrame(d).set_index("Name", drop=True)
    del df.index.name
    dfs["xsret"] = df.sort_values("$\\Delta$ 1M (bp)", ascending=False)

    return dfs


def aggregate_excess_returns(xsret, tret, start, end=None, derivative=False):
    # Combine data and drop any missing dates.
    data_df = pd.concat(
        [tret.rename("tret"), xsret.rename("xsret")], axis=1, sort=True
    ).dropna()

    if derivative:
        xsret_s = xsret.dropna()
        start_date = nearest_date(
            start, xsret_s.index, inclusive=False, after=False
        )
        start_val = xsret_s.loc[start_date]
        cur_val = xsret_s[-1] if end is None else xsret_s.loc[end]
        return 1e4 * (cur_val / start_val - 1)

    # Split DataFrame into months.
    month_dfs = [df for _, df in data_df.groupby(pd.Grouper(freq="M"))]
    tret_ix_0 = month_dfs[0]["tret"][-1]  # last value of prev month
    xs_col_month_list = []
    for df in month_dfs[1:]:  # first month is likely incomplete
        a = np.zeros(len(df))
        for i, row in enumerate(df.itertuples()):
            tret_ix, cum_xsret = row[1], row[2] / 100
            if i == 0:
                a[i] = cum_xsret
                tret = (tret_ix - tret_ix_0) / tret_ix_0
                prev_tret_ix = tret_ix
                prev_cum_rf_ret = 1 + tret - cum_xsret
            else:
                cum_tret = (tret_ix - tret_ix_0) / tret_ix_0
                tret = tret_ix / prev_tret_ix - 1
                rf_ret = (cum_tret - cum_xsret + 1) / prev_cum_rf_ret - 1
                a[i] = tret - rf_ret
                prev_tret_ix = tret_ix
                prev_cum_rf_ret *= 1 + rf_ret

        tret_ix_0 = tret_ix
        xs_col_month_list.append(pd.Series(a, index=df.index))

    xsret_s = pd.concat(xs_col_month_list, sort=True)
    tret_s = (data_df["tret"] / data_df["tret"].shift(1) - 1)[1:]
    xsret_s = xsret_s[xsret_s.index >= start]
    tret_s = tret_s[tret_s.index >= start]
    if end is not None:
        xsret_s = xsret_s[xsret_s.index <= end]
        tret_s = tret_s[tret_s.index <= end]

    rf_ret_s = tret_s - xsret_s
    total_ret = np.prod(1 + tret_s) - 1
    rf_total_ret = np.prod(1 + rf_ret_s) - 1
    return (total_ret - rf_total_ret) * 1e4


def update_cross_asset_trets():
    """
    Update plot comparing total returns among
    credit, equities, rates, and commoditites.
    """
    # Load data.
    db = Database()
    sec = secuirites()["cross_asset"]
    tret_df = db.load_bbg_data(sec, "tret", start=db.date("2m"))

    # Calculate total returns.
    d = defaultdict(list)
    for col in tret_df.columns:
        s = tret_df[col].dropna()
        d["Name"].append(db.bbg_names(col))
        start_date = nearest_date(db.date("1M"), s.index, after=False)
        start_val = s.loc[start_date]
        delta = (s[-1] - start_val) / start_val
        d["delta"].append(delta)
    df = pd.Series(d["delta"], index=d["Name"]).sort_values(ascending=False)

    # Make plot.
    vis.style()
    pal = sns.set_palette("coolwarm")
    fig, ax = vis.subplots(figsize=(12, 2.1))
    x_pos = np.arange(len(df))
    ax.bar(
        x_pos,
        df,
        color=vis.coolwarm(df.values[::-1], cmap="coolwarm_r")[::-1],
        alpha=0.8,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df.index, rotation=-45, ha="left", fontsize=8)

    pal = sns.diverging_palette(15, 250, n=1000, center="dark").as_hex()
    tick_colors = vis.coolwarm(df.values[::-1], pal=pal)[::-1]
    for xtick, color in zip(ax.get_xticklabels(), tick_colors):
        xtick.set_color(color)
        if xtick.get_text() in {"US Long Credit", "US Market Credit"}:
            xtick.set_fontweight("bold")

    ax.grid(False, axis="x")
    ax.locator_params(axis="y", nbins=5)
    ax.tick_params(axis="y", labelsize=8)
    vis.format_yaxis(ax, "{x:.0%}")
    fig_dir = root("latex/valuation_pack/fig")
    vis.savefig(fig_dir / "cross_asset_trets")
    vis.close()


def secuirites():
    """Securities for market analysis."""
    return {
        "equities": [
            "MSCI_ACWI",
            "SP500",
            "RUSSELL_2000",
            "MSCI_EU",
            "TOPIX",
            "MSCI_EM",
            "SP500_DISC",
            "SP500_STAP",
            "SP500_ENER",
            "SP500_FINS",
            "SP500_HEAL",
            "SP500_INDU",
            "SP500_MATS",
            "SP500_REAL",
            "SP500_INFO",
            "SP500_TELE",
            "SP500_UTIL",
        ],
        "equity_ratios": [
            "SP500",
            "SP500_MOM",
            "SP500_VALU",
            "SP500_GROW",
            "SP500_HIGHV",
            "SP500_LOWV",
            "RUSSELL_2000",
        ],
        "rates": [
            "UST_2Y",
            "UST_5Y",
            "UST_10Y",
            "UST_30Y",
            "UST_10Y_RY",
            "UST_10Y_BE",
            "UST_2Y_10Y",
            "UST_5Y_30Y",
            "BUND_10Y",
            "BUND_30Y",
            "JGB_10Y",
            "JGB_30Y",
            "UK_10Y",
            "UK_30Y",
        ],
        "commodities": ["GSCI", "OIL", "GOLD", "SILVER", "COPPER", "IRON"],
        "credit": [
            "US_IG",
            "US_IG_10+",
            "US_CORP",
            "US_HY",
            "US_BBB",
            "US_BB",
            "EU_IG",
            "EU_CORP",
            "EU_HY",
            "GBP_CORP",
            "EM_SOV",
            "EM_CORP",
            "EM_IG",
            "EM_HY",
            "CDX_IG",
            "CDX_HY",
            "CDX_EM",
            "ITRAXX_MAIN",
            "ITRAXX_XOVER",
        ],
        "cross_asset": [
            "MSCI_ACWI",
            "SP500",
            "RUSSELL_2000",
            "MSCI_EU",
            "MSCI_EM",
            "TOPIX",
            "US_IG",
            "US_IG_10+",
            "US_HY",
            "CDX_IG",
            "CDX_HY",
            "EU_CORP",
            "EU_HY",
            "GBP_CORP",
            "EM_CORP",
            "EM_IG",
            "EM_HY",
            "UST_5Y",
            "UST_10Y",
            "UST_30Y",
            "BUND_10Y",
            "JGB_10Y",
            "GSCI",
            "OIL",
            "GOLD",
        ],
    }


# %%

if __name__ == "__main__":
    fid = "first_draft"
    from datetime import datetime as dt

    fid = f"{dt.today().strftime('%Y-%m-%d')}_Valuation_Pack"
    update_market_review(fid)
