from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.models import simulate_rating_migrations
from lgimapy.stats import mode
from lgimapy.utils import root, to_list

# %%


def sectors():
    return [
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
        "INSURANCE_EX_HEALTHCARE",
        "REITS",
        "UTILITY",
        "SOVEREIGN",
        "NON_CORP_OTHER",
        "OTHER_INDUSTRIAL",
    ]


def potential_rising_star_dfs():
    db = Database()
    db.load_market_data()

    hy_ix = db.build_market_index(in_H0A0_index=True)
    bb_ix = hy_ix.subset(rating=("BB+", "BB-"))
    ix_rs = simulate_rating_migrations(
        bb_ix,
        "upgrade",
        threshold="BB+",
        notches=[1, 2],
        max_individual_agency_notches=1,
    )
    ix_rs.df["RS_Sector"] = np.nan
    hy_mv = hy_ix.total_value().iloc[0]

    d = defaultdict(list)
    d1 = defaultdict(list)
    d2 = defaultdict(list)
    hy_sectors = [s for s in sectors() if s != "SOVEREIGN"]
    for sector in hy_sectors:
        kwargs = db.index_kwargs(
            sector, unused_constraints=["in_stats_index", "OAS"]
        )
        # Find value of sector that is IG near a fallen angel.
        near_rs_ix = ix_rs.subset(**kwargs)
        ix_rs.df.loc[
            ix_rs.df["ISIN"].isin(near_rs_ix.isins), "RS_Sector"
        ] = kwargs["name"]
        near_rs_df = near_rs_ix.df.copy()

        if len(near_rs_df):
            near_rs_mv = near_rs_ix.total_value().iloc[0]
        else:
            near_rs_mv = 0
        # Find value of sector that is HY.
        hy_sector_ix = hy_ix.subset(**kwargs)
        if len(hy_sector_ix.df):
            hy_sector_mv = hy_sector_ix.total_value().iloc[0]
        else:
            hy_sector_mv = 0
        d["Sector"].append(kwargs["name"])
        for notches in [2, 1]:
            notch_df = near_rs_df[near_rs_df[f"{notches}Notch"] > 0]
            mv = notch_df["MarketValue"].sum() / 1e3
            d[f"{notches} Agency*MV (\\$B)"].append(mv)

        d["BB*MV (\\$B)"].append(hy_sector_mv / 1e3)

        one_notch_df = near_rs_df[near_rs_df["1Notch"] > 0]
        ticker_df = aggregate_tickers(one_notch_df)
        for i in range(len(ticker_df)):
            ticker, mv, rating = ticker_stats(ticker_df.iloc[i, :], "upgrade")
            d1["Ticker"].append(ticker)
            d1["Sector"].append(kwargs["name"])
            d1["MV (\\$B)"].append(mv)
            d1["Ratings*SP/Moody/Fitch"].append(rating)

        two_notch_df = near_rs_df[
            (near_rs_df["2Notch"] > 0) & (near_rs_df["1Notch"] == 0)
        ]
        ticker_df = aggregate_tickers(two_notch_df)
        for i in range(len(ticker_df)):
            ticker, mv, rating = ticker_stats(ticker_df.iloc[i, :], "upgrade")
            d2["Ticker"].append(ticker)
            d2["Sector"].append(kwargs["name"])
            d2["MV (\\$B)"].append(mv)
            d2["Ratings*SP/Moody/Fitch"].append(rating)

    sector_df = (
        pd.DataFrame(d)
        .sort_values("1 Agency*MV (\\$B)", ascending=False)
        .replace(0, np.nan)
    )
    sector_df = sector_df[sector_df.isna().sum(axis=1) < 2]
    one_notch_ticker_df = pd.DataFrame(d1).sort_values(
        "MV (\\$B)", ascending=False
    )
    two_notch_ticker_df = pd.DataFrame(d2).sort_values(
        "MV (\\$B)", ascending=False
    )
    return ix_rs.df, sector_df, one_notch_ticker_df, two_notch_ticker_df


def build_potential_fallen_angel_table():
    db = Database()
    db.load_market_data()

    ig_ix = db.build_market_index(in_stats_index=True)
    hy_ix = db.build_market_index(in_H0A0_index=True)
    bbb_ix = ig_ix.subset(rating=("BBB+", "BBB-"))
    ix_fa = simulate_rating_migrations(
        bbb_ix, "downgrade", threshold="BBB-", notches=[1, 2, 3]
    )

    ig_mv = ig_ix.total_value().iloc[0]
    hy_mv = hy_ix.total_value().iloc[0]

    d = defaultdict(list)
    for sector in sectors():
        kwargs = db.index_kwargs(
            sector, unused_constraints=["in_stats_index", "OAS"]
        )
        # Find value of sector that is IG near a fallen angel.
        near_fa_ix = ix_fa.subset(**kwargs)
        near_fa_df = near_fa_ix.df.copy()
        if len(near_fa_df):
            near_fa_mv = near_fa_ix.total_value().iloc[0]
        else:
            near_fa_mv = 0
        # Find value of sector that is HY.
        hy_sector_ix = hy_ix.subset(**kwargs)
        if len(hy_sector_ix.df):
            hy_sector_mv = hy_sector_ix.total_value().iloc[0]
        else:
            hy_sector_mv = 0
        d["Sector"].append(kwargs["name"])
        for notches in [3, 2, 1]:
            notch_df = near_fa_df[near_fa_df[f"{notches}Notch"] > 0]
            mv = notch_df["MarketValue"].sum() / 1e3
            d[f"{notches} Notch*MV (\\$B)"].append(mv)

        d["HY*MV (\\$B)"].append(hy_sector_mv / 1e3)
        d["MV %*of H0A0"].append(hy_sector_mv / hy_mv)

        one_notch_df = near_fa_df[near_fa_df["1Notch"] > 0]
        ticker, mv, rating = largest_ticker_stats(one_notch_df)
        d["1 Notch*Ticker"].append(ticker)
        d["MV (\\$B)"].append(mv)
        d["Ratings*SP/Moody/Fitch"].append(rating)

        two_notch_df = near_fa_df[near_fa_df["2Notch"] > 0]
        ticker, mv, rating = largest_ticker_stats(two_notch_df, ticker=ticker)
        d["2 Notch*Ticker"].append(ticker)
        d["MV (\\$B) "].append(mv)
        d["Ratings *SP/Moody/Fitch"].append(rating)

    df = (
        pd.DataFrame(d)
        .sort_values("2 Notch*MV (\\$B)", ascending=False)
        .replace(0, np.nan)
    )
    return df


def aggregate_tickers(df):
    agg_rules = {
        "MarketValue": np.sum,
        "SPRating": mode,
        "MoodyRating": mode,
        "FitchRating": mode,
        "SPOutlook": mode,
        "MoodyOutlook": mode,
        "FitchOutlook": mode,
    }
    return (
        df.groupby("Ticker", observed=True)
        .aggregate(agg_rules)
        .rename_axis(None)
    )


def largest_ticker_stats(df, migration="downgrade", ticker=None):
    ticker_df = aggregate_tickers(df)
    if ticker is not None:
        ignored_tickers = to_list(ticker, dtype=str)
        ticker_df = ticker_df[~ticker_df.index.isin(ignored_tickers)]
    if not len(ticker_df):
        # No potential fallen angels.
        return None, None, None

    largest_ticker = ticker_df.sort_values("MarketValue").iloc[-1, :]
    return ticker_stats(largest_ticker)


def ticker_stats(s, migration="downgrade"):
    ticker = s.name
    mv = s["MarketValue"] / 1e3
    rating = ""
    for agency in ["SP", "Moody", "Fitch"]:
        rating += s[f"{agency}Rating"]
        if (migration == "downgrade" and s[f"{agency}Outlook"] > 0) or (
            migration == "upgrade" and s[f"{agency}Outlook"] < 0
        ):
            rating += "*"
        if agency != "Fitch":
            rating += "/"
    return ticker, mv, rating


def last_12m_ratings_actions():
    db = Database()
    today = db.date("today")
    current_month = today.month
    agencies = ["SP", "Moody", "Fitch"]
    d = defaultdict(list)
    for months_ago in range(11, -1, -1):
        month = current_month - months_ago
        if month <= 0:
            month += 12
            year = today.year - 1
        else:
            year = today.year
        try:
            month_start = db.date("MONTH_START", f"{month}/15/{year}")
            month_end = db.date("month_end", month_start)
        except IndexError:
            month_start = db.date("MONTH_START")
            month_end = db.date("today")

        ratings_df = db.rating_changes(month_start, month_end)
        ratings_df.head()
        d["Month"].append(month_start.strftime("%b"))
        for agency in agencies:
            downgrades = ratings_df[
                (ratings_df["USCreditReturnsFlag"] == 1)
                & (ratings_df[f"{agency}Rating_CHANGE"] < 0)
            ]
            downgrade_notional = (
                (
                    downgrades[f"{agency}Rating_CHANGE"]
                    * downgrades["AmountOutstanding"]
                )
                .abs()
                .sum()
            )
            d[agency].append(downgrade_notional)
        fallen_angels = db.rating_changes(
            month_start, month_end, fallen_angels=True
        )
        fallen_angels = fallen_angels[
            fallen_angels["USCreditReturnsFlag"] == True
        ]
        d["Fallen Angels"].append(fallen_angels["AmountOutstanding"].sum())
        rising_stars = db.rating_changes(
            month_start, month_end, rising_stars=True
        )
        rising_stars = rising_stars[rising_stars["H0A0Flag"] == True]
        d["Rising Stars"].append(rising_stars["AmountOutstanding"].sum())
        df = pd.DataFrame(d).set_index("Month", drop=True).rename_axis(None)
        df["Net"] = df["Rising Stars"] - df["Fallen Angels"]
        df
    return df / 1e3


def plot_downgrades(df, path):
    plot_df = df[["SP", "Moody", "Fitch"]]
    colors = vis.colors("ryb")
    fig, ax = vis.subplots(figsize=(10, 4))
    vis.format_yaxis(ax, "${x:.0f}B")
    plot_df.plot.bar(rot=0, color=colors, ax=ax, alpha=0.9, width=0.7)
    ax.grid(False, axis="x")
    ax.set_title("Last 12m IG Downgrades\n", fontweight="bold")
    ax.set_ylabel("Notional * Notches")
    ax.legend(
        loc="upper center",
        fontsize=12,
        ncol=4,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.5, 1.12),
    )
    vis.savefig("recent_downgrades", path=path, dpi=200)
    vis.close()


def plot_fallen_angels_and_rising_stars(df, path):
    bar_plot_df = df[["Fallen Angels", "Rising Stars"]].copy()
    bar_plot_df["Fallen Angels"] *= -1
    fallen_angels = -df["Fallen Angels"]
    rising_stars = df["Rising Stars"]
    red, blue, yellow = vis.colors("ryb")
    colors = [red, blue]
    fig, ax = vis.subplots(figsize=(10, 4))
    vis.format_yaxis(ax, "${x:.0f}B")
    kwargs = {"rot": 0, "ax": ax, "alpha": 0.9, "width": 0.7}
    fallen_angels.plot.bar(color=red, **kwargs)
    rising_stars.plot.bar(color=blue, **kwargs)
    ax.plot(
        range(len(df)),
        df["Net"],
        lw=2,
        marker="s",
        ms=5,
        color="k",
        label="Net",
    )

    # Pad edges.
    vmin = df["Net"].min()
    vmin = min(vmin * 1.2, vmin - 5)
    vmax = df["Net"].max()
    vmax = max(vmax * 1.2, vmax + 5)
    ax.set_ylim(vmin, vmax)
    ax.grid(False, axis="x")
    ax.set_title("Last 12m Fallen Angels / Rising Stars\n\n", fontweight="bold")
    ax.legend(
        loc="upper center",
        fontsize=12,
        ncol=3,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.5, 1.23),
    )
    n_patches = len(ax.patches) / 2
    for i, p in enumerate(ax.patches):
        if i < n_patches:
            # Fallen Angels.
            color = red
            va = "top"
            height = p.get_height() - 2
        else:
            # Rising Stars.
            color = blue
            va = "bottom"
            height = p.get_height() + 1

        ax.annotate(
            f"{abs(p.get_height()):.0f}",
            (p.get_x() + p.get_width() / 2, height),
            va=va,
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=color,
        )
    vis.savefig("net_fallen_angels", path=path, dpi=200)
    vis.close()


def add_fallen_angel_table(df, doc):
    prec = {}
    for col in df.columns:
        if "HY" in col and "$B" in col:
            prec[col] = "0f"
        elif "$B" in col:
            prec[col] = "1f"
        elif "%" in col:
            prec[col] = "1%"

    cap = "Potential Fallen Angel Impact"
    footnote = """
        \\vspace{-2.5ex}
        \\scriptsize
        \\itemsep0.1em
        \\item
        Potential fallen angels are defined as issuers
        currently with negative outlook(s) from rating agencies
        where any combination of 1/2 notch downgrades from these
        outlooks would result in a downgrade to HY.

        \\item
        Individual tickers shown are the largest tickers (if any)
        by MV that meet the above criteria and would become a
        fallen angel with a cumulative 1 or 2 notch downgrade
        respectively.

        \\item
        Individual agency ratings market with an * indicate that
        the issuer is on negative outlook at the respective agency.
        """
    doc.add_table(
        df,
        caption=cap,
        table_notes=footnote,
        font_size="footnotesize",
        table_notes_justification="l",
        adjust=True,
        hide_index=True,
        multi_row_header=True,
        col_fmt="ll|rrr|rr|ccc|ccc",
        prec=prec,
    )


def add_rising_star_sector_table(df, doc):
    prec = {}
    for col in df.columns:
        if "BB*MV" in col and "$B" in col:
            prec[col] = "0f"
        elif "$B" in col:
            prec[col] = "1f"
        elif "%" in col:
            prec[col] = "1%"

    cap = "Potential Rising Star Sector Breakdown"
    doc.add_table(
        df,
        caption=cap,
        font_size="footnotesize",
        table_notes_justification="l",
        adjust=True,
        hide_index=True,
        multi_row_header=True,
        col_fmt="ll|rr|r",
        prec=prec,
    )


def add_rising_star_ticker_table(df, caption, doc, db):
    df = df.copy()

    # Find potential upside from comps.
    # First element of tuple is isin(s) for rising star candidate.
    # Second element is isin(s) for IG comps.
    comp_d = {
        "KHC": (["US50077LAV80"], ["US205887CC49", "US21036PBF45"]),
        "TOL": (["US88947EAU47"], ["US526057CD41"]),
        "OVV": (["US651290AR99"], ["US25179MAV54"]),
        "MDC": (["US552676AU23"], ["US526057CD41"]),
        "CLR": (["US212015AT84"], ["US136385BA87", "US25278XAR08"]),
        "MTNA": (
            ["US03938LBC72"],
            ["US50249AAC71", "US260543DC49"],
        ),
        "DB": (
            ["US251526CF47"],
            ["US06652KAB98", "US06738EBP97", "US639057AB46"],
        ),
        "NFLX": (["US64110LAV80"], ["US124857AZ68", "US25470DBJ72"]),
        "CNC": (["US15135BAX91"], ["US126650DN71", "US404119BX69"]),
        "WEIRLN": (["US94876QAA40"], ["US489170AF77"]),
        "PPC": (["US72147KAF57"], ["US832248BD93"]),
        "NWL": (
            ["US651229AY21"],
            ["US205887CE05", "US60871RAH30", "US418056AU19"],
        ),
        "FE": (["US337932AP26"], ["US059165EN63", "US69352PAQ63"]),
    }
    comp_offset = {"WEIRLN": 20}
    comps = []
    upside = []
    for ticker in df["Ticker"]:
        try:
            rep, comp = comp_d[ticker]
        except KeyError:
            # No comp.
            comps.append(np.nan)
            upside.append(np.nan)
        else:
            rep_oas = db.build_market_index(isin=rep).OAS().iloc[-1]
            comp_ix = db.build_market_index(isin=comp)
            comp_oas = comp_ix.OAS().iloc[-1] - comp_offset.get(ticker, 0)
            comp_tickers = sorted(comp_ix.df["Ticker"].unique())
            tickers = ", ".join(comp_tickers)
            comp_og_mat = comp_ix.df["OriginalMaturity"].mean()
            comp_curr_mat = comp_ix.df["MaturityYears"].mean()
            if comp_og_mat - comp_curr_mat > 4:
                comps.append(f"{tickers}: Rolled Down {comp_curr_mat:.0f}yr")
            else:
                comps.append(f"{tickers}: {comp_og_mat:.0f}yr")
            upside.append(rep_oas - comp_oas)
    df["BBB Comps"] = comps
    df["Potential*Upside (bp)"] = upside

    footnote = """
        \\vspace{-2.5ex}
        \\scriptsize
        \\itemsep0.8em
        \\item
        Potential rising stars are defined as issuers currently with
        positive outlook(s) from rating agencies where a positive
        outlook turned into a 1 notch upgrade from a single agency or
        two agencies both upgrading positive outlooks 1 notch
        would result in an upgrade to IG.

        \\item
        Individual agency ratings market with an * indicate that
        the issuer is on postive outlook at the respective agency.

        \\item
        Potential Upside assumes full compression to the comped
        bond(s) which are market value weighted to a single OAS value.
        """
    doc.add_table(
        df,
        caption=caption,
        font_size="footnotesize",
        table_notes=footnote if caption.startswith("2") else None,
        table_notes_justification="l",
        adjust=True,
        hide_index=True,
        multi_row_header=True,
        col_fmt="lll|r|c|cc",
        prec={"MV (\\$B)": "1f", "Potential*Upside (bp)": "0f"},
    )


def get_outlook_stats(df, name):
    d = {
        "2+ Pos": df[df["NetOutlook"] < -1]["MarketValue"].sum(),
        "1 Pos": df[df["NetOutlook"] == -1]["MarketValue"].sum(),
        "Stable": df[df["NetOutlook"] == 0]["MarketValue"].sum(),
        "1 Neg": df[df["NetOutlook"] == 1]["MarketValue"].sum(),
        "2+ Neg": df[df["NetOutlook"] > 1]["MarketValue"].sum(),
        "Pos": df[df["NetOutlook"] < 0]["MarketValue"].sum(),
        "Neg": df[df["NetOutlook"] > 0]["MarketValue"].sum(),
    }
    s = pd.Series(d, name=name)
    return s / s.drop(["Pos", "Neg"]).sum()


def plot_bb_outlooks(df, path):
    db = Database()
    df = df.copy()
    outlook_cols = [col for col in df.columns if "Outlook" in col]
    df["NetOutlook"] = df[outlook_cols].sum(axis=1)
    hy_sectors = [s for s in sectors() if s != "SOVEREIGN"]
    total_row = get_outlook_stats(df, "All BB")
    total_row["Pos"] = 100
    df_list = [total_row]
    for raw_sector in hy_sectors:
        sector = db.index_kwargs(raw_sector)["name"]
        sector_df = df[df["RS_Sector"] == sector]
        df_list.append(get_outlook_stats(sector_df, sector))

    plot_df = (
        pd.concat(df_list, axis=1)
        .T.sort_values(["Pos", "Neg"], ascending=[True, False])
        .drop(columns=["Pos", "Neg"])
    )

    red, blue, __ = vis.colors("ryb")
    colors = [blue, "skyblue", "grey", "firebrick", red]
    fig, ax = vis.subplots(figsize=[10, 10])
    ax.set_title("Current Outlooks for BB bonds (by MV)\n\n", fontweight="bold")
    plot_df.plot.barh(stacked=True, color=colors, alpha=0.9, ax=ax)
    vis.format_xaxis(ax, xtickfmt="{x:.0%}")
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(
        loc="upper center",
        fontsize=12,
        ncol=5,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.5, 1.065),
    )
    vis.savefig("BB_outlooks", path=path, dpi=200)
    vis.close()


def update_fallen_angel_analysis(fid):
    doc = Document(
        fid, path="reports/valuation_pack", fig_dir=True, load_tex=True
    )
    vis.style()
    path = root("reports/valuation_pack/fig")

    pfa_df = build_potential_fallen_angel_table()
    doc.start_edit("potential_fallen_angels_table")
    add_fallen_angel_table(pfa_df, doc)
    doc.end_edit()
    doc.save_tex()

    ra_df = last_12m_ratings_actions()
    plot_fallen_angels_and_rising_stars(ra_df, path)
    plot_downgrades(ra_df, path)


def update_fallen_angels_rising_stars(fid):
    doc = Document(fid, path="reports/HY", fig_dir=True)
    vis.style()
    db = Database()
    db.load_market_data()
    date = db.date("today")

    pfa_df = build_potential_fallen_angel_table()
    ra_df = last_12m_ratings_actions()
    outlook_df, rs_sector_df, rs_1_df, rs_2_df = potential_rising_star_dfs()

    plot_fallen_angels_and_rising_stars(ra_df, doc.fig_dir)
    plot_downgrades(ra_df, doc.fig_dir)
    plot_bb_outlooks(outlook_df, doc.fig_dir)

    doc.add_preamble(
        orientation="landscape",
        bookmarks=True,
        table_caption_justification="c",
        margin={"left": 0.5, "right": 0.5, "top": 2, "bottom": 1},
        header=doc.header(
            left="Potential Fallen Angels / Rising Stars",
            right=f"EOD {date.strftime('%B %#d, %Y')}",
            height=0.5,
        ),
        footer=doc.footer(logo="LG_umbrella", height=-0.5, width=0.05),
    )
    doc.add_section("Fallen Angels")
    add_fallen_angel_table(pfa_df, doc)

    doc.add_subfigures(figures=["net_fallen_angels", "recent_downgrades"])
    doc.add_pagebreak()
    doc.add_section("Rising Stars")
    rising_stars, bb_outlooks = doc.add_subfigures(n=2, widths=[0.45, 0.54])
    with doc.start_edit(rising_stars):
        add_rising_star_sector_table(rs_sector_df, doc)
        add_rising_star_ticker_table(
            rs_1_df, "1 Agency Upgrade Away Issuers", doc, db
        )
        add_rising_star_ticker_table(
            rs_2_df, "2 Agency Upgrades Away Issuers", doc, db
        )
    with doc.start_edit(bb_outlooks):
        doc.add_figure("BB_outlooks")
    doc.save()


# %%


def comp_check():
    # %%
    ticker = "FE"
    db = Database()
    db.load_market_data()
    ticker_ix = db.build_market_index(ticker=ticker, in_H0A0_index=True)
    sector = ticker_ix.df["Sector"].mode().iloc[-1]
    bbb_ix = db.build_market_index(
        in_stats_index=True, rating="BBB-", sector=sector
    )
    bbb_ix = db.build_market_index(
        in_stats_index=None,
        ticker=["PPL", "EXC"],
        # issue_years=(None, 2),
    )
    cols = ["ISIN", "Ticker", "MaturityYears", "IssueYears", "OAS"]
    ticker_ix.df[cols].sort_values(["Ticker", "MaturityYears"])
    # %%
    bbb_ix.df[cols].sort_values(["Ticker", "MaturityYears"])
    # %%


if __name__ == "__main__":
    fid = "Potential_Fallen_Angels_Rising_Stars"
    update_fallen_angels_rising_stars(fid)
