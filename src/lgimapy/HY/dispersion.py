from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import root

# %%
def update_HY_spread_dispersion():
    vis.style()
    df = load_data()
    fid = "HY_Spread_Dispersion"
    date = df.index[-1]
    doc = Document(fid, path="reports/HY", fig_dir=True, load_tex=fid)
    doc.add_preamble(
        margin={"left": 0.5, "right": 0.5, "top": 0.5, "bottom": 0.2},
        bookmarks=True,
        table_caption_justification="c",
        header=doc.header(
            left="HY Spread Dispersion",
            right=f"EOD {date.strftime('%B %#d, %Y')}",
        ),
        footer=doc.footer(logo="LG_umbrella"),
    )

    make_overview_tables(df, doc)
    plot_current_histograms(doc)
    for section in ["MV_BB", "Issuer_BB", "MV_B", "Issuer_B"]:
        make_section_plots(section, df, doc)
    make_decile_tables(df, doc)
    doc.save()


def load_data():
    """Load data, updated and save if required."""
    fid = root("data/HY/dispersion.parquet")
    try:
        old_df = pd.read_parquet(fid)
    except (FileNotFoundError, OSError):
        # Create file from scratch.
        df = compute_stats(start="1/1/2000")
    else:
        # Update data if required.
        last_date = old_df.index[-1]
        dates_to_compute = Database().trade_dates(exclusive_start=last_date)
        if dates_to_compute:
            new_df = compute_stats(start=dates_to_compute[0])
            df = pd.concat((old_df, new_df))
        else:
            df = old_df.copy()

    # Save Data.
    df.to_parquet(fid)
    return df


def compute_stats(start, end=None):
    db = Database()
    db.load_market_data(start=start, end=end, local=True)
    mv_ix = db.build_market_index(in_hy_stats_index=True)
    issuer_ix = mv_ix.issuer_index()

    ixs = {"MV": mv_ix, "Issuer": issuer_ix}
    rating_kwargs = {"BB": ("BB+", "BB-"), "B": ("B+", "B-")}

    col = "OAS"
    df_list = []
    for (name, ix), weights in zip(ixs.items(), ["MarketValue", None]):
        for rating, rating_kw in rating_kwargs.items():
            ix_r = ix.subset(rating=rating_kw)
            # Get stats.
            mean = ix_r.MEAN(col, weights=weights)
            median = ix_r.MEDIAN(col, weights=weights)
            std = ix_r.STD(col, weights=weights)
            iqr = ix_r.IQR(col, weights=weights)
            iqr_90_10 = ix_r.IQR(col, [10, 90], weights=weights)
            # Add columns to DataFrame.
            fmt = f"_{name}_{rating}_"
            df_list.append(mean.rename(f"mean{fmt}"))
            df_list.append(median.rename(f"median{fmt}"))
            df_list.append((mean - median).rename(f"mean-median{fmt}"))
            df_list.append(std.rename(f"std{fmt}"))
            df_list.append((std / mean).rename(f"RSD{fmt}"))
            df_list.append(iqr.rename(f"IQR{fmt}"))
            df_list.append(iqr_90_10.rename(f"IQR_90_10{fmt}"))
            df_list.append((iqr / median).rename(f"QCV{fmt}"))
            df_list.append((iqr_90_10 / median).rename(f"DCV{fmt}"))

    return pd.concat(df_list, axis=1)


def make_overview_tables(df, doc):
    """
    Make percentile tables for cover page showing current percentiles
    and changes since previous dates.
    """
    db = Database()
    tables = {}
    table_curr = get_percentile_table(df, db.date("today"))
    plot_table(table_curr, title="Current Percentiles", doc=doc)
    dates = {
        "1w": "1 Week Percentile Change",
        "1m": "1 Month Percentile Change",
    }
    for date_str, title in dates.items():
        table_prev = get_percentile_table(df, db.date(date_str))
        table_chg = table_curr - table_prev
        plot_table(table_chg, title, cbar_limits=(-20, 0, 20), doc=doc)


def get_percentile_table(df, date):
    """Get percentile stats for a given date."""
    subsets = {
        "_MV_BB_": "BB Market Value Weighted",
        "_Issuer_BB_": "BB Issuer Weighted",
        "_MV_B_": "B Market Value Weighted",
        "_Issuer_B_": "B Issuer Weighted",
    }
    df_list = []
    for subset, name in subsets.items():
        df_pctile = (100 * df.filter(regex=subset).rank(pct=True)).round(0)
        n = len(subset)
        df_pctile.columns = [col[:-n] for col in df_pctile.columns]
        df_list.append(df_pctile.loc[date].rename(name))

    col_names = {
        "mean-median": "$ \mu - \\tilde{x} $",
        "std": "Std Dev",
        "IQR_90_10": "IDR",
        "IQR": "IQR",
        "RSD": "RSD",
        "DCV": "DCV",
        "QCV": "QCV",
    }
    table = pd.concat(df_list, axis=1).T.rename(columns=col_names)[
        col_names.values()
    ]
    return table


def plot_table(df, title, doc, cbar_limits=(0, 50, 100)):
    fig, ax = vis.subplots(1, 1, figsize=[8, 3])
    sns.heatmap(
        df.astype(float),
        cmap="coolwarm",
        vmin=cbar_limits[0],
        center=cbar_limits[1],
        vmax=cbar_limits[2],
        linewidths=0.2,
        annot=True,
        annot_kws={"fontsize": 10},
        fmt=".0f",
        cbar=True,
        cbar_kws={"label": "%tile"},
        ax=ax,
    )
    ax.xaxis.tick_top()
    ax.set_xticklabels(df.columns, rotation=45, ha="left", fontsize=12)
    ax.set_yticklabels(df.index, ha="right", fontsize=12, va="center")
    ax.set_title(title, fontweight="bold")
    vis.savefig(doc.fig_dir / title.replace(" ", "_"), dpi=200)


def plot_current_histograms(doc):
    col = "OAS"
    db = Database()
    dates = {
        "Current": db.date("today"),
        "1 Month Ago": db.date("1m"),
        "Wides (3/23)": pd.to_datetime("3/23/2020"),
        "Tights (1/17)": pd.to_datetime("1/17/2020"),
    }
    spread_d = defaultdict(list)
    weight_d = defaultdict(list)
    for name, date in dates.items():
        db.load_market_data(date=date, local=True)
        mv_ix = db.build_market_index(in_hy_stats_index=True)
        titles = ["Market Value Weighted", "Issuer Weighted"]
        rating_kwargs = {"BB": ("BB+", "BB-"), "B": ("B+", "B-")}
        for rating, rating_kws in rating_kwargs.items():
            rating_ix = mv_ix.subset(rating=rating_kws)
            for title in titles:
                if title == "Issuer Weighted":
                    ix = rating_ix.issuer_index()
                else:
                    ix = rating_ix.copy()

                spread_d[f"{rating} {title}"].append(ix.df["OAS"].rename(name))
                weight_d[f"{rating} {title}"].append(
                    ix.df["MarketValue"].rename(name)
                )

    colors = ["k", "dodgerblue", "indigo", "darkgreen"]
    fig, axes = vis.subplots(2, 2, figsize=(12, 7))
    for ax, (title, spreads_list) in zip(axes.flat, spread_d.items()):
        weights_list = weight_d[title]
        for i, (spreads, weights, color) in enumerate(
            zip(spreads_list, weights_list, colors)
        ):
            data = pd.concat((spreads, weights / 1e3), axis=1)
            data.columns = ["OAS", "weights"]
            data = data[(data["OAS"] <= 2000) & (data["OAS"] > 15)]
            if "Market Value" in title:
                weight = "weights"
                ax.set_ylabel("Market Value")
                # vis.format_yaxis(ax, ytickfmt="${x:.0f}B")
            else:
                weight = None
                ax.set_ylabel("# Issuers")

            sns.kdeplot(
                data=data,
                x="OAS",
                label=spreads.name,
                alpha=0.6 if spreads.name == "Current" else 0.5,
                fill=True,
                weights=weight,
                color=color,
                ax=ax,
                zorder=5 - i,
            )
            ax.axes.yaxis.set_ticks([])
            ax.set_title(title, fontweight="bold", fontsize=14)
            ax.set_xlim((0, 2000))
    axes[0, 1].legend(shadow=True, fancybox=True)
    vis.savefig(doc.fig_dir / "current_histograms", dpi=200)


def make_section_plots(section, df, doc):
    db = Database()

    fig, axes = vis.subplots(2, 2, figsize=(12, 8))
    ytd = db.date("ytd")
    kwargs = {"lw": 1.5, "alpha": 0.8}
    abs_cols = {
        f"IQR_{section}_": ("IQR", "navy"),
        f"IQR_90_10_{section}_": ("IDR", "darkorchid"),
    }
    for col, (lbl, c) in abs_cols.items():
        s_tot = df[col].copy()
        s_1yr = s_tot[s_tot.index > ytd]
        last = s_1yr[-1]
        vis.plot_timeseries(s_tot, color=c, label=lbl, ax=axes[0, 0], **kwargs)
        axes[0, 0].axhline(last, color=c, lw=1, ls="--", label="_nolegend_")
        vis.plot_timeseries(s_1yr, color=c, ax=axes[1, 0], **kwargs)
        axes[1, 0].axhline(last, color=c, lw=1, ls="--", label="_nolegend_")

    axes[0, 0].set_title("Absolute Dispersion", fontweight="bold")
    axes[0, 0].legend(fancybox=True, shadow=True, fontsize=12)

    kwargs = {"lw": 1.5, "alpha": 0.8}
    abs_cols = {
        f"QCV_{section}_": ("QCV", "navy"),
        f"DCV_{section}_": ("DCV", "darkorchid"),
    }
    for col, (lbl, c) in abs_cols.items():
        s_tot = df[col].copy()
        s_1yr = s_tot[s_tot.index > ytd]
        last = s_1yr[-1]
        vis.plot_timeseries(s_tot, color=c, label=lbl, ax=axes[0, 1], **kwargs)
        axes[0, 1].axhline(last, color=c, lw=1, ls="--", label="_nolegend_")
        vis.plot_timeseries(s_1yr, color=c, ax=axes[1, 1], **kwargs)
        axes[1, 1].axhline(last, color=c, lw=1, ls="--", label="_nolegend_")

    axes[0, 1].set_title("Relative Dispersion", fontweight="bold")
    axes[0, 1].legend(fancybox=True, shadow=True, fontsize=12)
    vis.savefig(doc.fig_dir / f"{section}_timeseries", dpi=200)

    dates = {
        "Current": (db.date("today"), "k"),
        "1 Month Ago": (db.date("1m"), "dodgerblue"),
        "Wides (3/23)": (pd.to_datetime("3/23/2020"), "indigo"),
        "Tights (1/17)": (pd.to_datetime("1/17/2020"), "darkgreen"),
    }
    df_jp = df[[f"median_{section}_", f"IQR_90_10_{section}_"]].copy()
    df_jp.columns = ["Median OAS", "IDR"]

    g = sns.jointplot(
        data=df_jp, x="Median OAS", y="IDR", kind="hist", color="k"
    )
    # Plot lines for current values.
    g.ax_joint.axhline(
        df_jp["IDR"][-1], color="k", lw=0.5, ls="--", label="_nolegend_"
    )
    g.ax_joint.axvline(
        df_jp["Median OAS"][-1], color="k", lw=0.5, ls="--", label="_nolegend_"
    )
    # Plot colored points indicating
    for i, (lbl, (date, color)) in enumerate(dates.items()):
        df_date = df_jp.loc[date]
        g.ax_joint.plot(
            df_date["Median OAS"],
            df_date["IDR"],
            "o",
            ms=5,
            color=color,
            label=lbl,
            zorder=5 - i,
        )
    vis.savefig(doc.fig_dir / f"{section}_abs_jointplot", dpi=200)

    df_jp = df[[f"median_{section}_", f"DCV_{section}_"]].copy()
    df_jp.columns = ["Median OAS", "DCV"]

    g = sns.jointplot(
        data=df_jp, x="Median OAS", y="DCV", kind="hist", color="k"
    )
    # Plot lines for current values.
    g.ax_joint.axhline(
        df_jp["DCV"][-1], color="k", ls="--", lw=0.5, label="_nolegend_"
    )
    g.ax_joint.axvline(
        df_jp["Median OAS"][-1], color="k", lw=0.5, ls="--", label="_nolegend_"
    )
    # Plot colored points indicating
    for i, (lbl, (date, color)) in enumerate(dates.items()):
        df_date = df_jp.loc[date]
        g.ax_joint.plot(
            df_date["Median OAS"],
            df_date["DCV"],
            "o",
            ms=5,
            color=color,
            label=lbl,
            zorder=5 - i,
        )
    g.ax_joint.legend(fancybox=True, shadow=True, fontsize=8)
    vis.savefig(doc.fig_dir / f"{section}_rel_jointplot", dpi=200)


def get_decile_table(df, subset):
    df_sub = df.filter(regex=subset)
    df_sub.columns = [col[: -len(subset)] for col in df_sub.columns]

    col_names = {
        "mean-median": "$ \mu - \\tilde{x} $",
        "std": "Std Dev",
        "IQR_90_10": "IDR",
        "IQR": "IQR",
        "RSD": "RSD",
        "DCV": "DCV",
        "QCV": "QCV",
    }
    current_date = df_sub.index[-1].strftime("%m/%d/%Y")
    df_sub = df_sub.rename(columns=col_names)[col_names.values()]
    table_colors = {
        "babyblue": Database().date("today"),
    }

    color_locs = {}
    d = defaultdict(list)
    float_cols = {"RSD", "DCV", "QCV"}
    for i, col in enumerate(df_sub.columns):
        int_col = col not in float_cols
        vals = df_sub[col].dropna().round(2)
        d[current_date].append(int(np.round(vals[-1])) if int_col else vals[-1])
        d["Percentile"].append(np.round(100 * vals.rank(pct=True)[-1]))
        med = np.median(vals)
        d["Median"].append(int(np.round(med)) if int_col else med)
        d["Max"].append(
            int(np.round(np.max(vals))) if int_col else np.max(vals)
        )
        d["Min"].append(
            int(np.round(np.min(vals))) if int_col else np.min(vals)
        )
        quantiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
        __, bins = pd.qcut(vals, np.array(quantiles) / 100, retbins=True)
        if col in float_cols:
            bins = np.round(bins, 2)
        else:
            bins = np.rint(bins)
        percentile_labels = [
            "0-9",
            "10-19",
            "20-29",
            "30-39",
            "40-49",
            "50-59",
            "60-69",
            "70-79",
            "80-89",
            "90-94",
            "95-98",
            "99+",
        ]
        for bin, label in zip(bins[1:], percentile_labels):
            d[label].append(int(np.round(bin)) if int_col else bin)

        for color, date in table_colors.items():
            j = list(vals.index).index(date)
            bin = np.digitize(vals, bins)[j]
            color_locs[(min(16, int(bin + 4)), i)] = f"\\cellcolor{{{color}}}"

    table = pd.DataFrame(d, index=df_sub.columns).T

    return table, color_locs


def make_decile_tables(df, doc):
    subsets = {
        "_MV_BB_": "BB Market Value Weighted",
        "_Issuer_BB_": "BB Issuer Weighted",
        "_MV_B_": "B Market Value Weighted",
        "_Issuer_B_": "B Issuer Weighted",
    }
    doc.start_edit("decile_table")
    for subset, title in subsets.items():
        table, color_locs = get_decile_table(df, subset)
        # print(table)
        # print(color_locs)
        doc.add_table(
            table,
            caption=title,
            col_fmt="l|r|rrr|rrr",
            font_size="small",
            int_vals=True,
            midrule_locs=["Median"],
            specialrule_locs=["0-9"],
            loc_style=color_locs,
        )
    doc.end_edit()


# %%
if __name__ == "__main__":
    update_HY_spread_dispersion()
