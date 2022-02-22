from bisect import bisect_left
from collections import defaultdict
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import partial

import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.models import TLS
from lgimapy.utils import root

# %%


def update_macro_indicators(fid, plot=True):
    # Get tables for cash and CDX.
    cash_table, cdx_table = calculate_macro_indicators(plot=plot)
    cash_cap = "US Market Credit vs Macro Indicators"
    cdx_cap = "CDX IG vs Macro Indicators"

    # Open valuation pack document and update macro indicator page
    # with generated tables.
    doc = Document(
        fid, path="reports/valuation_pack", fig_dir=True, load_tex=True
    )
    doc.start_edit("macro_indicator_tables")
    doc.add_table(
        cash_table.df,
        caption=cash_cap,
        adjust=True,
        col_fmt="lcr",
        gradient_cell_col="Deviation",
        gradient_cell_kws={
            "vmin": -4,
            "vmax": 4,
            "cmax": "orchid",
            "cmin": "orange",
        },
    )
    doc.add_text("\\vskip1em")
    doc.add_table(
        cdx_table.df,
        caption=cdx_cap,
        adjust=True,
        col_fmt="lcr",
        gradient_cell_col="Deviation",
        gradient_cell_kws={
            "vmin": -4,
            "vmax": 4,
            "cmax": "orchid",
            "cmin": "orange",
        },
    )
    doc.end_edit()
    doc.save_tex()


class ValuationTable:
    """
    Valuation table for various indicators. Specifies credit
    index, indicator, leading/trailing designation, and Z-score.
    """

    def __init__(self):
        self.index = []
        self.d = defaultdict(list)

    def add_row(self, indicator, lead_lag, tls_res):
        """
        Add row to table. Calculate Z-score given a
        DataFrame of index and indicator.
        """
        # Remove unnecessary words from indicator.
        bad_words = ["YoY", "QoQ", "(inverted)"]
        for word in bad_words:
            indicator = indicator.replace(word, " ")
        indicator = " ".join(indicator.split())  # remove extra spaces

        # Calculate if deviation is wide or tight.
        deviation = np.abs(tls_res.norm_resid[-1])
        last_index_spread = tls_res.X[-1][0]
        last_indicator_val = tls_res.y[-1]
        theoretical_index_spread = last_indicator_val / tls_res.beta
        direction = last_index_spread - theoretical_index_spread

        self.index.append(indicator)
        self.d["$\Delta t$"].append(lead_lag.title())
        self.d["Deviation"].append(direction * deviation)
        self.d["Frobenius"].append(tls_res.frobenius_norm)

    @property
    def df(self, frobenius=False):
        df = (
            pd.DataFrame(self.d, index=self.index)
            .sort_values("Frobenius")
            .round(2)
        )
        if frobenius:
            return df
        else:
            cols = [c for c in df.columns if c != "Frobenius"]
            return df[cols]


def pct_change(s, delta_months):
    """
    Get percent change of series at given delta.

    Parameters
    ----------
    s: pd.Series
        Input timeseries.
    delta_months: int
        Number of months to take change from.

    Returns
    -------
    pd.Series
        Timeseries after applying change.
    """
    delta = partial(bisect_left, s.index)
    rel_delta = relativedelta(months=delta_months)
    dates = s.index[s.index >= s.index[0] + rel_delta]
    s_chg_a = np.zeros(len(dates))
    for i, (date, val) in enumerate(s.loc[dates].items()):
        prev_val = s.loc[s.index[delta(date - rel_delta)]]
        s_chg_a[i] = (val - prev_val) / prev_val
    return pd.Series(s_chg_a, dates, name=s.name)


def calculate_macro_indicators(plot=True):
    db = Database()
    path = root("reports/valuation_pack/fig")
    cdx_table = ValuationTable()
    cash_table = ValuationTable()
    figsize = (8, 5.5)

    # Industrials.
    left_axis = "US_CORP"
    title = db.bbg_names(left_axis)
    df = db.load_bbg_data(
        [left_axis, "US_IG", "CDX_IG"], "OAS", nan="drop", start="1/1/2010"
    )

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        table.add_row(title, "Coincident", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=f"{title} OAS",
            ylabel_right_inner="US Market Credit OAS",
            ylabel_right_outer="CDX IG Spread",
            title=title,
            alpha=0.7,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )

        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # High Yield CDX.
    left_axis = "CDX_HY"
    title = db.bbg_names(left_axis)
    df = db.load_bbg_data(
        [left_axis, "US_IG", "CDX_IG"], "OAS", nan="drop", start="1/1/2010"
    )

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        table.add_row(title, "Coincident", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=f"{title} Spread",
            ylabel_right_inner="US Market Credit OAS",
            ylabel_right_outer="CDX IG Spread",
            title=title,
            alpha=0.7,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )

        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # US economic surprise.
    left_axis = "US_ECO_SURP"
    inverted = True
    title = db.bbg_names(left_axis)
    l_lbl = f"{title}\n3m % Change"
    if inverted:
        l_lbl = f"{l_lbl} (inverted)"

    df = pd.concat(
        [
            pct_change(db.load_bbg_data("US_IG", "OAS"), 3),
            pct_change(db.load_bbg_data("CDX_IG", "OAS"), 3),
            100 * db.load_bbg_data(left_axis, "LEVEL"),
        ],
        axis=1,
        sort=True,
    )
    start = np.max([df[c].dropna().index[0] for c in df.columns])
    start = np.max([start, pd.to_datetime("1/1/2010")])
    df = df[df.index >= start]

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        table.add_row(title, "Leading", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=l_lbl,
            ylabel_right_inner="US Market Credit 3m % Change",
            ylabel_right_outer="CDX IG 3m % Change",
            ytickfmt_right_inner="{x:.0%}",
            ytickfmt_right_outer="{x:.0%}",
            # plot_kws_left={"marker": "o", "ls": "--", "lw": 1, "ms": 3},
            title=title,
            invert_left_axis=inverted,
            alpha=0.8,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )
        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # US GDP growth.
    left_axis = "US_GDP_QOQ"
    inverted = True
    title = db.bbg_names(left_axis)
    l_lbl = f"{title} 3m % Change"
    if inverted:
        l_lbl = f"{l_lbl} (inverted)"

    df = pd.concat(
        [
            db.load_bbg_data("US_IG", "OAS"),
            db.load_bbg_data("CDX_IG", "OAS"),
            db.load_bbg_data(left_axis, "LEVEL") / 100,
        ],
        axis=1,
        sort=True,
    )
    start = np.max([df[c].dropna().index[0] for c in df.columns])
    start = np.max([start, pd.to_datetime("1/1/2010")])
    df = df[df.index >= start]

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        table.add_row(title, "Trailing", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=l_lbl,
            ylabel_right_inner="US Market Credit OAS",
            ylabel_right_outer="CDX IG Spread",
            ytickfmt_left="{x:.0%}",
            plot_kws_left={"marker": "o", "ls": "--", "lw": 1, "ms": 3},
            title=title,
            invert_left_axis=inverted,
            alpha=0.8,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )
        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # US leading indicator.
    left_axis = "US_LEAD_IND"
    inverted = True
    title = db.bbg_names(left_axis)
    l_lbl = f"{title}\n3m % Change"
    if inverted:
        l_lbl = f"{l_lbl} (inverted)"

    df = pd.concat(
        [
            db.load_bbg_data("US_IG", "OAS"),
            db.load_bbg_data("CDX_IG", "OAS"),
            pct_change(db.load_bbg_data(left_axis, "LEVEL"), 3),
        ],
        axis=1,
        sort=True,
    )
    start = np.max([df[c].dropna().index[0] for c in df.columns])
    start = np.max([start, pd.to_datetime("1/1/2010")])
    df = df[df.index >= start]

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        table.add_row(title, "Trailing", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=l_lbl,
            ylabel_right_inner="US Market Credit OAS",
            ylabel_right_outer="CDX IG Spread",
            ytickfmt_left="{x:.1%}",
            plot_kws_left={"marker": "o", "ls": "--", "lw": 1, "ms": 3},
            title=title,
            invert_left_axis=inverted,
            alpha=0.8,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )
        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # US industrial production.
    left_axis = "US_IND_PROD_YOY"
    inverted = True
    title = db.bbg_names(left_axis)
    l_lbl = f"{title}\n% Change"
    if inverted:
        l_lbl = f"{l_lbl} (inverted)"

    df = pd.concat(
        [
            db.load_bbg_data("US_IG", "OAS"),
            db.load_bbg_data("CDX_IG", "OAS"),
            db.load_bbg_data(left_axis, "LEVEL") / 100,
        ],
        axis=1,
        sort=True,
    )
    start = np.max([df[c].dropna().index[0] for c in df.columns])
    start = np.max([start, pd.to_datetime("1/1/2010")])
    df = df[df.index >= start]

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        table.add_row(title, "Trailing", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=l_lbl,
            ylabel_right_inner="US Market Credit OAS",
            ylabel_right_outer="CDX IG Spread",
            ytickfmt_left="{x:.0%}",
            plot_kws_left={"marker": "o", "ls": "--", "lw": 1, "ms": 3},
            title=title,
            invert_left_axis=inverted,
            alpha=0.8,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )
        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # US unemployment rate.
    left_axis = "US_UNEMPLOY"
    inverted = False
    title = db.bbg_names(left_axis)
    l_lbl = f"{title} YoY % Change"
    if inverted:
        l_lbl = f"{l_lbl} (inverted)"

    df = pd.concat(
        [
            db.load_bbg_data("US_IG", "OAS"),
            db.load_bbg_data("CDX_IG", "OAS"),
            pct_change(db.load_bbg_data(left_axis, "LEVEL"), 12),
        ],
        axis=1,
        sort=True,
    )
    start = np.max([df[c].dropna().index[0] for c in df.columns])
    start = np.max([start, pd.to_datetime("1/1/2010")])
    df = df[df.index >= start]

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        resid = -tls.norm_resid[-1] if inverted else tls.norm_resid[-1]
        table.add_row(title, "Coincident", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=l_lbl,
            ylabel_right_inner="US Market Credit OAS",
            ylabel_right_outer="CDX IG Spread",
            ytickfmt_left="{x:.0%}",
            plot_kws_left={"marker": "o", "ls": "--", "lw": 1, "ms": 3},
            title=title,
            invert_left_axis=inverted,
            alpha=0.8,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )
        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # US manufacturing ISM
    left_axis = "US_MFG_ISM"
    inverted = True
    title = db.bbg_names(left_axis)
    l_lbl = f"{title}"
    if inverted:
        l_lbl = f"{l_lbl} (inverted)"

    df = pd.concat(
        [
            db.load_bbg_data("US_IG", "OAS"),
            db.load_bbg_data("CDX_IG", "OAS"),
            db.load_bbg_data(left_axis, "LEVEL"),
        ],
        axis=1,
        sort=True,
    )
    start = np.max([df[c].dropna().index[0] for c in df.columns])
    start = np.max([start, pd.to_datetime("1/1/2010")])
    df = df[df.index >= start]

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        table.add_row(title, "Coincident", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=l_lbl,
            ylabel_right_inner="US Market Credit OAS",
            ylabel_right_outer="CDX IG Spread",
            plot_kws_left={"marker": "o", "ls": "--", "lw": 1, "ms": 3},
            title=title,
            invert_left_axis=inverted,
            alpha=0.8,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )
        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # US ZEW economic expectations.
    left_axis = "US_ZEW"
    inverted = True
    title = db.bbg_names(left_axis).replace("th E", "th\nE")
    l_lbl = f"{title}"
    if inverted:
        l_lbl = f"{l_lbl} (inverted)"

    df = pd.concat(
        [
            db.load_bbg_data("US_IG", "OAS"),
            db.load_bbg_data("CDX_IG", "OAS"),
            db.load_bbg_data(left_axis, "LEVEL"),
        ],
        axis=1,
        sort=True,
    )
    start = np.max([df[c].dropna().index[0] for c in df.columns])
    start = np.max([start, pd.to_datetime("1/1/2010")])
    df = df[df.index >= start]

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        table.add_row(title, "Coincident", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=l_lbl,
            ylabel_right_inner="US Market Credit OAS",
            ylabel_right_outer="CDX IG Spread",
            plot_kws_left={"marker": "o", "ls": "--", "lw": 1, "ms": 3},
            title=title,
            invert_left_axis=inverted,
            alpha=0.8,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )
        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # US consumer confidence.
    left_axis = "US_CONS_CONF"
    inverted = True
    title = db.bbg_names(left_axis)
    l_lbl = f"{title}"
    if inverted:
        l_lbl = f"{l_lbl} (inverted)"

    df = pd.concat(
        [
            db.load_bbg_data("US_IG", "OAS"),
            db.load_bbg_data("CDX_IG", "OAS"),
            db.load_bbg_data(left_axis, "LEVEL"),
        ],
        axis=1,
        sort=True,
    )
    start = np.max([df[c].dropna().index[0] for c in df.columns])
    start = np.max([start, pd.to_datetime("1/1/2010")])
    df = df[df.index >= start]

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        table.add_row(title, "Trailing", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=l_lbl,
            ylabel_right_inner="US Market Credit OAS",
            ylabel_right_outer="CDX IG Spread",
            plot_kws_left={"marker": "o", "ls": "--", "lw": 1, "ms": 3},
            title=title,
            invert_left_axis=inverted,
            alpha=0.8,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )
        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # US policy uncertainty.
    left_axis = "US_POLICY_UNCERT"
    inverted = False
    title = db.bbg_names(left_axis)
    l_lbl = f"{title} 1m Rolling Avg"
    if inverted:
        l_lbl = f"{l_lbl} (inverted)"

    df = pd.concat(
        [
            db.load_bbg_data("US_IG", "OAS"),
            db.load_bbg_data("CDX_IG", "OAS"),
            db.load_bbg_data(left_axis, "LEVEL").rolling(20).mean(),
        ],
        axis=1,
        sort=True,
    )
    start = np.max([df[c].dropna().index[0] for c in df.columns])
    start = np.max([start, pd.to_datetime("1/1/2010")])
    df = df[df.index >= start]

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        table.add_row(title, "Coincident", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=l_lbl,
            ylabel_right_inner="US Market Credit OAS",
            ylabel_right_outer="CDX IG Spread",
            title=title,
            invert_left_axis=inverted,
            alpha=0.8,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )
        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # US economic sentiment.
    left_axis = "US_ECO_SENT"
    inverted = True
    title = db.bbg_names(left_axis)
    l_lbl = f"{title}"
    if inverted:
        l_lbl = f"{l_lbl} (inverted)"

    df = pd.concat(
        [
            db.load_bbg_data("US_IG", "OAS"),
            db.load_bbg_data("CDX_IG", "OAS"),
            db.load_bbg_data(left_axis, "LEVEL"),
        ],
        axis=1,
        sort=True,
    )
    start = np.max([df[c].dropna().index[0] for c in df.columns])
    start = np.max([start, pd.to_datetime("1/1/2010")])
    df = df[df.index >= start]

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        table.add_row(title, "Trailing", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=l_lbl,
            ylabel_right_inner="US Market Credit OAS",
            ylabel_right_outer="CDX IG Spread",
            plot_kws_left={"marker": "o", "ls": "--", "lw": 1, "ms": 3},
            title=title,
            invert_left_axis=inverted,
            alpha=0.8,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )
        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # US industrial confidence.
    left_axis = "US_IND_CONF"
    inverted = True
    title = db.bbg_names(left_axis)
    l_lbl = f"{title}"
    if inverted:
        l_lbl = f"{l_lbl} (inverted)"

    df = pd.concat(
        [
            db.load_bbg_data("US_IG", "OAS"),
            db.load_bbg_data("CDX_IG", "OAS"),
            db.load_bbg_data(left_axis, "LEVEL"),
        ],
        axis=1,
        sort=True,
    )
    start = np.max([df[c].dropna().index[0] for c in df.columns])
    start = np.max([start, pd.to_datetime("1/1/2010")])
    df = df[df.index >= start]

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        table.add_row(title, "Trailing", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=l_lbl,
            ylabel_right_inner="US Market Credit OAS",
            ylabel_right_outer="CDX IG Spread",
            plot_kws_left={"marker": "o", "ls": "--", "lw": 1, "ms": 3},
            title=title,
            invert_left_axis=inverted,
            alpha=0.8,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )
        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # US senior loan officer survey.
    left_axis = "US_SR_LOAN"
    inverted = False
    title = db.bbg_names(left_axis)
    l_lbl = f"{title}"
    if inverted:
        l_lbl = f"{l_lbl} (inverted)"

    df = pd.concat(
        [
            db.load_bbg_data("US_IG", "OAS"),
            db.load_bbg_data("CDX_IG", "OAS"),
            db.load_bbg_data(left_axis, "LEVEL"),
        ],
        axis=1,
        sort=True,
    )
    start = np.max([df[c].dropna().index[0] for c in df.columns])
    start = np.max([start, pd.to_datetime("1/1/2010")])
    df = df[df.index >= start]

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        table.add_row(title, "Trailing", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=l_lbl,
            ylabel_right_inner="US Market Credit OAS",
            ylabel_right_outer="CDX IG Spread",
            plot_kws_left={"marker": "o", "ls": "--", "lw": 1, "ms": 3},
            title=title,
            invert_left_axis=inverted,
            alpha=0.8,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )
        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # US demand for loans.
    left_axis = "US_LOAN_DEMAND"
    inverted = True
    title = db.bbg_names(left_axis)
    l_lbl = f"{title}"
    if inverted:
        l_lbl = f"{l_lbl} (inverted)"

    df = pd.concat(
        [
            db.load_bbg_data("US_IG", "OAS"),
            db.load_bbg_data("CDX_IG", "OAS"),
            db.load_bbg_data(left_axis, "LEVEL"),
        ],
        axis=1,
        sort=True,
    )
    start = np.max([df[c].dropna().index[0] for c in df.columns])
    start = np.max([start, pd.to_datetime("1/1/2010")])
    df = df[df.index >= start]

    for ix, table in zip(["US_IG", "CDX_IG"], [cash_table, cdx_table]):
        tls = TLS(df[[left_axis, ix]], left_axis).fit()
        table.add_row(title, "Trailing", tls)

    if plot:
        vis.plot_triple_y_axis_timeseries(
            df[left_axis].dropna(),
            df["US_IG"].dropna(),
            df["CDX_IG"].dropna(),
            ylabel_left=l_lbl,
            ylabel_right_inner="US Market Credit OAS",
            ylabel_right_outer="CDX IG Spread",
            plot_kws_left={"marker": "o", "ls": "--", "lw": 1, "ms": 3},
            title=title,
            invert_left_axis=inverted,
            alpha=0.8,
            lw=1,
            figsize=figsize,
            xtickfmt="auto",
        )
        vis.savefig(f"macro_indicator_{left_axis}", path=path)
        vis.close()

    # TODO
    # * change in 10yr real yield
    # * st louis fred stress index
    # * 10y tips breakeven % (inverted)
    # * global economic data change index vs YoY % change
    # * total centreal bank purchases vs 12m % change
    # * golmans CAI indicator
    # * 10y tips yield 6wk lead (%) vs 12m change spread (bp)

    return cash_table, cdx_table


if __name__ == "__main__":
    update_macro_indicators("first_draft", plot=False)


# %matplotlib qt
# db = Database()
# figsize = (8, 5.5)
# df = pd.concat(
#     [
#         db.load_bbg_data("US_MFG_ISM", "LEVEL"),
#         db.load_bbg_data("US_SR_LOAN", "LEVEL"),
#         db.load_bbg_data("US_IG", "OAS"),
#     ],
#     axis=1,
#     sort=True,
# )
# start = np.max([df[c].dropna().index[0] for c in df.columns])
# start = np.max([start, pd.to_datetime("1/1/2010")])
# df = df[df.index >= start]
#
# vis.plot_triple_y_axis_timeseries(
#     df["US_IG"].dropna(),
#     df["US_MFG_ISM"].dropna(),
#     df["US_SR_LOAN"].dropna(),
#     ylabel_left="US Market Credit OAS",
#     ylabel_right_inner=db.bbg_names("US_MFG_ISM") + " (inverted)",
#     ylabel_right_outer=db.bbg_names("US_SR_LOAN"),
#     invert_right_inner_axis=True,
#     # plot_kws_right_inner={'ms': 3},
#     plot_kws_left={"lw": 3},
#     alpha=0.7,
#     lw=2.5,
#     figsize=figsize,
#     xtickfmt="auto",
# )
# path = root("reports/valuation_pack/fig")
#
# vis.savefig("0_ISM_SNR_LOAN", path=path)
# vis.show()
