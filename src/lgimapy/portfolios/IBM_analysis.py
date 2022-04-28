import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

import lgimapy.vis as vis
from lgimapy.data import Database, concat_index_dfs
from lgimapy.latex import Document
from lgimapy.utils import to_labelled_buckets

# %%
account = "BLUELC"

current_year = 2022
current_month = 3  # {3, 6, 9, 12}
current_quarter = f"{current_month / 3:.0f}Q{current_year-2000:.0f}"

db = Database()
doc = Document(
    fid=f"IBM_{current_quarter}", path="reports/portfolios/IBM", fig_dir=True
)


def new_issuance(year, month):
    db = Database()
    date = pd.to_datetime(f"{month}/15/{year}")
    month_end = db.date("month_end", reference_date=date)
    next_month_start = db.date("next_month_start", reference_date=date)

    # Load data from the end of the month.
    db.load_market_data(start=month_end, end=next_month_start)
    ix_curr_month = db.build_market_index(date=month_end, in_returns_index=True)
    ix_next_month = db.build_market_index(
        date=next_month_start, in_returns_index=True
    )

    # Find CUSIPs that were added to returns index, and subset to those
    # which have been issued in last 40 days.
    ix_diff = ix_next_month.subset(
        isin=ix_curr_month.isins,
        issue_years=(None, 40 / 365),
        maturity=(11, None),
        special_rules="~ISIN",
    )
    return ix_diff


months = list(range(1, 13))
vals = []
for month in months:
    ix = new_issuance(current_year - 1, month)
    vals.append(np.sum(ix.df["AmountOutstanding"]))
s_prev = pd.Series(vals, index=months) / 1e3

months = list(range(1, current_month + 1))
vals = []
dfs = []
for month in months:
    ix = new_issuance(current_year, month)
    vals.append(np.sum(ix.df["AmountOutstanding"]))
    tdf = ix.ticker_df
    df_big = tdf[tdf["AmountOutstanding"] > 1000]
    dfs.append(ix.df[ix.df["Ticker"].isin(df_big.index)].copy())


s_curr = pd.Series(vals, index=months) / 1e3
cum_prev = np.cumsum(s_prev)
cum_curr = np.cumsum(s_curr)


df = concat_index_dfs(dfs).sort_values(["IssueDate", "Ticker", "MaturityDate"])
ref_date = f"{current_month}/15/{current_year}"
quarter_end = db.date("month_end", ref_date)
quarter_start = db.date("month_start", db.date("2m", ref_date))


def find_issue_date(isins, start, end):
    db = Database()
    db.load_market_data(start=start, end=end)
    ix = db.build_market_index(isin=isins)

    df_list = []
    for _, df in ix.df[["ISIN", "Date", "Ticker"]].groupby(
        "ISIN", observed=True
    ):
        df_list.append(df.iloc[0, :])
    return pd.concat(df_list, axis=1).T


new_issue_df = find_issue_date(df["ISIN"], start, end)

# %%
new_issue_port_weight = {}
new_issue_deal_size = {}
for date, date_df in new_issue_df.groupby("Date"):
    db.load_market_data(date=date)
    date_port = db.load_portfolio(account=account, date=date)
    ix = db.build_market_index()
    for isin in date_df["ISIN"]:
        try:
            p_weight = date_port.subset(isin=isin).df["P_Weight"].iloc[0]
        except IndexError:
            p_weight = 0
        new_issue_port_weight[isin] = p_weight
        new_issue_deal_size[isin] = (
            ix.df[ix.df["ISIN"] == isin].squeeze()["AmountOutstanding"] / 1e3
        )

# %%
new_issue_df["P_Weight"] = new_issue_df["ISIN"].map(new_issue_port_weight)
new_issue_df["deal_size"] = new_issue_df["ISIN"].map(new_issue_deal_size)

ticker_df = new_issue_df.groupby("Ticker").sum().rename_axis(None)

new_issue_table = pd.DataFrame(index=ticker_df.index)
participation = []
for weight in ticker_df["P_Weight"]:
    p = weight > 0
    color = "blue" if p else "red"
    text = "participated" if p else "passed"
    participation.append(f"\\color{{{color}}} {text}")

new_issue_table["LGIM America*Participation"] = participation
new_issue_table["Deal Size*(\$B)"] = ticker_df["deal_size"]
# %%
def make_patch_spines_invisible(ax):
    """
    Make matplotlib Axes edges invisible.

    Parameters
    ----------
    ax: matplotlib Axes, optional
        Axes in which to draw plot, otherwise activate Axes.
    """
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


vis.style()
fig, ax_left = vis.subplots(figsize=(10, 7))
ax_right = ax_left.twinx()

# Make spine visible for right axis.
make_patch_spines_invisible(ax_left)
make_patch_spines_invisible(ax_right)
ax_right.grid(False)
ax_right.spines["right"].set_position(("axes", 1.05))
ax_right.spines["right"].set_visible(True)
ax_right.spines["right"].set_linewidth(1.5)
ax_right.spines["right"].set_color("lightgrey")
ax_right.tick_params(right="on", length=5)

ax_left.bar(s_prev.index, s_prev.values, width=0.8, color="dimgray", alpha=0.6)
ax_left.bar(s_curr.index, s_curr.values, width=0.4, color="navy", alpha=0.7)

ax_right.plot(
    cum_prev, "-o", lw=2, ms=8, color="dimgray", label=(current_year - 1)
)
ax_right.plot(cum_curr, "-o", lw=2, ms=8, color="navy", label=current_year)

ax_left.set_ylabel("Monthly Issuance")
ax_right.set_ylabel("Yearly Cumulative Issuance")
ax_left.set_xticks(s_prev.index)
ax_right.set_xticks(s_prev.index)
ax_left.set_xticklabels(list("JFMAMJJASOND"))
ax_left.grid(False, axis="x")
ax_right.grid(False)
vis.legend(ax_right, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07))
vis.format_yaxis(ax_left, "${x:.0f}B")
vis.format_yaxis(ax_right, "${x:.0f}B")
new_issue_fid = "new_issuance"
vis.savefig(new_issue_fid, path=doc.fig_dir)
vis.show()

# %%
db = Database()
quarter_start = db.date(
    "next_month_start", reference_date=f"{current_month}/15/{current_year}"
)
port = db.load_portfolio(account=account, date=quarter_start)


# %%


def plot_contribution_by_tenor(port, metric, fig_dir):
    maturities = [0, 3, 5, 7, 10, 15, 20, 25, 30, 35]
    maturity_buckets = to_labelled_buckets(maturities, right_end_closed=False)
    credit_d = defaultdict(list)
    tsy_d = defaultdict(list)

    for bucket, maturity_kws in maturity_buckets.items():
        mat_port = port.subset(
            maturity=maturity_kws, drop_treasuries=False, df=port.full_df
        )

        port_credit_metric = mat_port.df[f"P_{metric}"].sum()
        bm_credit_metric = mat_port.df[f"BM_{metric}"].sum()

        credit_d["Portfolio Credit"].append(port_credit_metric)
        credit_d["Benchmark Credit"].append(bm_credit_metric)

        port_tsy_metric = mat_port.tsy_df[f"P_{metric}"].sum()
        bm_tsy_metric = mat_port.tsy_df[f"BM_{metric}"].sum()

        tsy_d["Portfolio Treasury/Cash"].append(
            port_credit_metric + port_tsy_metric
        )
        tsy_d["_no_legend_"].append(bm_credit_metric + bm_tsy_metric)

    credit_df = pd.DataFrame(credit_d, index=maturity_buckets.keys())
    tsy_df = pd.DataFrame(tsy_d, index=maturity_buckets.keys())

    fig, ax = vis.subplots(figsize=(10, 8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tsy_df.plot.bar(color=["lightskyblue", "gray"], ax=ax, alpha=0.7)
        credit_df.plot.bar(color=["navy", "gray"], ax=ax, alpha=0.8)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(False, axis="x")
    vis.legend(ax)
    if metric == "Weight":
        vis.format_yaxis(ax, ytickfmt="{x:.0%}")

    fid = f"{account}_{metric}_contribution_by_tenor"
    vis.savefig(fid, path=fig_dir)
    return fid


oad_plot_fid = plot_contribution_by_tenor(port, "OAD", doc.fig_dir)
mv_plot_fid = plot_contribution_by_tenor(port, "Weight", doc.fig_dir)
# %%
doc = Document(
    fid=f"IBM_{current_quarter}", path="reports/portfolios/IBM", fig_dir=True
)
doc.add_preamble(orientation="landscape", margin={"left": 0.5, "right": 0.5})
left, right = doc.add_minipages(n=2, widths=[0.65, 0.3], valign="t")
with doc.start_edit(left):
    doc.add_figure(new_issue_fid)

with doc.start_edit(right):
    doc.add_table(
        new_issue_table,
        col_fmt="lcr",
        multi_row_header=True,
        prec={"Deal Size*(\$B)": "2f"},
    )

doc.add_pagebreak()
doc.add_subfigures(figures=[oad_plot_fid, mv_plot_fid])
doc.save_tex()
doc.save()


# %%
new_quarter_start = db.date("NEXT_MONTH_START", ref_date)
port_old = db.load_portfolio(account=account, date=quarter_start)
port_new = db.load_portfolio(account=account, date=new_quarter_start)
# %%
port_df = pd.concat(
    (
        port_old.ticker_overweights(by="DTS").rename("old"),
        port_new.ticker_overweights(by="DTS").rename("new"),
    ),
    axis=1,
).fillna(0)
port_df["change"] = port_df["new"] - port_df["old"]
port_df.sort_values("change", inplace=True)

port_df.head(10)
port_df.tail(10)
