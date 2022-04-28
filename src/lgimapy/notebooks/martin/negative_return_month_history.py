from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document

vis.style()
# %%

selloff_threshold = 45
rating_buckets = ["BBB", "BB", "B", "CCC"]

db = Database()
hy_oas = db.load_bbg_data("US_HY", "OAS", start=None)

diff = hy_oas.diff(22).dropna()
selloff_dates = diff[diff < -selloff_threshold]
selloff_starts = []
while len(selloff_dates):
    date = selloff_dates.index[0]
    selloff_starts.append(date)
    try:
        one_month_past_date = db.date("+1M", date)
    except IndexError:
        break
    selloff_dates = selloff_dates[selloff_dates.index > one_month_past_date]


# %%
def max_consecutive_ones(a):
    max_consec = 0
    loc = 0
    count = 0
    for i in range(len(a)):
        if a[i] == 1:
            count = count + 1
            max_consec = max(count, max_consec)
        else:
            count = 0
    return max_consec


def get_max_consecutive_negative_weekly_returns(index, db):
    rets = db.load_bbg_data(index, field, start=None)
    rets = rets[rets.index > pd.to_datetime("2000")]
    dates = rets.groupby(pd.Grouper(freq="W-MON")).sum().index
    weekly_ret_list = []
    for i, end in enumerate(dates[1:]):
        start = dates[i]
        weekly_tret = db.load_bbg_data(
            index, field, start=start, end=end, aggregate=True
        )
        weekly_ret_list.append(weekly_tret)

    weekly_ret = pd.Series(weekly_ret_list, index=dates[1:])
    return max_consecutive_ones(weekly_ret.lt(0))


concecutive_negative_week_dict = defaultdict(dict)
for rating in rating_buckets:
    index = f"US_{rating}"
    for field in ["TRET", "XSRET"]:
        concecutive_negative_week_dict[rating][
            field
        ] = get_max_consecutive_negative_weekly_returns(index, db)

# %%
def get_forward_returns(field, months, selloff_starts, db):
    d = defaultdict(list)
    x_months_ago = db.date(f"{months+1}m")
    used_starts = [ss for ss in selloff_starts if ss < x_months_ago]
    for rating in rating_buckets:
        index = f"US_{rating}"
        for start in used_starts:
            d[rating].append(
                db.load_bbg_data(
                    index,
                    field,
                    start=db.date("+1m", start),
                    end=db.date(f"+{months+1}m", start),
                    aggregate=True,
                )
            )

    return pd.DataFrame(d, index=used_starts)


forward_returns_d = defaultdict(dict)
for months_ahead in [3, 6]:
    for field in ["TRET", "XSRET"]:
        forward_returns_d[months_ahead][field] = get_forward_returns(
            field, months_ahead, selloff_starts, db
        )

# %%


def make_consecutive_returns_table(d):
    df = pd.DataFrame(d)
    df.index = [format_ret(idx) for idx in df.index]
    return df


def make_latex_table(df):
    return pd.concat(
        (
            df.min().rename("Min"),
            df.median().rename("Median"),
            df.max().rename("Max"),
        ),
        axis=1,
    ).T


def format_ret(ret):
    return {"TRET": "Total Returns", "XSRET": "Excess Returns"}[ret]


# %%
df = forward_returns_d[3]["TRET"].copy()
make_latex_table(df)

# %%


doc = Document(
    "periods_of_negative_returns", path="latex/HY/2022", fig_dir=True
)
doc.add_preamble(
    table_caption_justification="c",
    margin={
        "left": 1,
        "right": 1,
        "top": 2,
        "bottom": 1,
    },
    header=doc.header(
        left="Periods of Negative Returns",
        right=f"EOD {db.date('today'):%b %d, %Y}",
    ),
)
doc.add_section("Methodology")
doc.add_text(
    f"""
Bloomberg HY indexes were used for the following analysis. To find periods of
negative returns, a threshold of a {selloff_threshold} bp selloff in the
LF98 index over a one month period was used. In total, {len(selloff_starts)}
selloffs of this magnitude were included with data beginning in
{hy_oas.index[0]:%b %d, %Y}.
"""
)

doc.add_section("Summary")
doc.add_text("""
Observing returns following large one month selloffs, credit markets tend
to undergo mean reversion, with median total and excess returns positive
at both 3 and 6 month horizons for each rating bucket studied. However,
it must be noted that there are very wide intervals between the maximum
and minimum observed returns for each rating bucket.
\\bigskip

The median return generally tends to increase as you go down in credit
quality, as does the spread between max and min returns. However, risk
adjusted returns seem to point to BB credit as an outperformer in
these periods. 3 months following selloffs, `BB's have similar downside
as `BBB's, but offer higher median returns and higher upside. In fact,
`BB's actually have a \\textit{lower} downside historically compared to
`BBB's, in excess return space, possibly due to the fact the `BB's represent
an up-in-quality trade for HY investors while `BBB's are a down-in-quality
trade for IG investors. On the other side, `B's offer similar median returns,
with some upside but significant downside relative to `BB's. At the lowest
end of the credit spectrum, `CCC's have incredible risk, which while skewed
towards the upside must be invested in cautiously only when there is strong
conviction that credit will improve.
\\bigskip

The 6 month horizon tends to be a bit more linear, with magnitudes for
min, median, and max returns increasing as credit quality decreases. However,
there still appears to be some preference for `BB's relative to `B's on a
risk-adjusted basis, as they offer similar median and upside, but `B's have
a significantly worse downside.

"""
)

doc.add_section("Results")
doc.add_table(
    make_consecutive_returns_table(concecutive_negative_week_dict),
    caption="Longest Period of Consecutive Negative Returns (weeks)",
    col_fmt="lcccc",
    font_size="Large",
)


doc.add_pagebreak()
sns.set(rc={"figure.figsize": (7, 3.5)})
vis.style()
for months_forward, ret_d in forward_returns_d.items():
    for raw_ret, df in ret_d.items():
        ret = format_ret(raw_ret)
        table_edit, plot_edit = doc.add_subfigures(
            n=2, valign="t", widths=[0.4, 0.57]
        )
        caption = f"{ret} {months_forward} Months After Selloff"
        with doc.start_edit(table_edit):
            doc.add_table(
                make_latex_table(df),
                col_fmt="lrrrr",
                prec="1%",
                caption=caption,
                adjust=True,
                font_size="Large",
            )
        with doc.start_edit(plot_edit):
            fid = f"{caption.replace(' ', '_')}"
            ax = sns.violinplot(
                data=df, inner="box", linewidth=1, alpha=0.4, color="steelblue"
            )
            ax.set_title(caption, fontweight="bold", fontsize=12)

            for violin in ax.collections[::2]:
                violin.set_alpha(0.4)
            vis.format_yaxis(ax, "{x:.1%}")
            doc.add_figure(fid, savefig=True)

doc.save()
