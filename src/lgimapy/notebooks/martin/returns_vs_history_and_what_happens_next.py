from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document

vis.style()

# %%
ret_types = ["TRET", "XSRET"]
lookbacks = ["3M", "6M"]
ratings = ["BB", "B", "CCC"]

lookback = "3M"
ret_type = "TRET"
rating = "CCC"


def get_lookback_return_table(lookback, ret_type, ratings):
    db = Database()
    d = defaultdict(list)
    data_df_list = []
    ends = db.date("MONTH_ENDS", start="1/1/2000", end="7/6/2022")
    for rating in ratings:
        ret_list = [
            db.load_bbg_data(
                f"US_{rating}",
                ret_type,
                start=db.date(lookback, end),
                end=end,
                aggregate=True,
            )
            for end in ends
        ]
        rets = pd.Series(ret_list, index=ends)
        historical_rets = rets[rets.index <= db.date(lookback)]

        worse_ret_starts = historical_rets[historical_rets < rets.iloc[-1]]

        foreword_rets = pd.Series(
            [
                db.load_bbg_data(
                    f"US_{rating}",
                    ret_type,
                    start=start,
                    end=db.date(f"+{lookback}", start),
                    aggregate=True,
                )
                for start in worse_ret_starts.index
            ],
            index=worse_ret_starts.index,
        )
        data_df_list.append(foreword_rets.rename(rating))

        d["Current"].append(rets.iloc[-1])
        d["N Worse Periods"].append(len(worse_ret_starts))
        d["Min"].append(foreword_rets.min())
        d["Median"].append(foreword_rets.median())
        d["Max"].append(foreword_rets.max())

    table_df = pd.DataFrame(d, index=ratings).T
    data_df = pd.concat(data_df_list, axis=1)
    return table_df, data_df


table_d = defaultdict(dict)
data_d = defaultdict(dict)
for ret_type in ret_types:
    for lookback in lookbacks:
        table_df, data_df = get_lookback_return_table(
            lookback, ret_type, ratings
        )
        table_d[ret_type][lookback] = table_df
        data_d[ret_type][lookback] = data_df


# %%


def format_ret(ret):
    return {"TRET": "Total Returns", "XSRET": "Excess Returns"}[ret]


db = Database()
doc = Document(
    "historical_context_of_current_returns", path="latex/HY/2022", fig_dir=True
)
doc.add_preamble(
    table_caption_justification="c",
    margin={
        "left": 1.5,
        "right": 1.5,
        "top": 0.5,
        "bottom": 1,
    },
    header=doc.header(
        left="Historical Context of Current Returns",
        right=f"EOD {db.date('today'):%b %d, %Y}",
    ),
)
doc.add_section("Methodology")
doc.add_text(
    f"""
Bloomberg HY rating bucket indexes were used for the following analysis.
Current returns were found at 3 and 6 month lookbacks horizons for each rating
bucket. For each rating bucket and horizon, cumulative returns were computed
monthly dating back to 2000. For each period with returns more negative
than currently observed, the cumulative return over the same horizon
immediately proceeding the selloff was recorded, and stats are provided.
Both total and excess returns are studied.
"""
)

doc.add_section("Summary")
doc.add_text(
    """
From a total returns perspective, we've rarely seen worse returns
over a 3 or 6-month period than we are currently experiencing. Since 2000,
only 3 times have BB-rated bonds had a worse 6-month period. Being aware
of this small sample size is an important caveat when analyzing forward
performance after each of these periods. For `BB' and `B's,
\\textit{every} historical period where total returns have been this bad
has been followed by strongly positive returns over the next 3/6 months.
This may be attributable to survivorship bias in previous episodes,
as such heavy losses are generally associated with an increase in
downgrades and defaults, leaving only the stronger companies in the respective
cohorts. However, there have been very few such actions
this year, leaving the possibility for heavy losses on the horizon
should there be a sudden increase in downgrades/defaults.
For `CCC's, returns over 6-month periods have always been positive, but
over 3-month periods there is still a large downside risk historically with
forward losses greater than 35\% observed in the data. Digging further,
this -35\% period occured in 4Q08, when defaults persisted for an extended
period of time, lending credence to the role of survivorship bias
in other analyzed periods.
\\bigskip

From an excess returns perspective, the historical risks were more balanced,
though still had a bias towards positive returns. Notably, the sample sizes
were also ~3x larger. The downside in 3-month periods following 3-month
periods with negative returns similar to those recently endured is apparent
for each HY rating bucket, although the median excess return is
positive for each. Of the historical risk-off periods analyzed, in the worst case
(GFC) `BB's and `B's lost 20\% and 25\% relative to treasuries, and there were
numerous periods where excess returns were negative on a forward basis.
\\bigskip

Looking at excess returns at a 6-month horizon (which has a slightly larger
sample size), there is evidence of upside risk to returns, with a long
positive tails, although again survivorship bias is expected to be at play.
Despite the long positive tail, in nearly 25\% of the studied periods, negative
excess returns continued over the forward 6-month horizon in each rating bucket.

"""
)

doc.add_pagebreak()
doc.add_section("Results")
sns.set(rc={"figure.figsize": (7, 4)})
vis.style()

column_edits = doc.add_minipages(n=2, valign="t")
for col_edit, (ret_type, lookback_df_d) in zip(column_edits, table_d.items()):
    with doc.start_edit(col_edit):
        ret = format_ret(ret_type)
        for lookback, df in lookback_df_d.items():
            prec = {idx: "1%" for idx in df.index}
            prec["N Worse Periods"] = "0f"
            doc.add_table(
                df,
                col_fmt="lrrr",
                row_prec=prec,
                midrule_locs="Min",
                caption=(
                    f"Cumulative {ret} {lookback[0]} Months After "
                    f"{lookback[0]}-Month {ret} of Current Magnitude"
                ),
                adjust=True,
                font_size="Large",
            )

            data_df = data_d[ret_type][lookback]
            ax = sns.violinplot(
                data=data_df,
                inner="box",
                linewidth=1,
                alpha=0.4,
                color="steelblue",
            )
            for violin in ax.collections[::2]:
                violin.set_alpha(0.4)
            vis.format_yaxis(ax, "{x:.1%}")
            fid = f"{lookback}_{ret_type}"
            doc.add_figure(fid, savefig=True)
            doc.add_vskip("3em")

doc.save()
