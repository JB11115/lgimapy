import pandas as pd
import seaborn as sns

import lgimapy.vis as vis
from lgimapy.data import Database
from lgimapy.latex import Document

vis.style()

# %%


def get_yields_and_coupons():
    db = Database()
    db.load_market_data(start=db.date("1y"), local=True)

    ix_bbb = db.build_market_index(rating="BBB-", in_stats_index=True)
    ix_bb = db.build_market_index(rating=("BB+", "BB-"), in_hy_stats_index=True)
    ix_b = db.build_market_index(rating=("B+", "B-"), in_hy_stats_index=True)

    d = {}
    d["BBB-_coupon"] = ix_bbb.market_value_weight("CouponRate")
    d["BB_coupon"] = ix_bb.market_value_weight("CouponRate")
    d["BB_yield"] = ix_bb.market_value_weight("YieldToWorst")
    d["B_yield"] = ix_b.market_value_weight("YieldToWorst")
    return d


def plot_yields_vs_coupons(data, doc, coupon_rating, yield_rating):
    yield_ts = data[f"{yield_rating}_yield"]
    coupon_ts = data[f"{coupon_rating}_coupon"]
    diff = (yield_ts - coupon_ts) / 100

    fig, axes = vis.subplots(
        2,
        1,
        figsize=(8, 4),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    vis.plot_timeseries(
        coupon_ts / 100,
        label=f"{coupon_rating} Coupon",
        color="navy",
        alpha=0.7,
        ax=axes[0],
        title=f"Current Cost = {diff.iloc[-1]:.1%}",
    )
    vis.plot_timeseries(
        yield_ts / 100,
        label=f"{yield_rating} Yield",
        color="darkorchid",
        alpha=0.7,
        ax=axes[0],
    )
    axes[0].legend(fancybox=True, shadow=True)
    axes[1].fill_between(
        diff.index,
        0,
        diff,
        color="grey",
        alpha=0.7,
    )
    vis.format_xaxis(axes[1], s=diff, xtickfmt="auto")
    for ax in axes:
        vis.format_yaxis(ax, ytickfmt="{x:.0%}")

    fid = f"{coupon_rating}_rating_vs_{yield_rating}_yield"
    doc.add_figure(fid, savefig=True, dpi=200)


def update_cost_of_downgrade():
    data = get_yields_and_coupons()
    doc = Document("Cost_of_Downgrade", path="reports/HY", fig_dir=True)
    doc.add_preamble(
        margin={"left": 0.5, "right": 0.5, "top": 1.5, "bottom": 1},
        bookmarks=True,
    )
    doc.add_section("Cost of Downgrade")
    doc.add_subsection("BBB to BB")
    plot_yields_vs_coupons(data, doc, coupon_rating="BBB-", yield_rating="BB")
    doc.add_subsection("BB to B")
    plot_yields_vs_coupons(data, doc, coupon_rating="BB", yield_rating="B")
    doc.save()


if __name__ == "__main__":
    update_cost_of_downgrade()
