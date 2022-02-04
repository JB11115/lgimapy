import numpy as np
import pandas as pd

from lgimapy.data import Database

# %%

db = Database()
db.load_market_data(start=db.date("ytd"))

# %%
port = db.load_portfolio(account="PMCHY")

ix = db.build_market_index(date=db.date("today"))
h4un = ix.subset(in_H4UN_index=True)
all_bb = h4un.subset(rating=("BB+", "BB-"))
bb = all_bb.subset(mod_dur_to_worst=(8, None))
all_bbb = ix.subset(
    in_stats_index=True,
    financial_flag=0,
    rating=("BBB+", "BBB-"),
)
bbb = all_bbb.subset(OAD=(8, None))
short_dur = all_bbb.subset(OAD=(None, 3)) + h4un.subset(
    mod_dur_to_worst=(None, 3)
)

ix_ytd = db.build_market_index()
h0a0_ytd = db.build_market_index(in_H0A0_index=True)

h0a0_tret = h0a0_ytd.aggregate_total_returns()
trets = ix_ytd.accumulate_individual_total_returns()

# %%
# Lowest $px bonds with duration > 8 years
low_px_bb_df = bb.subset(dirty_price=(None, 100)).df.sort_values("DirtyPrice")
low_px_bb_df.to_csv("low_px_bb.csv")
low_px_bbb_df = bbb.subset(dirty_price=(None, 100)).df.sort_values("DirtyPrice")
low_px_bbb_df.to_csv("low_px_bbb.csv")

# %%
# Bonds with lowest performance YTD
bb_df = bb.df.copy()
bb_df["YTD_TRet"] = trets[bb_df.index]
bb_df["YTD_TRet_vs_H0A0"] = bb_df["YTD_TRet"] - h0a0_tret
bb_underperform_df = bb_df[bb_df["YTD_TRet_vs_H0A0"] < -0.05].sort_values(
    "YTD_TRet_vs_H0A0"
)
bb_underperform_df.to_csv("bb_biggest_neg_performance_ytd.csv")

bbb_df = bbb.df.copy()
bbb_df["YTD_TRet"] = trets[bbb_df.index]
bbb_df["YTD_TRet_vs_H0A0"] = bbb_df["YTD_TRet"] - h0a0_tret
bbb_underperform_df = bbb_df[bbb_df["YTD_TRet_vs_H0A0"] < -0.05].sort_values(
    "YTD_TRet_vs_H0A0"
)
bbb_underperform_df.to_csv("bbb_biggest_neg_performance_ytd.csv")

# %%
wide_short_dur_df = short_dur.subset(OAS=(500, None)).df.sort_values(
    "OAS", ascending=False
)
wide_short_dur_df.to_csv("short_mat_bonds_wider_than_500bp.csv")

owned_cusips = port.port_df[port.port_df["P_Weight"] > 0.005]["CUSIP"]
unowned_wsd_df = wide_short_dur_df[~wide_short_dur_df.index.isin(owned_cusips)]
wide_short_dur_df.to_csv("small_position_short_mat_bonds_wider_than_500bp.csv")
