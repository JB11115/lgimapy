import numpy as np
import pandas as pd

from lgimapy.data import Database, SyntheticBond
from lgimapy.utils import load_json, root

# %%
account = "CITMC"
account_map = load_json("account_strategy")
strategy = account_map[account]

db = Database()
date = db.date("today")
acnt = db.load_portfolio(account=account, date=date, universe="returns")
ticker = "MEX"

print(f"{strategy}\n{ticker}")


# %%
# -------------------------------------------------------------------------- #
# Enter infor for new bonds.
#    Maturity (yrs),
#    Amount Outstanding ($M),
#    Coupon (%),
#    Dollar Price ($)]
#    Purchase Size for LD ($M)
# -------------------------------------------------------------------------- #
new_bonds_df = pd.DataFrame(
    [
        (12, 1750, 3.7, 100, 0),
        (30, 1750, 4.3, 100, 0),
    ],
    columns=["Maturity", "MarketValue", "Coupon", "Price", "P_MarketValue"],
)
# new_bonds_df = pd.DataFrame(
#     columns=["Maturity", "MarketValue", "Coupon", "Price", "P_MarketValue"]
# )
# # Enter tendered cusip: amt tendered
# tenders = {
#     "36962G5J9": 0.0477,
#     "36962G6F6": 0.0819,
#     "369604BD4": 0.2103,
#     "36962G6S8": 0.0963,
#     "36962G3A0": 0.5664,
#     "36962G3P7": 0.4777,
#     "36962G4B7": 0.47,
#     "369604BX0": 0.501,
#     "369604BF9": 0.4405,
#     "369604BH5": 0.3105,
#     "369604BY8": 0.6596,
# }

tenders = None
# -------------------------------------------------------------------------- #

new_bonds_df["Ticker"] = len(new_bonds_df) * [ticker]
new_bonds_df["P_MarketValue"] *= 1e6  # to $ Millions
new_bonds_df["OAD"] = [
    SyntheticBond(
        maturity=bond["Maturity"], coupon=bond["Coupon"], price=bond["Price"]
    ).duration
    for _, bond in new_bonds_df.iterrows()
]

# Get current benchmark data.
bm_df = acnt.df[acnt.df["BM_Weight"] > 0].copy()
bm_df["BM_OAD"] = (
    bm_df["OAD"] * bm_df["MarketValue"] / bm_df["MarketValue"].sum()
)
try:
    current_bm_ticker_oad = acnt.ticker_df.loc[ticker, "BM_OAD"]
    current_est_bm_ticker_oad = (
        bm_df.groupby("Ticker").sum().loc[ticker, "BM_OAD"]
    )
except KeyError:
    current_bm_ticker_oad = 0
    current_est_bm_ticker_oad = 0

print(f"Error: {current_est_bm_ticker_oad - current_bm_ticker_oad:.5f}")

# Combine data and calcualte a modified benchmark OAD using same
# methodology as for the new bonds.
if tenders is not None:
    for cusip, tender_amt in tenders.items():
        loc = (bm_df["CUSIP"] == cusip, "MarketValue")
        try:
            curr_mv = bm_df.loc[loc].iloc[0]
        except IndexError:
            continue  # Cusip not in index
        bm_df.loc[loc] = curr_mv * (1 - tender_amt)

# %%
bm_cols = ["Ticker", "OAD", "MarketValue"]
new_bm_df = pd.concat([bm_df[bm_cols], new_bonds_df[bm_cols]])
new_bm_df["BM_OAD"] = (
    new_bm_df["OAD"]
    * new_bm_df["MarketValue"]
    / np.sum(new_bm_df["MarketValue"])
)


new_est_bm_ticker_oad = new_bm_df.groupby("Ticker").sum().loc[ticker, "BM_OAD"]
bm_oad_chg = new_est_bm_ticker_oad - current_est_bm_ticker_oad
new_bm_ticker_oad = current_bm_ticker_oad + bm_oad_chg
print(
    f"\u0394BM OAD: {bm_oad_chg:+.3f}\n"
    f"{current_bm_ticker_oad:.3f} --> {new_bm_ticker_oad:.3f}"
)


# %%

strat_accounts = [
    acnt for acnt, strat in account_map.items() if strat == strategy
]
account_mv = pd.read_parquet(root("data/account_values.parquet")).loc[
    date, strat_accounts
]
account_strat_pct = account_mv[account] / account_mv.sum()
new_bonds_df_strat = new_bonds_df.copy()
new_bonds_df_strat["P_MarketValue"] *= account_strat_pct

p_cols = ["Ticker", "OAD", "P_MarketValue"]
p_df = acnt.full_df[acnt.full_df["P_Weight"] > 0][p_cols + ["CUSIP"]]


new_p_df = pd.concat([p_df[p_cols], new_bonds_df_strat[p_cols]])
new_p_df["P_OAD"] = (
    new_p_df["OAD"]
    * new_p_df["P_MarketValue"]
    / np.sum(new_p_df["P_MarketValue"])
)
new_p_df["P_MarketValue"] /= 1e6
new_p_ticker = new_p_df.groupby("Ticker").sum().loc[ticker].round(4)

try:
    current_ticker_ow = acnt.ticker_overweights()[ticker]
except KeyError:
    current_ticker_ow = 0

new_ticker_ow = new_p_ticker["P_OAD"] - new_bm_ticker_oad
month_end_ow = new_p_ticker["P_OAD"] - current_bm_ticker_oad

chg_in_ow = new_ticker_ow - current_ticker_ow
print(
    f"\u0394Port OW: {chg_in_ow:+.3f}\n"
    f"OW Until Month End: {current_ticker_ow:+.3f} --> {month_end_ow:+.3f}\n"
    f"OW Next Month: {current_ticker_ow:+.3f} --> {new_ticker_ow:+.3f}"
)


# %%
