from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database, Index

# %%


def get_index(name, ix):
    if name == "Market_Credit":
        return ix
    elif name == "Long_Credit":
        return ix.subset(maturity=(10, None))
    elif name == "Long_Corp":
        temp_ix = ix.subset(
            maturity=(10, None), sector=Database().long_corp_sectors,
        )
        df = temp_ix.df.copy()
        noncalls = (
            (df["Sector"] == "BANKING")
            & (df["MaturityYears"] > 10)
            & (df["MaturityYears"] < 11)
        )
        long_corp_df = df[~noncalls]
        return Index(long_corp_df)
    else:
        raise ValueError(f"{name} is not an index.")


def get_monthly_downgrades(ix, ratings, month, year, net=False):
    """Get notional downgrades for given index and month."""
    isins = ix.isins

    date = f"{month}/15/{year}"
    start = db.date("MONTH_START", date)
    end = db.date("MONTH_END", date)
    ratings_df = Database().rating_changes(start=start, end=end)
    numeric_ratings = Database().convert_letter_ratings(ratings)

    in_current_index = ratings_df["ISIN"].isin(isins)
    rating_change_df_list = []
    for agency in ["SP", "Moody"]:
        in_specied_ratings = (
            ratings_df[f"{agency}Rating_PREV"] >= numeric_ratings[0]
        ) & (ratings_df[f"{agency}Rating_PREV"] <= numeric_ratings[1])
        df = ratings_df[in_current_index & in_specied_ratings].copy()
        chg_col = f"{agency}Rating_CHANGE"
        # Weight individual rating changes by 50% * number of notches
        # to balance the two agencies impact.
        df["weighted_par"] = 0.5 * df["AmountOutstanding"] * df[chg_col]
        rating_change_df_list.append(df["weighted_par"])

    rating_change_df = pd.concat(rating_change_df_list)
    upgrades = rating_change_df[rating_change_df > 0].sum()
    downgrades = -rating_change_df[rating_change_df < 0].sum()
    if net:
        return downgrades - upgrades
    else:
        return downgrades


def get_montly_fallen_angels(ix, ix_next_month, month, year, net=False):
    """Get notional of fallen angels for given month and index."""
    if net:
        isins = set(ix.isins) | set(ix_next_month.isins)
    else:
        isins = ix.isins

    date = f"{month}/15/{year}"
    start = db.date("MONTH_START", date)
    end = db.date("MONTH_END", date)
    ratings_df = Database().rating_changes(start=start, end=end)

    in_current_index = ratings_df["ISIN"].isin(isins)
    is_fallen_angel = (
        ratings_df[f"NumericRating_PREV"] <= db.convert_letter_ratings("BBB-")
    ) & (ratings_df[f"NumericRating_NEW"] >= db.convert_letter_ratings("BB+"))
    is_rising_star = (
        ratings_df[f"NumericRating_PREV"] >= db.convert_letter_ratings("BB+")
    ) & (ratings_df[f"NumericRating_NEW"] <= db.convert_letter_ratings("BBB-"))

    fallen_angels = ratings_df.loc[
        (in_current_index & is_fallen_angel), "AmountOutstanding"
    ].sum()
    rising_stars = ratings_df.loc[
        (in_current_index & is_rising_star), "AmountOutstanding"
    ].sum()
    if net:
        return fallen_angels - rising_stars
    else:
        return fallen_angels


# %%
db = Database()
benchmarks = ["Long_Credit", "Long_Corp", "Market_Credit"]
years = range(2000, 2021)
rating_d = {"IG": ("AAA", "BBB-"), "A": ("AAA", "A-"), "BBB": ("BBB+", "BBB-")}
ix_d = {}
months = range(1, 12)
for year in tqdm(years):
    for month in months:
        date = db.nearest_date(f"{month}/15/{year}")
        db.load_market_data(date=date)
        ix_d[(year, month)] = db.build_market_index(in_returns_index=True)

# %%
d = defaultdict(list)
bm = benchmarks[2]
rating = "IG"
rating_kws = ("AAA", "BBB-")
months = range(1, 11)
years = range(2000, 2021)
for year in tqdm(years):
    ix = get_index(bm, ix_d[(year, 1)])
    mv = ix.total_value().squeeze()
    year_d = defaultdict(list)
    for month in months:
        ix = get_index(bm, ix_d[(year, month)])
        ix_next_month = get_index(bm, ix_d[(year, month + 1)])
        year_d["fallen_angels"].append(
            get_montly_fallen_angels(ix, ix_next_month, month, year)
        )
        year_d["net_fallen_angels"].append(
            get_montly_fallen_angels(ix, ix_next_month, month, year, net=True)
        )
        year_d["downgrades"].append(
            get_monthly_downgrades(ix, rating_kws, month, year)
        )
        year_d["net_downgrades"].append(
            get_monthly_downgrades(ix, rating_kws, month, year, net=True)
        )
    year_df = np.sum(pd.DataFrame(year_d) / mv)
    for var, val in year_df.items():
        d[f"{bm}_{var}"].append(val)


df = 100 * pd.DataFrame(d, index=years)

# %%
median = df.median().rename("Median")
df_final = df.append(median).round(2)
df_final.to_csv(f"{bm}.csv")

# %%
