import itertools as it

import numpy as np
import pandas as pd

from lgimapy.bloomberg import bdp
from lgimapy.data import Database
from lgimapy.utils import to_list

# %%


def add_rating_outlooks(ix):
    """
    Add the current outlook and current numeric rating for each
    rating agency in columns to :attr:`Index.df`.

    Parameters
    ----------
    ix: :class:`Index`
        Input index.

    Returns
    -------
    :class:`Index`:
        Index with rating outlook and individual numberic rating
        columns added for each rating agency.
    """
    db = Database()
    rating_agencies = ["SP", "Moody", "Fitch"]
    outlook_map = {"NEG": 1, "POS": -1, "STABLE": 0, np.nan: 0}

    for agency in rating_agencies:
        bbg_agency = {"Moody": "MDY"}.get(agency, agency).upper()
        outlook = bdp(ix.cusips, "Corp", f"RTG_{bbg_agency}_OUTLOOK").squeeze()
        ix.df[f"{agency}Outlook"] = outlook.map(outlook_map)
        ix.df[f"{agency}NumericRating"] = db._get_numeric_ratings(
            ix.df, [f"{agency}Rating"]
        )
    return ix


def compare(s, migration, numeric_thresh):
    if migration == "upgrade":
        return s < numeric_thresh
    elif migration == "downgrade":
        return s > numeric_thresh


def _simulate_rating_action(df, agency, n_notches):
    has_rating_loc = ~df[f"{agency}NumericRating"].isna()
    new_rating = pd.Series([np.nan] * len(df), index=df.index)
    new_rating.loc[has_rating_loc] = df[f"{agency}NumericRating"].loc[
        has_rating_loc
    ] + (n_notches * df[f"{agency}Outlook"].loc[has_rating_loc].fillna(0))
    return new_rating


def simulate_rating_migrations(
    ix,
    migration,
    threshold,
    notches,
    max_individual_agency_notches=2,
    new_column_name="{}Notch",
    add_rating_outlook_columns=True,
):
    """
    Simulate rating migrations from rating agencies using
    current outlooks. Create a column for each number of notches
    that would result in a downgrade past specified threshold.

    Parameters
    ----------
    ix: :class:`Index`
        Index to perform analysis on.
    threshold: str or int
        Rating or numeric rating for minimum (downgrade) or maximum
        (upgrade) allowable rating.
    migration ``{"upgrade", 'downgrade'}``
        Whether to find potential upgrades or downgrades.
    notches: int or List[int]:
        Number of notches of rating migrations to apply.
        Each notch results in its own column being added.
    max_individual_agency_notches: int, default=2
        Maximum number of notches for an individual rating agency
        to be simulated to upgrade/downgrade.
    new_column_name: str, default="{}Notch"
        New column name for results. the {} is formatted with
        the number of notches.
    add_rating_outlook_columns: bool, default=True
        If ``True``, add rating outlook columns using a Bloomberg
        scrape.

    Returns
    -------
    :class:`Index`:
        Index with boolean columns for each set of notches
        that result in a rating migration past given threshold.
    """
    db = Database()
    if add_rating_outlook_columns:
        ix = add_rating_outlooks(ix)

    rating_agencies = ["SP", "Moody", "Fitch"]
    rating_cols = [f"{agency}NumericRating" for agency in rating_agencies]
    outlook_cols = [f"{agency}Outlook" for agency in rating_agencies]
    new_rating_cols = [f"{agency}New" for agency in rating_agencies]

    df = ix.df[rating_cols + outlook_cols].copy()
    # Focus only on upgrades or downgrades.
    for col in outlook_cols:
        if migration == "downgrade":
            loc = df[col] < 0
        elif migration == "upgrade":
            loc = df[col] > 0
        df[col] = np.where(loc, 0, df[col])

    thresh = db.convert_input_ratings(threshold)[0]

    for notch in to_list(notches):
        breaks_threshold = np.zeros(len(df))
        if notch == 1:
            # One agency does 1 notch action.
            for agency in rating_agencies:
                # Perform 1 notch action for selected agency.
                df[f"{agency}New"] = _simulate_rating_action(
                    df, agency, n_notches=1
                )
                # Leave the other two agency ratings unchanged.
                unchanged_agencies = set(rating_agencies) - set([agency])
                for agency in unchanged_agencies:
                    df[f"{agency}New"] = df[f"{agency}NumericRating"].values
                # Store bonds which exceeded threshold.
                new_rating = db._get_numeric_ratings(df, new_rating_cols)
                loc = compare(new_rating, migration, thresh)
                breaks_threshold[loc] = 1

        elif notch == 2:
            # Two agencies each perform a 1 notch action.
            for action_agencies in it.combinations(rating_agencies, 2):
                # Perform 1 notch action for each action agency.
                for agency in action_agencies:
                    df[f"{agency}New"] = _simulate_rating_action(
                        df, agency, n_notches=1
                    )
                # Leave the other agency unchanged.
                agency = list(set(rating_agencies) - set(action_agencies))[0]
                df[f"{agency}New"] = df[f"{agency}NumericRating"].values
                # Store bonds which exceeded threshold.
                new_rating = db._get_numeric_ratings(df, new_rating_cols)
                loc = compare(new_rating, migration, thresh)
                breaks_threshold[loc] = 1

            if max_individual_agency_notches >= 2:
                # One agency does a 2 notch action.
                for agency in rating_agencies:
                    # Perform 2 notch action for selected agency.
                    df[f"{agency}New"] = _simulate_rating_action(
                        df, agency, n_notches=2
                    )
                    # Leave the other two agency ratings unchanged.
                    unchanged_agencies = set(rating_agencies) - set([agency])
                    for agency in unchanged_agencies:
                        df[f"{agency}New"] = df[f"{agency}NumericRating"].values
                    # Store bonds which exceeded threshold.
                    new_rating = db._get_numeric_ratings(df, new_rating_cols)
                    loc = compare(new_rating, migration, thresh)
                    breaks_threshold[loc] = 1

        elif notch == 3:
            # Each rating agency takes a 1 notch action.
            for agency in rating_agencies:
                df[f"{agency}New"] = _simulate_rating_action(
                    df, agency, n_notches=1
                )
            # Store bonds which exceeded threshold.
            new_rating = db._get_numeric_ratings(df, new_rating_cols)
            loc = compare(new_rating, migration, thresh)
            breaks_threshold[loc] = 1

            if max_individual_agency_notches >= 2:
                # One Rating agency takes a 2 notch
                # action and one takes 1 notch.
                for agencies in it.permutations(rating_agencies, 3):
                    # With each possible permutation, assign a 0 notch
                    # action to the first agency, 1 notch to the second,
                    # and 2 notches to the third.
                    for n_notches, agency in enumerate(agencies):
                        df[f"{agency}New"] = _simulate_rating_action(
                            df, agency, n_notches=n_notches
                        )
                        # Store bonds which exceeded threshold.
                        new_rating = db._get_numeric_ratings(
                            df, new_rating_cols
                        )
                        loc = compare(new_rating, migration, thresh)
                        breaks_threshold[loc] = 1

        ix.df[new_column_name.format(notch)] = breaks_threshold
    return ix


# %%
# max_individual_agency_notches = 2
# new_column_name = "{}Notch"
# add_rating_outlook_columns = True
# migration = "upgrade"
# threshold = "BB+"
# add_rating_outlook_columns = False
# notches = [2]

# db = Database()
# db.load_market_data()
# hy_ix = db.build_market_index(in_H0A0_index=True)
# ix = hy_ix.subset(rating=("BB+", "BB-"))
# ix = add_rating_outlooks(ix)


# %%

# ix_sim = simulate_rating_migrations(
#     ix,
#     "upgrade",
#     threshold="BB+",
#     notches=[2],
#     max_individual_agency_notches=1,
# )
# %%
# list(ix_sim.df[ix_sim.df["2Notch"] == 1]["Ticker"].unique())
