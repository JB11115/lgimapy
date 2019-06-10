def standardize_cusips(df):
    """
    Standardize CUSIPs, converting cusips which changed name
    to most recent cusip value for full history.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of selected index cusips.

    Returns
    -------
    df: pd.DataFrame
        DataFrame with all CUSIPs updated to current values.
    """
    dates = sorted(list(set(df["Date"])), reverse=True)
    df.set_index("CUSIP", inplace=True)

    # Build CUSIP map of CUSIPs which change.
    # TODO: add Issuer back
    multi_ix = [  # index for identifying CUSIP changes
        "Ticker",
        "CouponRate",
        "MaturityDate",
        "IssueDate",
    ]
    cusip_map = {}
    for i in range(len(dates) - 1):
        df_today = df[df["Date"] == dates[i]].copy()
        df_yesterday = df[df["Date"] == dates[i + 1]].copy()

        # Find CUSIPs that were dropped from yesterday to today
        # and CUSIPs that were newly added today.
        df_dropped = df_yesterday[
            ~df_yesterday.index.isin(df_today.index)
        ].copy()
        df_new = df_today[~df_today.index.isin(df_yesterday.index)].copy()

        # Store CUSIPs, and set index to change identifier.
        df_dropped["CUSIP_old"] = df_dropped.index
        df_new["CUSIP_new"] = df_new.index
        df_dropped.set_index(multi_ix, inplace=True)
        df_new.set_index(multi_ix, inplace=True)

        # Store instances where CUSIPs change in cusip_map dict.
        df_new[df_new.index.isin(df_dropped.index)]
        df_change = df_new[["CUSIP_new"]].join(
            df_dropped[["CUSIP_old"]], how="inner"
        )
        for _, row in df_change.iterrows():
            cusip_map[row["CUSIP_old"]] = row["CUSIP_new"]

    # Update CUSIP map to account for CUSIPs which change multiple times.
    rev_CUSIPs = list(cusip_map.keys())[::-1]
    for i, key in enumerate(rev_CUSIPs):
        for k in rev_CUSIPs[i:]:
            if cusip_map[k] == key:
                cusip_map[k] = cusip_map[key]

    # Map old CUSIPs to new CUSIPs and reset index.
    df.index = [cusip_map.get(ix, ix) for ix in df.index]
    df["CUSIP"] = df.index.astype("category")
    df.reset_index(inplace=True, drop=True)
    return df


def spread_diff(df1, df2):
    """
    Calculate spread difference of index values from df1 to df2.

    Parameters
    ----------
    df1: pd.DataFrame
        Older index DataFrame.
    df2: pd.DataFrame
        Newer index DataFrame.

    Returns
    -------
    df: pd.DataFrame

    """
    # Standardize cusips.
    combined_df = standardize_cusips(df1.append(df2))
    df1 = combined_df[combined_df["Date"] == df1["Date"].iloc[0]].copy()
    df2 = combined_df[combined_df["Date"] == df2["Date"].iloc[0]].copy()
    df1.set_index("CUSIP", inplace=True)

    # Add OAS column from df1 to df2, only keeping CUSIPS with values
    # in both DataFrames.
    df = df2.join(df1[["OAS"]], on=["CUSIP"], how="inner", rsuffix="_old")
    df["OAS_change"] = df["OAS"] - df["OAS_old"]
    df["OAS_pct_change"] = df["OAS"] / df["OAS_old"] - 1
    return df


# class Bond:
#     """
#     Class for bond math manipulation given
#     current state of the bond.
#
#     Parameters
#     ----------
#     series: pd.Series
#         Single bond row from `index_builder` DataFrame.
#
#
#     Attributes
#     ----------
#     coupon_dates: All coupon dates for given treasury.
#     coupon_years: Time to all coupons in years.
#
#     Methods
#     -------
#     calculate_price(rfr): Calculate price of bond with given risk free rate.
#     """
