import numpy as np
import pandas as pd

from lgimapy.bloomberg import bdp
from lgimapy.utils import root

# %%

fid = root("data/LGIMA_trade_report.csv")
df = pd.read_csv(fid, low_memory=False)
start_n = len(df)
df["date"] = pd.to_datetime(df["Trd Dt"], errors="coerce")


df = df[df["B/S"] == "B"].copy()

ignored_traders = {
    "BBUCHHOLZ3",
    "JTABELLIONE",
    "TBOWKER1",
    "NJOHANSSON11",
    "DCHAPMAN30",
    "DSCHULTE12",
    "KYU168",
    "MCOHEN187",
    "MKUSZYNSKI4",
    "NWATSON35",
    "RHULET",
    "RPAWAR17",
    "TKIM275",
    "LFERNANDES16",
}
df = df[~df["T Login"].isin(ignored_traders)].copy()

nine_digit_cusips = np.array([len(x) == 9 for x in df["Cusip"]])
df = df[nine_digit_cusips].copy()

cusips = list(df["Cusip"].unique())

df["sector"] = df["Cusip"].map(sectors)
df = df[df["sector"] != "Government"].copy()

dt_cusips = list(df["Cusip"].unique())
df["announce_date"] = pd.to_datetime(df["Cusip"].map(announce_dts))
df["issue_date"] = pd.to_datetime(df["Cusip"].map(issue_dts))


df["days_from_announce_to_trade"] = (df["date"] - df["announce_date"]).astype(
    "timedelta64[D]"
)
df["days_from_issue_to_trade"] = (df["date"] - df["issue_date"]).astype(
    "timedelta64[D]"
)


df = df[
    (df["days_from_issue_to_trade"] < 4)
    | (df["days_from_issue_to_trade"].isna())
].sort_values("days_from_issue_to_trade")

df.to_csv("lgima_trade_report_new_issues.csv")
