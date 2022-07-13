import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize


def get_clean_treasuries(df):

    d1 = pd.DataFrame(df, columns=["CurrentDate"])
    d2 = pd.DataFrame(df, columns=["MaturityDate"])

    tlist = []
    for i in range(0, len(d1)):
        tlist.append(
            float(
                round(
                    (
                        pd.to_datetime(d2["MaturityDate"].values[i]).timestamp()
                        - pd.to_datetime(
                            d1["CurrentDate"].values[i]
                        ).timestamp()
                    )
                    / (60 * 60 * 24 * 365),
                    1,
                )
            )
        )
    dt = pd.DataFrame(tlist)
    df["Tenor"] = dt

    df1 = df[(df["OAS"] < 40) & (df["OAS"] > -30)]

    df2 = df1[df1["CallType"] == "NONCALL"]

    df3 = df2[df2["OriginalTenor"].isin([2, 3, 5, 7, 10, 20, 30])]

    otlist = []
    for i in range(0, len(df3)):
        if df3["OriginalTenor"].values[i] == 30:
            otlist.append(20)
        elif df3["OriginalTenor"].values[i] == 20:
            otlist.append(10)
        elif df3["OriginalTenor"].values[i] == 10:
            otlist.append(7)
        elif df3["OriginalTenor"].values[i] == 7:
            otlist.append(5)
        elif df3["OriginalTenor"].values[i] == 5:
            otlist.append(3)
        elif df3["OriginalTenor"].values[i] == 3:
            otlist.append(2)
        elif df3["OriginalTenor"].values[i] == 2:
            otlist.append(0.25)
        else:
            print("Value Error")
    doriglower = pd.DataFrame(otlist, index=df3.index)
    doriglower.columns = ["Doriglower"]

    df4 = df3[
        (df3["Tenor"] < df3["OriginalTenor"])
        & (df3["Tenor"] > doriglower["Doriglower"])
    ]

    datelist = []
    for d in df4["MaturityDate"].values[:]:
        datelist.append(d[-2:])
    dls = pd.DataFrame(datelist, index=df4.index)
    dls.columns = ["DayVal"]

    df5list = []
    df5indexlist = []
    for i in range(0, len(df4)):
        if df4.iloc[i]["MaturityDate"][-2:] == "15" or df4.iloc[i][
            "Ticker"
        ] not in ["S", "SP", "SPX", "SPY"]:
            df5list.append(df4.iloc[i])
            df5indexlist.append(df4.index[i])

    df5 = pd.DataFrame(df5list)
    df5.index = df5indexlist
    df5.columns = [
        "CurrentDate",
        "MaturityDate",
        "OriginalTenor",
        "Ticker",
        "Sector",
        "CouponType",
        "CouponRate",
        "CallType",
        "OAS",
        "OASD",
        "Tenor",
    ]

    df6list = []
    df6indexlist = []
    for i in range(0, len(df5)):
        if (
            df5.iloc[i]["Sector"] == "TREASURIES"
            or df5.iloc[i]["CouponType"] == "ZEROCOUPON"
        ):
            df6list.append(df5.iloc[i])
            df6indexlist.append(df5.index[i])
        elif (
            df5.iloc[i]["CouponType"] == "FIXED"
            and df5.iloc[i]["CouponRate"] == 0
        ):
            df6list.append(df5.iloc[i])
            df6indexlist.append(df5.index[i])

    df6 = pd.DataFrame(df6list)
    df6.index = df6indexlist
    df6.columns = [
        "CurrentDate",
        "MaturityDate",
        "OriginalTenor",
        "Ticker",
        "Sector",
        "CouponType",
        "CouponRate",
        "CallType",
        "OAS",
        "OASD",
        "Tenor",
    ]

    return df6


def nearest_date(date, date_list, before=True, after=True):
    if before == True and after == True:
        mydict = {abs(date.timestamp() - d.timestamp()): d for d in date_list}
        nearest_date = mydict[min(mydict.keys())]
        return nearest_date
    elif before == True and after == False:
        mydict = {(date.timestamp() - d.timestamp()): d for d in date_list}
        mydict_before = [p for p in mydict if p > 0]
        new_date_list = date_list[0 : len(mydict_before)]
        mydict_new = {
            abs(date.timestamp() - d.timestamp()): d for d in new_date_list
        }
        nearest_date = mydict_new[min(mydict_new.keys())]
        return nearest_date
    elif before == False and after == True:
        mydict = {(date.timestamp() - d.timestamp()): d for d in date_list}
        mydict_after = [p for p in mydict if p < 0]
        new_date_list = date_list[len(mydict) - len(mydict_after) : len(mydict)]
        mydict_new = {
            abs(date.timestamp() - d.timestamp()): d for d in new_date_list
        }
        nearest_date = mydict_new[min(mydict_new.keys())]
        return nearest_date
    else:
        return None


def count_number_of_increasing_IDs(fid):
    with open(fid) as f:
        text_list = f.readlines()
    count = 0
    check = False
    for sometext in text_list:
        text_type = type(sometext)
        digits = text_type().join(filter(text_type.isdigit, sometext))
        if digits:
            for i in range(0, len(digits)):
                if i == 0:
                    prev = 0
                else:
                    prev = digits[i - 1]
                    curr = digits[i]
                    if curr > prev:
                        check = True
                    else:
                        check = False
                        break
            if check:
                count = count + 1
        else:
            check = False
    return count


class Bond:
    def __init__(self, price, cashflows):
        self._current_date = pd.to_datetime("1/1/2022")
        self._price = price

    @property
    def ytm(self):
        dlist1 = pd.to_datetime(cashflows.keys())
        tlist = {
            (self._current_date.timestamp() - d.timestamp())
            / (60 * 60 * 24 * 365): d
            for d in dlist1
        }
        temp = (
            lambda y: sum([c * exp(y * t) for c, t in zip(cashflows, tlist)])
            - self._price
        )
        self._ytm = optimize.newton(temp, 0.02)
        return self._ytm


class OLS:
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._fitcheck = False

    def fit(self):
        time.sleep(1)
        A = np.vstack([self._x, np.ones(len(self._x))]).T
        z, resid, rank, sigma = np.linalg.lstsq(A, self._y, rcond=None)
        self._resid = resid
        self._z = z
        self._r2 = float(1 - resid / sum((self._y - self._y.mean()) ** 2))
        self._fitcheck = True

    @property
    def resid(self):
        return self._resid

    def predict(self, x_pred):
        return self._z[1] * x_pred + self._z[0]

    def plot(self, x_pred=None):
        if self._fitcheck == False:
            ols.fit()
        plt.scatter(
            self._x,
            self._y,
            color="r",
            marker="o",
            edgecolors="black",
            s=35,
            label="Raw Data",
        )
        y_pred = self._z[1] * self._x + self._z[0]
        plt.plot(self._x, y_pred, color="green", label="Best Fit Line")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("OLS Best Fit Line")
        plt.grid()
        plt.text(0.5, 0.5, "$R^2 = {:.2f}$".format(self._r2), fontsize=10)
        plt.legend()
        plt.savefig("OLS_KM_plot.png")
        plt.close(plt.gcf())
