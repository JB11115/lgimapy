import itertools as it
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sms

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.utils import root

# %%


def update_voltility_model():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    vis.style()
    db = Database()
    start = "1/1/2010"
    y_var = "US_IG_10+"
    x_vars = ["VIX", "VIX_3M", "VIX_9M", "OIL_VOL", "VVIX", "MOVE"]

    y_temp = db.load_bbg_data(y_var, "OAS", start=start)
    x_temp = db.load_bbg_data(x_vars, "price", start=start)
    df = pd.concat([y_temp, x_temp], axis=1, sort=True).dropna()
    y = df[y_var]

    res = sms.OLS(y, sms.add_constant(df[x_vars])).fit(cov_type="HC1")
    pred = res.predict(sms.add_constant(df[x_vars]))

    fig, axes = vis.subplots(
        2,
        1,
        figsize=(9, 4.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    vis.plot_multiple_timeseries(
        [y.rename(db.bbg_names(y_var)), pred.rename("Vol Model")],
        c_list=["navy", "darkgreen"],
        lw=1.8,
        alpha=0.7,
        xtickfmt="auto",
        ylabel="OAS (bp)",
        ax=axes[0],
    )
    axes[0].legend(fancybox=True, shadow=True, loc="upper left")
    axes[0].set_ylim(None, 250)
    axes[1].fill_between(
        y.index,
        0,
        (y - pred),
        color="grey",
        alpha=0.7,
        label=f"Current: {(y-pred)[-1]:.0f} bp",
    )
    axes[1].legend(loc="upper left", fancybox=True, shadow=True)
    axes[1].set_ylabel("Resid (bp)")
    axes[1].set_ylim((-55, 55))
    axes[1].set_yticks([-50, -25, 0, 25, 50])

    # vis.show()
    vis.savefig(root("reports/valuation_pack/fig/long_credit_vol_model"))
    warnings.simplefilter(action="default", category=FutureWarning)


def find_best_model():
    """
    Find best combination of volatility securities to model
    US credit.
    """
    start = "1/1/2010"
    x_vars = ["VIX", "VIX_3M", "VIX_6M", "VIX_9M", "OIL_VOL", "VVIX", "MOVE"]
    y_var = "US_IG"

    db = Database()
    y_temp = db.load_bbg_data(y_var, "OAS", start=start)
    x_temp = db.load_bbg_data(x_vars, "price", start=start)
    df = pd.concat([y_temp, x_temp], axis=1, sort=True).dropna()
    df.drop(pd.to_datetime("5/28/2013"), axis=0, inplace=True)

    warnings.simplefilter(action="ignore", category=FutureWarning)
    x_var_groups = list(combos(x_vars))
    d = defaultdict(list)
    for x_group in x_var_groups:
        X = df[x_group]
        y = df[y_var]
        for mod in ["OLS", "WLS"]:
            if mod == "OLS":
                d["X"].append(", ".join(x_group))
                d["model"].append(mod)
                res = sms.OLS(y, sms.add_constant(X)).fit(cov_type="HC1")
                d["adj_R^2"].append(res.rsquared_adj)
            elif mod == "WLS":
                for weight_scheme in ["linear", "exp"]:
                    if weight_scheme == "linear":
                        d["X"].append(", ".join(x_group))
                        d["model"].append(f"{mod}-{weight_scheme}")
                        w = np.linspace(0, 1, len(y))
                        res = sms.WLS(y, sms.add_constant(X), weights=w).fit(
                            cov_type="HC1"
                        )
                        d["adj_R^2"].append(res.rsquared_adj)
                    elif weight_scheme == "exp":
                        for j in range(-6, 0):
                            d["X"].append(", ".join(x_group))
                            d["model"].append(f"{mod}-{weight_scheme} ({j})")
                            res = sms.WLS(
                                y, sms.add_constant(X), weights=w
                            ).fit(cov_type="HC1")
                            d["adj_R^2"].append(res.rsquared_adj)

    res_df = (
        pd.DataFrame(d).sort_values("adj_R^2", ascending=False).reset_index()
    )
    x_var_best = res_df["X"][0].split(", ")
    return x_var_best


def combos(iterable):
    """Find all combinations of items of any length from input list."""
    for i in range(1, len(iterable)):
        for combination in it.combinations(x_vars, i):
            yield list(combination)


if __name__ == "__main__":
    update_voltility_model()
