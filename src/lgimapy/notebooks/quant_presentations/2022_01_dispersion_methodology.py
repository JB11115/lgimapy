from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns

from lgimapy import vis

vis.style()
# %%

test_arrays = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 3, 3, 3, 3, 7, 7, 7, 7, 10],
    [1, 4, 4, 4, 4, 6, 6, 6, 6, 10],
    [1, 4, 4, 4, 5, 5, 5, 6, 6, 10],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 10],
    [1, 5, 5, 5, 5, 5, 5, 5, 5, 10],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 10],
]
x = np.array(test_arrays[0])


def mean_minus_median(x):
    return np.mean(x) - np.median(x)


def std_dev(x):
    return np.std(x)


def IDR(x):
    return np.percentile(x, q=90) - np.percentile(x, q=10)


def IQR(x):
    return np.percentile(x, q=75) - np.percentile(x, q=25)


def RSD(x):
    return np.std(x) / np.mean(x)


def DCV(x):
    return IDR(x) / np.median(x)


def QCV(x):
    return IQR(x) / np.median(x)


def MAD(x):
    return np.mean(np.abs(x - np.median(x)))


def RMAD(x):
    return MAD(x) / np.mean(x)


def IQR_plus_MAD(x):
    return IQR(x) + MAD(x)


def QCV_plus_RMAD(x):
    return QCV(x) + RMAD(x)


funcs_d = {
    "abs": [
        "mean_minus_median",
        "std_dev",
        "MAD",
        "IDR",
        "IQR",
        "IQR_plus_MAD",
    ],
    "rel": [
        "RSD",
        "RMAD",
        "DCV",
        "QCV",
        "QCV_plus_RMAD",
    ],
}

for key, funcs in funcs_d.items():
    d = defaultdict(list)
    for func in funcs:
        for array in test_arrays:
            d[func].append(eval(f"{func}(np.array(array))"))

    idx = [str(array) for array in test_arrays]
    df = pd.DataFrame(d, index=idx)
    # df = pd.DataFrame(d)

    fig, ax = vis.subplots(figsize=(8, 5))
    sns.heatmap(
        (df - df.min()) / (df.max() - df.min()),
        cmap="coolwarm",
        linewidths=0.4,
        annot=df,
        annot_kws={"fontsize": 10},
        fmt=".2f",
        ax=ax,
        cbar=False,
    )
    ax.xaxis.tick_top()
    ax.set_xticklabels(df.columns, rotation=45, ha="left", fontsize=12)
    ax.set_yticklabels(df.index, ha="right", fontsize=12, va="center")
    vis.savefig(f"{key}_dispersion_methodology")

# %%
