from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.models import BetaAdjustedPerformance
from lgimapy.utils import to_labelled_buckets

# %%
maturity_years = [2, 4, 6, 8, 10, 12, 15, 20, 24, 31]
maturity_buckets = to_labelled_buckets(
    maturity_years,
    right_end_closed=False,
)

db = Database()
start = "1/1/2001"
start_dates = db.date("MONTH_STARTS", start=start)
end_dates = db.date("MONTH_ENDS", start=start)

ret_d = defaultdict(lambda: defaultdict(list))
for start_date, end_date in tqdm(
    zip(start_dates, end_dates), total=len(start_dates)
):
    db.load_market_data(start=start_date, end=end_date)
    mod = BetaAdjustedPerformance(db)
    year_month = f"{start_date.year}-{start_date.month}"
    for label, maturity_kws in maturity_buckets.items():
        mod.train(
            date=end_date,
            predict_from_date=start_date,
            maturity=maturity_kws,
            universe="IG",
        )
        for ret_type in ["XSRet", "TRet"]:
            df = mod.get_sector_table(return_type=ret_type).sort_values(
                ["Rating", "Sector"]
            )
            df["idx"] = df["Rating"] + "-" + df["Sector"]
            df = df.set_index("idx").rename_axis(None)

            ret_d[f"real_{ret_type}"][label].append(
                df[f"Real*{ret_type}"].rename(year_month) / 1e4
            )
            ret_d[f"fcast_{ret_type}"].append(
                df[f"FCast*{ret_type}"].rename(year_month) / 1e4
            )


# %%
year = 2001
maturity_bucket = "24-31"
real_xsret_df_d = defaultdict(list)
out_performance_df_d = defaultdict(list)
years = list(set([dt.year for dt in start_dates]))
for maturity_bucket in maturity_buckets.keys():
    real_xsret_df = pd.concat(ret_d[f"real_XSRet"][maturity_bucket], axis=1).T
    fcast_xsret_df = pd.concat(ret_d[f"fcast_XSRet"][maturity_bucket], axis=1).T
    tret_df = pd.concat(ret_d[f"real_TRet"][maturity_bucket], axis=1).T
    for year in years:
        fcast_xsret_yr_df = fcast_xsret_df[
            fcast_xsret_df.index.str.startswith(str(year))
        ]
        real_xsret_yr_df = real_xsret_df[
            real_xsret_df.index.str.startswith(str(year))
        ]
        real_tret_yr_df = tret_df[tret_df.index.str.startswith(str(year))]
        fcast_tret_yr_df = (
            real_tret_yr_df + fcast_xsret_yr_df - real_xsret_yr_df
        )
        rf_ret_yr_df = real_tret_yr_df - real_xsret_yr_df

        # Accumulate excess returns.
        accum_rf_ret = np.prod(1 + rf_ret_yr_df) - 1
        accum_real_tret = np.prod(1 + real_tret_yr_df) - 1
        accum_fcast_tret = np.prod(1 + fcast_tret_yr_df) - 1
        accum_real_xsret = accum_real_tret - accum_rf_ret
        accum_fcast_xsret = accum_fcast_tret - accum_rf_ret
        accum_out_performance = accum_real_xsret - accum_fcast_xsret

        real_xsret_df_d[maturity_bucket].append(accum_real_xsret.rename(year))
        out_performance_df_d[maturity_bucket].append(
            accum_out_performance.rename(year)
        )


# %%
def make_pdf(d, fid):
    doc = Document(fid, path="reports/PM_meetings/2022_04")
    doc.add_preamble(
        bookmarks=True,
        margin={
            "paperheight": 60,
            "paperwidth": 35,
            "left": 0.5,
            "right": 0.5,
            "top": 0.5,
            "bottom": 0.2,
        },
    )
    color_thresh_d = {
        "2-4": 0.07,
        "4-6": 0.07,
        "6-8": 0.08,
        "8-10": 0.1,
        "10-12": 0.12,
        "12-15": 0.14,
        "15-20": 0.16,
        "20-24": 0.18,
        "24-31": 0.2,
        "31+": 0.22,
    }

    for maturity_bucket, df_list in d.items():
        doc.add_section(f"{maturity_bucket} yr")
        table = pd.concat(df_list, axis=1).replace(0, np.nan).sort_index()
        v = color_thresh_d[maturity_bucket]
        doc.add_table(
            table,
            col_fmt="l" + "r" * len(table.columns),
            prec="1%",
            adjust=True,
            gradient_cell_col=table.columns,
            gradient_cell_kws={
                "cmax": "steelblue",
                "cmin": "firebrick",
                "vmin": -v,
                "vmax": v,
            },
        )
        doc.add_pagebreak()
    doc.save()


make_pdf(real_xsret_df_d, "sector_excess_return_history")
make_pdf(out_performance_df_d, "beta_adjusted_excess_return_history")
