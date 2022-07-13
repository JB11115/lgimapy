from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.models import BetaAdjustedPerformance
from lgimapy.utils import to_labelled_buckets

# %%

db = Database()
start = db.date("YEAR_START")
start_dates = db.date("MONTH_STARTS", start=start)
end_dates = db.date("MONTH_ENDS", start=start)
maturity = (5, 10)
ret_d = defaultdict(lambda: defaultdict(list))
ret_d = defaultdict(list)

for start_date, end_date in tqdm(
    zip(start_dates, end_dates), total=len(start_dates)
):
    db.load_market_data(start=start_date, end=end_date)
    mod = BetaAdjustedPerformance(db)
    year_month = f"{start_date.year}-{start_date.month}"
    mod.train(
        date=end_date,
        predict_from_date=start_date,
        maturity=maturity,
        universe="IG",
    )
    if start_date == start_dates[0]:
        start_df = mod.get_sector_table(return_type="XSRet")
        start_df["idx"] = start_df["Rating"] + "-" + start_df["Sector"]
        start_df = start_df.set_index("idx").rename_axis(None)

    elif start_date == start_dates[-1]:
        end_df = mod.get_sector_table(return_type="XSRet")
        end_df["idx"] = end_df["Rating"] + "-" + end_df["Sector"]
        end_df = end_df.set_index("idx").rename_axis(None)

    for ret_type in ["XSRet", "TRet"]:
        df = mod.get_sector_table(return_type=ret_type).sort_values(
            ["Rating", "Sector"]
        )
        df["idx"] = df["Rating"] + "-" + df["Sector"]
        df = df.set_index("idx").rename_axis(None)

        ret_d[f"real_{ret_type}"].append(
            df[f"Real*{ret_type}"].rename(year_month) / 1e4
        )
        ret_d[f"fcast_{ret_type}"].append(
            df[f"FCast*{ret_type}"].rename(year_month) / 1e4
        )


# %%
real_xsret_df_d = defaultdict(list)
out_performance_df_d = defaultdict(list)

real_xsret_df = pd.concat(ret_d[f"real_XSRet"], axis=1).T
fcast_xsret_df = pd.concat(ret_d[f"fcast_XSRet"], axis=1).T
real_tret_df = pd.concat(ret_d[f"real_TRet"], axis=1).T

fcast_tret_df = real_tret_df + fcast_xsret_df - real_xsret_df
rf_ret_df = real_tret_df - real_xsret_df

# Accumulate excess returns.
accum_rf_ret = np.prod(1 + rf_ret_df) - 1
accum_real_tret = np.prod(1 + real_tret_df) - 1
accum_fcast_tret = np.prod(1 + fcast_tret_df) - 1
accum_real_xsret = accum_real_tret - accum_rf_ret
accum_fcast_xsret = accum_fcast_tret - accum_rf_ret
accum_out_performance = accum_real_xsret - accum_fcast_xsret


list(start_df.columns)
table_df = start_df.copy()
last_month_start_oas_col = end_df.columns[4]
first_month_start_oas_col = start_df.columns[4]
end_df
table_df["1M*$\Delta$OAS"] = (
    end_df[last_month_start_oas_col]
    + end_df["1M*$\Delta$OAS"]
    - start_df[first_month_start_oas_col]
)
table_df["FCast*XSRet"] = accum_fcast_xsret * 1e4
table_df["Real*XSRet"] = accum_real_xsret * 1e4
table_df["Out*Perform"] = accum_out_performance * 1e4
table_df = table_df.rename(
    columns={"1M*$\Delta$OAS": "YTD*$\Delta$OAS"}
).reset_index(drop=True)

# %%
doc = Document(
    "YTD_Sector_Beta_Adjusted_Performance",
    path="reports/YTD_Beta_Adjusted_Performance",
)
doc.add_preamble(
    orientation="landscape",
)
name = f"{maturity[0]}-{maturity[1]}yr"
table_captions = {
    f"{name} Non-Fin Sector YTD Performance": "Industrials",
    f"{name} Fin Sector YTD Performance": "Financials",
    f"{name} Non-Corp Sector YTD Performance": "Non-Corp",
}
table_df
left, right = doc.add_minipages(2, valign="t")

for caption, top_level_sector in table_captions.items():
    table = (
        table_df[table_df["TopLevelSector"].isin({top_level_sector, "-"})]
        .sort_values("Out*Perform", ascending=False)
        .reset_index(drop=True)
    )
    table.index += 1  ## start count at 1
    # Get row colors for level 3 sectors.
    sector_row_colors = {
        "ENERGY": "magicmint",
        "TMT": "opal",
        "CONSUMER_NON_CYCLICAL": "sage",
        "BANKS": "powderblue",
        "UTILITY": "magicmint",
        "SOVS": "sage",
        "OTHER": "opal",
    }
    sector_locs = {}
    for sector, color in sector_row_colors.items():
        bottom_sector_names = {
            "BANKS": {
                "SIFI_BANKS_SR",
                "SIFI_BANKS_SUB",
                "US_REGIONAL_BANKS",
                "YANKEE_BANKS",
            },
            "TMT": {
                "WIRELINES_WIRELESS",
                "CABLE_SATELLITE",
                "MEDIA_ENTERTAINMENT",
                "TECHNOLOGY",
            },
            "ENERGY": {
                "INDEPENDENT",
                "MIDSTREAM",
                "INTEGRATED",
                "OIL_FIELD_SERVICES",
                "REFINING",
            },
            "CONSUMER_NON_CYCLICAL": {
                "FOOD_AND_BEVERAGE",
                "HEALTHCARE_EX_MANAGED_CARE",
                "MANAGED_CARE",
                "PHARMACEUTICALS",
                "TOBACCO",
            },
            "UTILITY": {"UTILITY_HOLDCO", "UTILITY_OPCO"},
            "SOVS": {"OWNED_NO_GUARANTEE", "SOVEREIGN"},
            "OTHER": {"HOSPITALS", "MUNIS", "UNIVERSITY"},
        }[sector]

        locs = tuple(
            table.loc[table["raw_sector"].isin(bottom_sector_names)].index
        )
        if locs:
            sector_locs[locs] = color

    table.drop(["raw_sector", "TopLevelSector"], axis=1, inplace=True)
    if top_level_sector == "Industrials":
        doc.start_edit(left)
    elif top_level_sector == "Financials":
        doc.start_edit(right)

    mod._return_type = "XSRET"
    added_precs = {"YTD*$\Delta$OAS": "+0f", first_month_start_oas_col: "0f"}
    doc.add_table(
        table,
        prec=mod.table_prec(table, added_precs=added_precs),
        col_fmt="llc|cc|ccc",
        caption=caption,
        table_notes_justification="l",
        adjust=True,
        font_size="scriptsize",
        multi_row_header=True,
        row_color=sector_locs,
        gradient_cell_col="Out*Perform",
        gradient_cell_kws={"cmax": "steelblue", "cmin": "firebrick"},
    )

    if top_level_sector != "Financials":
        doc.end_edit()


doc.save()
