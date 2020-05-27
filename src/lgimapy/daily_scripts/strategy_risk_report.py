import numpy as np
import pandas as pd

import lgimapy.vis as vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import load_json, root, replace_multiple

# %%
def main():
    # %%
    db = Database()
    date = db.date("today")
    # date = db.date("MTD")
    prev_date = db.date("1w")
    # prev_date = db.date("YTD")
    pdf_path = root("reports/strategy_risk")

    strategy = "US Long Credit"
    strategy = "US Long Corporate"
    strategy = "US Credit"

    n_table_rows = 10

    strat_acnt = load_json("strategy_accounts")
    strat_df = pd.Series(
        {k: len(v) for k, v in strat_acnt.items()}
    ).sort_values(ascending=False)
    # strat_df.head(10)

    universe = "stats"
    universe = "returns"

    strategies = [
        "US Long Credit",
        "US Long Corporate",
        "US Long Corp 2% Cap",
        "US Long Credit Ex Emerging Market",
        "Custom RBS",
        "GM_Blend",
        "US Corporate IG",
        "US Intermediate Credit",
        "US Intermediate Credit A or better",
        "US Long GC 70/30",
        "US Long GC 75/25",
        "US Long GC 80/20",
        "US Long Government/Credit",
        "Liability Aware Long Duration Credit",
        "US Credit",
        "US Credit A or better",
        "US Credit Plus",
        "US Long A+ Credit",
        "80% US A or Better LC/20% US BBB LC",
        "Barclays-Russell LDI Custom - DE",
        "INKA",
        "US Long Corporate A or better",
        "US Corporate 7+ Years",
    ]
    # %%
    fid = f"{date.strftime('%Y-%m-%d')}_Risk_Report"
    if universe == "stats":
        fid = f"{fid}_stats"
    # fid = f"{date.strftime('%Y-%m-%d')}_Risk_Report_Q1_2020"
    doc = Document(fid, path=pdf_path, fig_dir=True)
    doc.add_preamble(margin=1, bookmarks=True, bar_size=7)
    res = []
    from tqdm import tqdm

    for strategy in tqdm(strategies):
        res.append(
            get_single_latex_risk_page(
                strategy, date, prev_date, pdf_path, universe
            )
        )

    df = (
        pd.DataFrame(res, columns=["strategy", "summary", "pages"])
        .set_index("strategy")
        .reindex(strategies)
    )
    summary = pd.concat(df["summary"].values, axis=1, sort=False).T
    summary.index = [ix.replace("_", " ") for ix in summary.index]
    summary.index.rename("Strategy", inplace=True)
    doc.add_section("Summary")
    doc.add_table(
        summary.reset_index(),
        col_fmt="ll|cc|cccc|cccc",
        prec={col: ("2f" if "OAD" in col else "2%") for col in summary.columns},
        multi_row_header=True,
        adjust=True,
        hide_index=True,
        midrule_locs=[1, 6, 13, 18, 19, 22],
    )
    doc.add_text("\\pagebreak")
    for page in df["pages"].values:
        doc.body += page

    doc.save(save_tex=False)

    # %%


def get_single_latex_risk_page(
    strategy, date, prev_date, pdf_path, universe="returns", n_table_rows=10
):

    GC_strategies = {
        "US Long GC 70/30",
        "US Long GC 75/25",
        "US Long GC 80/20",
        "US Long Government/Credit",
    }
    is_GC_strategy = strategy in GC_strategies

    db = Database()
    curr_strat = db.load_portfolio(
        strategy=strategy, date=date, universe=universe
    )
    prev_strat = db.load_portfolio(
        strategy=strategy, date=prev_date, universe=universe
    )
    date_fmt = date.strftime("%#m/%#d")
    prev_date_fmt = prev_date.strftime("%#m/%#d")

    # Build overview table.
    default_properties = ["dts_pct", "dts_abs", "credit_pct"]
    if is_GC_strategy:
        properties = [
            "dts_pct",
            "dts_abs",
            "dts_gc",
            "credit_pct",
            "curve_duration(5)",
            "curve_duration(7)",
            "curve_duration(10)",
        ]
    else:
        properties = ["dts_pct", "dts_abs", "credit_pct"]

    overview_table = pd.concat(
        [
            curr_strat.properties(properties).rename(date_fmt),
            np.std(curr_strat.account_properties(properties)).rename("Std Dev"),
            prev_strat.properties(properties).rename(prev_date_fmt),
        ],
        axis=1,
        sort=False,
    )
    prop_formats = curr_strat.property_latex_formats(properties)
    for prop, fmt in zip(overview_table.index, prop_formats):
        overview_table.loc[prop] = [
            "{:.{}}".format(v, fmt) for v in overview_table.loc[prop]
        ]

    # Save treasury summary figure.
    vis.style()
    repl = {" ": "_", "/": "_", "%": "pct"}
    tsy_fig_fid = "fig/" + replace_multiple(strategy, repl)
    curr_strat.plot_tsy_weights(figsize=(6, 3.1))
    vis.savefig(pdf_path / tsy_fig_fid)
    vis.close()

    # Build rating overweight table.
    rating_ow_df = pd.concat(
        [
            curr_strat.rating_overweights("OAD"),
            curr_strat.rating_overweights("OASD"),
        ],
        axis=1,
        sort=False,
    )
    top_level_sector_ow_df = pd.concat(
        [
            curr_strat.top_level_sector_overweights("OAD"),
            curr_strat.top_level_sector_overweights("OASD"),
        ],
        axis=1,
        sort=False,
    )
    gen_ow_table = pd.concat(
        [rating_ow_df, top_level_sector_ow_df], axis=0, sort=False
    )
    sub_cats = {
        "AAA",
        "AA",
        "A",
        "BBB",
        "Industrials",
        "Financials",
        "Utilities",
    }
    gen_ow_table.index = [
        f"~{ix}" if ix in sub_cats else ix for ix in gen_ow_table.index
    ]

    # Build account dispersion table.
    n_accounts = len(curr_strat.accounts)
    disp = {
        "dts": curr_strat.account_dts().sort_values(ascending=False),
        "tsy_oad": curr_strat.account_tsy_oad().sort_values(ascending=False),
        "cash": curr_strat.account_cash_pct().sort_values(ascending=False),
    }
    if n_accounts > 10:
        # Limit to top 5 and bottom 5 accounts with midrule between.
        n_accounts = 10
        midrule_loc = "5"
        for key, val in disp.items():
            disp[key] = pd.concat([val.head(), val.tail()])
    else:
        midrule_loc = None

    account_table = pd.DataFrame(index=range(n_accounts))
    account_table["Account"] = disp["dts"].index
    account_table["DTS*(%)"] = disp["dts"].values
    account_table["Account "] = disp["tsy_oad"].index
    account_table["Tsy*OAD"] = disp["tsy_oad"].values
    account_table[" Account "] = disp["cash"].index
    account_table["Cash*(%)"] = disp["cash"].values

    # Build bond change in overweight table.
    n = n_table_rows
    bond_ow_df = pd.concat(
        [
            curr_strat.bond_overweights().rename("curr"),
            prev_strat.bond_overweights().rename("prev"),
        ],
        axis=1,
        sort=False,
    ).fillna(0)
    bond_ow_df["diff"] = bond_ow_df["curr"] - bond_ow_df["prev"]

    def clean_index(df):
        return (
            pd.Series(df.index)
            .apply(lambda x: x.split("#")[0].replace("$", "\\$"))
            .values
        )

    bond_ow_table = pd.DataFrame(index=range(n_table_rows))
    bond_ow_df.sort_values("diff", inplace=True, ascending=False)
    bond_ow_table["Risk*Added"] = clean_index(bond_ow_df)[:n]
    bond_ow_table["$\\Delta$OAD*(yrs)"] = bond_ow_df["diff"].values[:n]
    bond_ow_df.sort_values("diff", inplace=True, ascending=True)
    bond_ow_table["Risk*Reduced"] = clean_index(bond_ow_df)[:n]
    bond_ow_table["$\\Delta$OAD*(yrs) "] = bond_ow_df["diff"].values[:n]

    # Build sector overweight table.
    n = n_table_rows
    sector_ow_df = pd.concat(
        [
            curr_strat.sector_overweights().rename("curr"),
            prev_strat.sector_overweights().rename("prev"),
        ],
        axis=1,
        sort=False,
    ).fillna(0)

    sector_ow_df["diff"] = sector_ow_df["curr"] - sector_ow_df["prev"]

    sector_ow_table = pd.DataFrame(index=range(n_table_rows))
    sector_ow_df.sort_values("curr", inplace=True, ascending=False)
    sector_ow_table["Largest*OW"] = sector_ow_df.index[:n]
    sector_ow_table["OAD*(yrs)"] = sector_ow_df["curr"].values[:n]
    sector_ow_df.sort_values("curr", inplace=True, ascending=True)
    sector_ow_table["Largest*UW"] = sector_ow_df.index[:n]
    sector_ow_table["OAD*(yrs) "] = sector_ow_df["curr"].values[:n]
    sector_ow_df.sort_values("diff", inplace=True, ascending=False)
    sector_ow_table["Risk*Added"] = sector_ow_df.index[:n]
    sector_ow_table["$\\Delta$OAD*(yrs)"] = sector_ow_df["diff"].values[:n]
    sector_ow_table["Current*OAD"] = sector_ow_df["curr"].values[:n]
    sector_ow_df.sort_values("diff", inplace=True, ascending=True)
    sector_ow_table["Risk*Reduced"] = sector_ow_df.index[:n]
    sector_ow_table["$\\Delta$OAD*(yrs) "] = sector_ow_df["diff"].values[:n]
    sector_ow_table["Current * OAD "] = sector_ow_df["curr"].values[:n]

    # Build ticker overweight table.
    n = n_table_rows
    ticker_ow_df = pd.concat(
        [
            curr_strat.ticker_overweights().rename("curr"),
            prev_strat.ticker_overweights().rename("prev"),
        ],
        axis=1,
        sort=False,
    ).fillna(0)
    ticker_ow_df["diff"] = ticker_ow_df["curr"] - ticker_ow_df["prev"]

    ticker_ow_table = pd.DataFrame(index=range(n_table_rows))
    ticker_ow_df.sort_values("curr", inplace=True, ascending=False)
    ticker_ow_table["Largest*OW"] = ticker_ow_df.index[:n]
    ticker_ow_table["OAD*(yrs)"] = ticker_ow_df["curr"].values[:n]
    ticker_ow_df.sort_values("curr", inplace=True, ascending=True)
    ticker_ow_table["Largest*UW"] = ticker_ow_df.index[:n]
    ticker_ow_table["OAD*(yrs) "] = ticker_ow_df["curr"].values[:n]
    ticker_ow_df.sort_values("diff", inplace=True, ascending=False)
    ticker_ow_table["Risk*Added"] = ticker_ow_df.index[:n]
    ticker_ow_table["$\\Delta$OAD*(yrs)"] = ticker_ow_df["diff"].values[:n]
    ticker_ow_table["Current*OAD"] = ticker_ow_df["curr"].values[:n]
    ticker_ow_df.sort_values("diff", inplace=True, ascending=True)
    ticker_ow_table["Risk*Reduced"] = ticker_ow_df.index[:n]
    ticker_ow_table["$\\Delta$OAD*(yrs) "] = ticker_ow_df["diff"].values[:n]
    ticker_ow_table["Current * OAD "] = ticker_ow_df["curr"].values[:n]

    # Create summary series for cover page.
    tsy_weights = curr_strat.tsy_weights()
    summary = pd.Series(name=strategy)
    summary[f"DTS (%)*{date_fmt}"] = curr_strat.dts()
    summary[f"$\\Delta$DTS*{prev_date_fmt}"] = (
        curr_strat.dts() - prev_strat.dts()
    )
    summary["Cash*(%)"] = curr_strat.cash_pct()
    summary["Tsy OAD*Total"] = curr_strat.tsy_oad()
    summary[f"Tsy $\\Delta$OAD*{prev_date_fmt}"] = (
        curr_strat.tsy_oad() - prev_strat.tsy_oad()
    )
    summary["Tsy (%)*Total"] = curr_strat.tsy_pct()
    summary["Tsy (%)*$\\leq$5y"] = tsy_weights.loc[[2, 3, 5]].sum()
    summary["Tsy (%)*7y"] = tsy_weights.loc[7]
    summary["Tsy (%)*10y"] = tsy_weights.loc[10]
    summary["Tsy (%)*30y"] = tsy_weights.loc[30]

    # Make LaTeX page for strategy.
    fid = "strategy_risk_page_template"

    page = Document(fid=fid, path=pdf_path, load_tex=True)
    strategy_name = replace_multiple(strategy, {"_": " ", "%": "\\%"})
    page.set_variable("strategy_name", strategy_name)
    page.set_variable("tsy_fig_fid", tsy_fig_fid)
    page.start_edit("overview")
    page.add_table(
        overview_table, caption="General Overview", col_fmt="lccc", adjust=True
    )
    page.end_edit()
    page.start_edit("gen_overweight")
    page.add_table(
        gen_ow_table,
        caption="Rating/Top Level Sector Overweights",
        col_fmt="lcc",
        prec=2,
        indent_subindexes=True,
        midrule_locs=["Corp", "Non-Corp"],
        adjust=True,
        multi_row_header=True,
    )
    page.end_edit()
    page.start_edit("account_dispersion")
    page.add_table(
        account_table,
        caption="Account Dispersion",
        col_fmt="lrc|rc|rc",
        multi_row_header=True,
        midrule_locs=midrule_loc,
        prec={"DTS*(%)": "1%", "Tsy*OAD": "2f", "Cash*(%)": "1%"},
        font_size="scriptsize",
        hide_index=True,
        align="left",
    )
    page.end_edit()
    page.start_edit("bond")
    page.add_table(
        bond_ow_table,
        caption="Individual Bonds",
        col_fmt="lrc|rc",
        prec=3,
        align="left",
        multi_row_header=True,
        hide_index=True,
        font_size="scriptsize",
    )
    page.end_edit()

    def col_prec(df):
        prec = {}
        for col in df.columns:
            if "OAD" in col:
                if "Delta" in col:
                    prec[col] = "3f"
                else:
                    prec[col] = "2f"
        return prec

    table_kwargs = {
        "col_fmt": "lrc|rc|rcc|rcc",
        "multi_row_header": True,
        "hide_index": True,
        "adjust": True,
        "font_size": "scriptsize",
    }
    page.start_edit("ticker")
    page.add_table(
        ticker_ow_table,
        caption="Issuers",
        align="left",
        prec=col_prec(ticker_ow_table),
        div_bar_col=["$\\Delta$OAD*(yrs)", "$\\Delta$OAD*(yrs) "],
        center_div_bar_header=False,
        **table_kwargs,
    )
    page.end_edit()
    page.start_edit("sector")
    page.add_table(
        sector_ow_table,
        caption="Sectors",
        prec=col_prec(sector_ow_table),
        **table_kwargs,
    )
    page.end_edit()

    # page.save_as("test", save_tex=True)
    return strategy, summary, page.body[16:]


if __name__ == "__main__":
    main()
