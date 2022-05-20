from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lgimapy import vis
from lgimapy.data import Database, TreasuryCurve
from lgimapy.latex import Document
from lgimapy.utils import get_ordinal


# %%


class LGRAValuationPack:
    def __init__(self, db):
        self._db = db
        self.date = self._db.date("today")
        fid = f"LGRA_Valuation_Pack_{self.date:%Y-%m-%d}"

        self.doc = Document(fid, path="reports/portfolios/LGRA", fig_dir=True)

    def _maturity_to_text(self, maturity):
        return {(7, 15): "7-15 yr", (15, None): "15+ yr"}[maturity]

    def _make_index_plot(self, maturity, figsize=(8, 6)):
        ix = self._db.build_market_index(in_stats_index=True, maturity=maturity)
        oas = ix.OAS()
        cp = 100 * oas.rank(pct=True).iloc[-1]
        range = [oas.min(), oas.max()]

        fig, ax = vis.subplots(figsize=figsize)
        title = f"{self._maturity_to_text(maturity)} IG"
        vis.plot_timeseries(
            oas,
            color="navy",
            lw=1.5,
            median_line=True,
            pct_lines=(5, 95),
            label=(
                f"Historical Stats Index\n"
                f"Last: {oas.iloc[-1]:.0f} ({cp:.0f}{get_ordinal(cp)} %tile)\n"
                f"Range: [{range[0]:.0f}, {range[1]:.0f}]"
            ),
            title=title,
            ax=ax,
        )
        vis.legend(ax, title="10yr Stats", loc="upper left")
        fid = title.replace(" ", "_")
        return fid

    def _make_fin_nonfin_ratio_plot(self, maturity, figsize=(9, 6)):
        ix = self._db.build_market_index(
            in_stats_index=True, maturity=maturity, start=db.date("5y")
        )
        nonfin_ix = ix.subset(financial_flag=0)
        nonfin_A_oas = nonfin_ix.subset(rating=("A+", "A-")).OAS()
        nonfin_BBB_oas = nonfin_ix.subset(rating=("BBB+", "BBB-")).OAS()
        bank_oas = ix.subset(**db.index_kwargs("SIFI_BANKS_SR")).OAS()
        maturity_text = self._maturity_to_text(maturity)
        vis.plot_double_y_axis_timeseries(
            (bank_oas / nonfin_A_oas).rename("vs A-Rated Industrials"),
            (bank_oas / nonfin_BBB_oas).rename("vs BBB-Rated Industrials"),
            color_left="navy",
            color_right="skyblue",
            alpha=0.8,
            lw=2,
            title=f"{maturity_text} US Sr SIFI Bank / Nonfin Ratio",
        )
        fid = f"{maturity_text}_bank_nonfin".replace(" ", "_")
        return fid

    def _make_bbb_a_ratio_plot(self, maturity, figsize=(9, 6)):
        ix_nonfin = self._db.build_market_index(
            in_stats_index=True,
            financial_flag=0,
            maturity=maturity,
            start=db.date("5y"),
        )
        a_oas = ix_nonfin.subset(rating=("A+", "A-")).OAS().rename("A")
        bbb_oas = ix_nonfin.subset(rating=("BBB+", "BBB-")).OAS().rename("BBB")
        df = pd.concat(
            [
                a_oas,
                bbb_oas,
            ],
            axis=1,
        )
        df["abs"] = df["BBB"] - df["A"]
        df["rel"] = df["BBB"] / df["A"]

        # Plot
        fig, ax_left = vis.subplots(figsize=figsize)
        ax_right = ax_left.twinx()
        ax_right.grid(False)

        vis.plot_timeseries(
            df["rel"], color="navy", alpha=0.9, lw=1.5, ax=ax_left
        )
        ax_left.set_ylabel("Ratio", color="navy")
        ax_left.tick_params(axis="y", colors="navy")
        ax_left.axhline(np.median(df["rel"]), ls=":", lw=1.5, color="navy")

        ax_right.fill_between(
            df.index,
            0,
            df["abs"],
            color="gray",
            alpha=0.4,
        )
        ax_right.axhline(
            np.median(df["abs"]),
            ls=":",
            lw=1.5,
            color="gray",
        )
        ax_right.set_ylabel("Absolute Difference (bp)", color="gray")
        ax_right.tick_params(axis="y", colors="gray")
        maturity_text = self._maturity_to_text(maturity)
        ax_left.set_title(
            f"{maturity_text} Nonfin BBB/A Ratio", fontweight="bold"
        )
        fid = f"BBB_A_ratio_{maturity_text}".replace(" ", "_")
        return fid

    def _make_full_bbb_a_ratio_plot(self, figsize=(8, 6)):
        ix_nonfin = self._db.build_market_index(
            in_stats_index=True, financial_flag=0
        )
        ixs = {
            "30_A": ix_nonfin.subset(rating=("A+", "A-"), maturity=(25, 32)),
            "30_BBB": ix_nonfin.subset(
                rating=("BBB+", "BBB-"), maturity=(25, 32)
            ),
            "10_A": ix_nonfin.subset(rating=("A+", "A-"), maturity=(8.25, 11)),
            "10_BBB": ix_nonfin.subset(
                rating=("BBB+", "BBB-"), maturity=(8.25, 11)
            ),
        }
        df = pd.concat(
            [
                ix.market_value_median("OAS").rename(key)
                for key, ix in ixs.items()
            ],
            axis=1,
            sort=True,
        ).dropna(how="any")
        df["10 yr"] = df["10_BBB"] / df["10_A"]
        df["30 yr"] = df["30_BBB"] / df["30_A"]

        # Plot
        fig, ax_left = vis.subplots(figsize=figsize)
        ax_right = ax_left.twinx()
        ax_right.grid(False)

        right_last = 100 * df["30 yr"].rank(pct=True).iloc[-1]
        right_label = f"30 yr: {right_last:.0f}{get_ordinal(right_last)} %tile"
        left_last = 100 * df["10 yr"].rank(pct=True).iloc[-1]
        left_label = f"10 yr: {left_last:.0f}{get_ordinal(left_last)} %tile"

        ax_left.plot(df["10 yr"], c="navy", alpha=0.9, lw=1.5)
        ax_right.plot(
            df["30 yr"].iloc[:2], c="navy", alpha=0.9, lw=1.5, label=left_label
        )
        ax_left.set_ylabel("10 yr", color="navy")
        ax_left.tick_params(axis="y", colors="navy")
        ax_left.axhline(np.median(df["10 yr"]), ls=":", lw=1.5, color="navy")

        ax_right.plot(
            df["30 yr"], c="skyblue", alpha=0.9, lw=1.5, label=right_label
        )
        ax_right.axhline(
            np.median(df["30 yr"]),
            ls=":",
            lw=1.5,
            color="skyblue",
            label="Median",
        )
        pct = {x: np.percentile(df["30 yr"], x) for x in [5, 95]}
        pct_line_kwargs = {"ls": "--", "lw": 1.5, "color": "dimgrey"}
        ax_right.axhline(pct[5], label="5/95 %tiles", **pct_line_kwargs)
        ax_right.axhline(pct[95], label="_nolegend_", **pct_line_kwargs)
        ax_right.set_title("Non-Fin BBB/A Ratio", fontweight="bold")
        ax_right.set_ylabel("30 yr", color="skyblue")
        ax_right.tick_params(axis="y", colors="skyblue")
        vis.format_xaxis(ax_right, df["30 yr"], "auto")
        vis.set_percentile_limits(
            [df["10 yr"], df["30 yr"]], [ax_left, ax_right]
        )
        vis.legend(ax_right, loc="upper left", title="5yr Stats", fontsize=10)
        fid = "BBB_A_ratio"
        return fid

    def _make_strategy_score_plot(self, figsize=(5, 10)):
        background_color = "gainsboro"
        vis.style(background=background_color)

        # Overall Scores
        scores_df = (
            pd.read_excel(
                self._db.local("chicago_strategy_meeting_scores.xlsx"),
                index_col=0,
                sheet_name="Summary",
            )
            .iloc[:-3, -2:]
            .dropna(how="all")
            .fillna(0)
            .astype(int)
        )
        # Bold Short and Long Term scores.
        scores_df.index = [
            " ".join(f"$\\bf{{{t}}}$" for t in idx.split())
            if "Term" in idx
            else idx
            for idx in scores_df.index
        ]
        # Create table for plotting.
        df = pd.DataFrame(np.zeros([len(scores_df), 7]), index=scores_df.index)
        df.columns = np.arange(-3, 4)
        for col in [0, 1]:
            for name, score in scores_df.iloc[:, col].items():
                df.loc[name, score] = col + 1

        fig, ax = vis.subplots(figsize=figsize)
        cmap = mpl.colors.ListedColormap(["w", "skyblue", "navy"])
        sns.heatmap(
            df,
            cmap=cmap,
            alpha=1,
            linewidths=2,
            linecolor=background_color,
            cbar=False,
            ax=ax,
        )
        ax.xaxis.tick_top()
        ax.set_xticklabels("-3 -2 -1 0 +1 +2 +3".split(), fontsize=10)
        ax.set_yticklabels(df.index, fontsize=10)

        legend_elements = [
            mpl.patches.Patch(facecolor=background_color, label=""),
            mpl.patches.Patch(facecolor=background_color, label=""),
            mpl.patches.Patch(
                facecolor="navy", label=scores_df.columns[-1].strftime("%B")
            ),
            mpl.patches.Patch(
                facecolor="skyblue", label=scores_df.columns[0].strftime("%B")
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.2, -0.02),
            ncol=4,
            fontsize=14,
            frameon=False,
        )
        ax.set_title("Strategy Scoring\n", fontsize=15, fontweight="bold")

        fid = "strategy_scores"
        vis.savefig(fid, path=self.doc.fig_dir, dpi=200)
        vis.close()
        vis.style()  # reset background color
        return fid

    def _make_hy_ig_ratio_plot(self, figsize=(8, 6)):
        bbg_df = self._db.load_bbg_data(
            ["US_HY", "CDX_IG", "CDX_HY"], "OAS", start=db.date("5y")
        )
        ix = self._db.build_market_index(
            in_stats_index=True, start=db.date("5y")
        )
        oas = ix.market_value_median("OAS").rename("US_IG")
        df = pd.concat([bbg_df, oas], axis=1, sort=True).dropna(how="any")
        df["HY/IG Cash"] = df["US_HY"] / df["US_IG"]
        df["HY/IG CDX"] = df["CDX_HY"] / df["CDX_IG"]

        right_last = 100 * df["HY/IG CDX"].rank(pct=True).iloc[-1]
        right_label = f"CDX: {right_last:.0f}{get_ordinal(right_last)} %tile"
        left_last = 100 * df["HY/IG Cash"].rank(pct=True).iloc[-1]
        left_label = f"Cash: {left_last:.0f}{get_ordinal(left_last)} %tile"

        # Plot
        fig, ax_left = vis.subplots(figsize=figsize)
        ax_right = ax_left.twinx()
        ax_right.grid(False)

        ax_left.plot(df["HY/IG Cash"], c="navy", alpha=0.9, lw=1.5)
        ax_right.plot(
            df["HY/IG Cash"].iloc[:2],
            c="navy",
            alpha=0.9,
            lw=1.5,
            label=left_label,
        )
        ax_left.set_ylabel("Cash", color="navy")
        ax_left.tick_params(axis="y", colors="navy")
        ax_left.axhline(
            np.median(df["HY/IG Cash"]), ls=":", lw=1.5, color="navy"
        )

        ax_right.plot(
            df["HY/IG CDX"], c="skyblue", alpha=0.9, lw=1.5, label=right_label
        )
        ax_right.axhline(
            np.median(df["HY/IG CDX"]),
            ls=":",
            lw=1.5,
            color="skyblue",
            label="Median",
        )
        pct = {x: np.percentile(df["HY/IG CDX"], x) for x in [5, 95]}
        pct_line_kwargs = {"ls": "--", "lw": 1.5, "color": "dimgrey"}
        ax_right.axhline(pct[5], label="5/95 %tiles", **pct_line_kwargs)
        ax_right.axhline(pct[95], label="_nolegend_", **pct_line_kwargs)
        ax_right.set_title("HY/IG Ratios", fontweight="bold")
        ax_right.set_ylabel("CDX", color="skyblue")
        ax_right.tick_params(axis="y", colors="skyblue")
        vis.format_xaxis(ax_right, df["HY/IG CDX"], "auto")
        vis.set_percentile_limits(
            [df["HY/IG Cash"], df["HY/IG CDX"]], [ax_left, ax_right]
        )
        vis.legend(ax_right, loc="upper left", title="5yr Stats", fontsize=10)
        fid = "HY_IG_ratio"
        return fid

    def _build_cover_page(self):
        left, center, right = self.doc.add_minipages(
            n=3, widths=[0.35, 0.35, 0.25]
        )
        self.doc.add_cover_page()
        with self.doc.start_edit(left):
            fid = self._make_index_plot(maturity=(7, 15))
            self.doc.add_figure(fid, savefig=True)
            self.doc.add_vskip()
            fid = self._make_hy_ig_ratio_plot()
            self.doc.add_figure(fid, savefig=True)

        with self.doc.start_edit(center):
            fid = self._make_index_plot(maturity=(15, None))
            self.doc.add_figure(fid, savefig=True)
            self.doc.add_vskip()
            fid = self._make_full_bbb_a_ratio_plot()
            self.doc.add_figure(fid, savefig=True)

        with self.doc.start_edit(right):
            fid = self._make_strategy_score_plot()
            self.doc.add_figure(fid)

        self.doc.add_pagebreak()

    def _build_fin_nonfin_bbb_a_ratio_page(self):
        left, right = self.doc.add_minipages(n=2)

        with self.doc.start_edit(left):
            fid = self._make_fin_nonfin_ratio_plot(maturity=(7, 15))
            self.doc.add_figure(fid, savefig=True)
            self.doc.add_vskip()
            fid = self._make_bbb_a_ratio_plot(maturity=(7, 15))
            self.doc.add_figure(fid, savefig=True)

        with self.doc.start_edit(right):
            fid = self._make_fin_nonfin_ratio_plot(maturity=(15, None))
            self.doc.add_figure(fid, savefig=True)
            self.doc.add_vskip()
            fid = self._make_bbb_a_ratio_plot(maturity=(15, None))
            self.doc.add_figure(fid, savefig=True)

        self.doc.add_pagebreak()

    def build_report(self):
        self.doc.add_preamble(
            orientation="landscape",
            margin={"top": 1.3, "bottom": 2, "left": 0.5, "right": 0.5},
            cover_page=True,
            header="LGRA Valuation Pack",
            page_numbers=True,
        )
        self._build_cover_page()
        self._build_fin_nonfin_bbb_a_ratio_page()
        self.doc.save()


# %%


def main():
    vis.style()
    db = Database()
    db.load_market_data(start=db.date("10y"))
    self = LGRAValuationPack(db)
    self.build_report()


if __name__ == "__main__":
    main()


# %%
db = Database()
db.load_market_data(start=db.date("10y"))

# %%
world_states = {"Recession": (60, 95), "Soft Landing": (40, 20)}
portfolio_maturities = {"7-15 yr": (7, 15), "15+ yr": (15, None)}


def forecast_spreads(
    db, portfolio_maturities, world_states=None, export_csv=False
):
    all_world_states = {"Current": (None, None)}
    all_world_states.update(**world_states)

    ix = db.build_market_index(in_stats_index=True)
    rating_kws = {
        "AAA": "AAA",
        "AA": ("AA+", "AA-"),
        "A": ("A+", "A-"),
        "BBB": ("BBB+", "BBB-"),
    }
    d = defaultdict(list)
    for state, (percent_chance, spread_percentile) in all_world_states.items():
        if state == "Current":
            d["idx"].append("Probability")
            d["OAD"].append(np.nan)
        d[state].append(percent_chance)
        for mat, mat_kw in portfolio_maturities.items():
            for rating, rating_kw in rating_kws.items():
                sub_ix = ix.subset(rating=rating_kw, maturity=mat_kw)
                oas_s = sub_ix.OAS()
                if state == "Current":
                    d["idx"].append(f"{mat}, {rating}")
                    d["OAD"].append(sub_ix.MEAN("OAD").iloc[-1])
                if spread_percentile is None:
                    oas = oas_s.iloc[-1]
                else:
                    oas = oas_s.quantile(spread_percentile / 100)
                d[state].append(oas)

    oas_df = pd.DataFrame(d).set_index("idx").rename_axis(None).round(2)
    tcurve = TreasuryCurve(db.date("today"))

    yields = tcurve.yields(oas_df["OAD"])

    yield_df = oas_df.copy()
    for col in yield_df.columns[1:]:
        yield_df[col] = (yields.values + oas_df[col] / 1e4) * 1e2

    yield_df = yield_df.round(2)
    for state, (percent_chance, spread_percentile) in all_world_states.items():
        yield_df.loc["Probability", state] = percent_chance

    if export_csv:
        oas_df.to_csv("spreads.csv")
        yield_df.to_csv("yields.csv")
    return oas_df, yield_df


oas_df, yield_df = forecast_spreads(db, portfolio_maturities, world_states)
# %%

oas_df
yield_df

# %%
maturity = "15+"
world_state = "Recession"
trade_duration = 2


def create_investment_matrix(oas_df, maturity, world_states, trade_duration):
    s_df = oas_df[oas_df.index.str.startswith(maturity)]
    s_df.index = [idx.split()[-1] for idx in s_df.index]
    bbb_s = s_df.loc["BBB"]
    a_s = s_df.loc["A"]
    n = 11
    recession_probabilities = np.linspace(1, 0, n)
    a_rated_investment_pctages = np.linspace(0, 1, n)
    a = np.zeros((n, n))
    for i, recession_prob in enumerate(recession_probabilities):
        for j, a_rated_pct in enumerate(a_rated_investment_pctages):
            xs_spread_required = 0
            for world_state in world_states.keys():
                ws_prob = (
                    recession_prob
                    if world_state == "Recession"
                    else 1 - recession_prob
                )
                bbb_carry_xsret = (
                    (bbb_s["Current"] + bbb_s[world_state]) / 2 * trade_duration
                )
                bbb_holding_xsret = (
                    bbb_s["Current"] - bbb_s[world_state]
                ) * bbb_s["OAD"]
                a_carry_xsret = (
                    (a_s["Current"] + a_s[world_state]) / 2 * trade_duration
                )
                a_holding_xsret = (a_s["Current"] - a_s[world_state]) * a_s[
                    "OAD"
                ]
                bbb_xsret = bbb_carry_xsret + bbb_holding_xsret
                a_xsret = a_rated_pct * (a_carry_xsret + a_holding_xsret)
                ws_implied_starting_spread_diff = (a_xsret - bbb_xsret) / bbb_s[
                    "OAD"
                ]
                xs_spread_required += ws_implied_starting_spread_diff * ws_prob

            a[i, j] = xs_spread_required

    def to_label(index):
        return [f"{idx:.0%}" for idx in index]

    df = pd.DataFrame(
        a,
        columns=to_label(a_rated_investment_pctages),
        index=to_label(recession_probabilities),
    )
    return df


vpack = LGRAValuationPack(db)

for maturity in portfolio_maturities.keys():
    df = create_investment_matrix(
        oas_df, maturity, world_states, trade_duration=0.25
    )

    fig, ax = vis.subplots(figsize=(14, 11))
    sns.heatmap(
        df,
        cmap="PiYG",
        fmt=".0f",
        annot=True,
        center=0,
        cbar_kws={"label": "Spread Pickup (bp)"},
        ax=ax,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_ylabel(f"Probability of Recession within {trade_duration} Years")
    ax.set_xlabel("% of Ultimate BBB Allocation Initially Allocated to A's")
    ax.set_title(
        (
            f"Spread Pickup to BBB Allocation by Delaying Investment into BBB's"
            f"\n (in expected bp), {maturity} Portfolio"
        ),
        fontweight="bold",
    )
    fid = f"spread_pickup_{maturity.replace(' ', '_')}"
    # vis.savefig(fid, vpack.doc.fig_dir)
    vis.show()
