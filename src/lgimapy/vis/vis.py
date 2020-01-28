import os
import re
from datetime import datetime as dt
from datetime import timedelta
from inspect import cleandoc

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sms
import squarify
from pandas._libs.tslibs.timestamps import Timestamp
from tqdm import tqdm

from lgimapy.bloomberg import bdh
from lgimapy.data import Database
from lgimapy.utils import mkdir, root, to_list

# pd.plotting.register_matplotlib_converters()
# %matplotlib qt
# os.chdir(root("src/lgimapy/vis"))

# %%
def style(style="fivethirtyeight", background="white"):
    """Apply preferred style to interactive and saved plots."""
    pd.plotting.register_matplotlib_converters()
    plt.style.use(style)
    mpl.rcParams["axes.facecolor"] = background
    mpl.rcParams["axes.edgecolor"] = background
    mpl.rcParams["figure.facecolor"] = background
    mpl.rcParams["figure.edgecolor"] = background
    mpl.rcParams["savefig.facecolor"] = background
    mpl.rcParams["savefig.edgecolor"] = background


def show():
    """Display current figure."""
    plt.tight_layout()
    plt.show()


def close(fig=None):
    """
    Clear current figure and all axes.

    Parameters
    ----------
    fig: mpl.Figure
        Figure instance to close.

    """
    plt.cla()
    plt.clf()
    if fig is not None:
        plt.close(fig)
    else:
        plt.close(plt.gcf())
    plt.close("all")


def savefig(fid, path=None, dpi=300):
    """Save figure to specified location."""
    if path is not None:
        mkdir(path)
        full_fid = f"{str(path)}/{fid}.png"
    else:
        full_fid = f"{fid}.png"
    plt.tight_layout()
    plt.savefig(full_fid, dpi=dpi, bbox_inches="tight")


def subplots(*args, **kwargs):
    """
    Create a matplotlib figure and set of subplots.

    See Also
    --------
    matplotlib.pyplot.subplots

    Parameters
    ----------
    args:
        Positional arguments for matplotlib.pyplot.subplots
    kwargs:
        Keyword arguments for matplotlib.pyplot.subplots

    Returns
    -------
    fig: matplotlib.Figure
        Figure instance.
    ax: matplotlib.axes.Axes
        Object or array of Axes objects.
    """
    plot_kwargs = {"figsize": (8, 6)}
    plot_kwargs.update(**kwargs)
    return plt.subplots(*args, **plot_kwargs)


def colors(color):
    """
    LGIMA umbrella colors.

    Parameters
    ----------
    color: str
        Umbrella color or first letter of color to return.

    Returns
    -------
    str:
        Specified color hex code.
    """
    return {
        "red": "#DD2026",
        "r": "#DD2026",
        "blue": "#209CD8",
        "b": "#209CD8",
        "yellow": "#FDD302",
        "y": "#FDD302",
        "green": "#09933C",
        "g": "#09933C",
    }[color.lower()]


# %%
def coolwarm(
    x, center=0, symettric=False, quality=1000, cmap="coolwarm", pal=None
):
    """
    Create custom diverging color palette centered around
    specified value.

    Parameters
    ----------
    x: array-like
        Sorted values to be converted to colors.
    center: float, default=0
        Central value for divergin colorbar.
    symmetric: # TODO bool, default=False
        If True, use symmetric gradient for each side of
        the color range, such that full color range
        would not be used for skewed data.
        If False, use full color gradient to both min and
        max of the value range, so that full color bar
        is used regardless of skew in data.
    quality: int, default=1000
        Total number of uniuqe colors to include in colorbar.
        Higher number gives better gradient between colors.

    Retuns
    ------
    List[str]:
        List of custom color pallete hex codes for each
        respective input value.
    """
    if pal is None:
        pal = sns.color_palette(cmap, quality).as_hex()

    # Convert x into array and split by center value.
    x = np.array(x)
    x_neg = np.append(x[x <= center], [center])
    x_pos = np.append([center], x[x > center])

    # Center both sides around given center.
    x_neg_center = (x_neg - np.min(x_neg)) / (np.max(x_neg) - np.min(x_neg))
    x_pos_center = (x_pos - np.min(x_pos)) / (np.max(x_pos) - np.min(x_pos))

    # Scale each side to color pallete index.
    x_neg_norm = (quality * 0.5 * x_neg_center[:-1]).astype(int)
    x_pos_norm = ((quality - 1) * (0.5 + 0.5 * x_pos_center[1:])).astype(int)

    return [pal[ix] for ix in np.concatenate([x_neg_norm, x_pos_norm])]


def hex_to_rgb(h):
    """
    Convert hex color codes to RGB.

    Parameters
    ----------
    h: str or List[str].
        Hex codes for conversion.

    Returns
    -------
    Tuple[int] or List[Tuple[int]]:
        RGB triple(s) for input hex code(s).
    """
    hex_codes = to_list(h, dtype=str)
    rgb_codes = []
    for hex_code in hex_codes:
        rgb_codes.append(
            tuple(int(hex_code.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        )
    if len(rgb_codes) == 1:
        return rgb_codes[0]
    else:
        return rgb_codes


def spider_plots():
    # %%
    prev_df = pd.DataFrame(
        {
            "Economic Backdrop": [1, 0, 1, 0],
            "Liquidiyt/Monetary Policy": [1, 1, 1, 0],
            "Geopolitics": [-2, -1, -2, -2],
            "Corporate Strength (Fins)": [0, -1, -1, 0],
            "Corporate Strength (Non Fins)": [0, 0, 0, -1],
            "Supply/Demand (Fins)": [1, 1, 1, 1],
            "Supply/Demand (Non Fins)": [1, 1, 2, 1],
            "Short Term": [0, 0, 1, 0],
            "Long Term": [0, -1, 0, -2],
        },
        index=["US", "Europe", "UK", "Global"],
    ).T

    col = "US"

    df = pd.DataFrame(
        {
            "Economic Backdrop": [0, 0, 1, 0],
            "Liquidiyt/Monetary Policy": [1, 2, 1, 1],
            "Geopolitics": [-1, -1, -1, -1],
            "Corporate Strength (Fins)": [0, -1, -1, 0],
            "Corporate Strength (Non Fins)": [0, 0, 0, -1],
            "Supply/Demand (Fins)": [1, 1, 1, 0],
            "Supply/Demand (Non Fins)": [1, 1, 1, 1],
            "Short Term": [0, 1, 1, 1],
            "Long Term": [0, -1, 0, -2],
        },
        index=["US", "Europe", "UK", "Global"],
    ).T

    col = "US"

    # %%

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={"polar": True})

    colors = ["steelblue", "darkgreen", "firebrick", "darkorchid"]
    for ax, col, c in zip(axes.flat, prev_df.columns, colors):
        labels = list(prev_df.index)
        vals = prev_df[col].values
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        vals = np.concatenate((vals, [vals[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        ax.plot(angles, vals, "o-", color="grey", linewidth=2, alpha=0.4)
        ax.fill(angles, vals, alpha=0.15, color="grey")
        ax.set_title(f"{col}\n", fontsize=14, fontweight="bold")
        ax.set_thetagrids(angles * 180 / np.pi, labels, fontsize=8)
        ax.set_yticks(np.arange(-3, 4))
        ax.set_yticklabels(np.arange(-3, 4), size=7)
        ax.grid(True)

    for ax, col, c in zip(axes.flat, df.columns, colors):
        labels = list(df.index)
        vals = df[col].values
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        vals = np.concatenate((vals, [vals[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        ax.plot(angles, vals, "o-", color=c, linewidth=2)
        ax.fill(angles, vals, alpha=0.25, color=c)
        ax.set_title(f"{col}\n", fontsize=14, fontweight="bold")
        ax.set_thetagrids(angles * 180 / np.pi, labels, fontsize=8)
        ax.set_yticks(np.arange(-3, 4))
        ax.set_yticklabels(np.arange(-3, 4), size=7)
        ax.grid(True)
    plt.tight_layout()
    plt.show()
    # %%


def treemap():
    from lgimapy.data import Database

    db = Database()
    db.load_market_data()
    ix = db.build_market_index(
        rating=("BBB+", "BBB-"), financial_flag="non-financial"
    )

    # %%
    ix.df["market_value"] = ix.df["AmountOutstanding"] * ix.df["DirtyPrice"]
    df = ix.df[["Ticker", "market_value"]]
    df.set_index("Ticker", inplace=True)
    df = df.groupby(["Ticker"]).sum()
    df.sort_values("market_value", inplace=True, ascending=False)
    total = np.sum(df["market_value"])
    cumsum = np.cumsum(df.values) / total
    labels = [
        f"{i}\n${mv * 1e-6:.1f}B" if cs < 0.493 else ""
        for i, mv, cs in zip(df.index, df.values.ravel(), cumsum)
    ]

    df.sort_values("market_value", inplace=True)
    # %%
    plt.figure(figsize=[22, 14])
    colors = ["#DD2026", "#209CD8", "#FDD302", "#09933C"]
    squarify.plot(
        sizes=df["market_value"],
        pad=0.0001,
        # pad=False,
        label=labels[::-1],
        color=colors,
        alpha=0.7,
        text_kwargs={"fontsize": 7},
    )
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        labelleft=False,
        labelbottom=False,
    )

    savefig("BBB treemap")
    plt.show()


def linreg():
    # %%
    x = np.array([46, 12, 14, 18, 15, 19, 20, 30, 20, 20])
    bm = np.array([16.8, 10.7, 17.1, 12.7, -6.6, 16.4, -4.6, 10.2, 12.2, -6.8])
    lgim = np.array([19.8, 12.2, 18.3, 14.5, -4.7, 16.6, -4.1, 11, 12.3, -6.5])
    alpha = lgim - bm

    ols = sms.OLS(y, sms.add_constant(x)).fit()

    # Plot results.
    fig, ax = plt.subplots(1, 1, figsize=[8, 8])
    ax.plot(x, y, "o", ms=6, c="#209CD8")
    ax.plot(x, x * ols.params[1] + ols.params[0], lw=1.5, c="#FDD302")
    ax.set_xlabel("Downgrades in US Credit")
    ax.set_ylabel("LGIMA Outperformance")
    tick = mpl.ticker.StrMethodFormatter("{x:.1f}%")
    ax.yaxis.set_major_formatter(tick)
    savefig("dongrades vs outperformance")
    plt.show()
    # %%


# %%
def calculate_ticks(ax, ticks, round_to=0.1, center=False):
    """Naive solution to aligning grids for multi y-axis plot."""
    upperbound = np.ceil(ax.get_ybound()[1] / round_to)
    lowerbound = np.floor(ax.get_ybound()[0] / round_to)
    dy = upperbound - lowerbound
    fit = np.floor(dy / (ticks - 1)) + 1
    dy_new = (ticks - 1) * fit
    if center:
        offset = np.floor((dy_new - dy) / 2)
        lowerbound = lowerbound - offset
    values = np.linspace(lowerbound, lowerbound + dy_new, ticks)
    return values * round_to


def plot_double_y_axis_timeseries(
    s_left,
    s_right,
    start=None,
    end=None,
    invert_right_axis=False,
    ax=None,
    figsize=(8, 6),
    xtickfmt=None,
    xlabel=None,
    ytickfmt_left=None,
    ytickfmt_right=None,
    ylabel_left=None,
    ylabel_right=None,
    title=None,
    legend=False,
    plot_kwargs=None,
    left_plot_kwargs=None,
    right_plot_kwargs=None,
    color_yticks=True,
    **kwargs,
):
    """Plot two timeseries, one on each y-axis."""
    # Fix start and end dates of series.
    if start is not None:
        s_left = s_left[s_left.index >= start]
        s_right = s_right[s_right.index >= start]
    if end is not None:
        s_left = s_left[s_left.index <= end]
        s_right = s_right[s_right.index <= end]

    # Create axis objects.
    if ax is None:
        fig, ax_left = plt.subplots(1, 1, figsize=figsize)
    else:
        ax_left = ax
    ax_right = ax_left.twinx()
    ax_right.grid(False)

    # Update kwargs and plot.
    kwargs_left = {
        "color": "steelblue",
        "alpha": 0.9,
        "lw": 1.5,
        "label": s_left.name,
    }
    kwargs_left.update(**kwargs)
    if left_plot_kwargs is not None:
        kwargs_left.update(**left_plot_kwargs)
    kwargs_right = {
        "color": "firebrick",
        "alpha": 0.9,
        "lw": 1.5,
        "label": s_right.name,
    }
    kwargs_right.update(**kwargs)
    if right_plot_kwargs is not None:
        kwargs_right.update(**right_plot_kwargs)
    ln_left = ax_left.plot(s_left, **kwargs_left)
    ln_right = ax_right.plot(s_right, **kwargs_right)

    # Apply custom formatting on x and y-axis if specified.
    format_yaxis(ax_left, ytickfmt_left)
    format_yaxis(ax_right, ytickfmt_right)
    format_xaxis(ax_right, s_right, xtickfmt)

    # Add extra info and color labels.
    if title is not None:
        ax_right.set_title(title, fontweight="bold")
    if xlabel is not None:
        ax_right.set_xlabel(xlabel)
    if ylabel_left is not None:
        ax_left.set_ylabel(ylabel_left, color=kwargs_left["color"])
    if ylabel_right is not None:
        ax_right.set_ylabel(ylabel_right, color=kwargs_right["color"])
    if color_yticks:
        ax_left.tick_params(axis="y", colors=kwargs_left["color"])
        ax_right.tick_params(axis="y", colors=kwargs_right["color"])
    if legend:
        lns = ln_left + ln_right
        labs = [ln.get_label() for ln in lns]
        ax_right.legend(lns, labs)
    if invert_right_axis:
        ax_right.set_ylim(*ax_right.get_ybound()[::-1])
    plt.tight_layout()


def plot_triple_y_axis_timeseries(
    s_left,
    s_right_inner,
    s_right_outer,
    start=None,
    end=None,
    invert_left_axis=False,
    ax=None,
    figsize=(8, 6),
    xtickfmt=None,
    xlabel=None,
    ytickfmt_left=None,
    ytickfmt_right_inner=None,
    ytickfmt_right_outer=None,
    ylabel_left=None,
    ylabel_right_inner=None,
    ylabel_right_outer=None,
    title=None,
    legend=False,
    grid=True,
    plot_kwargs=None,
    plot_kwargs_left=None,
    plot_kwargs_right_inner=None,
    plot_kwargs_right_outer=None,
    color_yticks=True,
    **kwargs,
):
    """Plot two timeseries, one on each y-axis."""
    # Fix start and end dates of series.
    if start is not None:
        s_left = s_left[s_left.index >= start]
        s_right_inner = s_right_inner[s_right_inner.index >= start]
        s_right_outer = s_right_outer[s_right_outer.index >= start]
    if end is not None:
        s_left = s_left[s_left.index <= end]
        s_right_inner = s_right_inner[s_right_inner.index <= end]
        s_right_outer = s_right_outer[s_right_outer.index <= end]

    # Create axis objects.
    if ax is None:
        fig, ax_left = plt.subplots(1, 1, figsize=figsize)
    else:
        ax_left = ax
    ax_right_inner = ax_left.twinx()
    ax_right_outer = ax_left.twinx()
    if not grid:
        ax_left.grid(False)

    # Make spine visible for outer right axis and push it
    # outside the inner axis.
    ax_right_inner.spines["right"].set_position(("axes", 1.05))
    ax_right_outer.spines["right"].set_position(("axes", 1.25))
    make_patch_spines_invisible(ax_left)
    for axis in [ax_right_inner, ax_right_outer]:
        make_patch_spines_invisible(axis)
        axis.grid(False)
        axis.spines["right"].set_visible(True)
        axis.spines["right"].set_linewidth(1.5)
        axis.spines["right"].set_color("lightgrey")
        axis.tick_params(right="on", length=5)

    # Update kwargs and plot.
    kwargs_left = {
        "color": "darkorchid",
        "alpha": 0.9,
        "lw": 1.5,
        "label": s_left.name,
    }
    kwargs_left.update(**kwargs)
    if plot_kwargs_left is not None:
        kwargs_left.update(**plot_kwargs_left)
    kwargs_right_inner = {
        "color": "navy",
        "alpha": 0.9,
        "lw": 1.5,
        "label": s_right_inner.name,
    }
    kwargs_right_inner.update(**kwargs)
    if plot_kwargs_right_inner is not None:
        kwargs_right_inner.update(**plot_kwargs_right_inner)
    kwargs_right_outer = {
        "color": "dodgerblue",
        "alpha": 0.9,
        "lw": 1.5,
        "label": s_right_outer.name,
    }
    kwargs_right_outer.update(**kwargs)
    if plot_kwargs_right_outer is not None:
        kwargs_right_outer.update(**plot_kwargs_right_outer)
    ln_left = ax_left.plot(s_left, **kwargs_left)
    ln_right_inner = ax_right_inner.plot(s_right_inner, **kwargs_right_inner)
    ln_right_outer = ax_right_outer.plot(s_right_outer, **kwargs_right_outer)

    # Apply custom formatting on x and y-axis if specified.
    format_yaxis(ax_left, ytickfmt_left)
    format_yaxis(ax_right_inner, ytickfmt_right_inner)
    format_yaxis(ax_right_outer, ytickfmt_right_outer)
    format_xaxis(ax_right_outer, s_right_outer, xtickfmt)

    # Add extra info and color labels.
    if title is not None:
        ax_right_outer.set_title(title, fontweight="bold")
    if xlabel is not None:
        ax_right_outer.set_xlabel(xlabel)
    if ylabel_left is not None:
        ax_left.set_ylabel(ylabel_left, color=kwargs_left["color"])
    if ylabel_right_inner is not None:
        ax_right_inner.set_ylabel(
            ylabel_right_inner, color=kwargs_right_inner["color"]
        )
    if ylabel_right_outer is not None:
        ax_right_outer.set_ylabel(
            ylabel_right_outer, color=kwargs_right_outer["color"]
        )
    if color_yticks:
        ax_left.tick_params(axis="y", colors=kwargs_left["color"])
        ax_right_inner.tick_params(axis="y", colors=kwargs_right_inner["color"])
        ax_right_outer.tick_params(axis="y", colors=kwargs_right_outer["color"])
    if legend:
        lns = ln_left + ln_right_inner + ln_right_outer
        labs = [ln.get_label() for ln in lns]
        ax_right_outer.legend(lns, labs)
    if invert_left_axis:
        ax_left.set_ylim(*ax_left.get_ybound()[::-1])
    plt.tight_layout()


def plot_timeseries(
    s,
    start=None,
    end=None,
    stats_start=None,
    stats=False,
    bollinger=False,
    mean_line=False,
    pct_lines=False,
    ax=None,
    figsize=(8, 6),
    ytickfmt=None,
    xtickfmt=None,
    ylabel=None,
    xlabel=None,
    title=None,
    legend=True,
    **kwargs,
):
    """Plot simple timeseries."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_kwargs = {"color": "steelblue", "alpha": 0.9, "lw": 2.5}
    plot_kwargs.update(**kwargs)

    # Subset series to stat collection period and get stat formats.
    if stats_start is not None:
        s = s[s.index >= pd.to_datetime(stats_start)].copy()
    if ytickfmt is None:
        n = 0
        f = "f"
    else:
        n = int(re.findall("\d+", ytickfmt)[0])
        if "%" in ytickfmt:
            n += 1
            f = "%"
        else:
            f = "f"

    # Get stats and format them into the legend.
    if stats:
        plot_kwargs["label"] = cleandoc(
            f"""
             Last: {s[-1]:.{n}{f}}
             %tile: {s.rank(pct=True)[-1]:.0%}
             Range: [{np.min(s):.{n}{f}}, {np.max(s):.{n}{f}}]
             """
        )
    else:
        if "label" not in plot_kwargs:
            plot_kwargs["label"] = "_nolegend_"

    if mean_line:
        avg = np.mean(s)
        ax.axhline(
            avg,
            ls="--",
            lw=1.5,
            color="firebrick",
            label=f"Mean: {avg:.{n}{f}}",
        )

    if pct_lines:
        pct_line_kwargs = {"ls": "--", "lw": 1.5, "color": "dimgrey"}
        low = np.percentile(s, pct_lines[0])
        high = np.percentile(s, pct_lines[1])
        label = f"{pct_lines[0]}/{pct_lines[1]} %tiles"
        ax.axhline(low, label=label, **pct_line_kwargs)
        ax.axhline(high, label="_nolegend_", **pct_line_kwargs)

    # Add Bollinger bands if specified.
    if bollinger:
        upper_band, lower_band = bollinger_bands(s, window_size=20, n_std=2)
        ax.fill_between(
            s.index,
            lower_band,
            upper_band,
            color="lightgrey",
            alpha=0.7,
            label="Bollinger Bands",
        )

    # Plot main series.
    start = s.index[0] if start is None else pd.to_datetime(start)
    end = s.index[-1] if end is None else pd.to_datetime(end)
    s = s[(s.index >= start) & (s.index <= end)].copy()
    ax.plot(s.index, s.values, **plot_kwargs)

    # Apply custom formatting on x and y-axis if specified.
    format_yaxis(ax, ytickfmt)
    format_xaxis(ax, s, xtickfmt)
    ax.set_xlim(start, end)

    # Add extra info.
    if title is not None:
        ax.set_title(title, fontweight="bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend and (
        plot_kwargs["label"] != "_nolegend_" or mean_line or pct_lines
    ):
        ax.legend()
    plt.tight_layout()


def plot_multiple_timeseries(
    s_list,
    c_list=None,
    ls_list=None,
    start=None,
    end=None,
    ax=None,
    figsize=(8, 6),
    ytickfmt=None,
    xtickfmt=None,
    ylabel=None,
    xlabel=None,
    title=None,
    legend=True,
    **kwargs,
):
    """Plot multiple timeseries on same axis."""
    # Set default plot params and update with specified params.
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if isinstance(s_list, pd.DataFrame):
        s_list = [s_list[col] for col in s_list]

    plot_kwargs = {"alpha": 0.8, "lw": 2}
    plot_kwargs.update(**kwargs)

    default_c_list = [
        "steelblue",
        "firebrick",
        "darkgreen",
        "darkorange",
        "darkorchid",
        "deepskyblue",
        "navy",
        "coral",
    ]
    c_list = default_c_list if c_list is None else c_list

    if ls_list is None:
        ls_list = [plot_kwargs.get("ls", "-")] * len(s_list)
    plot_kwargs.pop("ls", None)

    # Format start date for all series.
    if start is not None:
        old_s_list = s_list.copy()
        s_list = []
        start_date = pd.to_datetime(start)
        s_list = [[s.index >= start_date].copy() for s in old_s_list]

    # Plot data and format axis.
    for s, c, ls in zip(s_list, c_list, ls_list):
        ax.plot(s.index, s.values, c=c, ls=ls, **plot_kwargs, label=s.name)
    format_xaxis(ax, s_list[0], xtickfmt)
    format_yaxis(ax, ytickfmt)

    # Add extra info.
    if title is not None:
        ax.set_title(title, fontweight="bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    plt.tight_layout()


# %%
def format_xaxis(ax, s, xtickfmt):
    """Format dates on x axis if necessary."""
    if not isinstance(s.index[0], Timestamp):
        return
    if xtickfmt is None:
        day_range = (s.index[-1] - s.index[0]).days
        if day_range > 365 * 35:
            loc = mpl.dates.YearLocator(10)
            date_fmt = mpl.dates.DateFormatter("%Y")
        elif day_range > 365 * 20:
            loc = mpl.dates.YearLocator(5)
            date_fmt = mpl.dates.DateFormatter("%Y")
        elif day_range > 365 * 12:
            loc = mpl.dates.YearLocator(4)
            date_fmt = mpl.dates.DateFormatter("%Y")
        elif day_range > 365 * 6:
            loc = mpl.dates.YearLocator(2)
            date_fmt = mpl.dates.DateFormatter("%Y")
        elif day_range > 1000:
            loc = mpl.dates.MonthLocator(bymonth=1, bymonthday=1)
            date_fmt = mpl.dates.DateFormatter("%Y")
        elif day_range > 720:
            loc = mpl.dates.MonthLocator(bymonth=(1, 7), bymonthday=1)
            date_fmt = mpl.dates.DateFormatter("%b-%Y")
        elif day_range > 360:
            loc = mpl.dates.MonthLocator(bymonth=(1, 4, 7, 10), bymonthday=1)
            date_fmt = mpl.dates.DateFormatter("%b-%Y")
        elif day_range > 220:
            loc = mpl.dates.MonthLocator(bymonthday=1, interval=2)
            date_fmt = mpl.dates.DateFormatter("%b-%d")
        elif day_range > 90:
            loc = mpl.dates.MonthLocator(bymonthday=1)
            date_fmt = mpl.dates.DateFormatter("%b-%d")
        elif day_range > 45:
            loc = mpl.dates.MonthLocator(bymonthday=(1, 15))
            date_fmt = mpl.dates.DateFormatter("%b-%d")
        elif day_range > 25:
            loc = mpl.dates.MonthLocator(bymonthday=(1, 10, 20))
            date_fmt = mpl.dates.DateFormatter("%b-%d")
        else:
            loc = mpl.dates.AutoDateLocator(minticks=3, maxticks=9)
            date_fmt = mpl.dates.ConciseDateFormatter(loc)
    elif xtickfmt == "auto":
        loc = mpl.dates.AutoDateLocator(minticks=3, maxticks=9)
        date_fmt = mpl.dates.ConciseDateFormatter(loc)

    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(date_fmt)


def format_yaxis(ax, ytickfmt):
    """Formate y axis to specified formatter"""
    if ytickfmt is not None:
        tick = mpl.ticker.StrMethodFormatter(ytickfmt)
        ax.yaxis.set_major_formatter(tick)


def bollinger_bands(vals, window_size, n_std):
    """Bollinger band calculation."""
    rolling_mean = vals.rolling(window=window_size, min_periods=1).mean()
    rolling_std = vals.rolling(window=window_size, min_periods=1).std()
    upper_band = rolling_mean + (rolling_std * n_std)
    lower_band = rolling_mean - (rolling_std * n_std)
    return upper_band, lower_band


# from lgimapy.bloomberg import bdh
# df = bdh("LULCOAS", "Index", "PX_LAST", start="1/1/2000")
# # plot_timeseries(100 * df_temp["PX_LAST"], xtickfmt=None, start="1/1/2018")
# plot_timeseries(
#     100 * df_temp["PX_LAST"],
#     xtickfmt="auto",
#     bollinger=True,
#     stats=True,
#     start="1/1/2018",
# )
# plt.show()


def rolling_correlation():
    # %%
    from lgimapy.data import Database, Index
    from lgimapy.bloomberg import bdh

    # %%
    db = Database()
    start = "1/1/2007"
    db.load_market_data(start=start, local=True)

    # %%
    ix = db.build_market_index(
        start="1/1/2007", in_stats_index=True, maturity=(10, None)
    )

    # ix.df['mv'] = ix.df['AmountOutstanding'] * ix.df['DirtyPrice']
    #
    # gdf = ix.df.groupby('Issuer')
    #
    # mv_df = pd.DataFrame(gdf['mv'].agg(np.sum))
    # mv_df.sort_values('mv', ascending=False, inplace=True)
    # issuers = mv_df.index[:50]

    oas_df = ix.get_value_history("OAS")
    oas_a = oas_df.values
    window = 30
    corrs = []
    for i in range(window, len(oas_df)):
        a_i = oas_a[i : i + window, :].T
        a_i = a_i[~np.isnan(a_i).any(axis=1)]
        corrs.append(np.mean(np.corrcoef(a_i)))

    corr = (
        pd.Series(corrs, index=ix.dates[window:])
        .rolling(window=60, min_periods=60)
        .mean()
    )

    # %%
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(corr, c="#209CD8")
    tick = mpl.ticker.StrMethodFormatter("{x:.0%}")
    ax.yaxis.set_major_formatter(tick)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Correlation")
    fig.autofmt_xdate()
    savefig("rolling correlation")
    plt.show()

    # %%
    corr.index[0].year
    years = np.arange(corr.index[0].year, corr.index[-1].year + 1)

    corr_df = pd.DataFrame(corr, columns=["corr"])
    corr_df["year"] = [i.year for i in corr.index]
    gdf = pd.DataFrame(corr_df.groupby("year")["corr"].agg(np.mean))

    # %%
    x = np.array([46, 12, 14, 18, 15, 19, 20, 30, 20, 20])
    y = gdf.values[2:-1]
    ols = sms.OLS(y, sms.add_constant(x)).fit()

    # Plot results.
    fig, ax = plt.subplots(1, 1, figsize=[8, 8])
    ax.plot(x, y, "o", ms=6, c="#209CD8")
    ax.plot(x, x * ols.params[1] + ols.params[0], lw=1.5, c="#FDD302")
    ax.set_xlabel("Rolling Correlation")
    ax.set_ylabel("LGIMA Outperformance")
    tick = mpl.ticker.StrMethodFormatter("{x:.1f}%")
    ax.yaxis.set_major_formatter(tick)
    savefig("ro")
    plt.show()

    # %%
    luacoas = 100 * bdh("LULCOAS", "Index", start="1/1/2007", fields="PX_BID")

    # %%
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax2 = ax.twinx()
    bar_dates = [pd.to_datetime(f"6/1/{y}") for y in gdf.index]
    ticks = [pd.to_datetime(f"1/1/{y}") for y in gdf.index]
    ax2.bar(bar_dates, gdf["corr"].values, width=200, color="grey", alpha=0.4)
    tick = mpl.ticker.StrMethodFormatter("{x:.0%}")
    ax2.yaxis.set_major_formatter(tick)
    ax2.grid(False)
    ax.set_ylim((None, 700))
    ax2.set_ylim((0.3, 0.65))
    ax.plot(luacoas.index, luacoas.values, lw=5, c="#209CD8")
    ax.grid(False, axis="x")
    ax.set_xticks(bar_dates)
    for t in ticks:
        ax.axvline(t, ymax=0.01, lw=0.5, c="k")
    ax.set_ylabel("Long Credit Index OAS", color="#209CD8", weight="bold")
    ax2_ylab = "Average Annual Rolling Correlation"
    ax2.set_ylabel(ax2_ylab, color="grey", weight="bold")
    ax.set_xlabel("Date", weight="bold")
    savefig("yearly correlation with index OAS")
    plt.show()

    luacoas.to_csv("long_credit_index.csv")
    index_corr_df = pd.DataFrame(
        {"date": bar_dates, "annual correlation": gdf["corr"].values}
    )
    index_corr_df.to_csv("annual_index_correlation.csv")

    # %%

    disps = []
    for i in range(len(oas_df)):
        a_i = oas_a[i]
        a_i = np.sort(a_i[~np.isnan(a_i)])
        n = len(a_i)
        disps.append(a_i[int(0.75 * n)] - a_i[int(0.25 * n)])

    disp = (
        pd.Series(disps, index=ix.dates).rolling(window=1, min_periods=1).mean()
    )

    # %%
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(corr, c="#209CD8")
    tick = mpl.ticker.StrMethodFormatter("{x:.0%}")
    ax.yaxis.set_major_formatter(tick)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Correlation")
    fig.autofmt_xdate()
    # savefig("rolling correlation")
    plt.show()
    # %%
    years = np.arange(disp.index[0].year, disp.index[-1].year + 1)

    disp_df = pd.DataFrame(disp, columns=["disp"])
    disp_df["year"] = [i.year for i in disp.index]
    disp_gdf = pd.DataFrame(disp_df.groupby("year")["disp"].agg(np.mean))
    disp_gdf

    # %%
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax2 = ax.twinx()
    bar_dates = [pd.to_datetime(f"6/1/{y}") for y in disp_gdf.index]
    ticks = [pd.to_datetime(f"1/1/{y}") for y in disp_gdf.index]
    ax2.bar(
        bar_dates, disp_gdf["disp"].values, width=200, color="grey", alpha=0.4
    )
    fs = 24
    tick = mpl.ticker.StrMethodFormatter("{x:.0f}")
    ax2.yaxis.set_major_formatter(tick)
    ax2.grid(False)
    ax.set_ylim((None, 500))
    # ax2.set_ylim((0.3, 0.65))
    ax.plot(luacoas.index, luacoas.values, lw=5, c="#209CD8")
    ax.grid(False, axis="x")
    ax.set_xticks(bar_dates)
    for t in ticks:
        ax.axvline(t, ymax=0.01, lw=0.5, c="k")
    ax.set_ylabel(
        "Long Credit Index OAS", color="#209CD8", weight="bold", fontsize=fs
    )
    ax2_ylab = "Annual Dispersion (bp)"
    ax2.set_ylabel(ax2_ylab, color="grey", weight="bold", fontsize=fs)
    # savefig("yearly dispersion with index OAS")
    plt.show()

    # %%
    index_disp_df = pd.DataFrame(
        {"date": bar_dates, "annual dispersion": disp_gdf["disp"].values}
    )
    index_disp_df.to_csv("annual_index_dispersion.csv")

    gdf[2:-1]
    lr_df = pd.DataFrame(
        {
            "downgrades": x,
            "correlation": gdf["corr"].values[2:-1],
            "dispersion": disp_gdf["disp"].values[2:-1],
        }
    )

    ols = sms.OLS(alpha, sms.add_constant(lr_df)).fit()
    ols.summary()

    # %%
    disp_gdf
    plt.figure()
    plt.plot(disp_gdf["disp"].values[2:-1], alpha, "o")
    plt.show()

    # %%

    alpha_df = pd.DataFrame(
        {
            "date": disp_gdf["disp"].index[2:-1],
            "dispersion": disp_gdf["disp"].values[2:-1],
            "alpha": alpha,
        }
    )
    alpha_df.to_csv("dispersion_vs_alpha.csv")


def libor_vs_long_credit_yield():
    two_yrs_ago = dt.today() - timedelta(365 * 2)
    lc_yield = bdh("LULCYW", "Index", fields="PX_LAST", start=two_yrs_ago) / 100
    libor = bdh("US0003M", "Index", fields="PX_LAST", start=two_yrs_ago) / 100
    diff = (lc_yield - libor).dropna()

    # %%
    lw = 3
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(lc_yield, c="steelblue", lw=lw, label="Long Credit Yield")
    axes[0].plot(libor, c="firebrick", lw=lw, label="3 Month Libor")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim((None, 0.058))
    tick = mpl.ticker.StrMethodFormatter("{x:.1%}")
    axes[0].yaxis.set_major_formatter(tick)
    axes[1].plot(diff, c="k", lw=lw, label="Long Credit Yield - Libor")
    axes[1].legend()
    axes[1].yaxis.set_major_formatter(tick)
    axes[1].set_xlabel("Date")
    savefig("libor_vs_long_credit_yield")
    plt.show()
    # %%


def highlighted_sector_downgrades():
    n_highlight = 5  # number to highlight
    start_year = 2006

    # Load and clean data.
    df = pd.read_csv("Moody_sector_downgrades.csv", index_col=0)
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    cols = [c for c in df.columns if int(c) >= start_year]
    df = df[cols].fillna(0)
    sorted(list(df.index))

    # Combine sectors that are similar.
    def combine_sectors(df, sectors, new_sector):
        all_sectors = list(df.index)
        other_sectors = [s for s in all_sectors if s not in sectors]
        df_sectors = pd.DataFrame(
            df.loc[sectors, :].sum(), columns=[new_sector]
        ).T
        return df.loc[other_sectors, :].append(df_sectors)

    ute_sectors = ["ELECTRIC", "NATURAL_GAS", "UTILITY_OTHER"]
    energy_sectors = [
        "INDEPENDENT",
        "INTEGRATED",
        "OIL_FIELD_SERVICES",
        "REFINING",
        "MIDSTREAM",
    ]
    df = combine_sectors(df, ute_sectors, "Utilities")
    df = combine_sectors(df, energy_sectors, "Energy")

    # Find most downgraded sectors and aggregate others together.
    least_downgraded = df.sum(axis=1).sort_values().index[:-n_highlight]
    df = combine_sectors(df, least_downgraded, "Other")
    df.index = [ix.replace("_", " ").title() for ix in df.index]

    # %%
    clrs = [colors(c) for c in "bgry"] + ["#002D72", "grey"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    df.T.plot.bar(stacked=True, ax=ax, color=clrs, rot=0, alpha=0.8)
    ax.set_ylabel("Number of Downgrades", fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(False, axis="x")
    savefig("Moody_sector_downgrades")
    plt.show()


def number_of_issuers_timeseries():
    db = Database()
    db.load_market_data(start="1/1/2014", local=True)
    ix_a = db.build_market_index(rating=("AAA", "A-"), in_stats_index=True)
    ix_b = db.build_market_index(rating=("BBB+", "BBB-"), in_stats_index=True)
    n_issuers = np.zeros([len(ix_a.dates), 2])
    for i, day in enumerate(ix_a.dates):
        df_a = ix_a.day(day)
        df_b = ix_b.day(day)
        n_issuers[i, 0] = len(df_a["Issuer"].dropna().unique())
        n_issuers[i, 1] = len(df_b["Issuer"].dropna().unique())

    # %%
    df = pd.DataFrame(n_issuers, columns=["A", "BBB"], index=ix_a.dates)
    for col in df.columns:
        df[col] = df[col] / df[col][0]
    df *= 100

    df = df[df.index < pd.to_datetime("7/1/2019")]

    df.plot(alpha=0.5, color=["steelblue", "firebrick"])


def market_value_timeseries():
    df = pd.read_csv(
        "data.csv", index_col=0, parse_dates=True, infer_datetime_format=True
    )
    df.sort_index(inplace=True)
    for col in df.columns:
        df[col] = [float(cell.replace(",", "")) for cell in df[col]]
    df = df.divide(df.sum(axis=1), axis=0)
    df = df[df.index > pd.to_datetime("6/1/2008")]

    # %%
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.stackplot(
        df.index,
        df["BBB"],
        df["A"],
        df["AA"],
        df["AAA"],
        # labels=,
        colors=[colors(c) for c in "rbyg"],
        alpha=0.8,
    )
    tick = mpl.ticker.StrMethodFormatter("{x:.0%}")
    ax.yaxis.set_major_formatter(tick)
    plt.margins(0, 0)
    ax.grid(False)
    ax.set_title("Market Value Contribution to Index")
    savefig("market_value_contribution_to_index")
    plt.show()


def ebitda_by_sector():
    # %%
    df = pd.read_clipboard(
        names=["sector", "change_mn", "change", "level"], sep="\s{2,}"
    )
    df["sector"] = df["sector"].str.replace(" e", "e")
    df["change"] = [float(val.replace("%", "")) for val in df["change"]]

    # %%
    pal = sns.set_palette("coolwarm")
    fig, ax = plt.subplots(1, 1, figsize=[8, 8])
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["change"], color=coolwarm(df["change"]), alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["sector"])
    ax.grid(False, axis="y")
    tick = mpl.ticker.StrMethodFormatter("{x:.0f}%")
    ax.xaxis.set_major_formatter(tick)
    ax.set_title("EBITDA Growth by Sector (y/y)")
    savefig("EBITDA_by_sector")
    plt.show()
    # %%


def make_patch_spines_invisible(ax):
    """Make axis edges invisible."""
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def set_percentile_limits(series, axes, percentiles=(5, 95)):
    """Set limits such that 5/95 percentils are equal."""
    vals = {}
    for s, ax_side in zip(series, ["left", "right"]):
        vals[ax_side] = {
            "min": np.min(s),
            "max": np.max(s),
            "pct_min": np.percentile(s, percentiles[0]),
            "pct_max": np.percentile(s, percentiles[1]),
            "tgt_min": 0.98 * np.min(s),
            "tgt_max": 1.02 * np.max(s),
        }
    mult_offset = 0
    while (
        vals["left"]["pct_min"] * (1 - mult_offset) > vals["left"]["tgt_min"]
        or vals["right"]["pct_min"] * (1 - mult_offset)
        > vals["right"]["tgt_min"]
        or vals["left"]["pct_max"] * (1 + mult_offset) < vals["left"]["tgt_max"]
        or vals["right"]["pct_max"] * (1 + mult_offset)
        < vals["right"]["tgt_max"]
    ):
        mult_offset += 0.001
    axes[0].set_ylim(
        (1 - mult_offset) * vals["left"]["pct_min"],
        (1 + mult_offset) * vals["left"]["pct_max"],
    )
    axes[1].set_ylim(
        (1 - mult_offset) * vals["right"]["pct_min"],
        (1 + mult_offset) * vals["right"]["pct_max"],
    )


def hist(ax, a, bins=20, weights=None, normed=True, bin_width=1, **kwargs):
    # Find histogram bin locations and sizes.
    res, edges = np.histogram(a, bins=bins, weights=weights, density=normed)
    bw = edges[1] - edges[0]
    pct = res * bw

    # Update plot arguments.
    plot_kwargs = {"color": "steelblue", "alpha": 0.7}
    plot_kwargs.update(**kwargs)
    ax.bar(edges[:-1], pct, width=(bin_width * bw), **plot_kwargs)
    if normed:
        tick = mpl.ticker.StrMethodFormatter("{x:.0%}")
        ax.yaxis.set_major_formatter(tick)
