import re
from functools import partial

import numpy as np
import pandas as pd

from lgimapy.latex import greeks_to_latex
from lgimapy.utils import replace_multiple, to_list


def combine_error_table(vals, errs, prec=3, latex_format=True):
    """
    Combines standard errors and values into one DataFrame.

    Parameters
    ----------
    vals: pd.DataFrame
        Values for table.
    errs: pd.DataFrame
        Errors for respective values.
    prec: int, defualt=3
        Precision for
    latex_format: bool, default=True
        Characters to use to separate values and errors
        {True: ' \pm ', False: ' ± '}.

    Returns
    -------
    df: pd.DataFrame
        DataFrame of values +/- errors.
    """
    if vals.size != errs.size:
        raise ValueError(
            f"vals size {vals.size} does not match errs {err.size}"
        )

    df = pd.DataFrame(index=vals.index)
    pm = "$\pm$" if latex_format else "±"
    for col in vals.columns:
        df[col] = [
            f"{val:.{prec}f} {pm} {abs(error):.{prec}f}"
            for val, error in zip(vals[col], errs[col])
        ]
    return df


def latex_matrix(obj, prec=3, matrix_type="b"):
    """
    Format a numpy matrix for printing in LaTeX.

    Parameters
    ----------
    obj: np.matrix
        Matrix values.
    prec: int, default=3,
        Precision for numbers in matrix.
    matrix_type: {'b', 'p'}, defualt='b'
        Style for matrix in latex
        {'b': bracket, 'p': paranthesis}.

    Returns
    -------
    fout: str
        Formatted matrix in LaTeX syntax.
    """
    # Convert object to 2D matrix.
    try:
        obj = np.matrix(obj)
    except ValueError:
        msg = f"Value of type {type(obj)} cannot be converted to matrix."
        raise TypeError(msg)
    if len(obj.shape) > 2:
        raise ValueError("Matrix can at most display two dimensions")

    robj = obj.round(prec)
    lines = str(robj).replace("[", "").replace("]", "").splitlines()
    fout_list = ["\\begin{" + f"{matrix_type}matrix" + "}"]
    fout_list += ["  " + " & ".join(l.split()) + r"\\" for l in lines]
    fout_list += ["\\end{" + f"{matrix_type}matrix" + "}"]
    fout = "\n".join(fout_list)
    return fout


def latex_array(obj, prec=3):
    """
    Format a list for printing in LaTeX.

    Parameters
    ----------
    obj: list-like
        Values for LaTeX array.
    prec: int, default=3,
        Precision for numbers in array.

    Returns
    -------
    fout: str
        Formatted array in LaTeX syntax.
    """
    fout_list = ["{:.{}f}".format(elem, prec) for elem in list(obj)]
    fout = "[" + ",\ ".join(fout_list) + "]"
    return fout


def latex_color_gradient(
    vals,
    vmin=None,
    vmax=None,
    center=0,
    cmin="steelblue",
    cmid="white",
    cmax="firebrick",
    alphamax=70,
    symmetric=False,
):
    """
    Create gradient colormap for coloring cells in LaTeX.

    Parameters
    ----------
    vals: array-like
        Numeric values to assign color codes to.
    vmin: float, default=None
        Value to anchor the colormap minimum, otherwise it is
        inferred from the data.
    vmax: float, default=None
        Value to anchor the colormap maximum, otherwise it is
        inferred from the data.
    center: float, default=0
        The value at which to center the colormap when
        plotting divergant data.
    cmin: str, default='steelblue'
        Pre-defined color in LaTeX preamble for values below center.
    cmid: str, default="white"
        Pre-defined color in LaTeX preamble for central color.
    cmax: str, default='firebrick'
        Pre-defined color in LaTeX preamble for values above center.
    alphamax: int, default=80
        Maximum blending parameter for cmin and cmax colors.
    symmetric: bool, default=False
        If True, set vmin and vmax to the same distance away
        from the center such that the color scheme diverges
        symettrically in value.

    Returns
    -------
    gradient_colors: List[str].
        List of colors matching gradient for given input values.
    """
    vmin = np.min(vals) if vmin is None else vmin
    vmax = np.max(vals) if vmax is None else vmax
    if symmetric:
        vdist = max(np.abs(vmax - center), np.abs(vmin - center))
        vmin = center - vdist
        vmax = center + vdist

    gradient_colors = []
    for val in vals:
        if np.isnan(val) or val is None:
            gradient_colors.append(f"{{{cmin}!0!{cmid}}}")
        elif val < center:
            val = max(val, vmin)
            alpha = int(alphamax * (center - val) / (center - vmin))
            gradient_colors.append(f"{{{cmin}!{alpha}!{cmid}}}")
        else:
            color = cmax
            val = min(val, vmax)
            alpha = int(alphamax * (val - center) / (vmax - center))
            gradient_colors.append(f"{{{cmax}!{alpha}!{cmid}}}")

    return gradient_colors


def latex_diverging_bars(
    vals, vmax=None, center=0, cmin="steelblue", cmax="firebrick"
):
    """
    Create divergent bar plot corresponding to specified values
    in a LaTeX table.

    Parameters
    ----------
    vals: array-like
        Numeric values to assign color codes to.
    vmax: float, default=None
        Value to anchor the maximum bar size, otherwise it is
        inferred from the data.
    center: float, default=0
        The value at which to center the bars.
    cmin: str, default='steelblue'
        Pre-defined color in LaTeX preamble for values below center.
    cmax: str, default='firebrick'
        Pre-defined color in LaTeX preamble for values above center.

    Returns
    -------
    divergent_bars: List[str].
        List of commands to make divergent bars in LaTeX table.
    """
    vmax = np.max(np.abs(vals)) if vmax is None else vmax

    divergent_bars = []
    for val in vals:
        if val < center:
            val = max(val, -vmax)
            bar_size = int(50 * (val - center) / (vmax - center))
            divergent_bars.append(f"\\divbar{{{bar_size}}}{{{cmin}}}")
        else:
            val = min(val, vmax)
            bar_size = int(50 * (val - center) / (vmax - center))
            divergent_bars.append(f"\\divbar{{{bar_size}}}{{{cmax}}}")

    return divergent_bars


def latex_table(
    df,
    caption=None,
    table_notes=None,
    table_notes_justification="j",
    col_fmt=None,
    prec=3,
    align="center",
    max_width=120,
    adjust=False,
    hide_index=False,
    add_blank_column=False,
    indent_subindexes=False,
    int_vals=False,
    nan_value="-",
    font_size=None,
    midrule_locs=None,
    specialrule_locs=None,
    greeks=False,
    multi_row_header=False,
    loc_style=None,
    row_color=None,
    row_font=None,
    col_style=None,
    gradient_cell_col=None,
    gradient_cell_kws=None,
    div_bar_col=None,
    div_bar_kws=None,
    arrow_col=None,
    arrow_kws=None,
    center_div_bar_header=True,
    alternating_colors=(None, None),
):
    r"""
    Return table with syntax formatted for LaTeX.

    Parameters
    ----------

    df: pd.DataFrame
        DataFrame object to convert for printing in LaTeX.
    caption: str, default=None
        Caption for table.
    table_notes: str, default=None
        Notes to place below table.
    table_notes_justification: ``{'l', 'c'}``, default='l'
        Justification for table notes text. ``'l'`` is for
        justify left, 'c'`` is for center.
    col_fmt: str, default=None
        Column format for latex, (e.g., 'lrc').
    prec: int or Dict[str: str], default=3
        Precision for printing decimals.
        If Dict is given, input is expected to be column name
        keys with respective string formatting as values.

        * ```prec={col1: ':.1f', col2: ':,.0f', col3: ':.0%'}```
    align: ``{'center', 'left', 'right'}``, default='center'
        How to horizontally align table.
    adjust: bool, default=False
        If True, fit table to page width, especially usefull for
        tables with many columns.
    max_width: int, default=120
        Maximum character count for a single cell.
    hide_index: bool, default=False
        If True, hide index from printing in LaTeX.
    add_blank_column: bool, default=False
        If True, append a blank column to the right side of the table.
    indent_subindexes: bool, default=False
        If ``True``, indent index values beginning with ``~``.
    int_vals: bool, default=False
        If True, remove decimal places for number where
        all trailing values are 0 (e.g., 3.00 --> 3).
    nan_value: str, default='-'
        Value to fill NaNs with in LaTeX table.
        For blank, use ```" "```.
    font_size: str, default=None
        Font size for tables.
        ``{'tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize',
        'large', 'Large', 'LARGE', 'huge', 'Huge'}``.
    midrule_locs: str or List[str], default=None
        Index or list of index values to place a midrule line above.
    specialrule_locs: str or List[str], default=None
        Index or list of index values to place a midrule line above.
    greeks: bool, default=False
        If True, convert all instances of greek letters to LaTeX syntax.
    multi_row_header: bool, default=False
        if True, use \\thead to make header column multiple lines.
        Expects ```*``` as separator between lines in header column.
    loc_style: Dict[Tuple[int] or Tuple[Tuple[int]]: str], default=None
        Special styling for specific locations.
        Input is expected as dict with tuple of iloc tuples for
        DataFrame elements to apply the style to as keys and respective
        LaTeX formatting as values.

        * Bold: ```loc_style={((0, 0), (1, 3), (4, 6')): \\textbf'}```:
        * Red Font: ```loc_style={((0, 1), (1, 5)): '\\color{red}'}```
        * Blue Cell: ```loc_style={(5, 6): '\\cellcolor{blue}'}
    row_font: Dict[str: str].
        Special styling for a specific row.
        Input is expected as a dict with row names as keys and
        respective LaTeX formatting as value.
    row_color: Dict[str or Tuple[str]: str].
        Background colors for specific rows.
        Input is expected as a dict with row names as keys and
        respective colors as values.
    col_style: Dict[str: str].
        Special styling for specific columns.
        Input is expexted as dict with column names as keys
        and respective LaTeX formatting as values.

        * Bar plot: ```[col: \pctbar]```
        * Bar plot with font size of 5: ```[col: \pctbar{5}]```
        * Bold: ```[col: \textbf]```
    gradient_cell_col: str or List[str], default=None
        Columns(s) to apply background cell color gradient to based
        on column values.
    gradient_cell_kws: Dict, default=None
        Keyword arguments for columns to apply cell gradients on.
        If all columns have the same arguments simply specify
        ``gradient_cell_kws=`Dict[**kwargs]``.  To customize
        arguments by column specify
        ``gradient_cell_kws=`Dict[col: Dict[**col_kwargs]``
        for each column.
    div_bar_col: str or List[str], default=None
        Columns(s) to apply divergent bar plots to based
        on column values.
    div_bar_kws: Dict, default=None
        Keyword arguments for columns to apply divergent bar plots on.
        If all columns have the same arguments simply specify
        ``div_bar_kws=`Dict[**kwargs]``.  To customize
        arguments by column specify
        ``div_bar_kws=`Dict[col: Dict[**col_kwargs]``
        for each column.
    arrow_col: str or List[str], default=None
        Columns(s) to apply arrows to based
        on column values.
    arrow_kws: Dict, default=None
        Keyword arguments for columns to apply arrows to.
        If all columns have the same arguments simply specify
        ``gradient_cell_kws=`Dict[**kwargs]``.  To customize
        arguments by column specify
        ``gradient_cell_kws=`Dict[col: Dict[**col_kwargs]``
        for each column.
    center_div_bar_header: bool, default=True
        If ``True`` automatically center the header for div bar columns.
    alternating_colors: Tuple(str), default=(None, None).
        Color 1 and Color 2 for for alternating row colors.

    Returns
    -------
    fout: str
        Table formatted with LaTeX syntax.

    Notes
    -----
    Requires the folloiwng LaTeX packages in preamble

        * adjustbox: Fit table to page width.
        * caption: Add caption.
        * makecell: Create multi-row headers.
        * tabu: Build table with fancy formatting options.
        * xcolor: Change font color of table values.
    """
    df = df.copy()

    # Increase width of columns for up to 120 characters.
    pd.set_option("display.max_colwidth", max_width)

    # Format columns.
    n_cols = len(df.columns)
    if col_fmt is not None:
        if len(col_fmt) == 1:
            cfmt = "l" + col_fmt * n_cols
        else:
            cfmt = col_fmt
    else:
        cfmt = "l" + "c" * n_cols

    # Collect gradient values if needed.
    if gradient_cell_col is not None:
        gradient_cell_cols = to_list(gradient_cell_col, dtype=str)
        if gradient_cell_kws is None:
            # Use default kwargs for making gradients for all columns.
            gradient_cells = {
                col: latex_color_gradient(df[col]) for col in gradient_cell_cols
            }
        else:
            if isinstance(list(gradient_cell_kws.values())[0], dict):
                # Use column specific kwargs for each columns' gradients.
                gradient_cells = {}
                for col, kws in gradient_cell_kws.items():
                    kwargs = {"vals": df[col]}
                    kwargs.update(**kws)
                    gradient_cells[col] = latex_color_gradient(**kwargs)
            else:
                # Use single set of kwargs for all column's gradients.
                gradient_cells = {}
                for col in gradient_cell_cols:
                    kwargs = {"vals": df[col]}
                    kwargs.update(**gradient_cell_kws)
                    gradient_cells[col] = latex_color_gradient(**kwargs)
    else:
        gradient_cells = None

    # Collect arrow values if needed.
    if arrow_col is not None:
        arrow_cols = to_list(arrow_col, dtype=str)
        if arrow_kws is None:
            # Use default kwargs for making gradients for all columns.
            arrows = {col: latex_color_gradient(df[col]) for col in arrow_cols}
        else:
            if isinstance(list(arrow_kws.values())[0], dict):
                # Use column specific kwargs for each columns' gradients.
                arrows = {}
                for col, kws in arrow_kws.items():
                    kwargs = {"vals": df[col]}
                    kwargs.update(**kws)
                    arrows[col] = latex_color_gradient(**kwargs)
            else:
                # Use single set of kwargs for all column's gradients.
                arrows = {}
                for col in arrow_cols:
                    kwargs = {"vals": df[col]}
                    kwargs.update(**arrow_kws)
                    arrows[col] = latex_color_gradient(**kwargs)
    else:
        arrows = None

    # Collect diverging bar values if needed.
    if div_bar_col is not None:
        div_bar_cols = to_list(div_bar_col, str)
        if div_bar_kws is None:
            # Use default kwargs for making divergent bars for all columns.
            div_bars = {
                col: latex_diverging_bars(df[col]) for col in div_bar_cols
            }
        else:
            if isinstance(list(div_bar_kws.values())[0], dict):
                # Use column specific kwargs for each columns' div bars.
                div_bars = {
                    col: latex_diverging_bars(df[col], **kwargs)
                    for col, kwargs in div_bar_kws.items()
                }
            else:
                # Use single set of kwargs for all column's div bars.
                div_bars = {
                    col: latex_diverging_bars(df[col], **div_bar_kws)
                    for col in div_bar_cols
                }
    else:
        div_bars = None

    # Round values toi specified precision.
    if isinstance(prec, int):
        df = df.round(prec).copy()
    elif isinstance(prec, dict):
        for col, fmt in prec.items():
            df[col] = ["{:.{}}".format(v, fmt) for v in df[col]]

    # Hide index if required.
    if hide_index:
        df.index = ["{}" for _ in df.index]

    # Add emtpy row if required.
    if add_blank_column:
        df["{ }"] = ["{ }"] * len(df)

    # Replace NaN's.
    df = df.replace(np.nan, nan_value, regex=True)

    # Store table settings.
    fout = "\\begin{table}[H]\n"
    if font_size:
        fout += f"\{font_size}\n"
    if caption:
        fout += f"\caption{{{caption}}}\n"
    else:
        fout += "%\caption{}\n"
    if alternating_colors != (None, None):
        c1, c2 = alternating_colors
        c1 = "" if c1 is None else c1
        c2 = "" if c2 is None else c2
        fout += f"\\rowcolors{{1}}{{{c1}}}{{{c2}}}\n"
    if align == "center":
        fout += "\\centering\n"
    elif align == "left":
        fout += "\\raggedright\n"
    elif align == "right":
        fout += "\\raggedleft\n"
    else:
        raise ValueError("`align` must be 'center', 'left', or 'right'")

    if adjust:
        fout += "\\begin{adjustbox}{width =\\textwidth}\n"

    # Apply special formatting if required.
    if loc_style is not None:
        for locs, fmt in loc_style.items():
            if isinstance(locs[0], tuple):
                # Multiple locations given.
                for loc in locs:
                    df.iloc[loc] = f"{fmt}{{{df.iloc[loc]}}}"
            elif isinstance(locs[0], int):
                # One location given.
                df.iloc[locs] = f"{fmt}{{{df.iloc[locs]}}}"
            else:
                raise ValueError("Incorrect format for `loc_style`")

    # Apply column formatting if required.
    if col_style is not None:
        for col, fmt in col_style.items():
            # Add default font size for percentile bars.
            fmt = fmt + "{10}" if fmt in {"\pctbar", "\\pctbar"} else fmt
            df[col] = [f"{fmt}{{{v}}}" for v in df[col]]

    # Apply column gradient coloring if required.
    if gradient_cells is not None:
        for col, colors in gradient_cells.items():
            df[col] = [
                "\\cellcolor{} {}".format(color, val)
                for color, val in zip(colors, df[col].values)
            ]

    # Apply column arrows coloring if required.
    if arrows is not None:
        for col, colors in arrows.items():
            new_col = []
            cell = (
                "\\cellcolor{{white}} \\color{}{{\\{}arrow}} "
                "\\color{{black}} {}"
            )
            for color, val in zip(colors, df[col].values):
                try:
                    float_val = float(val)
                except ValueError:
                    new_col.append(f"\\cellcolor{{white}} {val}")
                if float_val > 0:
                    new_col.append(cell.format(color, "UP", val))
                elif float_val < 0:
                    new_col.append(cell.format(color, "DOWN", val))
                else:
                    new_col.append(f"\\cellcolor{{white}} {val}")

            df[col] = new_col

    # Create divergent bar plots if required.
    if div_bars is not None:
        # Create bar plots.
        for col, bars in div_bars.items():
            df[col] = [
                "{} {}".format(val, bar)
                for bar, val in zip(bars, df[col].values)
            ]
        # Center column titles.
        if center_div_bar_header:
            col_map = {
                col: f"\\multicolumn{{1}}{{c}}{{{col}}}"
                for col in div_bars.keys()
            }
            df.rename(columns=col_map, inplace=True)

    # Add indent to index column for sub-indexes if requried.
    if indent_subindexes:
        og_ix = list(df.index)
        final_ix = []
        for i, ix in enumerate(og_ix):
            if ix.startswith("~"):
                try:
                    if og_ix[i + 1].startswith("~"):
                        final_ix.append("\hspace{1mm} $\\vdash$ " + ix[1:])
                    else:
                        final_ix.append("\hspace{1mm} $\lefthalfcup$ " + ix[1:])
                except IndexError:
                    # Last item in table, must be halfcup.
                    final_ix.append("\hspace{1mm} $\lefthalfcup$ " + ix[1:])
            else:
                final_ix.append(ix)
        df.index = final_ix
        # Save a map of old to new index values.
        ix_map = {og: final for og, final in zip(og_ix, final_ix)}
    else:
        # Init empty map of old to new index values.
        ix_map = {}

    # Replace improperly formatted values in table.
    repl = {
        "\$": "$",
        "\\textbackslash ": "\\",
        "\\{": "{",
        "\{": "{",
        "\\}": "}",
        "\}": "}",
        "\_": "_",
        ">": "$>$",
        "<": "$<$",
        "\\textasciicircum ": "^",
        "\\begin{tabular}": "\\begin{tabu} to \linewidth",
        "\end{tabular}": "\end{tabu}",
    }

    if int_vals:
        for i in range(1, 10):
            repl[f".{'0' * i} "] = "  " + " " * i
            repl[f".{'0' * i}}}"] = "}" + " " * i

    latex_fmt = partial(replace_multiple, repl_dict=repl)

    # Create multi-row header if specified.
    if multi_row_header:
        header = df.to_latex().split("toprule\n")[1].split("\\\\\n\midrule")[0]
        new_header = ""
        for i, h in enumerate(header.split("&")):
            if i == 0:
                new_header += "{}\n"
            elif i == len(header.split("&")) - 1:
                new_header += "& \\thead{{{}}}".format(h).replace("*", "\\\\")
            else:
                new_header += "& \\thead{{{}}}\n".format(h).replace("*", "\\\\")
        repl[latex_fmt(header)] = latex_fmt(new_header)

    # Build table with LaTeX formatting.
    fout += latex_fmt(df.to_latex(column_format=cfmt))

    # Add midrules if necesary.
    if midrule_locs is not None:
        fmt_str = "\midrule \n"
        midrule_locs = to_list(midrule_locs, str)
        for loc in midrule_locs:
            loc = ix_map.get(loc, loc)
            if hide_index:
                n = 0
                body = fout[fout.find("\\midrule\n") :][9:]
                match = body.split("\\\\\n")[int(loc)]
            else:
                n = 1
                if loc == "header":
                    match = "\n{}"
                else:
                    match = f"\n{loc} "
            ix = fout.find(match)
            if ix != -1:
                fout = fmt_str.join((fout[: ix + n], fout[ix + n :]))

    # Add specialrules if necesary.
    if specialrule_locs is not None:
        fmt_str = "\specialrule{2.5pt}{1pt}{1pt} \n"
        specialrule_locs = to_list(specialrule_locs, str)
        for loc in specialrule_locs:
            loc = ix_map.get(loc, loc)
            if loc == "header":
                match = "\n{}"
                n = 1
            else:
                match = f"\n{loc} "
                n = 1
            ix = fout.find(match)
            if ix != -1:
                fout = fmt_str.join((fout[: ix + n], fout[ix + n :]))

    # Apply row formatting if necessary.
    if row_font is not None:
        for rows, row_fmt in row_font.items():
            fmt_str = f"\\rowfont{{{row_fmt}}} \n"
            rows = to_list(rows, str)
            for loc in rows:
                loc = ix_map.get(loc, loc)
                if loc == "header":
                    match = "\n{}"
                    n = 1
                else:
                    match = f"\n{loc} "
                    n = 1
                ix = fout.find(match)
                if ix != -1:
                    fout = fmt_str.join((fout[: ix + n], fout[ix + n :]))

    # Apply row background coloring if necessary.
    if row_color is not None:
        for rows, color in row_color.items():
            fmt_str = f"\\rowcolor{{{color}}} \n"
            rows = to_list(rows, str)
            for loc in rows:
                loc = ix_map.get(loc, loc)
                if loc == "header":
                    match = "\n{}"
                    n = 1
                else:
                    match = f"\n{loc} "
                    n = 1
                ix = fout.find(match)
                if ix != -1:
                    fout = fmt_str.join((fout[: ix + n], fout[ix + n :]))

    # Replace all greek letters if required.
    if greeks:
        fout = greeks_to_latex(fout)

    # Adujst table to page width if specified.
    if adjust:
        fout += "\end{adjustbox}\n"
    if table_notes is not None:
        justification = {"l": "justify", "j": "justify", "c": "center"}[
            table_notes_justification
        ]
        fout += f"{{\\{justification} {table_notes} \\par}}"
        fout += "\n"
    fout += "\end{table}"
    # One final check to remove nans.
    if nan_value:
        fout = re.sub(" nan\\\\% ", nan_value, fout)
        fout = re.sub(" nan ", nan_value, fout)

    return fout
