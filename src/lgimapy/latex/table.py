from functools import partial

import numpy as np
import pandas as pd

from lgimapy.latex import greeks_to_latex
from lgimapy.utils import replace_multiple, tolist


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

    def build_error_column(val, err, latex_format, prec):
        """Combine single column of values & errors with specifed separator."""
        v_e = []
        for i, (v, e) in enumerate(zip(val, err)):
            if latex_format:
                v_e.append(f"{v:.{prec}f} $\pm$ {abs(e):.{prec}f}")
            else:
                v_e.append(f"{v:.{prec}f} ± {abs(e):.{prec}f}")
        return np.asarray(v_e)

    if vals.size != errs.size:
        raise ValueError(
            f"vals size {vals.size} does not match errs {err.size}"
        )
    df = pd.DataFrame(index=vals.index)

    for col in vals.columns:
        df[col] = build_error_column(vals[col], errs[col], latex_format, prec)
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


# %%
def latex_table(
    df,
    caption=None,
    col_fmt=None,
    prec=3,
    adjust=False,
    hide_index=False,
    add_blank_column=False,
    int_vals=False,
    nan_value="-",
    font_size=None,
    midrule_locs=None,
    specialrule_locs=None,
    greeks=True,
    multi_row_header=False,
    loc_style=None,
    row_color=None,
    row_font=None,
    col_style=None,
):
    r"""
    Return table with syntax formatted for LaTeX.

    Parameters
    ----------

    df: pd.DataFrame
        DataFrame object to convert for printing in LaTeX.
    caption: str, default=None
        Caption for table.
    col_fmt: str, default=None
        Column format for latex, (e.g., 'lrc').
    prec: int or Dict[str: str], default=3
        Precision for printing decimals.
        If Dict is given, input is expected to be column name
        keys with respective string formatting as values.

        * ```prec={col1: ':.1f', col2: ':,.0f', col3: ':.0%'}```

    adjust: bool, default=False
        If True, fit table to page width, especially usefull for
        tables with many columns.
    hide_index: bool, default=False
        If True, hide index from printing in LaTeX.
    add_blank_column: bool, default=False
        If True, append a blank column to the right side of the table.
    int_vals: bool, default=True
        If True, remove decimal places for number where
        all trailing values are 0 (e.g., 3.00 --> 3).
    nan_value: str, default='-'
        Value to fill NaNs with in LaTeX table.
        For blank, use ```" "```.
    font_size: str, default=None
        Font size for tables.
        {'tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize',
        'large', 'Large', 'LARGE', 'huge', 'Huge'}.
    midrule_locs: str or List[str], default=None
        Index or list of index values to place a midrule line above.
    specialrule_locs: str or List[str], default=None
        Index or list of index values to place a midrule line above.
    greeks: bool, default=True
        If True, convert all instances of greek letters to LaTeX syntax.
    multi_row_header: bool, default=False
        if True, use \\thead to make header column multiple lines.
        Expects ```*``` as separator between lines in header column.
    loc_style: Dict[Tuple[Tuple[int]]: str], default=None
        Special styling for specific locations.
        Input is expexted as dict with tuple of iloc tuples for
        DataFrame elements to apply the style to as keys and respective
        LaTeX formatting as values.

        * Bold: ```loc_style={'\textbf': [(0, 0), (1, 3), (4, 6)]}```
        * Red Font: ```loc_style={'\color{red}': [(0, 1), (1, 5)]}```
    row_font: Dict[str: str].
        Special styling for a specific row.
        Input is expected as a dict with row names as keys and
        respective LaTeX formatting as value.
    row_color: Dict[str: str].
        Background colors for specific rows.
        Input is expected as a dict with row names as keys and
        respective colors as values.
    col_style: Dict[str: str].
        Special styling for specific columns.
        Input is expexted as dict with column names as keys
        and respective LaTeX formatting as values.

        * Bar Plot: ```[col: \mybar]```
        * Bold: ```[col: \textbf]```

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

    # Format columns.
    n_cols = len(df.columns)
    if col_fmt is not None:
        if len(col_fmt) == 1:
            cfmt = "l" + col_fmt * n_cols
        else:
            cfmt = col_fmt
    else:
        cfmt = "l" + "c" * n_cols

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
    fout += "\centering\n"
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
            df[col] = [f"{fmt}{{{v}}}" for v in df[col]]

    # Replace improperly formatted values in table.
    repl = {
        "\$": "$",
        "\\textbackslash ": "\\",
        "\\{": "{",
        "\{": "{",
        "\\}": "}",
        "\}": "}",
        "\_": "_",
        "\\textasciicircum ": "^",
        "\\begin{tabular}": "\\begin{tabu} to \linewidth",
        "\end{tabular}": "\end{tabu}",
    }
    latex_fmt = partial(replace_multiple, repl_dict=repl)

    if int_vals:
        int_repl = {f".{'0' * i} ": "  " + " " * i for i in range(1, 10)}
        repl = {**repl, **int_repl}

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
        midrule_locs = tolist(midrule_locs, str)
        for loc in midrule_locs:
            if loc == "header":
                match = "\n{}"
                n = 1
            else:
                match = f"\n{loc}"
                n = 1
            ix = fout.find(match)
            if ix != -1:
                fout = fmt_str.join((fout[: ix + n], fout[ix + n :]))

    # Add specialrules if necesary.
    if specialrule_locs is not None:
        fmt_str = "\specialrule{2.5pt}{1pt}{1pt} \n"
        specialrule_locs = tolist(specialrule_locs, str)
        for loc in specialrule_locs:
            if loc == "header":
                match = "\n{}"
                n = 1
            else:
                match = f"\n{loc}"
                n = 1
            ix = fout.find(match)
            if ix != -1:
                fout = fmt_str.join((fout[: ix + n], fout[ix + n :]))

    # Apply row formatting if necessary.
    if row_font is not None:
        for rows, row_fmt in row_font.items():
            fmt_str = f"\\rowfont{{{row_fmt}}} \n"
            rows = tolist(rows, str)
            for loc in rows:
                if loc == "header":
                    match = "\n{}"
                    n = 1
                else:
                    match = f"\n{loc}"
                    n = 1
                ix = fout.find(match)
                if ix != -1:
                    fout = fmt_str.join((fout[: ix + n], fout[ix + n :]))

    # Apply row background coloring if necessary.
    if row_color is not None:
        for rows, color in row_color.items():
            fmt_str = f"\\rowcolor{{{color}}} \n"
            rows = tolist(rows, str)
            for loc in rows:
                if loc == "header":
                    match = "\n{}"
                    n = 1
                else:
                    match = f"\n{loc}"
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

    fout += "\end{table}"
    return fout
