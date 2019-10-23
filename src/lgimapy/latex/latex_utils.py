import re

import numpy as np
import pandas as pd


def greeks_to_latex(text):
    """
    Transform text containing greeks letters with/without
    subscripts to proper LaTeX representation.

    Parameters
    ----------
    text: str
        Text containing greek letter(s) to be formatted to LaTeX.

    Returns
    -------
    text: str
        Text with all greek letter(s) formatted into LaTeX syntax.

    Examples
    --------
    greeks_to_latex('alpha=0.5, Beta=2, \tgamma_=3, \nDelta^4=16')
    >>> $\alpha$= 0.5, $\Beta$= 2, 	gamma_=3,
    >>> $\Delta^{4}$=16

    greeks_to_latex('partial u / partial t - alpha_gold nabla^2 u = 0')
    >>> $\partial$ u / $\partial$ t - $\alpha_{gold}$ $\nabla^{2}$ u = 0
    """

    greeks = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
        "varepsilon",
        "vartheta",
        "varpi",
        "varrho",
        "varsigma",
        "varphi",
        "digamma",
        "partial",
        "eth",
        "hbar",
        "nabla",
        "infty",
        "aleph",
        "beth",
        "gimel",
    ]

    # Capitalize each greek letter and append to greeks list.
    greeks = greeks + [greek.title() for greek in greeks]
    text = " " + text  # add leading space

    for greek in greeks:
        # Case 1: 'greek_123'
        p = re.compile("\s" + greek + "[_]\w+")
        findings = p.findall(text)
        for finding in findings:
            lead_char = re.search("\s", finding).group(0)
            split = finding.split("_")
            new = r"$\{}_{{{}}}$".format(*split).replace(lead_char, "")
            text = text.replace(finding, lead_char + new)

        # Case 2: 'greek123'
        p = re.compile("\s" + greek + "\d+")
        findings = p.findall(text)
        for finding in findings:
            lead_char = re.search("\s", finding).group(0)
            letter = finding[: len(lead_char + greek)]
            num = finding[len(lead_char + greek) :]
            new = r"$\{}_{{{}}}$".format(letter, num).replace(lead_char, "")
            text = text.replace(finding, lead_char + new)

        # Case 3: 'greek^123'
        p = re.compile("\s" + greek + "[\^]\d+")
        findings = p.findall(text)
        for finding in findings:
            lead_char = re.search("\s", finding).group(0)
            split = finding.split("^")
            new = r"$\{}^{{{}}}$".format(*split).replace(lead_char, "")
            text = text.replace(finding, lead_char + new)

        # Case 4: 'greek'
        p = re.compile("\s" + greek + "[^_]")
        findings = p.findall(text)
        for finding in findings:
            lead_char = re.search("\s", finding).group(0)
            letter = re.sub(r"\W+", "", finding)  # remove non-alphanum chars
            new = r"$\{}${}".format(letter, finding[-1]).replace(lead_char, "")
            text = text.replace(finding, lead_char + new + " ")

    return text[1:]  # remove leading space


def find_max_locs(df, axis=0):
    """
    Find iloc location of maximum values in input DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to find maximum locations.
    axis: {0, 1}, default=0
        Axis to find max values over.

    Returns
    -------
    List[Tuple[int]]:
        List of maximum locations.
    """
    num_df = df.apply(pd.to_numeric, errors="coerce").values
    ix = 1 - axis
    slice_max = num_df.max(axis=ix)
    ij_max = []
    for i, i_max in enumerate(slice_max):
        if np.isnan(i_max):  # ignore nan values
            continue
        for j in range(num_df.shape[ix]):
            if axis == 0:
                if num_df[i, j] == i_max:
                    ij_max.append((i, j))
            elif axis == 1:
                if num_df[j, i] == i_max:
                    ij_max.append((j, i))
    return ij_max


def find_min_locs(df, axis=0):
    """
    Find iloc location of minimum values in input DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to find minimum locations.
    axis: {0, 1}, default=0
        Axis to find min values over.

    Returns
    -------
    List[Tuple[int]]:
        List of minimum locations.
    """
    num_df = df.apply(pd.to_numeric, errors="coerce").values
    ix = 1 - axis
    slice_min = num_df.min(axis=ix)
    ij_min = []
    for i, i_min in enumerate(slice_min):
        if np.isnan(i_min):  # ignore nan values
            continue
        for j in range(num_df.shape[ix]):
            if axis == 0:
                if num_df[i, j] == i_min:
                    ij_min.append((i, j))
            elif axis == 1:
                if num_df[j, i] == i_min:
                    ij_min.append((j, i))
    return ij_min
