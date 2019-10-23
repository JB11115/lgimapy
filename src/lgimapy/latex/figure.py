from lgimapy.utils import tolist


def latex_figure(fids, caption=None, subcaptions=None, width=0.9, tab=2):
    r"""
    Returns figure(s) syntax formatted for LaTeX.

    Notes
    -----
    Requires \usepackage{graphicx, caption, subcaption} in LaTeX preamble.

    Parameters
    ----------
    fids: str or List[str].
        Filenames for figure(s).
    caption: str, default=None
        Main caption for figure(s).
    subcaptions: List[str], defualt=None
        Subcaptions for each subfigure.
    width: float, default=0.9
        Fraction of page width for figure to fill.
    tab: int, default=2
        Number of spaces to indent each tab.

    """
    fids = tolist(fids, dtype=str)
    n = len(fids)
    subcaps = subcaptions if subcaptions else [None] * n
    t = " " * tab  # tab size

    # Add figures to string to be printed.
    fout = f"\\begin{{figure}}[H]\n{t}\centering\n"
    if n == 1:
        fout += f"{t}\includegraphics[width={width}\\textwidth]{{{fids[0]}}}\n"
        fout += f"{t}\caption{{{caption}}}" if caption else "{t}%\caption{}"
    else:
        for fid, subcap in zip(fids, subcaps):
            fout += f"{t}\\begin{{subfigure}}[b]{{{width/n:.3f}\\textwidth}}"
            fout += f"\n{t}{t}\centering\n{t}{t}"
            fout += f"\includegraphics[width=1\\textwidth]{{{fid}}}\n{t}{t}"
            fout += f"\caption{{{subcap}}}" if subcap else "%\caption{}"
            fout += f"\n{t}\end{{subfigure}}\n{t}"
            if fid != fids[-1]:
                fout += "\hfill\n"
            else:
                fout += f"\caption{{{caption}}}" if caption else "%\caption{}"

    fout += "\n\end{figure}"
    print(fout)
