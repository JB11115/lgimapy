import os
import subprocess
import sys
from inspect import cleandoc
from pathlib import Path

from lgimapy.latex import latex_array, latex_matrix, latex_table, latex_figure
from lgimapy.utils import mkdir, root

# %%


class Document:
    """
    Class for building a document in LaTeX.

    TODO: add other types such as
        * array
        * matrix
        * figure

    TODO: add LaTeX functions such as
        * appendix
        * header
        * equation
        * bibliography
        * bodies of text

    Parameters
    ----------
    fid: str
        Filename.
    path: Path or str, default=None
        Path to directory for file.

        * None: create in current directory.
        * Path: create at specified directory, create dir if required.
        * str: create `root/latex/{str}` directory, create dir if required.
    """

    def __init__(self, fid, path=None):
        # Format fid and path.
        self.fid = fid[:-4] if fid.endswith(".tex") else fid
        if path is None:
            self.path = ""
        elif isinstance(path, str):
            self.path = root(f"latex/{path}")
            mkdir(self.path)
        else:
            self.path = path
            mkdir(self.path)

        self.body = "\n\n\\begin{document}\n\n"
        self.add_preamble()
        self.bibliography = ""
        self.appendix = ""
        self.background_image = ""

    def add_preamble(
        self,
        font_size=12,
        margin=None,
        margin_unit="cm",
        page_numbers=False,
        ignore_bottom_margin=True,
    ):

        # Format margin.
        if margin is None:
            margin = f"margin=2.5{margin_unit}"
        elif isinstance(margin, (int, float)):
            margin = f"margin={margin}{margin_unit}"
        elif isinstance(margin, dict):
            margin = ", ".join(
                f"{k}={v}{margin_unit}" for k, v, in margin.items()
            )

        # Format other options.
        page_numbers = "" if page_numbers else "\pagenumbering{gobble}"
        ignore_bmargin = (
            "\enlargethispage{100\\baselineskip}"
            if ignore_bottom_margin
            else ""
        )

        self.preamble = cleandoc(
            f"""
            \documentclass[{font_size}pt]{{article}}
            \\usepackage[{margin}]{{geometry}}
            \\usepackage[table]{{xcolor}}
            \\usepackage{{
                amsmath,
                amsthm,
                amssymb,
                adjustbox,
                array,
                background,
                booktabs,
                bm,
                caption,
                dsfont,
                enumitem,
                epigraph,
                fancyhdr,
                float,
                graphicx,
                makecell,
                MnSymbol,
                nicefrac,
                subcaption,
                tabu,
                titlesec,
                transparent,
                xcolor,
            }}
            \PassOptionsToPackage{{hyphens}}{{url}}\\usepackage{{hyperref}}

            \\backgroundsetup{{contents={{}}}}

            \captionsetup{{
                justification=raggedright,
                singlelinecheck=false,
                font={{footnotesize, bf}},
                aboveskip=0pt,
                belowskip=0pt,
                labelformat=empty
            }}
            \setlength{{\\textfloatsep}}{{0.1mm}}
            \setlength{{\\floatsep}}{{1mm}}
            \setlength{{\intextsep}}{{2mm}}

            \\newcommand{{\\N}}{{\mathbb{{N}}}}
            \\newcommand{{\Z}}{{\mathbb{{Z}}}}
            \DeclareMathOperator*{{\\argmax}}{{argmax}}
            \DeclareMathOperator*{{\\argmin}}{{argmin}}
            \setcounter{{MaxMatrixCols}}{{20}}

            \\newlength{{\\barw}}
            \setlength{{\\barw}}{{0.15mm}}
            \def\mybar#1{{%%
            	{{\color{{gray}}\\rule{{#1\\barw}}{{10pt}}}} #1\%}}

            \definecolor{{lightgray}}{{gray}}{{0.9}}

            {page_numbers}
            {ignore_bmargin}
            """
        )

    def add_bibliography(self):
        """TODO"""
        self.bibliography = ""

    def add_appendix(self):
        self.appendix = ""

    def add_section(self, section):
        self.body = "\n\n".join([self.body, f"\section{{{section}}}"])

    def add_paragraph(self, paragraph):
        self.body = "\n\n".join([self.body, paragraph, "\\bigskip"])

    def add_background_image(
        self,
        image,
        scale=1,
        width=None,
        height=None,
        angle=0,
        vshift=0,
        hshift=0,
        unit="cm",
        alpha=None,
    ):
        """
        Add a background image to the document.
        """

        # Format custom dimensions if specified.
        if width is not None and height is not None:
            dims = f"[width={width}{unit}, height={height}{unit}]"
        elif width is not None:
            dims = f"[width={width}{unit}]"
        elif height is not None:
            dims = f"[height={height}{unit}]"
        else:
            dims = ""

        transparent = f"\\transparent{{{alpha}}}" if alpha is not None else ""
        self.background_image = cleandoc(
            f"""
            \\backgroundsetup{{
                scale={scale},
                angle={angle},
                vshift={vshift}{unit},
                hshift={hshift}{unit},
                contents={{
                    {transparent}\includegraphics{dims}{{{image}}}}}\\\\
                }}

            """
        )
        self.body = "\n\n".join([self.body, "\BgThispage"])

    def add_table(self, table_or_df, **kwargs):
        """
        Add table to document using :func:`latex_table`.

        Parameters
        ----------
        table_or_df: str or pd.DataFrame
            If str is provided, it is expected to be a
            pre-formatted table for LaTeX. Otherwise,
            a DataFrame is expected to be input into
            :func:`latex_table`.
        kwargs:
            Keyword arguments for :func:`latex_table`.


        See Also
        --------
        :func:`latex_table`: Return table with syntax formatted for LaTeX.
        """
        if isinstance(table_or_df, str):
            self.body = "\n\n".join((self.body, table_or_df))
        else:
            table = latex_table(table_or_df, **kwargs)
            self.body = "\n\n".join((self.body, table))

    def save(self, save_tex=False):
        """
        Save `.tex` file and compile to `.pdf`.

        Parameters
        ----------
        save_tex: bool, default=False
            If True, save `.tex` as well as `.pdf`.
        """
        # Combine all parts of the document.
        doc = "\n\n".join(
            [
                self.preamble,
                self.background_image,
                self.body,
                "\end{document}",
                self.bibliography,
                self.appendix,
            ]
        )

        # Clean file path and move to proper directory.
        if self.path:
            os.chdir(self.path)
            fid = Path(self.path).joinpath(self.fid)
        else:
            fid = self.fid

        # Save tex file, compile, and delete intermediate files.
        with open(f"{fid}.tex", "w") as f:
            f.write(doc)

        # Documents with transparent images must be compiled twice.
        n = 2 if "\\transparent" in doc else 1
        for _ in range(n):
            subprocess.check_call(
                ["pdflatex", f"{self.fid}.tex", "-interaction=nonstopmode"],
                shell=False,
                stderr=subprocess.STDOUT,
                stdout=subprocess.DEVNULL,
            )
        if not save_tex:
            os.remove(f"{fid}.tex")

        os.remove(f"{fid}.aux")
        os.remove(f"{fid}.log")
        os.remove(f"{fid}.out")
