import os
import subprocess
import sys
from inspect import cleandoc
from pathlib import Path

import lgimapy.vis as vis
from lgimapy.latex import latex_array, latex_matrix, latex_table, latex_figure
from lgimapy.utils import mkdir, root, to_list

# %%


class Document:
    """
    Class for building a document in LaTeX.

    TODO: add other types such as
        * array
        * matrix

    TODO: add LaTeX functions such as
        * appendix
        * header
        * equation
        * bibliography

    TODO: documentation

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

    def __init__(self, fid, path=None, fig_dir=False):
        # Format fid and path.
        self.fid = fid[:-4] if fid.endswith(".tex") else fid
        if path is None:
            self.path = Path(os.getcwd())
        elif isinstance(path, str):
            self.path = root(f"latex/{path}")
            mkdir(self.path)
        else:
            self.path = Path(path)
            mkdir(self.path)

        # Create figure directory if required.
        self._fig_dir_bool = fig_dir
        if fig_dir:
            self.fig_dir = self.path.joinpath("fig/")
            mkdir(self.fig_dir)
        else:
            self.fig_dir = self.path

        # Initialize document components.
        self.body = "\n\n\\begin{document}\n\n"
        self.add_preamble()
        self.bibliography = ""
        self.appendix = ""
        self.background_image = ""

    def add_preamble(
        self,
        packages=None,
        orientation="portrait",
        font_size=12,
        margin=None,
        margin_unit="cm",
        page_numbers=False,
        ignore_bottom_margin=False,
    ):
        """
        Add preamble with custom specifications.

        Parameters
        ----------
        packages: str, List[str], default=None
            Additonal packages to include for rendering document.
            By default no additional packages are included.
        orientation: ``{'landscape', 'portrait'}``, default='portrait'
            Page orientation.
        font_size: int, default=12
            Base font size for general text.
        margin: float or Dict[str: float], default=None
            Page margins. One universal margin can be used
            or individual margins can be specified:

            *``margin=2``
            *``margin={'top': 1, 'bottom': 0.5, 'left': 0.3, 'right': 0.3}``
        margin_unit: str, default='cm'
            Unit for margin size(s).
        page_numbers: bool, default=False
            If True include centered page numbers in the footer.
        ignore_bottom_margin: bool, default=False
            If True completely ignore bottom margin such that extra
            content runs off the page but does not start on a new page.
        """

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
        orient = {"portrait": "", "landscape": ", landscape"}[orientation]
        page_numbers = "" if page_numbers else "\pagenumbering{gobble}"
        ignore_bmargin = (
            "\enlargethispage{100\\baselineskip}"
            if ignore_bottom_margin
            else ""
        )
        default_packages = [
            "amsmath",
            "amsthm",
            "amssymb",
            "adjustbox",
            "array",
            "background",
            "booktabs",
            "bm",
            "caption",
            "colortbl",
            "dsfont",
            "enumitem",
            "epigraph",
            "fancyhdr",
            "float",
            "graphicx",
            "makecell",
            "MnSymbol",
            "nicefrac",
            "ragged2e",
            "subcaption",
            "tabu",
            "titlesec",
            "transparent",
            "xcolor",
        ]
        if packages is not None:
            all_packages = sorted(
                list(set(default_packages + to_list(packages, dtype=str)))
            )
        else:
            all_packages = default_packages
        use_packages = ",\n\t\t".join(all_packages)

        self.preamble = cleandoc(
            f"""
            \documentclass[{font_size}pt]{{article}}
            \\usepackage[{margin}{orient}]{{geometry}}
            \\usepackage[table]{{xcolor}}
            \\usepackage{{
                {use_packages}
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

    def add_figure(self, fids, savefig=False, **kwargs):
        """
        Add table to document using :func:`latex_table`.

        Parameters
        ----------
        fids: fids: str or List[str].
            Filenames for figure(s).
        savefig: bool, default=False
            If True save single Figure to proper directory.
        kwargs:
            Keyword arguments for :func:`latex_figure`.


        See Also
        --------
        :func:`latex_figure`: Return figure with syntax formatted for LaTeX.
        """
        fids = to_list(fids, dtype=str)
        if savefig:
            if len(fids) == 1:
                vis.savefig(self.fig_dir / fids[0])
                vis.close()
            else:
                msg = "Saving can only be done on individual Figures."
                raise ValueError(msg)

        if self._fig_dir_bool:
            fids = [f"fig/{fid}" for fid in fids]

        fig = latex_figure(fids, **kwargs)
        self.body = "\n\n".join((self.body, fig))

    @property
    def doc(self):
        """
        Combine all parts of the document.

        Returns
        -------
        str:
            Entire document contents.
        """
        return "\n\n".join(
            [
                self.preamble,
                self.background_image,
                self.body,
                "\end{document}",
                self.bibliography,
                self.appendix,
            ]
        )

    def write_tex(self):
        """Save .tex file of current document."""
        # Clean file path and move to proper directory.
        if self.path:
            os.chdir(self.path)
            fid = self.path.joinpath(self.fid)
        else:
            fid = self.fid

        # Save tex file.
        with open(f"{fid}.tex", "w") as f:
            f.write(self.doc)

    def save(self, save_tex=False):
        """
        Save `.tex` file and compile to `.pdf`.

        Parameters
        ----------
        save_tex: bool, default=False
            If True, save `.tex` as well as `.pdf`.
        """
        # Clean file path and move to proper directory.
        if self.path:
            os.chdir(self.path)
            fid = self.path.joinpath(self.fid)
        else:
            fid = self.fid

        # Save tex file.
        doc = self.doc
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


# %%
