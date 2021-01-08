import os
import re
import subprocess
import sys
import time
from inspect import cleandoc
from pathlib import Path

import lgimapy.vis as vis
from lgimapy.latex import latex_array, latex_matrix, latex_table, latex_figure
from lgimapy.utils import mkdir, root, to_list

# %%


class OpenEditError(Exception):
    """Raised when trying to save before finishing current edit."""

    pass


class NoEditFoundError(Exception):
    """Raised when no current edit has been found."""

    pass


class KeywordNotFoundError(Exception):
    """Raised when no current edit has been found."""

    pass


class LatexScript:
    """
    Base class for writing LaTeX in Python.

    TODO: add other types such as
        * array
        * matrix

    TODO: add LaTeX functions such as
        * appendix
        * header
        * equation
        * bibliography

    Parameters
    ----------
    fid: str
        Filename.
    path: Path or str, default=None
        Path to directory for file.

        * None: create in current directory.
        * Path: create at specified directory, create dir if required.
        * str: create `root/latex/{str}` directory, create dir if required.
    fig_dir: bool, defautl=False
        If True, create a directory named ``fig`` to store all
        figures in the specified path.
    load_tex: bool, default=False
        If True, load a tex file with the specified filename
        to edit.
    """

    def __init__(self, path, fig_dir, tab_indent):
        """Format path for file and figure directory."""
        if path is None:
            self.path = Path(os.getcwd())
        elif isinstance(path, str):
            self.path = root(path)
        else:
            self.path = Path(path)

        self._fig_dir_bool = fig_dir
        if fig_dir:
            self.fig_dir = self.path.joinpath("fig/")
        else:
            self.fig_dir = self.path

        self._currently_open_edit = False
        self._tab_indent = tab_indent

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.end_edit()

    def _add_to_body(self, script):
        """Add additional script to current :param:`body`."""
        self.body = "\n\n".join([self.body, script])

    def start_edit(self, keyword):
        """
        Begin editing saved document at keyword location.

        Parameters
        ----------
        keyword: str
            Keyword in .tex file to start edit.
            In .tex file use ``%%\begin{<keyword>}`` to indicate
            starting location for edit. Use ``%%\end{<keyword>}``
            to indicate the end of the portion to be replaced.

        See Also
        --------
        :meth:`Document.end_edit`
        """
        # Find editing location and raise error if start or end
        # points are not declared.
        start_phrase = f"%%\\begin{{{keyword}}}"
        end_phrase = f"%%\\end{{{keyword}}}"
        start_ix = self.body.find(start_phrase)
        end_ix = self.body.find(end_phrase)
        if start_ix == -1 or end_ix == -1:
            msg = f"'{keyword}' not found as editing keyword in {self.fid}.tex"
            raise KeywordNotFoundError(msg)

        # Find amount of white space before before
        pre_edit = self.body[:start_ix].rstrip("\n")
        no_white_space = pre_edit.rstrip(" \t")
        n = len(pre_edit) - len(no_white_space)
        prefix = pre_edit[-n:]
        self._prefix = prefix

        # Store keyword and sections of the document surrounding
        # area to be edited to recombine later.
        self._start_phrase = f"{prefix}%%\\begin{{{keyword}}}"
        self._pre_edit = pre_edit[:-n].rstrip()
        self._post_edit = f"{prefix}{self.body[end_ix:]}"
        self.body = ""

        # Make safety flag to avoid saving file before edit is complete.
        self._currently_open_edit = True
        return self

    def end_edit(self):
        """
        Finish previously started edit.

        See Also
        --------
        :meth:`Document.start_edit`
        """
        # Make sure an edit has been started.
        if not self._currently_open_edit:
            msg = "Document.start_edit() has not been started."
            raise NoEditFoundError(msg)

        # Combine pre- and post-edited sections of the .tex document
        # with the newly created edited section.
        body = self._prefix + self.body.lstrip().replace(
            "\n", "\n" + self._prefix
        )
        self.body = "\n".join(
            [self._pre_edit, self._start_phrase, body, self._post_edit]
        )

        # Close edit so document can be safely saved.
        self._currently_open_edit = False

    def add_section(self, section):
        """
        Add a new section to :attr:`Document.body`.

        Parameters
        ----------
        section: str
            Title of new section.
        """
        self._add_to_body(f"\section{{{section}}}")

    def add_subsection(self, section):
        """
        Add a new subsection to :attr:`Document.body`.

        Parameters
        ----------
        section: str
            Title of new section.
        """
        self._add_to_body(f"\subsection{{{section}}}")

    def add_paragraph(self, paragraph):
        """
        Add text as a new paragraph to :attr:`Document.body`.

        Parameters
        ----------
        paragraph: str
            Individual paragraph of text.
        """
        self.body = "\n".join([self.body, "\n", paragraph, "\\bigskip"])

    def add_text(self, text):
        """
        Add text to :attr:`Document.body`.

        Parameters
        ----------
        text: str
            Text to add to :attr:`Document.body`.
        """
        self._add_to_body(text)

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
            self._add_to_body(table_or_df)
        else:
            table = latex_table(table_or_df, **kwargs)
            self._add_to_body(table)

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
        self._add_to_body(fig)

    def add_pagebreak(self):
        """Add a pagebreak."""
        self.add_text("\\pagebreak")

    def add_vskip(self, vskip="1em"):
        """Add verticle skip"""
        self.add_text(f"\\vskip{vskip}")

    def add_vfill(self, vskip="1em"):
        """Add verticle skip"""
        self.add_text(f"\\vfill")

    def set_variable(self, var_name, var_val):
        """
        Set a value for a variable in LaTeX file denoted
        by ``~{VARIABLE_NAME}~`` in LaTeX file.

        Parameters
        ----------
        var_name: str
            Name of variable in LaTeX file.
        var_val: str or scalar
            Value to set variable equal to.
        """
        regex = f"~{{{var_name}}}~"
        self.body = re.sub(regex, str(var_val), self.body)

    @property
    def _epoch(self):
        """int: current epoch, used for unique identifier."""
        epoch = int(time.time() * 1e3)
        time.sleep(0.001)
        return epoch

    def _format_caption(self, caption, indent=None):
        """str: Format caption for a figure."""
        indent = self._tab_indent if indent is None else indent
        t = " " * indent  # tab length
        if caption is None:
            return f"{t}%\\caption{{}}\n"
        else:
            return f"{t}\\caption{{{caption}}}\n"

    def add_caption(self, caption, indent=None):
        """Add a caption to current figure."""
        self._add_to_body(self._format_caption(caption, indent=indent))

    def _format_plot(self, fid, width=0.95, indent=None):
        """str: Format plot fid to insert into a figure."""
        indent = self._tab_indent if indent is None else indent
        t = " " * indent  # tab length
        return f"{t}\\includegraphics[width={width}\\textwidth]{{{fid}}}\n"

    def add_plot(self, fid, width=0.95, indent=None):
        """Add a plot to current figure."""
        self._add_to_body(self._format_plot(fid, width=width, indent=indent))

    def add_subfigures(
        self,
        n=None,
        figures=None,
        widths=None,
        valign="b",
        caption=None,
        caption_on_top=True,
        subcaptions=None,
        subcaptions_on_top=True,
        indent=2,
    ):
        # Determine number of subfigures.
        if n is None:
            if figures is None:
                raise ValueError(
                    "Either a value for `n` or a list of figures is required."
                )
            else:
                figure_list = to_list(figures, dtype=str)
                n = len(figure_list)
        else:
            if figures is not None:
                figure_list = to_list(figures, dtype=str)
                if len(subcaps) != n:
                    raise ValueError(
                        "Length of `figures` must equal number of subfigures."
                    )
            else:
                figure_list = [None] * n

        # Get subfigure widths and ensure they fit constraints.
        if widths is None:
            widths = [round(0.95 / n, 3)] * n
        else:
            if sum(widths) > 1:
                raise ValueError("Sum of widths is greater than 1.")
            if len(widths) != n:
                raise ValueError(
                    "Length of `widths` must equal number of subfigures."
                )

        # Get subcaptions and ensure they fit constraints.
        if subcaptions is None:
            subcaps = [None] * n
        else:
            subcaps = to_list(subcaptions, dtype=str)
            if len(subcaps) != n:
                raise ValueError(
                    "Length of `subcaptions` must equal number of subfigures."
                )

        unique_edit_IDs = []

        # Create subfigure string.
        indent = self._tab_indent if indent is None else indent
        double_indent = 2 * indent
        t = " " * indent  # tab length
        fout = f"\\begin{{figure}}[H]\n"
        fout += f"{t}\\centering\n"
        if caption_on_top:
            fout += self._format_caption(caption, indent=indent)
        for i, (fig, subcap, w) in enumerate(zip(figure_list, subcaps, widths)):
            fout += f"{t}\\begin{{subfigure}}[{valign}]{{{w}\\textwidth}}\n"
            fout += f"{2*t}\\centering\n"
            if subcaptions_on_top:
                fout += self._format_caption(subcap, indent=double_indent)
            if fig is None:
                # make editable location and store keyword.
                edit_id = self._epoch
                unique_edit_IDs.append(edit_id)
                fout += f"{2*t}%%\\begin{{{edit_id}}}\n"
                fout += f"{2*t}%%\\end{{{edit_id}}}\n"
            else:
                # Add plot.
                fout += self._format_plot(fig, indent=double_indent, width=1)
            if not subcaptions_on_top:
                fout += self._format_caption(subcap, indent=double_indent)
            fout += f"{t}\\end{{subfigure}}\n"
            if i != n - 1:
                # Not last figure, so add horizontal space.
                fout += f"{t}\\hfill\n"
        if not caption_on_top:
            fout += self._format_caption(caption, indent=indent)
        fout += "\\end{figure}\n"
        self._add_to_body(fout)

        if len(unique_edit_IDs) == 1:
            return unique_edit_IDs[0]
        else:
            return tuple(unique_edit_IDs)


class Document(LatexScript):
    """
    Class for building a document in LaTeX.

    Parameters
    ----------
    fid: str
        Filename.
    path: Path or str, default=None
        Path to directory for file.

        * None: create in current directory.
        * Path: create at specified directory, create dir if required.
        * str: create `root/latex/{str}` directory, create dir if required.
    fig_dir: bool, defautl=False
        If True, create a directory named ``fig`` to store all
        figures in the specified path.
    load_tex: bool, default=False
        If True, load a tex file with the specified filename
        to edit.
    """

    def __init__(
        self, fid, path=None, fig_dir=False, load_tex=False, tab_indent=2
    ):
        super().__init__(path=path, fig_dir=fig_dir, tab_indent=tab_indent)

        # Format fid and create directories for saving file and figures.
        self.fid = fid[:-4] if fid.endswith(".tex") else fid
        mkdir(self.path)
        mkdir(self.fig_dir)

        # Initialize document components.
        if isinstance(load_tex, str):
            # Set _loaded_file to be False, otherwise a .tex file
            # will be saved to the `path` and not deleted.
            self._loaded_file = False
            self.preamble, self.body = self._load_tex(load_tex)
        elif load_tex:
            self._loaded_file = True
            self.preamble, self.body = self._load_tex()
        else:
            self._loaded_file = False
            self.body = "\n\n\\begin{document}\n\n"
            self.add_preamble()

        # Initialize other document properties.
        self.bibliography = ""
        self.appendix = ""
        self.background_image = ""

    def _load_tex(self, tex_fid=None):
        """
        Load .tex file and separate preamble from body.

        Parameters
        ----------
        tex_fid: str, optional
            Fid from `\tex` directory to load.

        Returns
        -------
        preamble: str
            Preamble of loaded .tex file.
        body: str
            Body of loaded .tex file.
        """
        # Load file.
        if tex_fid is None:
            fid = f"{self.path.joinpath(self.fid)}.tex"
        else:
            fid = root(f"src/lgimapy/tex/{tex_fid}.tex")
        with open(fid, "r") as f:
            doc = f.read()

        # Split into file parts.
        ix_begin_body = doc.find("\\begin{document}")
        ix_end_body = doc.find("\end{document}")
        preamble = doc[:ix_begin_body]
        body = doc[ix_begin_body:ix_end_body]
        return preamble, body

    def add_preamble(
        self,
        packages=None,
        orientation="portrait",
        font_size=12,
        margin=None,
        margin_unit="cm",
        bookmarks=False,
        page_numbers=False,
        ignore_bottom_margin=False,
        table_caption_justification="l",
        bar_size=10,
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
        bookmarks: bool, default=False,
            If ``True`` automatically create bookmarks for chapters,
            sections, and subsections. Open the bookmark toolbar
            by default in PDF readers.
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
        bookmarks = "\\usepackage[open]{{bookmark}}" if bookmarks else ""
        orient = {"portrait": "", "landscape": ", landscape"}[orientation]
        page_numbers = "" if page_numbers else "\\pagenumbering{gobble}"
        ignore_bmargin = (
            "\\enlargethispage{100\\baselineskip}"
            if ignore_bottom_margin
            else ""
        )
        tcj = {
            "l": "raggedright",
            "r": "raggedleft",
            "c": "centering",
            "j": "justified",
        }[table_caption_justification.lower()]

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
            "marvosym",
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
            \\documentclass[{font_size}pt]{{article}}
            \\usepackage[{margin}{orient}]{{geometry}}
            \\usepackage[table]{{xcolor}}
            {bookmarks}
            \\usepackage{{
                {use_packages}
            }}
            \\PassOptionsToPackage{{hyphens}}{{url}}\\usepackage{{hyperref}}

            \\backgroundsetup{{contents={{}}}}

            %% Define default caption settings for figures and tables.
            \\captionsetup{{
                justification={tcj},
                singlelinecheck=false,
                font={{footnotesize, bf}},
                aboveskip=0pt,
                belowskip=0pt,
                labelformat=empty
            }}
            \\captionsetup[subfigure]{{
                justification=raggedright,
                singlelinecheck=false,
                font={{footnotesize, bf}},
                aboveskip=0pt,
                belowskip=0pt,
                labelformat=empty
            }}
            %% Define columns with fixed widths for tables.
            \\newcolumntype{{L}}[1]{{>{{\\raggedright\\arraybackslash}}p{{#1}}}}
            \\newcolumntype{{C}}[1]{{>{{\\centering\\arraybackslash}}p{{#1}}}}
            \\newcolumntype{{R}}[1]{{>{{\\raggedleft\\arraybackslash}}p{{#1}}}}

            %% Change separations distances for better presentation.
            \\setlength{{\\textfloatsep}}{{0.1mm}}
            \\setlength{{\\floatsep}}{{1mm}}
            \\setlength{{\\intextsep}}{{2mm}}

            %% Declare operators for mathematical expressions.
            \\newcommand{{\\N}}{{\\mathbb{{N}}}}
            \\newcommand{{\\Z}}{{\\mathbb{{Z}}}}
            \\DeclareMathOperator*{{\\argmax}}{{argmax}}
            \\DeclareMathOperator*{{\\argmin}}{{argmin}}
            \\setcounter{{MaxMatrixCols}}{{20}}

            %% Define custom colors.
            \\definecolor{{lightgray}}{{gray}}{{0.9}}
            \\definecolor{{steelblue}}{{HTML}}{{0C70D5}}
            \\definecolor{{firebrick}}{{HTML}}{{E85650}}
            \\definecolor{{orchid}}{{HTML}}{{9A2BE6}}
            \\definecolor{{orange}}{{HTML}}{{E69A2B}}
            \\definecolor{{babyblue}}{{HTML}}{{85C7DB}}
            \\definecolor{{salmon}}{{HTML}}{{DB8585}}
            \\definecolor{{eggplant}}{{HTML}}{{815a71}}
            \\definecolor{{mauve}}{{HTML}}{{a78197}}
            \\definecolor{{oldmauve}}{{HTML}}{{4C243B}}

            %% Define function for perctile bars in tables.
            \\newlength{{\\barw}}
            \\setlength{{\\barw}}{{0.15mm}}
            \\def\\pctbar#1#2{{%%
            	{{\\color{{gray}}\\rule{{#2\\barw}}{{#1pt}}}} #2\%}}

            %% Define function for divergent bars in tables.
            \\def\\bar#1#2{{%
            	\\color{{#2}}\\rule{{#1\\barw}}{{{bar_size}pt}}
            }}

            \\def\\divbar#1#2{{%
            	\\ifnum#1<0{{%
            		\\bar{{\\the\\numexpr50+#1}}{{white}}
                    \\bar{{-#1}}{{#2}}
                    \\bar{{50}}{{white}}
                }}
            	\\else{{%
            		\\bar{{50}}{{white}}
                    \\bar{{#1}}{{#2}}
                    \\bar{{\\the\\numexpr50-#1}}{{white}}
                }}
            	\\fi
            }}

            {page_numbers}
            {ignore_bmargin}
            """
        )

    def add_bibliography(self):
        """TODO"""
        self.bibliography = ""

    def add_appendix(self):
        """TODO"""
        self.appendix = ""

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
                    {transparent}\\includegraphics{dims}{{{image}}}}}\\\\
                }}

            """
        )
        self._add_to_body("\BgThispage")

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
                self.preamble.rstrip("\n"),
                self.background_image,
                self.body.rstrip("\n"),
                "\\end{document}",
                self.bibliography,
                self.appendix,
            ]
        )

    def _delete_temp_files(self, fid):
        """
        Delete temporary files created during LaTeX rendering.

        Parameters
        ----------
        fid: str
            Filename with path of saved .pdf file.
        """
        for extension in ["aux", "log", "out"]:
            try:
                os.remove(f"{fid}.{extension}")
            except FileNotFoundError:
                continue

    def create_page(self):
        """
        Create a separate :class:`Page` for the current :class:`Document`
        to be added back later.

        Returns
        -------
        :class:`Page`:
            New separate page.
        """
        return Page(
            path=self.path, fig_dir=self.fig_dir, tab_indent=self._tab_indent
        )

    def add_page(self, page):
        """
        Add created :class:`Page` back into :class:`Document`.

        Parameters
        ----------
        page: :class:`Page`
            Page to be added to document.
        """
        if isinstance(page, Page):
            self._add_to_body(page.body)
        else:
            raise TypeError(f"input of type {type(page)} is not a Page.")

    def save_tex(self):
        """Save .tex file of current document."""
        if self._currently_open_edit:
            raise OpenEditError("Cannot save before current edit is completed.")

        # Clean file path and move to proper directory.
        if self.path:
            os.chdir(self.path)
            fid = self.path.joinpath(self.fid)
        else:
            fid = self.fid

        # Save tex file.
        with open(f"{fid}.tex", "w") as f:
            f.write(self.doc)

    def save(self, fid=None, save_tex=False):
        """
        Save `.tex` file and compile to `.pdf`.

        Parameters
        ----------
        fid: str, optional
            Filename to save PDF.
        save_tex: bool, default=False
            If True, save `.tex` as well as `.pdf`.
        """
        fid = self.fid if fid is None else fid

        if self._currently_open_edit:
            raise OpenEditError("Cannot save before current edit is completed.")

        # Clean file path and move to proper directory.
        if self.path:
            os.chdir(self.path)
            full_fid = self.path.joinpath(fid)
        else:
            full_fid = fid

        # Save tex file.
        doc = self.doc
        with open(f"{full_fid}.tex", "w") as f:
            f.write(doc)

        # Documents with transparent images must be compiled twice.
        n = 2 if "\\transparent" in doc else 1
        for _ in range(n):
            subprocess.check_call(
                ["pdflatex", f"{fid}.tex", "-interaction=nonstopmode"],
                shell=False,
                stderr=subprocess.STDOUT,
                stdout=subprocess.DEVNULL,
            )

        # Clean up temporary files.
        if not save_tex and not self._loaded_file:
            os.remove(f"{full_fid}.tex")
        self._delete_temp_files(full_fid)

    def save_as(self, fid, save_tex=False):
        """
        Save `.tex` to new file and compile to `.pdf`.

        Parameters
        ----------
        fid: str, optional
            Filename to save PDF.
        save_tex: bool, default=False
            If True, save `.tex` as well as `.pdf`.
        """
        self.save(fid, save_tex)


class Page(LatexScript):
    """
    Separate page(s) of a LaTeX Document, which can be edited
    apart from a main :class:`Document` and attached later.
    """

    def __init__(self, path, fig_dir, tab_indent):
        super().__init__(path=path, fig_dir=fig_dir, tab_indent=tab_indent)
        self.body = ""
