# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../src/lgimapy"))
sys.setrecursionlimit(1500)

# -- Project information -----------------------------------------------------

project = "lgimapy"
copyright = "2019, Jason R. Becker"
author = "Jason R. Becker"

# The full version, including alpha/beta/rc tags
release = "0.0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.napoleon", "rinoh.frontend.sphinx"]

# Napoleon settings
napoleon_numpy_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Master toctree document.
master_doc = "index"
source_suffix = ".rst"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Syntax highlighting.
pygments_style = "sphinx"

# Include TODO's in documentation.
todo_include_todos = True
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "nature"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "mainDoc"

# --Options for LaTeX output -------------------------------------------------
# inside conf.py
latex_engine = "pdflatex"
latex_elements = {
    "papersize": "letterpaper",
    "fncychap": "\\usepackage{fncychap}",
    "fontpkg": "\\usepackage{amsmath,amsfonts,amssymb,amsthm}",
    "figure_align": "htbp",
    "pointsize": "10pt",
    "printindex": r"\footnotesize\raggedright\printindex",
    "fontpkg": r"""
        \setmainfont{DejaVu Serif}
        \setsansfont{DejaVu Sans}
        \setmonofont{DejaVu Sans Mono}
        """,
    "preamble": r"""
        %%%%%%%%%%%%%%%%%%%% Meher %%%%%%%%%%%%%%%%%%
        %%%add number to subsubsection 2=subsection, 3=subsubsection
        %%% below subsubsection is not good idea.
        \setcounter{secnumdepth}{3}
        %
        %%%% Table of content upto 2=subsection, 3=subsubsection
        \setcounter{tocdepth}{2}

        \usepackage{amsmath,amsfonts,amssymb,amsthm}
        \usepackage{graphicx}

        %%% reduce spaces for Table of contents, figures and tables
        %%% it is used "\addtocontents{toc}{\vskip -1.2cm}" etc. in the document
        \usepackage[notlot,nottoc,notlof]{}

        \usepackage{color}
        \usepackage{transparent}
        \usepackage{eso-pic}
        \usepackage{lipsum}

        \usepackage{footnotebackref} %%link at the footnote to go to the place of footnote in the text

        %% spacing between line
        \usepackage{setspace}
        %%%%\onehalfspacing
        %%%%\doublespacing
        \singlespacing


        %%%%%%%%%%% datetime
        \usepackage{datetime}

        \newdateformat{MonthYearFormat}{%
            \monthname[\THEMONTH], \THEYEAR}


        %% RO, LE will not work for 'oneside' layout.
        %% Change oneside to twoside in document class
        \usepackage{fancyhdr}
        \pagestyle{fancy}
        \fancyhf{}

        %%% Alternating Header for oneside
        \fancyhead[L]{\ifthenelse{\isodd{\value{page}}}{ \small \nouppercase{\leftmark} }{}}
        \fancyhead[R]{\ifthenelse{\isodd{\value{page}}}{}{ \small \nouppercase{\rightmark} }}

        %%% Alternating Header for two side
        %\fancyhead[RO]{\small \nouppercase{\rightmark}}
        %\fancyhead[LE]{\small \nouppercase{\leftmark}}

        %% for oneside: change footer at right side. If you want to use Left and right then use same as header defined above.
        \fancyfoot[R]{\ifthenelse{\isodd{\value{page}}}{{\tiny Jason R. Becker} }{\href{http://pythondsp.readthedocs.io/en/latest/pythondsp/toc.html}{\tiny PythonDSP}}}

        %%% Alternating Footer for two side
        %\fancyfoot[RO, RE]{\scriptsize Jason R. Becker (jason.becker@lgima.com}

        %%% page number
        \fancyfoot[CO, CE]{\thepage}

        \renewcommand{\headrulewidth}{0.5pt}
        \renewcommand{\footrulewidth}{0.5pt}

        \RequirePackage{tocbibind} %%% comment this to remove page number for following
        \addto\captionsenglish{\renewcommand{\contentsname}{Table of contents}}
        \addto\captionsenglish{\renewcommand{\listfigurename}{List of figures}}
        \addto\captionsenglish{\renewcommand{\listtablename}{List of tables}}
        % \addto\captionsenglish{\renewcommand{\chaptername}{Chapter}}


        %%reduce spacing for itemize
        \usepackage{enumitem}
        \setlist{nosep}

        %%%%%%%%%%% Quote Styles at the top of chapter
        \usepackage{epigraph}
        \setlength{\epigraphwidth}{0.8\columnwidth}
        \newcommand{\chapterquote}[2]{\epigraphhead[60]{\epigraph{\textit{#1}}{\textbf {\textit{--#2}}}}}
        %%%%%%%%%%% Quote for all places except Chapter
        \newcommand{\sectionquote}[2]{{\quote{\textit{``#1''}}{\textbf {\textit{--#2}}}}}
    """,
    "maketitle": r"""
        \pagenumbering{Roman} %%% to avoid page 1 conflict with actual page 1

        \begin{titlepage}
            \centering

            \vspace*{40mm} %%% * is used to give space from top
            \textbf{\Huge {lgimapy}}

            \vspace{0mm}

            \vspace{0mm}
            \Large \textbf{{Jason R.Becker}}

            \small Created on : May, 2019

            \vspace*{0mm}
            \small  Last updated : \MonthYearFormat\today


            %% \vfill adds at the bottom
            \vfill
        \end{titlepage}

        \clearpage
        \pagenumbering{roman}
        \tableofcontents
        \listoffigures
        \listoftables
        \clearpage
        \pagenumbering{arabic}

        """,
    "sphinxsetup": "hmargin={0.7in,0.7in}, vmargin={1in,1in}, \
            verbatimwithframe=true, \
            TitleColor={rgb}{0,0,0}, \
            HeaderFamily=\\rmfamily\\bfseries, \
            InnerLinkColor={rgb}{0,0,1}, \
            OuterLinkColor={rgb}{0,0,1}",
    "tableofcontents": " ",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, f"{project}_documentation.tex", project, author, "report")
]
