from lgimapy.latex.latex_utils import (
    find_max_locs,
    find_min_locs,
    greeks_to_latex,
    drop_consecutive_duplicates,
)
from lgimapy.latex.table import (
    combine_error_table,
    latex_array,
    latex_matrix,
    latex_table,
)
from lgimapy.latex.figure import latex_figure
from lgimapy.latex.document import Document, Page, merge_pdfs


__all__ = [
    "find_max_locs",
    "find_min_locs",
    "greeks_to_latex",
    "drop_consecutive_duplicates",
    "combine_error_table",
    "latex_array",
    "latex_matrix",
    "latex_table",
    "latex_figure",
    "Document",
    "Page",
    "merge_pdfs",
]
