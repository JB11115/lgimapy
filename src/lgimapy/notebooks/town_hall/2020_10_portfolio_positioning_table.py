from lgimapy.data import Database
from lgimapy.latex import Document

# %%

db = Database()
strat = db.load_portfolio(strategy="US LONG CREDIT")

# %%

df = strat.sector_overweights().rename_axis(None).rename("{}").to_frame()
highlighted_rows = [
    "Midstream",
    "Wirelines/Wireless",
    "Cable Satellite",
    "Utilities",
    "Integrated",
    "Independent",
    "Media",
    "Pharma",
    "Managed Care",
    "Refining",
    "Oil Field Services",
]
row_colors = {row: "babyblue" for row in highlighted_rows}

doc = Document(fid="2020_10_portfolio_positioning", path="reports/town_hall")
doc.add_preamble(table_caption_justification="c")
doc.add_table(
    df,
    col_fmt="rl",
    prec=2,
    caption="Sector Overweights",
    font_size="scriptsize",
    row_color=row_colors,
)
doc.save()
