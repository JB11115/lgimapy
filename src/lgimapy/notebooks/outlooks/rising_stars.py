import pandas as pd

from lgimapy.latex import Document, drop_consecutive_duplicates
from lgimapy.utils import root

# %%
year = 2022
fid = root(f'data/potential_rising_stars_{year}.csv')
repl_val = 999999
df = pd.read_csv(fid).replace('+', repl_val)

new_sector_idx = []
for i, prev_sector in enumerate(df['Sector'].iloc[1:]):
    curr_sector = df['Sector'].iloc[i]
    if curr_sector != prev_sector:
        new_sector_idx.append(i+1)

df['Sector'] = drop_consecutive_duplicates(df['Sector'])
bank_cols = list(df.columns)[4:]

# %%

doc = Document('Rising_Star_Consensus', path='reports/HY')
doc.add_preamble(
    margin={"paperheight": 35, "left": 0.5, "right": 0.5, "top": 2, "bottom": 1},
    header=doc.header(
        left="Rising Star Sell-Side Consensus",
        right=year,
        height=0.5,
    ),
    footer=doc.footer(logo="LG_umbrella", height=-0.5, width=0.08),
)
debt_col = r'Debt (\$B)*Outstanding '
prec = {debt_col: '1f'}
for col in bank_cols:
    prec[col] = '0f'


caption = f"Total sum of potential rising star debt: \\${df[debt_col].sum():.1f}B"
table_notes = '*JP Morgan rising star candidates are through 1H23.'
doc.add_table(
    df,
    caption=caption,
    table_notes=table_notes,
    col_fmt='ll|ll|r|c|c|c|c|c|c',
    multi_row_header=True,
    hide_index=True,
    prec=prec,
    adjust=True,
    midrule_locs=new_sector_idx,
    nan_value=" ",
    gradient_cell_col=bank_cols,
    gradient_cell_kws={'cmax': 'navy'},
    special_replace_rules={str(repl_val): ''}
    )

doc.save()
