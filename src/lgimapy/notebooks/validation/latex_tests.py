import pandas as pd

from lgimapy.latex import Document

# %%
df = pd.DataFrame(
    {"a": [1, 2, 3, 4], "b": [4, 5, 6, 7], "c": [5, 4, 3, 6], "d": [9, 8, 7, 0]}
)
df_2 = df + 29
# %%
doc = Document("subfig_test", path="latex/tests")
edit1, edit2, edit3 = doc.add_subfigures(n=3, widths=[0.25, 0.25, 0.45],)

doc.start_edit(edit1)
doc.add_table(df, adjust=True)
doc.end_edit()

doc.start_edit(edit2)
doc.add_table(df, adjust=True)
doc.end_edit()

doc.start_edit(edit3)
doc.add_caption("plot cap")
doc.add_plot("a")
doc.end_edit()


page = doc.create_page()
page.add_pagebreak()
page.add_plot("a")
page.add_table(df_2)

page_2 = doc.create_page()
page_2.add_pagebreak()
e1, e2 = page_2.add_subfigures(2)
with page_2.start_edit(e1):
    page_2.add_table(df)
with page_2.start_edit(e2):
    page_2.add_table(df * 5, adjust=True)

doc.add_page(page_2)

doc.add_page(page)


doc.save(save_tex=True)
