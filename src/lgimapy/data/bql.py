import pandas as pd

from win32com import client
from time import sleep
from lgimapy.utils import Time, root

# %%


def load_bql(fid, header=None):
    # Load excel sheet.
    xl = client.gencache.EnsureDispatch("Excel.Application")
    xl.Visible = False
    xl_wb = xl.Workbooks.Open(fid)
    xl_sh = xl_wb.Worksheets("Sheet1")

    # Find number of rows and columns in excel sheet.
    row_num = 0
    cell_val = ""
    while cell_val != None:
        row_num += 1
        cell_val = xl_sh.Cells(row_num, 1).Value
    n_rows = row_num - 1

    col_num = 0
    cell_val = ""
    while cell_val != None:
        col_num += 1
        cell_val = xl_sh.Cells(1, col_num).Value
    n_cols = col_num - 1

    # Convert excel sheet to pandas DataFrame and close excel.
    sleep(5)
    data = xl_sh.Range(xl_sh.Cells(1, 1), xl_sh.Cells(n_rows, n_cols)).Value
    xl_wb.Close(False)
    if header is None:
        df = pd.DataFrame(list(data[1:]), columns=data[0])
    elif header is False:
        df = pd.DataFrame(list(data))
    else:
        df = pd.DataFrame(list(data), columns=header)
    return df


with Time():
    fid = root("src/lgimapy/data/test.xlsx")
    header = False
    df = load_bql(fid, header)
