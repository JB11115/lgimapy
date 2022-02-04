import pandas as pd
from tqdm import tqdm

from lgimapy.utils import root
from lgimapy.data import Database

# %%


def main():
    """
    Create .txt file of SQL update statements to update
    amount outstanding history in DataMart.
    """
    filename = root("data/SQL_amount_outstanding_updates.txt")
    cusip_dir = root("data/amt_outstanding_history")
    fids = sorted(list(cusip_dir.glob("*.csv")))
    updates = "\n".join(make_statements(fids))
    with open(filename, "w") as fid:
        fid.write(updates)


def make_statements(fids):
    """
    Make generator of single SQL update statemnts
    for each cusip filename.
    """
    statement = (
        "update ia set AmountOutstanding = {} "
        "from LGIMADatamart.DBO.instrumentanalytics ia "
        "inner join LGIMADatamart.DBO.diminstrument"
        " i on ia.instrumentkey = i.instrumentkey where "
        "cusip = '{}' and effectivedatekey{}{}{}{}{};"
    )
    edk = " and effectivedatekey"
    for fid in tqdm(fids):
        cusip = fid.stem
        df = pd.read_csv(fid)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime("%Y%m%d")
        for i, row in enumerate(df.itertuples()):
            date = row[1]
            amt = row[2]
            if i == 0:
                # First amount outstading update.
                yield statement.format(amt, cusip, " < ", date, "", "", "")
            else:
                # Intermediate amount outstanding update.
                yield statement.format(
                    prev_amt, cusip, " > ", prev_date, edk, " < ", date
                )
            yield statement.format(amt, cusip, " = ", date, "", "", "")
            # Store previous values.
            prev_date = date
            prev_amt = amt
        # Last amount outstanding update.
        yield statement.format(amt, cusip, " > ", date, "", "", "")

if __name__ == '__main__':
    main()
