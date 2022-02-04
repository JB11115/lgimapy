from lgimapy.utils import load_json, root


json_fid = "ratings_changes"
txt_fid = "ratings_changes.txt"

json = load_json(json_fid)


file_contents = []
for cusip, val in json.items():
    for agency in ["Fitch", "SP", "Moody"]:
        try:
            dates = val[agency]["date"]
            ratings = val[agency]["rating"]
        except KeyError:
            continue
        for date, rating in zip(dates, ratings):
            file_contents.append(f"{cusip}|{date}|{rating}|{agency}Rating\n")

with open(root(f"data/{txt_fid}"), "w") as fid:
    fid.writelines(file_contents)
