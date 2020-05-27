from lgimapy.utils import dump_json, load_json, root


def main():
    """
    Prune synthetic difference history file map json files to remove
    any keys that don't have an associated file.
    """
    # Get all saved histoyr files.
    fid_dir = root("data/synthetic_difference/history")
    history_fids = set([f"{fid.stem}.csv" for fid in fid_dir.glob("*.csv")])

    # Search each json file looking for keys to prune.
    json_dir = root("data/synthetic_difference/file_maps")
    json_filenames = [fid.stem for fid in json_dir.glob("*.json")]
    for filename in json_filenames:
        json_fid = f"synthetic_difference/file_maps/{filename}"
        d = load_json(json_fid)
        expected_fids = set(d.values())
        fids_to_prune = expected_fids - history_fids
        pruned_d = {
            key: fid for key, fid in d.items() if fid not in fids_to_prune
        }
        dump_json(pruned_d, json_fid)

if __name__ == "__main__":
    main()
