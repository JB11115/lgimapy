from lgimapy.latex import Document

# %%


def main():
    fid = "global_valuation_pack"
    doc = Document(
        fid, path="reports/global_valuation_pack", fig_dir=True, load_tex=fid,
    )
    doc.save()


# %%

if __name__ == "__main__":
    main()
