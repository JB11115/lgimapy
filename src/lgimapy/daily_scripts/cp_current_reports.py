import shutil

from lgimapy.utils import mkdir, root, X_drive

# %%

def update_current_reports_on_X_drive():
    src_dir = root('reports/current_reports')
    dst_dir = X_drive("Credit Strategy/current_reports")

    mkdir(dst_dir)
    for fid in src_dir.glob('*.pdf'):
        shutil.copyfile(fid, dst_dir / f"{fid.stem}.pdf")

if __name__ == '__main__':
    update_current_reports_on_X_drive()
