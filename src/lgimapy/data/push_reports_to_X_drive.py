import subprocess

from lgimapy.utils import root

rsync = [
    "rsync",
    "-crP",
    "--exclude",
    "fig/",
    f"{root('reports')}/",
    "/mnt/x/Credit Strategy/lgimapy/reports/",
]

subprocess.check_call(rsync, shell=False)
