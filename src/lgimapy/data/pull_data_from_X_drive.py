import subprocess

from lgimapy.data import Database


rsync = [
    "rsync",
    "-rvu",
    "--exclude",
    "trace/",
    "/mnt/x/Credit Strategy/lgimapy/data/",
    f"{Database().local()}/",
]

subprocess.check_call(rsync, shell=False)
