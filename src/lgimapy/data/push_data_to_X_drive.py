import subprocess

from lgimapy.data import Database


rsync = [
    "rsync",
    "-rvu",
    "--exclude",
    "trace/",
    f"{Database().local()}/",
    "/mnt/x/Credit Strategy/lgimapy/data/",
]

subprocess.check_call(rsync, shell=False)
