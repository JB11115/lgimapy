import subprocess

from lgimapy.data import Database


rsync = [
    "rsync",
    "-crv",
    "--exclude",
    "{trace/, synthetic_difference/}",
    "/mnt/x/Credit Strategy/lgimapy/data/",
    f"{Database().local()}/",
]

subprocess.check_call(rsync, shell=False)
