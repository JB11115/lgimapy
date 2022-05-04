import subprocess

from lgimapy.data import Database


rsync = [
    "rsync",
    "-crv",
    "--exclude",
    "{trace/, synthetic_difference/}",
    f"{Database().local()}/",
    "/mnt/x/Credit Strategy/lgimapy/data/",
]

subprocess.check_call(rsync, shell=False)
