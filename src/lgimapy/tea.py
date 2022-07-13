"""
Beep when tea is done steeping.
"""

import argparse
from time import sleep

from lgimapy.utils import beep


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--minutes", help="Minutes before beep")
    parser.set_defaults(minutes="3.5")
    return parser.parse_args()


def beep_when_tea_is_done_steeping():
    args = parse_args()
    minutes = float(args.minutes)
    seconds = minutes
    sleep(seconds)
    beep()


if __name__ == "__main__":
    beep_when_tea_is_done_steeping()
