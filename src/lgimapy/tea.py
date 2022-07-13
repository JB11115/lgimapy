import argparse
from time import sleep

from lgimapy.utils import beep

def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--minutes", help="Minutes before beep")
    parser.set_defaults(minutes='3.5')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    minutes = float(args.minutes)
    seconds = minutes
    sleep(seconds)
    beep()
