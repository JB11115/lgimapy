from time import perf_counter

import pandas as pd


class Timer:
    def __init__(self, unit="s", name=""):
        self._name = name
        self._start = perf_counter()
        self._last = perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.run_time("Total")

    def split(self, name=""):
        split = perf_counter() - self._last
        self._last = perf_counter()
        print(f"{self._name} {name}: {split:.3f}")

    def run_time(self, name=""):
        elapsed = (perf_counter() - self._start) / self._unit_div
        self._last = perf_counter()
        print(f"{self._name} {name}: {elapsed:.3f}")


# %%
