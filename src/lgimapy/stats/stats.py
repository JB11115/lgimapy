import warnings

import numpy as np
from scipy import stats


def mode(x):
    """Get most frequent occurance in Pandas aggregate by."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            return stats.mode(x)[0][0]
        except IndexError:
            return np.nan
