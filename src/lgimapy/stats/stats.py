import warnings
from scipy import stats

def mode(x):
    """Get most frequent occurance in Pandas aggregate by."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return stats.mode(x)[0][0]
