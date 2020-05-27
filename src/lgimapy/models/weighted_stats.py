import numpy as np


def weighted_percentile(a, weights=None, q=50):
    """
    Weighted version of ``np.percentile``.

    Parameters
    ----------
    a: array_like
        Input array or object that can be converted to an array.
    weights: array_like, optional
        An array of weights associated with the values in `a`.
    q: array_like or float, default=50.
        Percentile or sequence of percentiles to compute,
        which must be between 0 and 100 inclusive.
        Defaults to median.

    Returns
    -------
    scalar or ndarray:
        If `q` is a single percentile the result is a scalar
        corresponding to the percentile. If `q` contains multiple
        percentiles the output is an array of corresponding
        values.
    """
    a = np.array(a)
    q = np.array(q) / 100
    if weights is None:
        weights = np.ones(len(a))
    weights = np.array(weights)

    sorter = np.argsort(a)
    sorted_a = a[sorter]
    sorted_weights = weights[sorter]

    weighted_q = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    weighted_q -= weighted_q[0]
    weighted_q /= weighted_q[-1]
    return np.interp(q, weighted_q, sorted_a)
