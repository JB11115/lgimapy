import numpy as np

# %%


def _weight_array(w, a):
    weights = np.ones(len(a)) if w is None else w
    return np.array(weights)


def mean(a, weights=None):
    return np.average(a, weights=weights)


def var(a, weights=None, ddof=None):
    """
    A measure of weighted variance derived from
    http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
    where the uncertainty of each value is known but
    the importance of each value (weight) differs.
    """
    a = np.array(a)
    w = _weight_array(weights, a)

    w_mean = mean(a, w)
    n = len(a)
    if weights is None:
        if ddof is None:
            bias_correction = 1 / (n - 1)
        else:
            bias_correction = 1 / (n - ddof)
    else:
        w_sum = np.sum(w)
        n_eff = w_sum ** 2 / np.sum(w ** 2)
        bias_correction = n_eff / (n_eff - 1) / w_sum

    return np.sum(w * (a - w_mean) ** 2) * bias_correction


def std(a, weights=None, ddof=None):
    return var(a, weights, ddof=ddof) ** 0.5


def RSD(a, weights=None, ddof=None):
    return std(a, weights, ddof=ddof) / mean(a, weights)


def percentile(a, weights=None, q=50, dropna=True):
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
    dropna: bool, default=True
        If ``True`` remove nan's from either array.

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
    weights = _weight_array(weights, a)

    if dropna:
        mask = ~np.isnan(a) & ~np.isnan(weights)
        a = a[mask]
        weights = weights[mask]

    sorter = np.argsort(a)
    sorted_a = a[sorter]
    sorted_weights = weights[sorter]

    weighted_q = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    weighted_q -= weighted_q[0]
    weighted_q /= weighted_q[-1]
    return np.interp(q, weighted_q, sorted_a)


def median(a, weights=None):
    return percentile(a, weights, q=50)


def IDR(a, weights=None):
    q_10, q_90 = percentile(a, weights, q=[10, 90])
    return q_90 - q_10


def DCV(a, weights=None):
    q_10, q_50, q_90 = percentile(a, weights, q=[10, 50, 90])
    return (q_90 - q_10) / q_50


def IQR(a, weights=None):
    q_25, q_75 = percentile(a, weights, q=[25, 75])
    return q_75 - q_25


def QCV(a, weights=None):
    q_25, q_50, q_75 = percentile(a, weights, q=[25, 50, 75])
    return (q_75 - q_25) / q_50


def MAD(a, weights=None):
    resid = np.abs(np.array(a) - median(a, weights))
    return mean(resid, weights)


def RMAD(a, weights=None):
    return MAD(a, weights) / mean(a, weights)
