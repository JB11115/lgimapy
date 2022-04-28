from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd


def tracking_error(
    performance, lookback_months=3, date=None, date_performance=None
):
    """Annualized tracking error."""
    if date is None:
        performance_s = performance.copy()
    else:
        performance_s = pd.concat(
            (
                performance[performance.index < date],
                pd.Series({date: date_performance}),
            )
        )
    te = (
        np.sqrt(252)
        * performance_s.rolling(
            window=(21 * lookback_months), min_periods=21
        ).std()
    )

    if date is None:
        return te
    else:
        return te.iloc[-1]


def normalized_tracking_error(tracking_error, bm_oas):
    """Tracking error normalized for volatility level."""
    return tracking_error / bm_oas
