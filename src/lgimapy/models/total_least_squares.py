import numpy as np
import matplotlib.pyplot as plt


class TLS:
    """
    Class to implement Total Least Squares (TLS) regression
    on one or more variables.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of all X and Y data to be used
        in TLS regression.
    y_col: str
        Column name for Y variable.

    Attributes
    ----------
    norm_df: pd.DataFrame
        Normalized DataFrame of inputs with missing data removed.
        Data is normalized using full series before remoivng missing
        data as to not lose information.
    y: ndarray
        Normalized y values.
    X: ndarray
        Nomalized X values.
    n: int
        Number of samples.
    beta: ndarray
        Estimated model parameter values.
    X_resid: ndarray
        Estimated errors in input X variables, same shape as `X`.
    y_resid: ndarray
        Estimated error in input y values, same shape as `y`.
    resid: ndarray
        Estimated orthogonal error.
    norm_resid: ndarray
        Normalized estimated orthogonal error.
    frobenius_norm: float
        Frobenius norm of of residuals.
    """

    def __init__(self, df, y_col):
        self._y_col = y_col
        self.norm_df = ((df - df.mean()) / df.std()).dropna()
        self.y = self.norm_df[y_col].values
        self._X_cols = [col for col in df.columns if col != y_col]
        self.n = len(self._X_cols)
        self.X = self.norm_df[self._X_cols].values

    def fit(self):
        """
        Fit TLS model.

        Returns
        -------
        :class:`TLS`:
            Fit TLS model with result attributes.
        """
        # Concatenate data and perform singular value decomposition.
        Z = np.vstack((self.X.T, self.y)).T
        U, S, Vt = np.linalg.svd(Z, full_matrices=True)
        V = Vt.T

        # Solve total least squares solution vector beta.
        n = self.n
        Vxy = V[:n, n:]
        Vyy = V[n:, n:]
        beta = -Vxy / Vyy
        self.beta = np.squeeze(beta)

        # Compute residuals and other statistics.
        Xy_tilde = -Z @ V[:, n:] @ V[:, n:].T
        X_tilde = Xy_tilde[:, :n]
        X_resid = np.squeeze(X_tilde)
        y_resid = np.squeeze(Xy_tilde[:, n:])
        resid = (
            np.sign(y_resid)
            * X_resid
            * y_resid
            / (X_resid ** 2 + y_resid ** 2) ** 0.5
        )
        self.X_resid = X_resid
        self.y_resid = y_resid
        self.resid = resid
        self.norm_resid = (resid - np.mean(resid)) / np.std(resid)
        self.y_tls = np.squeeze((self.X + X_tilde) @ beta)
        self.frobenius_norm = np.linalg.norm(X_tilde, "fro")
        return self

    def plot(self, highlight_last=True, ax=None, figsize=(8, 6)):
        """
        Plot TLS results in normalized space.

        Parameters
        ----------
        highlight_last: bool, default=True
            If true, highlight last data point from input DataFrame.
        ax: matplotlib axis, default=None
            Matplotlib axis to plot figure, if None one is created.
        figsize: list or tuple, default=(6, 6).
            Figure size.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel(self._X_cols[0])
        ax.set_ylabel(self._y_col)

        # Plot raw data.
        ax.plot(self.X, self.y, "o", color="steelblue")
        if highlight_last:
            ax.plot(self.X[-1], self.y[-1], "o", color="firebrick")

        # Plot TLS line of best fit.
        all_x = np.concatenate([self.X.flatten(), self.y / self.beta])
        ix = [np.argmin(all_x), np.argmax(all_x)]
        ax.plot(all_x[ix], all_x[ix] * self.beta, lw=2, c="k", alpha=0.5)
