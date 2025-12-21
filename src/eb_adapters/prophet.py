from __future__ import annotations

"""
Prophet adapter.

This module provides `ProphetAdapter`, a thin wrapper around `prophet.Prophet`
with a scikit-learn-like interface (`fit`, `predict`).

Unlike tabular regressors, Prophet expects a pandas DataFrame with a timestamp
column named `ds` and a target column named `y`. This adapter converts common
array-like inputs into that canonical format for use within the
ElectricBarometer ecosystem.
"""

from typing import Any, Optional

import numpy as np

from .base import BaseAdapter


class ProphetAdapter(BaseAdapter):
    """
    Adapter for `prophet.Prophet`.

    This adapter enables Prophet models to be used inside ElectricBarometer or
    other CWSL-based evaluation workflows by exposing a scikit-learn-like API.

    Parameters
    ----------
    model : Any | None
        Optional pre-configured `prophet.Prophet` instance. If None, a default
        Prophet model is constructed. If the `prophet` package is not installed,
        constructing a default model will raise `ImportError`.

    Notes
    -----
    Input conventions:

    - `X` encodes the time index as either:
      - shape (n_samples,) of datetime-like values, or
      - shape (n_samples, n_features) where the first column is datetime-like
    - `y` is a one-dimensional array-like of numeric targets.

    At fit time, the adapter constructs a DataFrame with columns:

    - `ds`: timestamps parsed from `X`
    - `y`: targets from `y`

    and calls `Prophet.fit(df)`.

    At predict time, the adapter constructs a DataFrame with column `ds` and
    returns the `yhat` predictions as a one-dimensional numpy array.

    Examples
    --------
    >>> from prophet import Prophet
    >>> base = Prophet()
    >>> model = ProphetAdapter(model=base)
    >>> # X contains datetimes, y contains numeric targets
    >>> # model.fit(X, y).predict(X)

    """

    def __init__(self, model: Optional[Any] = None) -> None:
        if model is None:
            try:
                from prophet import Prophet as _Prophet  # type: ignore
            except Exception as e:  # pragma: no cover - import failure path
                raise ImportError(
                    "ProphetAdapter requires the optional 'prophet' package. "
                    "Install it via `pip install prophet`."
                ) from e

            model = _Prophet()

        self.model = model

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,  # ignored
    ) -> "ProphetAdapter":
        """
        Fit the underlying Prophet model.

        Parameters
        ----------
        X : numpy.ndarray
            Time index values. Accepted forms are:
            - shape (n_samples,) of datetime-like values, or
            - shape (n_samples, n_features) where the first column is datetime-like
        y : numpy.ndarray
            Target vector of shape (n_samples,).
        sample_weight : numpy.ndarray | None
            Accepted for API compatibility but ignored by this adapter.

        Returns
        -------
        ProphetAdapter
            The fitted adapter (self), allowing method chaining.

        Notes
        -----
        This method imports pandas locally to avoid making pandas a hard
        dependency at module import time.
        """
        _ = sample_weight  # intentionally unused; kept for API compatibility

        # Local import to avoid making pandas a hard dependency for the module
        import pandas as pd

        X_arr = np.asarray(X)

        # Use the first column if 2D
        if X_arr.ndim > 1:
            X_arr = X_arr[:, 0]

        ds = pd.to_datetime(X_arr)
        y_arr = np.asarray(y, dtype=float)

        df = pd.DataFrame({"ds": ds, "y": y_arr})
        self.model.fit(df)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted Prophet model.

        Parameters
        ----------
        X : numpy.ndarray
            Time index values in the same format accepted by `fit`.

        Returns
        -------
        numpy.ndarray
            Predicted values of shape (n_samples,), taken from Prophet's `yhat`
            output column.

        Raises
        ------
        RuntimeError
            If the Prophet forecast output does not contain the `yhat` column.

        Notes
        -----
        This method imports pandas locally to avoid making pandas a hard
        dependency at module import time.
        """
        import pandas as pd

        X_arr = np.asarray(X)
        if X_arr.ndim > 1:
            X_arr = X_arr[:, 0]

        ds = pd.to_datetime(X_arr)
        df_future = pd.DataFrame({"ds": ds})
        forecast = self.model.predict(df_future)

        if "yhat" not in forecast.columns:
            raise RuntimeError("ProphetAdapter: expected 'yhat' column in forecast output.")

        return np.asarray(forecast["yhat"], dtype=float)