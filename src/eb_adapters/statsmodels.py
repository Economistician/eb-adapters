from __future__ import annotations

"""
Statsmodels adapters.

This module provides thin wrappers around univariate statsmodels time-series models so
they can be used inside the ElectricBarometer ecosystem via a scikit-learn-like API.

Adapters in this module intentionally treat `X` as an index placeholder:

- `fit(X, y)` fits the underlying time-series model to `y` only.
- `predict(X)` forecasts `len(X)` steps ahead from the end of the training sample.

This design supports evaluation workflows that expect the `predict(X)` signature while
remaining faithful to how classic univariate ARIMA-family models operate.
"""

from typing import Any, Dict, Optional

import numpy as np

from .base import BaseAdapter

# Optional statsmodels support ------------------------------------------------
try:  # pragma: no cover - import guard
    import statsmodels.api as _sm  # type: ignore[import]

    HAS_STATSMODELS = True
except Exception:  # pragma: no cover - import guard
    _sm = None
    HAS_STATSMODELS = False


class SarimaxAdapter(BaseAdapter):
    """
    Adapter for `statsmodels` SARIMAX.

    This wrapper fits a univariate SARIMAX model on `y` and produces forecasts for
    `len(X)` steps ahead when `predict(X)` is called.

    Parameters
    ----------
    order : tuple[int, int, int], default (1, 0, 0)
        ARIMA (p, d, q) order.
    seasonal_order : tuple[int, int, int, int], default (0, 0, 0, 0)
        Seasonal (P, D, Q, s) order.
    trend : str | None, default None
        Trend specification forwarded to SARIMAX.
    enforce_stationarity : bool, default True
        Whether to enforce stationarity in the SARIMAX model.
    enforce_invertibility : bool, default True
        Whether to enforce invertibility in the SARIMAX model.

    Notes
    -----
    - `X` is ignored during fitting. It is only used at prediction time to
      determine the forecast horizon (`n_steps = len(X)`).
    - This adapter stores initialization parameters via `get_params()` so cloning
      utilities can reconstruct the adapter.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 0),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: Optional[str] = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
    ) -> None:
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        self._result = None

        # Failure is intentionally delayed until fit(); this flag supports
        # feature detection and optional-dependency behavior.
        if not HAS_STATSMODELS:
            pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "SarimaxAdapter":
        """
        Fit a univariate SARIMAX model on `y`.

        Parameters
        ----------
        X : numpy.ndarray
            Ignored. Present for API compatibility.
        y : numpy.ndarray
            Target series of shape (n_samples,).
        sample_weight : numpy.ndarray | None
            Accepted for API compatibility but ignored by this adapter.

        Returns
        -------
        SarimaxAdapter
            The fitted adapter (self), allowing method chaining.

        Raises
        ------
        ImportError
            If `statsmodels` is not installed.
        """
        _ = X  # intentionally unused; kept for API compatibility
        _ = sample_weight  # intentionally unused; kept for API compatibility

        if not HAS_STATSMODELS:
            raise ImportError(
                "SarimaxAdapter requires the optional 'statsmodels' package. "
                "Install it via `pip install statsmodels`."
            )

        y_arr = np.asarray(y, dtype=float)

        model = _sm.tsa.statespace.SARIMAX(
            y_arr,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )

        # Keep fitting lightweight for typical adapter usage and tests.
        self._result = model.fit(disp=False, maxiter=50)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Forecast `len(X)` steps ahead from the end of the training sample.

        Parameters
        ----------
        X : numpy.ndarray
            Array-like placeholder used to determine the forecast horizon.

        Returns
        -------
        numpy.ndarray
            Forecast values of shape (len(X),).

        Raises
        ------
        RuntimeError
            If the adapter has not been fit yet.
        """
        if self._result is None:
            raise RuntimeError("SarimaxAdapter has not been fit yet. Call `fit(X, y)` first.")

        n_steps = len(X)
        if n_steps <= 0:
            return np.array([], dtype=float)

        forecast = self._result.forecast(steps=n_steps)
        return np.asarray(forecast, dtype=float)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Return initialization parameters for cloning utilities.

        Parameters
        ----------
        deep : bool
            Included for scikit-learn compatibility.

        Returns
        -------
        dict[str, Any]
            Initialization parameters that can be passed back to `__init__`.
        """
        _ = deep  # intentionally unused; kept for API compatibility
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
            "enforce_stationarity": self.enforce_stationarity,
            "enforce_invertibility": self.enforce_invertibility,
        }

    def set_params(self, **params: Any) -> "SarimaxAdapter":
        """
        Update adapter parameters.

        Parameters
        ----------
        **params
            Parameters to set as attributes on the adapter instance.

        Returns
        -------
        SarimaxAdapter
            The updated adapter instance (self).
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        return (
            f"SarimaxAdapter(order={self.order}, "
            f"seasonal_order={self.seasonal_order}, trend={self.trend!r})"
        )


class ArimaAdapter(BaseAdapter):
    """
    Adapter for `statsmodels` ARIMA.

    This wrapper fits a univariate ARIMA model on `y` and produces forecasts for
    `len(X)` steps ahead when `predict(X)` is called.

    Parameters
    ----------
    order : tuple[int, int, int], default (1, 0, 0)
        ARIMA (p, d, q) order.
    trend : str | None, default None
        Trend specification forwarded to `statsmodels.tsa.ARIMA`.

    Notes
    -----
    - `X` is ignored during fitting. It is only used at prediction time to
      determine the forecast horizon (`n_steps = len(X)`).
    - This adapter stores initialization parameters via `get_params()` so cloning
      utilities can reconstruct the adapter.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 0),
        trend: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.order = order
        self.trend = trend
        self._result = None

        # Failure is intentionally delayed until fit(); this flag supports
        # feature detection and optional-dependency behavior.
        if not HAS_STATSMODELS:
            pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ArimaAdapter":
        """
        Fit a univariate ARIMA model on `y`.

        Parameters
        ----------
        X : numpy.ndarray
            Ignored. Present for API compatibility.
        y : numpy.ndarray
            Target series of shape (n_samples,).
        sample_weight : numpy.ndarray | None
            Accepted for API compatibility but ignored by this adapter.

        Returns
        -------
        ArimaAdapter
            The fitted adapter (self), allowing method chaining.

        Raises
        ------
        ImportError
            If `statsmodels` is not installed.
        """
        _ = X  # intentionally unused; kept for API compatibility
        _ = sample_weight  # intentionally unused; kept for API compatibility

        if not HAS_STATSMODELS:
            raise ImportError(
                "ArimaAdapter requires the optional 'statsmodels' package. "
                "Install it via `pip install statsmodels`."
            )

        y_arr = np.asarray(y, dtype=float)

        model = _sm.tsa.ARIMA(
            y_arr,
            order=self.order,
            trend=self.trend,
        )

        self._result = model.fit()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Forecast `len(X)` steps ahead from the end of the training sample.

        Parameters
        ----------
        X : numpy.ndarray
            Array-like placeholder used to determine the forecast horizon.

        Returns
        -------
        numpy.ndarray
            Forecast values of shape (len(X),).

        Raises
        ------
        RuntimeError
            If the adapter has not been fit yet.
        """
        if self._result is None:
            raise RuntimeError("ArimaAdapter has not been fit yet. Call `fit(X, y)` first.")

        n_steps = len(X)
        if n_steps <= 0:
            return np.array([], dtype=float)

        forecast = self._result.forecast(steps=n_steps)
        return np.asarray(forecast, dtype=float)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Return initialization parameters for cloning utilities.

        Parameters
        ----------
        deep : bool
            Included for scikit-learn compatibility.

        Returns
        -------
        dict[str, Any]
            Initialization parameters that can be passed back to `__init__`.
        """
        _ = deep  # intentionally unused; kept for API compatibility
        return {
            "order": self.order,
            "trend": self.trend,
        }

    def set_params(self, **params: Any) -> "ArimaAdapter":
        """
        Update adapter parameters.

        Parameters
        ----------
        **params
            Parameters to set as attributes on the adapter instance.

        Returns
        -------
        ArimaAdapter
            The updated adapter instance (self).
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        return f"ArimaAdapter(order={self.order}, trend={self.trend!r})"