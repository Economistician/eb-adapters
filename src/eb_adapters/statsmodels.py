from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np

from .base import BaseAdapter

# Optional statsmodels / SARIMAX / ARIMA support
try:  # pragma: no cover - import guard
    import statsmodels.api as _sm  # type: ignore[import]

    HAS_STATSMODELS = True
except Exception:  # pragma: no cover - import guard
    _sm = None
    HAS_STATSMODELS = False


class SarimaxAdapter(BaseAdapter):
    """
    Adapter for statsmodels SARIMAX to make it EB-compatible.

    This wrapper:

      - Ignores X entirely (as is typical for univariate SARIMAX).
      - Fits on y as a simple time series.
      - On predict(X_val), it forecasts len(X_val) steps ahead from the end
        of the training series.

    This is enough to use SARIMAX inside Electric Barometer or other
    CWSL-based workflows without EB needing to know anything about
    statsmodels.

    Parameters
    ----------
    order : tuple, default (1, 0, 0)
        ARIMA (p, d, q) order.

    seasonal_order : tuple, default (0, 0, 0, 0)
        Seasonal (P, D, Q, s) order.

    trend : str or None, default None
        Trend parameter passed to SARIMAX.

    enforce_stationarity, enforce_invertibility : bool, default True
        Passed through to SARIMAX.
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

        # We delay failure until fit(); this flag is for tests / feature detection.
        if not HAS_STATSMODELS:
            pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "SarimaxAdapter":
        if not HAS_STATSMODELS:
            raise ImportError(
                "SarimaxAdapter requires 'statsmodels' to be installed. "
                "Install it with `pip install statsmodels`."
            )

        y_arr = np.asarray(y, dtype=float)

        # Basic univariate SARIMAX on y only.
        model = _sm.tsa.statespace.SARIMAX(
            y_arr,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )

        # Use a small maxiter to keep tests lightweight.
        self._result = model.fit(disp=False, maxiter=50)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._result is None:
            raise RuntimeError(
                "SarimaxAdapter has not been fit yet. Call .fit(X, y) first."
            )

        n_steps = len(X)
        if n_steps <= 0:
            return np.array([], dtype=float)

        # Forecast n_steps ahead from the end of the training sample.
        forecast = self._result.forecast(steps=n_steps)
        return np.asarray(forecast, dtype=float)

    # Minimal param API so sklearn.clone or a simple clone can reconstruct it.
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
            "enforce_stationarity": self.enforce_stationarity,
            "enforce_invertibility": self.enforce_invertibility,
        }

    def set_params(self, **params: Any) -> "SarimaxAdapter":
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
    Adapter for statsmodels ARIMA to make it EB-compatible.

    This wrapper:

      - Ignores X entirely (univariate time-series model).
      - Fits on y as a simple ARIMA(p, d, q) series.
      - On predict(X_val), it forecasts len(X_val) steps ahead from the end
        of the training series.

    That is enough for Electric Barometer to perform holdout or CV selection
    without knowing anything about statsmodels itself.

    Parameters
    ----------
    order : tuple, default (1, 0, 0)
        ARIMA (p, d, q) order.

    trend : str or None, default None
        Trend parameter passed to statsmodels.tsa.ARIMA.
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

        if not HAS_STATSMODELS:
            # Delay failure until fit(); flag is used for feature detection.
            pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ArimaAdapter":
        if not HAS_STATSMODELS:
            raise ImportError(
                "ArimaAdapter requires 'statsmodels' to be installed. "
                "Install it with `pip install statsmodels`."
            )

        y_arr = np.asarray(y, dtype=float)

        # Classic ARIMA interface
        model = _sm.tsa.ARIMA(
            y_arr,
            order=self.order,
            trend=self.trend,
        )

        # Default fit is fine for synthetic / test use
        self._result = model.fit()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._result is None:
            raise RuntimeError(
                "ArimaAdapter has not been fit yet. Call .fit(X, y) first."
            )

        n_steps = len(X)
        if n_steps <= 0:
            return np.array([], dtype=float)

        forecast = self._result.forecast(steps=n_steps)
        return np.asarray(forecast, dtype=float)

    # Minimal param API so cloning helpers can reconstruct it.
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "order": self.order,
            "trend": self.trend,
        }

    def set_params(self, **params: Any) -> "ArimaAdapter":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        return f"ArimaAdapter(order={self.order}, trend={self.trend!r})"