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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .base import BaseAdapter

# Optional statsmodels support ------------------------------------------------
if TYPE_CHECKING:
    # Resolution for reportMissingImports: ignore missing optional library
    import statsmodels.api as _sm  # type: ignore

    # Resolution for reportUndefinedVariable: define flag for type checker
    HAS_STATSMODELS = True
else:
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
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 0),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str | None = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
    ) -> None:
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        self._result: Any | None = None

        # Failure is intentionally delayed until fit(); this flag supports
        # feature detection and optional-dependency behavior.
        if not HAS_STATSMODELS:
            pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> SarimaxAdapter:
        """
        Fit a univariate SARIMAX model on `y`.
        """
        _ = X  # intentionally unused; kept for API compatibility
        _ = sample_weight  # intentionally unused; kept for API compatibility

        if not HAS_STATSMODELS:
            raise ImportError(
                "SarimaxAdapter requires the optional 'statsmodels' package. "
                "Install it via `pip install statsmodels`."
            )

        y_arr = np.asarray(y, dtype=float)

        # Resolution for reportOptionalMemberAccess: cast the module to Any
        sm_mod = cast(Any, _sm)
        model = sm_mod.tsa.statespace.SARIMAX(
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
        """
        if self._result is None:
            raise RuntimeError("SarimaxAdapter has not been fit yet. Call `fit(X, y)` first.")

        n_steps = len(X)
        if n_steps <= 0:
            return np.array([], dtype=float)

        # Resolution for reportAttributeAccessIssue: cast result to Any
        res = cast(Any, self._result)
        forecast = res.forecast(steps=n_steps)
        return np.asarray(forecast, dtype=float)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        _ = deep
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
            "enforce_stationarity": self.enforce_stationarity,
            "enforce_invertibility": self.enforce_invertibility,
        }

    def set_params(self, **params: Any) -> SarimaxAdapter:
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
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 0),
        trend: str | None = None,
    ) -> None:
        super().__init__()
        self.order = order
        self.trend = trend
        self._result: Any | None = None

        if not HAS_STATSMODELS:
            pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> ArimaAdapter:
        """
        Fit a univariate ARIMA model on `y`.
        """
        _ = X
        _ = sample_weight

        if not HAS_STATSMODELS:
            raise ImportError(
                "ArimaAdapter requires the optional 'statsmodels' package. "
                "Install it via `pip install statsmodels`."
            )

        y_arr = np.asarray(y, dtype=float)

        # Resolution for reportOptionalMemberAccess
        sm_mod = cast(Any, _sm)
        model = sm_mod.tsa.ARIMA(
            y_arr,
            order=self.order,
            trend=self.trend,
        )

        self._result = model.fit()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Forecast `len(X)` steps ahead.
        """
        if self._result is None:
            raise RuntimeError("ArimaAdapter has not been fit yet. Call `fit(X, y)` first.")

        n_steps = len(X)
        if n_steps <= 0:
            return np.array([], dtype=float)

        # Resolution for reportAttributeAccessIssue
        res = cast(Any, self._result)
        forecast = res.forecast(steps=n_steps)
        return np.asarray(forecast, dtype=float)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        _ = deep
        return {
            "order": self.order,
            "trend": self.trend,
        }

    def set_params(self, **params: Any) -> ArimaAdapter:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        return f"ArimaAdapter(order={self.order}, trend={self.trend!r})"
