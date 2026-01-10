"""
XGBoost adapter.

This module provides `XGBoostRegressorAdapter`, a thin wrapper around
`xgboost.XGBRegressor` with a scikit-learn-like interface (`fit`, `predict`).

The adapter is designed for use within the ElectricBarometer ecosystem and aims to be:

- Lightweight: minimal behavior beyond input normalization and parameter storage.
- Cloneable: constructor parameters are preserved so cloning utilities can
  reconstruct the instance consistently.
- Optional-dependency safe: importing this module does not require XGBoost, but
  instantiating `XGBoostRegressorAdapter` does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .base import BaseAdapter

# Optional XGBoost dependency guard -------------------------------------------
if TYPE_CHECKING:
    # Resolution for reportMissingImports: ignore missing optional library
    from xgboost import XGBRegressor  # type: ignore

    # Resolution for reportUndefinedVariable: define flag for type checker
    HAS_XGBOOST = True
else:
    try:  # pragma: no cover - optional dependency
        from xgboost import XGBRegressor

        HAS_XGBOOST = True
    except Exception:
        XGBRegressor = None  # type: ignore[assignment]
        HAS_XGBOOST = False


class XGBoostRegressorAdapter(BaseAdapter):
    """
    Adapter for `xgboost.XGBRegressor`.

    Parameters
    ----------
    **xgb_params
        Keyword arguments forwarded to `xgboost.XGBRegressor`.

    Notes
    -----
    - `X` and `y` are treated as standard tabular regression inputs.
    - If provided, `sample_weight` is passed through to XGBoost training.
    - All initialization parameters are stored in `self.xgb_params` so
      cloning utilities can reconstruct the instance consistently.
    """

    def __init__(self, **xgb_params: Any) -> None:
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoostRegressorAdapter requires the optional 'xgboost' dependency. "
                'Install it via `pip install "eb-adapters[xgboost]"`.'
            )

        # Store init params so the adapter is cloneable.
        self.xgb_params: dict[str, Any] = dict(xgb_params)

        # Underlying XGBoost model instance
        # Resolution for reportOptionalCall: Cast the constructor to Any to bypass None-check
        ctor = cast(Any, XGBRegressor)
        self.model: Any | None = ctor(**self.xgb_params)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.xgb_params})"

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters (sklearn-style) for clone compatibility."""
        _ = deep
        return dict(self.xgb_params)

    def set_params(self, **params: Any) -> XGBoostRegressorAdapter:
        """Set constructor parameters (sklearn-style) for clone compatibility."""
        self.xgb_params.update(params)
        ctor = cast(Any, XGBRegressor)
        self.model = ctor(**self.xgb_params)
        return self

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> XGBoostRegressorAdapter:
        """
        Fit the underlying `xgboost.XGBRegressor`.
        """
        if self.model is None:
            raise RuntimeError(
                "XGBoostRegressorAdapter internal model is missing. This should not happen."
            )

        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        m = cast(Any, self.model)
        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=float)
            m.fit(X_arr, y_arr, sample_weight=sw_arr)
        else:
            m.fit(X_arr, y_arr)

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted XGBoost model.
        """
        if self.model is None:
            raise RuntimeError(
                "XGBoostRegressorAdapter has not been fit yet. Call `fit(...)` first."
            )

        X_arr = np.asarray(X)
        m = cast(Any, self.model)
        preds = m.predict(X_arr)
        return np.asarray(preds, dtype=float).ravel()
