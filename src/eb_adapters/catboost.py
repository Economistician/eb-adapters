from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import BaseAdapter


# Optional CatBoost dependency guard -----------------------------------------
try:  # pragma: no cover - optional dependency
    from catboost import CatBoostRegressor  # type: ignore
    HAS_CATBOOST = True
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None
    HAS_CATBOOST = False


class CatBoostAdapter(BaseAdapter):
    """
    Adapter for `catboost.CatBoostRegressor`, providing a clean
    scikit-learn-like interface for use inside ElectricBarometer.

    Notes
    -----
    * X and y are treated as normal tabular regression data.
    * `sample_weight` is passed through when provided.
    * Verbose training output is disabled by default.
    * All initialization kwargs are stored so the object is fully cloneable
      via `clone_model()` (sklearn clone or our fallback).

    Example
    -------
    >>> models = {
    ...     "catboost": CatBoostAdapter(
    ...         depth=4,
    ...         learning_rate=0.1,
    ...         iterations=200,
    ...         loss_function="RMSE",
    ...     )
    ... }
    """

    def __init__(self, **params: Any) -> None:
        if not HAS_CATBOOST:
            raise ImportError(
                "CatBoostAdapter requires the optional 'catboost' package. "
                "Install it via `pip install catboost`."
            )

        # Store params for clone() compatibility
        self.params: Dict[str, Any] = dict(params)

        # Default: no spammy training logs
        if "verbose" not in self.params:
            self.params["verbose"] = False

        # Instantiate the underlying CatBoost model
        self.model: Optional[CatBoostRegressor] = CatBoostRegressor(**self.params)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "CatBoostAdapter":
        if not HAS_CATBOOST or self.model is None:
            raise RuntimeError(
                "CatBoostAdapter cannot train — CatBoost is not available or "
                "the internal model was not initialized correctly."
            )

        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)

        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=float)
            self.model.fit(X_arr, y_arr, sample_weight=sw_arr)
        else:
            self.model.fit(X_arr, y_arr)

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "CatBoostAdapter has not been fit yet. Call .fit(...) first."
            )

        X_arr = np.asarray(X)
        preds = self.model.predict(X_arr)
        return np.asarray(preds, dtype=float).ravel()

    # ------------------------------------------------------------------
    # Param API for clone_model() compatibility
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Ensure sklearn.clone or our `clone_model` can reconstruct the adapter.
        """
        return dict(self.params)

    def set_params(self, **params) -> "CatBoostAdapter":
        """
        Update parameters and rebuild underlying CatBoostRegressor.
        """
        self.params.update(params)
        if HAS_CATBOOST:
            self.model = CatBoostRegressor(**self.params)
        return self

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"CatBoostAdapter(params={self.params})"