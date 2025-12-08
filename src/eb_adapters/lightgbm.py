from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import BaseAdapter


# Optional LightGBM dependency guard ------------------------------------------
try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMRegressor  # type: ignore
    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None
    HAS_LIGHTGBM = False


class LightGBMRegressorAdapter(BaseAdapter):
    """
    Adapter for `lightgbm.LGBMRegressor` so it can be used inside
    the Electric Barometer engine.

    Design goals
    ------------
    * Keep LightGBM as an optional dependency.
    * Present a simple sklearn-style interface: fit(X, y, sample_weight=None),
      predict(X) -> 1D float ndarray.
    * Maintain a clean param API so `clone_model()` (or sklearn.clone) can
      reconstruct the adapter from its init kwargs.

    Example
    -------
    >>> models = {
    ...     "lgbm": LightGBMRegressorAdapter(
    ...         n_estimators=200,
    ...         learning_rate=0.05,
    ...         max_depth=-1,
    ...     ),
    ... }
    """

    def __init__(self, **lgbm_params: Any) -> None:
        if not HAS_LIGHTGBM:
            raise ImportError(
                "LightGBMRegressorAdapter requires the optional 'lightgbm' "
                "package. Install it via `pip install lightgbm`."
            )

        # Store init params so the adapter is cloneable.
        self.lgbm_params: Dict[str, Any] = dict(lgbm_params)

        # Underlying LightGBM model instance
        self.model: Optional[LGBMRegressor] = LGBMRegressor(**self.lgbm_params)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LightGBMRegressorAdapter":
        if not HAS_LIGHTGBM or self.model is None:
            raise RuntimeError(
                "LightGBMRegressorAdapter cannot train — LightGBM is not "
                "available or the internal model was not initialized."
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
                "LightGBMRegressorAdapter has not been fit yet. "
                "Call .fit(...) first."
            )

        X_arr = np.asarray(X)
        preds = self.model.predict(X_arr)
        return np.asarray(preds, dtype=float).ravel()

    # ------------------------------------------------------------------
    # Param API for clone_model() compatibility
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Return initialization parameters so that sklearn.clone or our
        `clone_model` helper can reconstruct this adapter.
        """
        return dict(self.lgbm_params)

    def set_params(self, **params: Any) -> "LightGBMRegressorAdapter":
        """
        Update stored parameters and rebuild underlying LGBMRegressor.
        """
        self.lgbm_params.update(params)
        if HAS_LIGHTGBM:
            self.model = LGBMRegressor(**self.lgbm_params)
        return self

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"LightGBMRegressorAdapter(params={self.lgbm_params})"