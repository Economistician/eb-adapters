"""
LightGBM adapter.

This module provides `LightGBMRegressorAdapter`, a thin wrapper around
`lightgbm.LGBMRegressor` with a scikit-learn-like interface (`fit`, `predict`).

The adapter is designed for use within the ElectricBarometer ecosystem and aims to be:

- Lightweight: minimal behavior beyond input normalization and parameter storage.
- Cloneable: constructor parameters are preserved so cloning utilities can
  reconstruct the instance consistently.
- Optional-dependency safe: importing this module does not require LightGBM, but
  instantiating `LightGBMRegressorAdapter` does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .base import BaseAdapter

# Optional LightGBM dependency guard ------------------------------------------
if TYPE_CHECKING:
    # Resolution for reportMissingImports: ignore missing optional library
    from lightgbm import LGBMRegressor  # type: ignore

    # Resolution for reportUndefinedVariable: define flag for type checker
    HAS_LIGHTGBM = True
else:
    try:  # pragma: no cover - optional dependency
        from lightgbm import LGBMRegressor

        HAS_LIGHTGBM = True
    except Exception:  # pragma: no cover - optional dependency
        LGBMRegressor = None
        HAS_LIGHTGBM = False


class LightGBMRegressorAdapter(BaseAdapter):
    """
    Adapter for `lightgbm.LGBMRegressor`.

    This adapter exposes a scikit-learn-like API and stores initialization parameters
    so the instance can be reconstructed by cloning utilities (for example, an internal
    `clone_model()` helper or `sklearn.base.clone`).

    Parameters
    ----------
    **lgbm_params
        Keyword arguments forwarded to `lightgbm.LGBMRegressor`.

    Notes
    -----
    - `X` and `y` are treated as standard tabular regression inputs.
    - If provided, `sample_weight` is passed through to LightGBM training.
    - All initialization parameters are stored in `self.lgbm_params`.

    Examples
    --------
    >>> model = LightGBMRegressorAdapter(
    ...     n_estimators=200,
    ...     learning_rate=0.05,
    ...     max_depth=-1,
    ... )
    >>> # X, y are numpy arrays (or array-like)
    >>> # model.fit(X, y).predict(X)
    """

    def __init__(self, **lgbm_params: Any) -> None:
        if not HAS_LIGHTGBM:
            raise ImportError(
                "LightGBMRegressorAdapter requires the optional 'lightgbm' package. "
                "Install it via `pip install lightgbm`."
            )

        # Store init params so the adapter is cloneable.
        self.lgbm_params: dict[str, Any] = dict(lgbm_params)

        # Underlying LightGBM model instance
        # Resolution for reportOptionalCall: Cast the constructor to Any to bypass None-check
        ctor = cast(Any, LGBMRegressor)
        self.model: Any | None = ctor(**self.lgbm_params)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> LightGBMRegressorAdapter:
        """
        Fit the underlying `lightgbm.LGBMRegressor`.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : numpy.ndarray
            Target vector of shape (n_samples,).
        sample_weight : numpy.ndarray | None
            Optional per-sample weights of shape (n_samples,). If provided, this is
            forwarded to LightGBM training.

        Returns
        -------
        LightGBMRegressorAdapter
            The fitted adapter (self), allowing method chaining.

        Raises
        ------
        RuntimeError
            If LightGBM is not available or the internal model is not initialized.
        """
        if not HAS_LIGHTGBM or self.model is None:
            raise RuntimeError(
                "LightGBMRegressorAdapter cannot train: LightGBM is not available "
                "or the internal model was not initialized."
            )

        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)

        # Resolution for reportOptionalCall: Use cast to ensure callability
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
        Predict using the fitted LightGBM model.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predicted values of shape (n_samples,).

        Raises
        ------
        RuntimeError
            If the adapter has not been fit yet.
        """
        if self.model is None:
            raise RuntimeError(
                "LightGBMRegressorAdapter has not been fit yet. Call `fit(...)` first."
            )

        X_arr = np.asarray(X)
        # Resolution for reportOptionalCall: Use cast to ensure callability
        m = cast(Any, self.model)
        preds = m.predict(X_arr)
        return np.asarray(preds, dtype=float).ravel()

    # ------------------------------------------------------------------
    # Param API for clone_model() compatibility
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Return initialization parameters for cloning utilities.

        Parameters
        ----------
        deep : bool
            Included for scikit-learn compatibility. This adapter does not expose
            nested estimators, so the value does not change the output.

        Returns
        -------
        dict[str, Any]
            A shallow copy of the stored initialization parameters.
        """
        _ = deep  # intentionally unused; kept for API compatibility
        return dict(self.lgbm_params)

    def set_params(self, **params: Any) -> LightGBMRegressorAdapter:
        """
        Update parameters and rebuild the underlying LightGBM model.

        Parameters
        ----------
        **params
            Keyword parameters to merge into the stored initialization parameters.

        Returns
        -------
        LightGBMRegressorAdapter
            The updated adapter instance (self).

        Notes
        -----
        This method updates `self.lgbm_params` and then re-instantiates
        `lightgbm.LGBMRegressor` using the merged parameter set.
        """
        self.lgbm_params.update(params)
        if HAS_LIGHTGBM:
            ctor = cast(Any, LGBMRegressor)
            self.model = ctor(**self.lgbm_params)
        return self

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"LightGBMRegressorAdapter(params={self.lgbm_params})"
