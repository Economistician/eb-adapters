from __future__ import annotations

"""
CatBoost adapter.

This module provides `CatBoostAdapter`, a thin wrapper around
`catboost.CatBoostRegressor` with a scikit-learn-like interface (`fit`, `predict`).

The adapter is designed for use within the ElectricBarometer ecosystem and aims to be:

- Lightweight: minimal behavior beyond input normalization and parameter storage.
- Cloneable: constructor parameters are preserved in `self.params` so cloning utilities
  can reconstruct the instance consistently.
- Optional-dependency safe: importing this module does not require CatBoost, but
  instantiating `CatBoostAdapter` does.

"""

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
    Adapter for `catboost.CatBoostRegressor`.

    This adapter exposes a scikit-learn-like API and stores initialization parameters
    so the instance can be reconstructed by cloning utilities (for example, an internal
    `clone_model()` helper or `sklearn.base.clone`).

    Parameters
    ----------
    **params
        Keyword arguments forwarded to `catboost.CatBoostRegressor`.

    Notes
    -----
    - `X` and `y` are treated as standard tabular regression inputs.
    - If provided, `sample_weight` is passed through to CatBoost training.
    - Training verbosity is disabled by default (`verbose=False`) unless the caller
      supplies `verbose` explicitly.
    - All initialization parameters are stored in `self.params`.

    Examples
    --------
    >>> model = CatBoostAdapter(
    ...     depth=4,
    ...     learning_rate=0.1,
    ...     iterations=200,
    ...     loss_function="RMSE",
    ... )
    >>> # X, y are numpy arrays (or array-like)
    >>> # model.fit(X, y).predict(X)

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
        """
        Fit the underlying `catboost.CatBoostRegressor`.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : numpy.ndarray
            Target vector of shape (n_samples,).
        sample_weight : numpy.ndarray | None
            Optional per-sample weights of shape (n_samples,). If provided, this is
            forwarded to CatBoost training.

        Returns
        -------
        CatBoostAdapter
            The fitted adapter (self), allowing method chaining.

        Raises
        ------
        RuntimeError
            If CatBoost is not available or the internal model is not initialized.

        """
        if not HAS_CATBOOST or self.model is None:
            raise RuntimeError(
                "CatBoostAdapter cannot train: CatBoost is not available or "
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
        """
        Predict using the fitted CatBoost model.

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
            raise RuntimeError("CatBoostAdapter has not been fit yet. Call `fit(...)` first.")

        X_arr = np.asarray(X)
        preds = self.model.predict(X_arr)
        return np.asarray(preds, dtype=float).ravel()

    # ------------------------------------------------------------------
    # Param API for clone_model() compatibility
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
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
        return dict(self.params)

    def set_params(self, **params: Any) -> "CatBoostAdapter":
        """
        Update parameters and rebuild the underlying CatBoost model.

        Parameters
        ----------
        **params
            Keyword parameters to merge into the stored initialization parameters.

        Returns
        -------
        CatBoostAdapter
            The updated adapter instance (self).

        Notes
        -----
        This method updates `self.params` and then re-instantiates
        `catboost.CatBoostRegressor` using the merged parameter set.

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