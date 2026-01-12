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

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .base import BaseAdapter

# Optional CatBoost dependency guard -----------------------------------------
if TYPE_CHECKING:
    # Resolution for reportMissingImports: ignore missing optional library
    from catboost import CatBoostRegressor  # type: ignore

    # Optional: pandas is not required at import-time; only used if provided at runtime

    HAS_CATBOOST = True
else:
    try:  # pragma: no cover - optional dependency
        from catboost import CatBoostRegressor

        HAS_CATBOOST = True
    except Exception:  # pragma: no cover - optional dependency
        CatBoostRegressor = None
        HAS_CATBOOST = False


def _is_pandas_df(obj: Any) -> bool:
    """Return True if obj is a pandas DataFrame (without importing pandas globally)."""
    return hasattr(obj, "__dataframe__") or obj.__class__.__name__ == "DataFrame"


def _infer_cat_features_from_df(X: Any) -> list[int]:
    """
    Infer categorical feature indices from a pandas DataFrame.

    We treat object/category dtype columns as categorical by default.
    (Booleans are *not* inferred as categorical here; callers may pass them explicitly.)
    """
    # Import pandas lazily to keep module import optional-dependency safe.
    import pandas as pd  # type: ignore

    if not isinstance(X, pd.DataFrame):
        return []

    cat_cols: list[str] = []
    for c in X.columns:
        dt = X[c].dtype
        if dt == "object" or str(dt) == "category":
            cat_cols.append(str(c))

    if not cat_cols:
        return []

    col_index = {str(c): i for i, c in enumerate(X.columns)}
    return [col_index[c] for c in cat_cols if c in col_index]


def _normalize_cat_features(
    X: Any,
    cat_features: Sequence[int] | Sequence[str] | None,
) -> list[int] | None:
    """
    Normalize `cat_features` into the form expected by CatBoost: a list of indices.

    - If X is a DataFrame:
        - cat_features may be column names or indices.
        - if cat_features is None, infer from object/category dtype columns.
    - If X is an ndarray:
        - cat_features must be indices (or None).
    """
    if cat_features is None:
        if _is_pandas_df(X):
            inferred = _infer_cat_features_from_df(X)
            return inferred or None
        return None

    # If provided, normalize based on X type
    if _is_pandas_df(X):
        import pandas as pd  # type: ignore

        if isinstance(X, pd.DataFrame):
            col_index = {str(c): i for i, c in enumerate(X.columns)}

            # Names -> indices
            if len(cat_features) > 0 and isinstance(cat_features[0], str):  # type: ignore[index]
                names = cast(Sequence[str], cat_features)
                unknown = [n for n in names if str(n) not in col_index]
                if unknown:
                    raise ValueError(
                        f"Unknown cat_features column name(s): {unknown}. "
                        "Pass valid DataFrame column names or integer indices."
                    )
                return [col_index[str(n)] for n in names]

            # Indices as-is
            return [int(i) for i in cast(Sequence[int], cat_features)]

    # ndarray path: indices only
    if len(cat_features) > 0 and isinstance(cat_features[0], str):  # type: ignore[index]
        raise ValueError(
            "cat_features as names is only supported when X is a pandas DataFrame. "
            "For numpy arrays, pass cat_features as integer indices."
        )
    return [int(i) for i in cast(Sequence[int], cat_features)]


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
    - Supports X as either numpy arrays or pandas DataFrames.
    - If X is a pandas DataFrame, categorical features can be specified via:
        - `fit(..., cat_features=[...])` as column names or indices, OR
        - inferred automatically from object/category dtype columns when `cat_features=None`.
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
    >>> # X may be numpy arrays or a pandas DataFrame
    >>> # model.fit(X, y, cat_features=["STATE", "SEASON"]).predict(X)
    """

    def __init__(self, **params: Any) -> None:
        if not HAS_CATBOOST:
            raise ImportError(
                "CatBoostAdapter requires the optional 'catboost' package. "
                "Install it via `pip install catboost`."
            )

        # Store params for clone() compatibility
        self.params: dict[str, Any] = dict(params)

        # Default: no spammy training logs
        if "verbose" not in self.params:
            self.params["verbose"] = False

        # Instantiate the underlying CatBoost model
        # Resolution for reportOptionalCall: Cast the constructor to Any to bypass None-check
        ctor = cast(Any, CatBoostRegressor)
        self.model: Any | None = ctor(**self.params)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: Any,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        cat_features: Sequence[int] | Sequence[str] | None = None,
    ) -> CatBoostAdapter:
        """
        Fit the underlying `catboost.CatBoostRegressor`.

        Parameters
        ----------
        X : Any
            Feature matrix. May be a numpy.ndarray of shape (n_samples, n_features)
            or a pandas.DataFrame.
        y : numpy.ndarray
            Target vector of shape (n_samples,).
        sample_weight : numpy.ndarray | None
            Optional per-sample weights of shape (n_samples,). If provided, this is
            forwarded to CatBoost training.
        cat_features : Sequence[int] | Sequence[str] | None
            Categorical feature specification.
            - If X is a pandas.DataFrame: may be column names or integer indices.
            - If X is a numpy.ndarray: must be integer indices.
            - If None and X is a pandas.DataFrame: inferred from object/category dtypes.

        Returns
        -------
        CatBoostAdapter
            The fitted adapter (self), allowing method chaining.

        Raises
        ------
        RuntimeError
            If CatBoost is not available or the internal model is not initialized.
        ValueError
            If cat_features are invalid for the given X type.
        """
        if not HAS_CATBOOST or self.model is None:
            raise RuntimeError(
                "CatBoostAdapter cannot train: CatBoost is not available or "
                "the internal model was not initialized correctly."
            )

        y_arr = np.asarray(y, dtype=float).ravel()
        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=float).ravel()
        else:
            sw_arr = None

        cat_idx = _normalize_cat_features(X, cat_features)

        # Resolution for reportOptionalCall: use a local typed variable
        m = cast(Any, self.model)

        # If pandas DataFrame provided, pass through directly so CatBoost can use it.
        if _is_pandas_df(X):
            if sw_arr is not None:
                m.fit(X, y_arr, sample_weight=sw_arr, cat_features=cat_idx)
            else:
                m.fit(X, y_arr, cat_features=cat_idx)
            return self

        # numpy / array-like fallback
        X_arr = np.asarray(X)
        if sw_arr is not None:
            m.fit(X_arr, y_arr, sample_weight=sw_arr, cat_features=cat_idx)
        else:
            m.fit(X_arr, y_arr, cat_features=cat_idx)

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X: Any) -> np.ndarray:
        """
        Predict using the fitted CatBoost model.

        Parameters
        ----------
        X : Any
            Feature matrix. May be a numpy.ndarray of shape (n_samples, n_features)
            or a pandas.DataFrame.

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

        # Resolution for reportOptionalCall
        m = cast(Any, self.model)
        preds = m.predict(X)
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
        return dict(self.params)

    def set_params(self, **params: Any) -> CatBoostAdapter:
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
            ctor = cast(Any, CatBoostRegressor)
            self.model = ctor(**self.params)
        return self

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"CatBoostAdapter(params={self.params})"
