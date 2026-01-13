"""
Base adapter interfaces and cloning utilities.

This module defines the minimal adapter contract used throughout the
ElectricBarometer ecosystem, along with a lightweight cloning helper for
estimator-like objects.

Adapters are intended to wrap non-scikit-learn forecasting or regression
libraries (for example, statsmodels, Prophet, or custom models) and expose
a scikit-learn-like interface so they can be used interchangeably inside
ElectricBarometer evaluation and selection workflows.

Important
---------
Model adapters in eb-adapters are intentionally "array-level" by default:
they expect `X` and `y` as numpy arrays. Electric Barometer contracts
(e.g., PanelDemandV1) must be converted to arrays by the caller (or by a
higher-level orchestration layer) before calling `fit()` / `predict()`.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _clone_model(model: Any) -> Any:
    """
    Lightweight cloning utility for estimator-like or adapter-like objects.

    This function attempts to reconstruct a fresh instance of a model using a
    best-effort strategy that favors compatibility with scikit-learn-style APIs
    while remaining usable for custom adapters.

    Cloning strategy
    ----------------
    The following steps are attempted in order:

    1. If scikit-learn is available, call `sklearn.base.clone(model)`.
    2. Otherwise, if the object implements `get_params()`, re-instantiate via::

           model.__class__(**model.get_params())

    3. As a final fallback, instantiate the class with no arguments::

           model.__class__()

    Notes
    -----
    For custom adapters, the most reliable approach is to make the adapter
    configuration-only at initialization time and implement `get_params()`
    so that the instance can be reconstructed deterministically.

    If a model cannot be meaningfully cloned using parameters alone, callers
    may choose to bypass this helper and explicitly construct fresh adapter
    instances before passing them into ElectricBarometer workflows.
    """
    # Try sklearn.clone if available
    try:  # pragma: no cover - optional dependency path
        from sklearn.base import clone as sk_clone  # type: ignore

        return sk_clone(model)
    except Exception:
        pass

    # Fallback: re-create via class + get_params
    if hasattr(model, "get_params"):
        try:
            params = model.get_params()  # type: ignore[assignment]
            return model.__class__(**params)
        except Exception:
            # If get_params exists but reconstruction fails, fall through
            # to the final fallback below.
            pass

    # Last resort: call class with no args
    return model.__class__()


# Optional public alias for convenience / backwards compatibility
clone_model = _clone_model


def _type_label(obj: Any) -> str:
    """Return a compact type label suitable for error messages."""
    try:
        return f"{obj.__class__.__module__}.{obj.__class__.__name__}"
    except Exception:  # pragma: no cover
        return str(type(obj))


def _looks_like_eb_contract(obj: Any) -> bool:
    """
    Best-effort detection for EB contract-like objects.

    We intentionally avoid importing eb-contracts here to keep base.py
    lightweight and free of hard dependencies.

    Heuristic:
    - has attribute `frame` (common for panel contracts), and
    - class name contains "Panel" or ends with "V1" (contract naming convention)
    """
    try:
        name = obj.__class__.__name__
    except Exception:  # pragma: no cover
        return False
    if not hasattr(obj, "frame"):
        return False
    return ("Panel" in name) or name.endswith("V1")


def validate_fit_inputs(
    X: Any,
    y: Any,
    sample_weight: Any | None = None,
    *,
    adapter_name: str | None = None,
) -> None:
    """
    Validate that adapter `fit()` inputs match the EB model-adapter contract.

    This is a small, reusable guard intended to be called by subclasses
    at the top of their `fit()` implementations.

    Parameters
    ----------
    X, y
        Expected to be numpy arrays. `y` should be one-dimensional.
    sample_weight
        If provided, expected to be a numpy array.
    adapter_name
        Optional name to use in error messages (defaults to "Adapter").

    Raises
    ------
    TypeError
        If inputs are not numpy arrays, or if an EB contract-like object was passed.
    ValueError
        If array shapes are inconsistent or y is not one-dimensional.
    """
    name = adapter_name or "Adapter"

    # Helpful error when a user passes an EB contract by mistake.
    if (
        _looks_like_eb_contract(X)
        or _looks_like_eb_contract(y)
        or _looks_like_eb_contract(sample_weight)
    ):
        raise TypeError(
            f"{name}.fit expects numpy arrays (X: np.ndarray, y: np.ndarray). "
            "You passed an Electric Barometer contract-like object. "
            "Extract arrays from the contract (e.g., from `panel.frame`) before calling fit(). "
            f"Got X={_type_label(X)}, y={_type_label(y)}."
        )

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError(
            f"{name}.fit expects numpy arrays (X: np.ndarray, y: np.ndarray). "
            f"Got X={_type_label(X)}, y={_type_label(y)}."
        )

    if y.ndim != 1:
        raise ValueError(
            f"{name}.fit expects y as a 1D numpy array. "
            f"Got y with shape={getattr(y, 'shape', None)}."
        )

    if sample_weight is not None and not isinstance(sample_weight, np.ndarray):
        raise TypeError(
            f"{name}.fit expects sample_weight as a numpy array when provided. "
            f"Got sample_weight={_type_label(sample_weight)}."
        )

    if sample_weight is not None:
        if sample_weight.ndim != 1:
            raise ValueError(
                f"{name}.fit expects sample_weight as a 1D numpy array. "
                f"Got sample_weight with shape={getattr(sample_weight, 'shape', None)}."
            )
        if len(sample_weight) != len(y):
            raise ValueError(
                f"{name}.fit expects sample_weight to have the same length as y. "
                f"Got len(sample_weight)={len(sample_weight)}, len(y)={len(y)}."
            )


class BaseAdapter:
    """
    Minimal base class defining the adapter contract for ElectricBarometer.

    This class documents the expected interface for wrapping non-scikit-learn
    forecasting or regression engines so they can be evaluated and selected
    alongside native scikit-learn estimators.

    Subclasses are expected to present a scikit-learn-like API:

    - `fit(X, y, sample_weight=None)` returning `self`
    - `predict(X)` returning a one-dimensional numpy array

    Notes
    -----
    This adapter contract is array-level: `X` and `y` are numpy arrays.
    Electric Barometer contracts (e.g., PanelDemandV1) should be converted to
    arrays before calling `fit()` / `predict()`.
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> BaseAdapter:
        """
        Fit the underlying forecasting or regression model.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix. For pure time-series models, this may be ignored
            or used only for alignment.
        y : numpy.ndarray
            One-dimensional target vector.
        sample_weight : numpy.ndarray | None
            Optional per-sample weights. Adapters may ignore this argument if
            weighting is not supported by the underlying model.

        Returns
        -------
        BaseAdapter
            The fitted adapter instance (self).

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError(
            "BaseAdapter subclasses must implement fit(X, y, sample_weight=None)."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions from the fitted model.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix used to generate predictions.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of predictions.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("BaseAdapter subclasses must implement predict(X).")
