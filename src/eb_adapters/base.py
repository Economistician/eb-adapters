from __future__ import annotations

"""
Base adapter interfaces and cloning utilities.

This module defines the minimal adapter contract used throughout the
ElectricBarometer ecosystem, along with a lightweight cloning helper for
estimator-like objects.

Adapters are intended to wrap non-scikit-learn forecasting or regression
libraries (for example, statsmodels, Prophet, or custom models) and expose
a scikit-learn-like interface so they can be used interchangeably inside
ElectricBarometer evaluation and selection workflows.
"""

from typing import Any, Optional

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


class BaseAdapter:
    """
    Minimal base class defining the adapter contract for ElectricBarometer.

    This class documents the expected interface for wrapping non-scikit-learn
    forecasting or regression engines so they can be evaluated and selected
    alongside native scikit-learn estimators.

    Subclasses are expected to present a scikit-learn-like API:

    - `fit(X, y, sample_weight=None)` returning `self`
    - `predict(X)` returning a one-dimensional numpy array

    The ElectricBarometer engine does not distinguish between native
    scikit-learn estimators and adapters; it simply calls `fit` and `predict`.
    This base class serves as a clear, documented contract for adapter authors.
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BaseAdapter":
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
        raise NotImplementedError(
            "BaseAdapter subclasses must implement predict(X)."
        )