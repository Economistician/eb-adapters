from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _clone_model(model: Any) -> Any:
    """
    Lightweight clone for sklearn-style estimators or adapter-like objects.

    Cloning strategy
    ----------------
    1. If scikit-learn is available, try ``sklearn.base.clone(model)``.
    2. Otherwise, if the object implements ``get_params()``, re-instantiate via::

           model.__class__(**model.get_params())

    3. As a last resort, call ``model.__class__()`` with no arguments.

    Notes for custom adapters
    -------------------------
    If you implement your own adapter (e.g., for statsmodels, Prophet, ARIMA),
    you have two main options to play nicely with Electric Barometer:

    - Make your adapter "configuration-only" at ``__init__`` time and implement
      ``get_params()`` so that it can be reconstructed via ``__class__(**params)``.

    - Or, if you need custom cloning logic, simply instantiate fresh adapters in
      your own code before passing them into the selection / evaluation engine.
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


# Optional public alias for convenience / backwards-compat
clone_model = _clone_model


class BaseAdapter:
    """
    Minimal base class for non-sklearn-style forecast engines.

    Subclasses are expected to wrap libraries like statsmodels, Prophet,
    ARIMA/SARIMAX, or any custom forecasting engine, and present a
    scikit-learn-like interface:

        adapter = MyAdapter(...)
        adapter.fit(X, y)       # returns self
        y_pred = adapter.predict(X)

    Requirements
    ------------
    - ``fit(self, X, y, sample_weight=None) -> BaseAdapter``

        * ``X`` : array-like or ignored (for pure time-series models)
        * ``y`` : 1D array-like of targets
        * ``sample_weight`` : optional, can be ignored if not supported.

    - ``predict(self, X) -> array-like``

        Should return a 1D numpy array of predictions for each row in ``X``.

    The Electric Barometer engine does *not* need to know whether a model is a
    "real" sklearn estimator or an adapter; it simply calls ``.fit()`` and
    ``.predict()``. This base class is provided as a clear, documented contract
    for users who want to adapt non-sklearn libraries into the EB ecosystem.
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BaseAdapter":
        """
        Fit the underlying forecasting model.

        Subclasses must override this method and return ``self``.
        """
        raise NotImplementedError(
            "BaseAdapter subclasses must implement .fit(X, y, sample_weight=None)."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions from the fitted model.

        Subclasses must override this method and return a 1D numpy array.
        """
        raise NotImplementedError(
            "BaseAdapter subclasses must implement .predict(X)."
        )