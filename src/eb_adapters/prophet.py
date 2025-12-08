from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import BaseAdapter


class ProphetAdapter(BaseAdapter):
    """
    Adapter for `prophet.Prophet` so it can be used inside Electric Barometer
    or any CWSL-based evaluation workflow.

    Assumptions
    -----------
    * X encodes the time index:
        - shape (n_samples,) of datetime-like values, or
        - shape (n_samples, n_features) where the *first column* is datetime-like.
    * y is a 1D array-like of numeric targets (demand, load, etc).

    Behavior
    --------
    - At fit-time, constructs a DataFrame with columns:
          ds = timestamps from X
          y  = targets from y
      and calls `Prophet.fit(df)`.

    - At predict-time, constructs a DataFrame with:
          ds = timestamps from X
      and returns the Prophet `yhat` column as a 1D numpy array.

    Optional dependency
    -------------------
    Prophet is *not* required for core EB metrics. If the `prophet` package is
    not installed, constructing ProphetAdapter() without an explicit model
    will raise an ImportError with a helpful message.

    You can also pass an already-configured Prophet instance:

        from prophet import Prophet
        base = Prophet(...)
        adapter = ProphetAdapter(model=base)
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        if model is None:
            try:
                from prophet import Prophet as _Prophet  # type: ignore
            except Exception as e:  # pragma: no cover - import failure path
                raise ImportError(
                    "ProphetAdapter requires the optional 'prophet' package. "
                    "Install it via `pip install prophet`."
                ) from e

            model = _Prophet()

        self.model = model

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,  # ignored
    ) -> "ProphetAdapter":
        # Local import to avoid making pandas a hard dependency for the module
        import pandas as pd

        X_arr = np.asarray(X)

        # Use the first column if 2D
        if X_arr.ndim > 1:
            X_arr = X_arr[:, 0]

        ds = pd.to_datetime(X_arr)
        y_arr = np.asarray(y, dtype=float)

        df = pd.DataFrame({"ds": ds, "y": y_arr})
        self.model.fit(df)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        import pandas as pd

        X_arr = np.asarray(X)
        if X_arr.ndim > 1:
            X_arr = X_arr[:, 0]

        ds = pd.to_datetime(X_arr)
        df_future = pd.DataFrame({"ds": ds})
        forecast = self.model.predict(df_future)

        if "yhat" not in forecast.columns:
            raise RuntimeError(
                "ProphetAdapter: expected 'yhat' column in forecast output."
            )

        return np.asarray(forecast["yhat"], dtype=float)