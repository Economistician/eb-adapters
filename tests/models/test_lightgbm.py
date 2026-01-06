import warnings

import numpy as np
import pytest

# Skip this entire module if lightgbm is not installed
pytest.importorskip("lightgbm", reason="lightgbm not installed")

from eb_adapters.models.lightgbm import LightGBMRegressorAdapter


def test_lightgbm_adapter_fit_and_predict_basic():
    """
    Smoke test: LightGBMRegressorAdapter should fit on small data and predict
    without errors, returning numeric 1D outputs of the right shape.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    adapter = LightGBMRegressorAdapter(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=-1,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"X does not have valid feature names, but LGBMRegressor was fitted "
                r"with feature names"
            ),
            category=UserWarning,
        )
        adapter.fit(X, y)
        y_pred = adapter.predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert np.all(np.isfinite(y_pred))


def test_lightgbm_adapter_respects_sample_weight_argument():
    """
    The adapter should accept sample_weight and not raise. We just check
    that it runs and returns predictions of the right shape.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 2.0, 1.0, 0.5])

    adapter = LightGBMRegressorAdapter(
        n_estimators=5,
        learning_rate=0.05,
        max_depth=3,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"X does not have valid feature names, but LGBMRegressor was fitted "
                r"with feature names"
            ),
            category=UserWarning,
        )
        adapter.fit(X, y, sample_weight=w)
        y_pred = adapter.predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert np.all(np.isfinite(y_pred))


def test_lightgbm_adapter_get_params_and_set_params_roundtrip():
    """
    get_params should expose the initialization kwargs, and set_params should
    rebuild the internal model with updated parameters.
    """
    adapter = LightGBMRegressorAdapter(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=4,
    )

    params = adapter.get_params()
    # At minimum, these keys should be present
    assert params["n_estimators"] == 50
    assert params["learning_rate"] == 0.1
    assert params["max_depth"] == 4

    # Update params and ensure they are reflected
    adapter = adapter.set_params(n_estimators=10, max_depth=2)
    new_params = adapter.get_params()
    assert new_params["n_estimators"] == 10
    assert new_params["max_depth"] == 2


def test_lightgbm_adapter_repr_contains_class_name():
    """
    __repr__ should contain some identifying info and be stable enough
    for debugging. (If you didn't implement __repr__, this will still
    pass as long as the default repr includes the class name.)
    """
    adapter = LightGBMRegressorAdapter(n_estimators=10, learning_rate=0.1)
    rep = repr(adapter)
    # We're not strict on exact formatting, just that it's useful.
    assert "LightGBMRegressorAdapter" in rep or "LightGBM" in rep
