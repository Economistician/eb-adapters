import numpy as np
import pytest

# Skip this entire module if catboost is not installed
pytest.importorskip("catboost", reason="catboost not installed")

from eb_adapters.catboost import CatBoostAdapter


def test_catboost_adapter_fit_and_predict_basic():
    """
    Smoke test: CatBoostAdapter should fit on small data and predict
    without errors, returning numeric 1D outputs of the right shape.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    adapter = CatBoostAdapter(
        iterations=10,
        depth=2,
        loss_function="RMSE",
        verbose=False,
    )

    adapter.fit(X, y)
    y_pred = adapter.predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert np.all(np.isfinite(y_pred))


def test_catboost_adapter_respects_sample_weight_argument():
    """
    The adapter should accept sample_weight and not raise. We just check
    that it runs and returns predictions of the right shape.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 2.0, 1.0, 0.5])

    adapter = CatBoostAdapter(
        iterations=5,
        depth=2,
        loss_function="RMSE",
        verbose=False,
    )

    adapter.fit(X, y, sample_weight=w)
    y_pred = adapter.predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert np.all(np.isfinite(y_pred))


def test_catboost_adapter_get_params_and_set_params_roundtrip():
    """
    get_params should expose the initialization kwargs (plus defaults like
    verbose), and set_params should rebuild the internal model with updated
    parameters.
    """
    adapter = CatBoostAdapter(
        iterations=20,
        depth=3,
        loss_function="RMSE",
        learning_rate=0.1,
    )

    params = adapter.get_params()
    # At minimum, these keys should be present
    assert params["iterations"] == 20
    assert params["depth"] == 3
    assert params["learning_rate"] == 0.1
    # verbose should be present (defaulted to False if not provided)
    assert "verbose" in params

    # Update params and ensure they are reflected
    adapter = adapter.set_params(iterations=5, depth=2)
    new_params = adapter.get_params()
    assert new_params["iterations"] == 5
    assert new_params["depth"] == 2


def test_catboost_adapter_repr_contains_class_name():
    """
    __repr__ should contain the class name and be stable enough for debugging.
    """
    adapter = CatBoostAdapter(iterations=10, depth=2, loss_function="RMSE")
    rep = repr(adapter)
    assert "CatBoostAdapter" in rep
    assert "iterations" in rep or "params=" in rep
