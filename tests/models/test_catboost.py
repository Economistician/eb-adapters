from __future__ import annotations

import numpy as np
import pytest

# Skip this entire module if catboost is not installed
pytest.importorskip("catboost", reason="catboost not installed")

from eb_adapters.models.catboost import CatBoostAdapter


def test_catboost_adapter_fit_and_predict_basic() -> None:
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


def test_catboost_adapter_respects_sample_weight_argument() -> None:
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


def test_catboost_adapter_supports_dataframe_categoricals_via_names() -> None:
    """
    Regression test: CatBoostAdapter should accept a pandas DataFrame with string
    categorical columns, and allow `cat_features` to be passed as column names.

    This specifically guards against the historical failure mode:
    "Cannot convert 'LEGACY' to float".
    """
    pd = pytest.importorskip("pandas", reason="pandas not installed")

    rng = np.random.default_rng(0)

    X_df = pd.DataFrame(
        {
            "STORE_AGE_BUCKET": ["LEGACY", "MATURE", "NEW", "EARLY", "LEGACY"],
            "STATE": ["FL", "FL", "TX", "TX", "FL"],
            "INTERVAL_30_INDEX": [0, 1, 2, 3, 4],
            "LAG_1_COMMODITY_USAGE_QTY": rng.integers(0, 5, size=5).astype(float),
            "HAS_LAG_1_COMMODITY_USAGE_QTY": [1, 1, 0, 1, 0],
        }
    )

    y = np.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype=float)

    adapter = CatBoostAdapter(
        iterations=20,
        depth=4,
        learning_rate=0.1,
        loss_function="RMSE",
        random_seed=0,
        verbose=False,
    )

    adapter.fit(X_df, y, cat_features=["STORE_AGE_BUCKET", "STATE"])
    y_pred = adapter.predict(X_df)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert np.all(np.isfinite(y_pred))


def test_catboost_adapter_supports_dataframe_categoricals_via_indices() -> None:
    """
    Regression test: CatBoostAdapter should accept `cat_features` as integer indices
    when fitting on a pandas DataFrame.

    This ensures both name-based and index-based categorical specs work.
    """
    pd = pytest.importorskip("pandas", reason="pandas not installed")

    X_df = pd.DataFrame(
        {
            "A_CAT": ["x", "y", "x", "z"],
            "B_NUM": [0.0, 1.0, 2.0, 3.0],
        }
    )
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)

    adapter = CatBoostAdapter(
        iterations=10,
        depth=2,
        loss_function="RMSE",
        random_seed=0,
        verbose=False,
    )

    # "A_CAT" is column index 0
    adapter.fit(X_df, y, cat_features=[0])
    y_pred = adapter.predict(X_df)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert np.all(np.isfinite(y_pred))


def test_catboost_adapter_get_params_and_set_params_roundtrip() -> None:
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


def test_catboost_adapter_repr_contains_class_name() -> None:
    """
    __repr__ should contain the class name and be stable enough for debugging.
    """
    adapter = CatBoostAdapter(iterations=10, depth=2, loss_function="RMSE")
    rep = repr(adapter)
    assert "CatBoostAdapter" in rep
    assert "iterations" in rep or "params=" in rep
