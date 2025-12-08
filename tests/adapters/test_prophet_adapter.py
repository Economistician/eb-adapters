import numpy as np
import pytest

# Skip this entire module if prophet is not installed
pytest.importorskip("prophet", reason="prophet not installed")

from prophet import Prophet  # type: ignore
from eb_adapters.prophet import ProphetAdapter


def _make_time_index(n: int = 10):
    """Helper to build a simple increasing datetime index."""
    # Use numpy datetime64 for simplicity
    base = np.datetime64("2024-01-01")
    return base + np.arange(n).astype("timedelta64[D]")


def test_prophet_adapter_fit_and_predict_basic():
    """
    Smoke test: ProphetAdapter should fit on small data and predict
    without errors, returning numeric 1D outputs of the right shape.
    """
    n = 20
    X = _make_time_index(n)
    y = np.linspace(10.0, 20.0, n)

    adapter = ProphetAdapter()  # will construct an internal Prophet model
    adapter.fit(X, y)

    # Predict on the same horizon
    y_pred = adapter.predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert np.all(np.isfinite(y_pred))


def test_prophet_adapter_accepts_external_model_instance():
    """
    You can pass an already-configured Prophet instance into the adapter.
    The adapter should still fit and predict correctly.
    """
    n = 15
    X = _make_time_index(n)
    y = np.linspace(5.0, 8.0, n)

    base_model = Prophet()  # user-configured Prophet instance
    adapter = ProphetAdapter(model=base_model)

    adapter.fit(X, y)
    y_pred = adapter.predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert np.all(np.isfinite(y_pred))


def test_prophet_adapter_ignores_sample_weight_argument():
    """
    ProphetAdapter.fit accepts sample_weight but currently ignores it.
    This test just ensures that passing sample_weight does not raise
    and that predictions still look sane.
    """
    n = 12
    X = _make_time_index(n)
    y = np.linspace(0.0, 1.0, n)
    w = np.linspace(1.0, 2.0, n)

    adapter = ProphetAdapter()
    adapter.fit(X, y, sample_weight=w)

    y_pred = adapter.predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert np.all(np.isfinite(y_pred))