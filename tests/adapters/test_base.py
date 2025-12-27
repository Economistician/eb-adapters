import numpy as np
import pytest

from eb_adapters.base import BaseAdapter, _clone_model


def test_base_adapter_fit_raises_not_implemented():
    """BaseAdapter.fit should raise NotImplementedError by default."""
    adapter = BaseAdapter()
    with pytest.raises(NotImplementedError):
        adapter.fit(np.array([[1.0]]), np.array([1.0]))


def test_base_adapter_predict_raises_not_implemented():
    """BaseAdapter.predict should raise NotImplementedError by default."""
    adapter = BaseAdapter()
    with pytest.raises(NotImplementedError):
        adapter.predict(np.array([[1.0]]))


def test_clone_model_with_sklearn_estimator_if_available():
    """
    If scikit-learn is installed, _clone_model should delegate to sklearn.clone
    and produce an independent estimator with the same parameters.
    """
    pytest.importorskip("sklearn", reason="sklearn not installed")
    from sklearn.dummy import DummyRegressor  # type: ignore

    original = DummyRegressor(strategy="mean", constant=None)
    clone = _clone_model(original)

    # Same class, same params, but not the same object
    assert isinstance(clone, DummyRegressor)
    assert clone is not original
    assert clone.get_params() == original.get_params()

    # Mutating cloned params should not affect the original
    clone.set_params(strategy="median")
    assert original.get_params()["strategy"] == "mean"
    assert clone.get_params()["strategy"] == "median"


def test_clone_model_with_get_params_fallback():
    """
    If sklearn.clone is unavailable or fails, and the object exposes get_params(),
    _clone_model should reconstruct via model.__class__(**get_params()).
    """

    class ToyModel:
        def __init__(self, a: int = 1, b: float = 2.0) -> None:
            self.a = a
            self.b = b

        def get_params(self, deep: bool = True):
            return {"a": self.a, "b": self.b}

    m = ToyModel(a=5, b=3.5)
    cloned = _clone_model(m)

    assert isinstance(cloned, ToyModel)
    assert cloned is not m
    assert cloned.a == 5
    assert cloned.b == 3.5


def test_clone_model_without_get_params_uses_default_ctor():
    """
    If the object has no get_params(), _clone_model should fall back to calling
    model.__class__() with no arguments.
    """

    class NoParams:
        def __init__(self):
            self.flag = True

    obj = NoParams()
    cloned = _clone_model(obj)

    assert isinstance(cloned, NoParams)
    assert cloned is not obj
    assert cloned.flag is True
