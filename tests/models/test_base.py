import numpy as np
import pytest

from eb_adapters.models.base import BaseAdapter, _clone_model, validate_fit_inputs


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


def test_validate_fit_inputs_accepts_numpy_arrays():
    """validate_fit_inputs should accept well-typed numpy arrays."""
    X = np.zeros((10, 2), dtype=float)
    y = np.zeros((10,), dtype=float)
    sw = np.ones((10,), dtype=float)
    validate_fit_inputs(X, y, sw, adapter_name="ToyAdapter")


def test_validate_fit_inputs_rejects_non_ndarray_X():
    """validate_fit_inputs should raise TypeError when X is not an ndarray."""
    X = [[1.0, 2.0]]  # list, not ndarray
    y = np.zeros((1,), dtype=float)
    with pytest.raises(TypeError, match=r"expects numpy arrays"):
        validate_fit_inputs(X, y, adapter_name="ToyAdapter")


def test_validate_fit_inputs_rejects_non_ndarray_y():
    """validate_fit_inputs should raise TypeError when y is not an ndarray."""
    X = np.zeros((1, 2), dtype=float)
    y = [1.0]  # list, not ndarray
    with pytest.raises(TypeError, match=r"expects numpy arrays"):
        validate_fit_inputs(X, y, adapter_name="ToyAdapter")


def test_validate_fit_inputs_rejects_non_ndarray_sample_weight():
    """validate_fit_inputs should raise TypeError when sample_weight is not an ndarray."""

    X = np.zeros((3, 1), dtype=float)
    y = np.zeros((3,), dtype=float)
    sw = [1.0, 1.0, 1.0]  # list, not ndarray

    with pytest.raises(TypeError, match=r"sample_weight"):
        validate_fit_inputs(X, y, sw, adapter_name="ToyAdapter")


def test_validate_fit_inputs_rejects_contract_like_input():
    """validate_fit_inputs should provide actionable guidance for contract-like objects."""

    class FakePanelDemandV1:
        # Heuristic in base.py: has `.frame` and class name looks contract-ish.
        def __init__(self) -> None:
            self.frame = "not-a-real-frame"

    contract_like = FakePanelDemandV1()
    y = np.zeros((1,), dtype=float)

    with pytest.raises(TypeError, match=r"contract-like|Electric Barometer"):
        validate_fit_inputs(contract_like, y, adapter_name="ToyAdapter")
