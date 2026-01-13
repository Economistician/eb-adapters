import numpy as np
import pytest

# Skip the whole module if statsmodels is not installed
pytest.importorskip("statsmodels", reason="statsmodels not installed")

from eb_adapters.models.statsmodels import ArimaAdapter, SarimaxAdapter


def _make_series(n: int = 30) -> np.ndarray:
    """Simple helper to build a numeric time series."""
    x = np.linspace(0.0, 4.0 * np.pi, n)
    y = 10.0 + np.sin(x)
    return y.astype(float)


def test_sarimax_adapter_fit_and_predict_basic():
    """
    Smoke test: SarimaxAdapter should fit and forecast the right number
    of steps with finite numeric outputs.
    """
    y = _make_series(40)
    X_train = np.zeros((len(y), 1))  # ignored by the adapter
    X_val = np.zeros((10, 1))  # just controls forecast horizon

    adapter = SarimaxAdapter(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    adapter.fit(X_train, y)

    y_pred = adapter.predict(X_val)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (len(X_val),)
    assert np.all(np.isfinite(y_pred))


def test_sarimax_adapter_records_fit_diagnostics():
    """
    After fit(), SarimaxAdapter should expose statsmodels fit diagnostics.
    """
    y = _make_series(40)
    X_train = np.zeros((len(y), 1))

    adapter = SarimaxAdapter(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    adapter.fit(X_train, y)

    assert adapter.fit_diagnostics is not None
    assert "converged" in adapter.fit_diagnostics
    assert isinstance(adapter.fit_diagnostics["converged"], bool)

    diag = adapter.get_fit_diagnostics()
    assert diag is not None
    assert diag["converged"] == adapter.fit_diagnostics["converged"]


def test_sarimax_adapter_get_and_set_params_roundtrip():
    """
    get_params/set_params should provide a minimal sklearn-like API so
    that clone-like behavior can reconstruct equivalent adapters.
    """
    adapter = SarimaxAdapter(
        order=(2, 1, 0),
        seasonal_order=(1, 0, 0, 7),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    params = adapter.get_params()
    # Basic keys present
    assert params["order"] == (2, 1, 0)
    assert params["seasonal_order"] == (1, 0, 0, 7)
    assert params["trend"] == "c"
    assert params["enforce_stationarity"] is False
    assert params["enforce_invertibility"] is False

    # Change some params via set_params
    adapter.set_params(order=(1, 0, 0), trend=None)
    params2 = adapter.get_params()
    assert params2["order"] == (1, 0, 0)
    assert params2["trend"] is None


def test_arima_adapter_fit_and_predict_basic():
    """
    Smoke test: ArimaAdapter should fit and forecast the right number
    of steps with finite numeric outputs.
    """
    y = _make_series(30)
    X_train = np.zeros((len(y), 1))  # ignored by the adapter
    X_val = np.zeros((5, 1))

    adapter = ArimaAdapter(order=(1, 0, 0), trend=None)
    adapter.fit(X_train, y)

    y_pred = adapter.predict(X_val)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (len(X_val),)
    assert np.all(np.isfinite(y_pred))


def test_arima_adapter_records_fit_diagnostics():
    """
    After fit(), ArimaAdapter should expose statsmodels fit diagnostics.
    """
    y = _make_series(30)
    X_train = np.zeros((len(y), 1))

    adapter = ArimaAdapter(order=(1, 0, 0), trend=None)
    adapter.fit(X_train, y)

    assert adapter.fit_diagnostics is not None
    assert "converged" in adapter.fit_diagnostics
    assert isinstance(adapter.fit_diagnostics["converged"], bool)

    diag = adapter.get_fit_diagnostics()
    assert diag is not None
    assert diag["converged"] == adapter.fit_diagnostics["converged"]


def test_arima_adapter_get_and_set_params_roundtrip():
    """
    get_params/set_params should allow cloning-style reconstruction
    for the ARIMA adapter as well.
    """
    adapter = ArimaAdapter(order=(2, 1, 1), trend="c")
    params = adapter.get_params()

    assert params["order"] == (2, 1, 1)
    assert params["trend"] == "c"

    adapter.set_params(order=(1, 0, 0), trend=None)
    params2 = adapter.get_params()

    assert params2["order"] == (1, 0, 0)
    assert params2["trend"] is None


def test_sarimax_adapter_fit_rejects_contract_like_inputs():
    """
    SarimaxAdapter should provide an actionable error when a contract-like object
    is passed instead of numpy arrays.
    """

    class FakePanelDemandV1:
        def __init__(self) -> None:
            self.frame = "not-a-real-frame"

    y = _make_series(20)

    adapter = SarimaxAdapter(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))

    with pytest.raises(TypeError, match=r"expects numpy arrays|contract-like|Electric Barometer"):
        adapter.fit(FakePanelDemandV1(), y)  # type: ignore[arg-type]


def test_arima_adapter_fit_rejects_contract_like_inputs():
    """
    ArimaAdapter should provide an actionable error when a contract-like object
    is passed instead of numpy arrays.
    """

    class FakePanelDemandV1:
        def __init__(self) -> None:
            self.frame = "not-a-real-frame"

    y = _make_series(20)

    adapter = ArimaAdapter(order=(1, 0, 0), trend=None)

    with pytest.raises(TypeError, match=r"expects numpy arrays|contract-like|Electric Barometer"):
        adapter.fit(FakePanelDemandV1(), y)  # type: ignore[arg-type]


def test_arima_adapter_fit_rejects_non_ndarray_inputs():
    """
    ArimaAdapter should reject non-ndarray inputs with a clear TypeError.
    """
    y = _make_series(20).tolist()
    X_train = np.zeros((20, 1))

    adapter = ArimaAdapter(order=(1, 0, 0), trend=None)

    with pytest.raises(TypeError, match=r"expects numpy arrays"):
        adapter.fit(X_train, y)  # type: ignore[arg-type]
