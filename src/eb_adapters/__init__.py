"""
eb_adapters.

Adapter classes for integrating external forecasting and regression engines
(Prophet, statsmodels, CatBoost, LightGBM, etc.) into the ElectricBarometer
ecosystem using a consistent scikit-learn-like interface.

All adapters exposed by this package implement:

- `fit(X, y, sample_weight=None)` returning `self`
- `predict(X)` returning a one-dimensional numpy array

This allows ElectricBarometer evaluation, selection, and cloning utilities to
treat native scikit-learn estimators and wrapped external models uniformly.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .contracts.demand_panel.v1.qsr import (
    QSRIntervalPanelDemandSpecV1,
    to_panel_demand_v1,
)
from .models.base import BaseAdapter, clone_model
from .models.catboost import CatBoostAdapter
from .models.lightgbm import LightGBMRegressorAdapter
from .models.prophet import ProphetAdapter
from .models.statsmodels import ArimaAdapter, SarimaxAdapter
from .models.xgboost import XGBoostRegressorAdapter


def _resolve_version() -> str:
    """Return the installed version of the eb-adapters distribution."""
    try:
        return version("eb-adapters")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _resolve_version()

__all__ = [
    "ArimaAdapter",
    "BaseAdapter",
    "CatBoostAdapter",
    "LightGBMRegressorAdapter",
    "ProphetAdapter",
    "QSRIntervalPanelDemandSpecV1",
    "SarimaxAdapter",
    "XGBoostRegressorAdapter",
    "__version__",
    "clone_model",
    "to_panel_demand_v1",
]
