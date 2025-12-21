from __future__ import annotations

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

from .base import BaseAdapter, _clone_model, clone_model
from .prophet import ProphetAdapter
from .statsmodels import SarimaxAdapter, ArimaAdapter
from .catboost import CatBoostAdapter
from .lightgbm import LightGBMRegressorAdapter

__all__ = [
    "BaseAdapter",
    "_clone_model",
    "clone_model",
    "ProphetAdapter",
    "SarimaxAdapter",
    "ArimaAdapter",
    "CatBoostAdapter",
    "LightGBMRegressorAdapter",
]