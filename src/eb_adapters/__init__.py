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

from .base import BaseAdapter, _clone_model, clone_model
from .catboost import CatBoostAdapter
from .lightgbm import LightGBMRegressorAdapter
from .prophet import ProphetAdapter
from .statsmodels import ArimaAdapter, SarimaxAdapter

__all__ = [
    "ArimaAdapter",
    "BaseAdapter",
    "CatBoostAdapter",
    "LightGBMRegressorAdapter",
    "ProphetAdapter",
    "SarimaxAdapter",
    "_clone_model",
    "clone_model",
]
