"""
Model adapters.

This subpackage contains thin adapter wrappers around third-party forecasting
and regression libraries (for example, statsmodels, Prophet, CatBoost, LightGBM,
and XGBoost). Adapters expose a scikit-learn-like interface so they can be used
interchangeably in Electric Barometer workflows.

Exports here are provided for convenience so callers can import adapters via:

    from eb_adapters.models import LightGBMRegressorAdapter
"""

from __future__ import annotations

from .base import BaseAdapter, _clone_model, clone_model
from .catboost import CatBoostAdapter
from .lightgbm import LightGBMRegressorAdapter
from .prophet import ProphetAdapter
from .statsmodels import ArimaAdapter, SarimaxAdapter
from .xgboost import XGBoostRegressorAdapter

__all__ = [
    "ArimaAdapter",
    "BaseAdapter",
    "CatBoostAdapter",
    "LightGBMRegressorAdapter",
    "ProphetAdapter",
    "SarimaxAdapter",
    "XGBoostRegressorAdapter",
    "_clone_model",
    "clone_model",
]
