"""
eb_adapters: Adapter classes for bringing external forecast engines
(Prophet, statsmodels, CatBoost, LightGBM, etc.) into the Electric
Barometer ecosystem with a consistent .fit/.predict interface.
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