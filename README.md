# eb-adapters

**eb-adapters** is the official adapter layer for the Electric Barometer ecosystem.  
It provides a unified, lightweight interface that allows external forecasting libraries  
(Prophet, statsmodels, CatBoost, LightGBM, and others) to plug directly into the  
CWSL-based model selection workflow without requiring deep integration changes.

The goal of this package is simple:

> **Make any forecasting engine behave like a scikit-learn regressor —**  
> providing a consistent `.fit(X, y)` / `.predict(X)` API so Electric Barometer  
> can evaluate, compare, and select models using cost-weighted service loss (CWSL).

---

## Features

### Unified Adapter API
All adapters subclass `BaseAdapter`, enforcing a minimal contract:

- `fit(X, y, sample_weight=None)`
- `predict(X)`

Adapters behave like scikit-learn estimators, regardless of the underlying library.

### Robust Model Cloning
`clone_model()` reconstructs models safely using:

1. `sklearn.clone` (when available)  
2. A model’s `get_params()` method  
3. A no-arg constructor fallback  

This allows Electric Barometer to clone models for CV, selection, and refitting.

### Optional Dependency Handling
Heavy libraries (Prophet, LightGBM, statsmodels, CatBoost) are **not required** unless  
you use their corresponding adapters. Import errors raise clean, descriptive messages.

### Fully Tested & Modular
Each adapter has a dedicated test suite, with optional-dependency skipping and  
synthetic-data smoke tests.

---

## Included Adapters

| Adapter | Library | Purpose |
|--------|---------|---------|
| **BaseAdapter** | — | Contract for all adapters; ensures EB compatibility |
| **ProphetAdapter** | `prophet` | Wraps Prophet for timestamp-indexed forecasting |
| **SarimaxAdapter** | `statsmodels` | SARIMAX time-series forecasting |
| **ArimaAdapter** | `statsmodels` | Classic ARIMA(p,d,q) forecasting |
| **CatBoostAdapter** | `catboost` | Gradient boosting for tabular regression |
| **LightGBMRegressorAdapter** | `lightgbm` | Fast gradient-boosted decision trees |

Each adapter supports clean cloning and optional sample weights (if applicable).

---

## Installation

```bash
pip install eb-adapters
```

Optional dependencies can be installed as needed:

```bash
pip install prophet
pip install statsmodels
pip install catboost
pip install lightgbm
```

---

## Basic Usage

```python
from eb_adapters.prophet import ProphetAdapter
from eb_adapters.statsmodels import SarimaxAdapter
from eb_adapters.lightgbm import LightGBMRegressorAdapter

from eb_evaluation import ElectricBarometer

models = {
    "prophet": ProphetAdapter(),
    "sarimax": SarimaxAdapter(order=(1,1,1)),
    "lgbm": LightGBMRegressorAdapter(n_estimators=200)
}

eb = ElectricBarometer(models=models, cu=2.0, co=1.0)
eb.fit(X_train, y_train, X_val, y_val)

print("Winner:", eb.best_name_)
print("Validation CWSL:", eb.validation_cwsl_)
preds = eb.predict(X_test)
```

---

## Philosophy

Adapters should:

- Be **minimal**: no hiding of underlying model hyperparameters
- Be **reconstructible**: all config stored in get_params()
- Be **pure wrappers**: no alteration of core model logic
- Be **transparent**: prediction outputs match the native library

Electric Barometer does the evaluation — adapters simply standardize fit/predict.

---

## License

This project is released under the **BSD-3 License**, consistent with other Electric Barometer ecosystem packages.