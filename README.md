# Electric Barometer · Adapters (`eb-adapters`)

[![CI](https://github.com/Economistician/eb-adapters/actions/workflows/ci.yml/badge.svg)](https://github.com/Economistician/eb-adapters/actions/workflows/ci.yml)
![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/eb-adapters)
![PyPI](https://img.shields.io/pypi/v/eb-adapters)

Adapter interfaces that normalize forecasting model APIs for consistent evaluation within the Electric Barometer ecosystem.

---

## Overview

`eb-adapters` provides a thin adapter layer that normalizes the interfaces of heterogeneous forecasting libraries into a consistent, evaluation-ready API. It enables models from different frameworks to be trained, predicted, and evaluated using a common contract, without requiring downstream systems to account for library-specific behaviors.

Within the Electric Barometer ecosystem, `eb-adapters` serves as the bridge between model implementations and evaluation logic. By isolating framework-specific details behind stable adapter interfaces, the package allows forecasting models to be compared, selected, and assessed consistently across diverse modeling stacks, while remaining usable outside the ecosystem in standalone evaluation workflows.

---

## Role in the Electric Barometer Ecosystem

`eb-adapters` defines the model interface normalization layer used throughout the Electric Barometer ecosystem. It is responsible for wrapping heterogeneous forecasting libraries behind a consistent training and prediction contract, allowing models from different frameworks to be evaluated in a uniform manner.

This package focuses exclusively on adapting model APIs and handling framework-specific behaviors. It does not perform feature construction, forecast evaluation, metric definition, or decision logic. Those responsibilities are handled by adjacent layers in the ecosystem that generate inputs, apply evaluation logic, or interpret results.

By separating model integration concerns from evaluation and metric semantics, `eb-adapters` enables fair, reproducible comparison of forecasting approaches across diverse modeling stacks, while keeping downstream systems agnostic to underlying implementation details.

---

## Installation

`eb-adapters` is distributed as a standard Python package.

```bash
pip install eb-adapters
```

The package supports Python 3.10 and later.

---

## Core Concepts

- **Interface normalization** — Forecasting models from different libraries are wrapped behind a common training and prediction contract, enabling uniform downstream evaluation.
- **Thin adaptation layer** — Adapters aim to be minimal and non-invasive, preserving native model behavior while standardizing how models are invoked.
- **Framework isolation** — Library-specific configuration, defaults, and quirks are contained within adapters, preventing leakage into evaluation or orchestration layers.
- **Explicit lifecycle boundaries** — Model fitting, prediction, and state management are clearly separated to support reproducibility and controlled execution.
- **Comparability over abstraction** — Adapters do not attempt to hide meaningful differences between modeling approaches; they exist to make comparison feasible, not to enforce uniformity.

---

## Minimal Example

The example below shows how a forecasting model is wrapped behind a standardized adapter interface so it can be trained and evaluated consistently alongside other models.

```python
from eb_adapters.statsmodels import StatsModelsAdapter
from statsmodels.tsa.arima.model import ARIMA

# Define a native model
model = ARIMA

# Wrap the model with an adapter
adapter = StatsModelsAdapter(
    model_class=model,
    model_kwargs={"order": (1, 1, 1)}
)

# Fit and predict using a standardized interface
adapter.fit(y_train)
y_pred = adapter.predict(horizon=7)
```

The same interface can be used with tree-based or machine-learning models via a different adapter:

```python
from eb_adapters.xgboost import XGBoostAdapter
from xgboost import XGBRegressor

# Define a native model
model = XGBRegressor

# Wrap the model with an adapter
adapter = XGBoostAdapter(
    model_class=model,
    model_kwargs={"n_estimators": 200, "max_depth": 4}
)

# Fit and predict using the same contract
adapter.fit(X_train, y_train)
y_pred = adapter.predict(X_future)
```

---

## License

BSD 3-Clause License.  
© 2025 Kyle Corrie.