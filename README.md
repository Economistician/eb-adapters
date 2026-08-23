# Electric Barometer · Adapters (`eb-adapters`)

[![CI](https://github.com/Economistician/eb-adapters/actions/workflows/ci.yml/badge.svg)](https://github.com/Economistician/eb-adapters/actions/workflows/ci.yml)
![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/eb-adapters)
![PyPI](https://img.shields.io/pypi/v/eb-adapters)

Adapter interfaces that normalize forecasting model APIs for consistent evaluation within the Electric Barometer ecosystem.

---

## Overview

`eb-adapters` wraps heterogeneous forecasting libraries behind a common `fit` / `predict` contract for uniform evaluation. It isolates framework-specific behavior; metrics and evaluation live in sibling packages.

---

## Installation

`eb-adapters` is distributed as a standard Python package.

```bash
pip install eb-adapters
```

The package supports Python 3.11 and later.

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
import numpy as np
from eb_adapters import ArimaAdapter, SarimaxAdapter

y_train = np.array([10.0, 12.0, 11.0, 13.0, 14.0, 15.0], dtype=float)
X_train = np.arange(len(y_train), dtype=float).reshape(-1, 1)
X_horizon = np.arange(7, dtype=float).reshape(-1, 1)

arima = ArimaAdapter(order=(1, 1, 1))
arima.fit(X_train, y_train)
y_pred = arima.predict(X_horizon)

sarimax = SarimaxAdapter(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
sarimax.fit(X_train, y_train)
y_pred = sarimax.predict(X_horizon)
```

The same `fit(X, y)` / `predict(X)` contract is used with tree-based models:

```python
import numpy as np
from eb_adapters import XGBoostRegressorAdapter

X_train = np.arange(20, dtype=float).reshape(-1, 1)
y_train = (2.0 * X_train[:, 0] + 1.0)
X_future = np.arange(20, 27, dtype=float).reshape(-1, 1)

adapter = XGBoostRegressorAdapter(n_estimators=50, max_depth=3)
adapter.fit(X_train, y_train)
y_pred = adapter.predict(X_future)
```

---

## License

BSD 3-Clause License.
© 2026 Kyle Corrie.
