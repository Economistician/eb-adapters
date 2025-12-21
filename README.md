# Electric Barometer Adapters (`eb-adapters`)

![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Python Versions](https://img.shields.io/badge/Python-3.10%2B-blue)
[![Docs](https://img.shields.io/badge/docs-electric--barometer-blue)](https://economistician.github.io/eb-docs/)
![Project Status](https://img.shields.io/badge/Status-Alpha-yellow)

This repository contains the **model adapter layer** of the *Electric Barometer*
ecosystem.

`eb-adapters` provides standardized interfaces that adapt common forecasting and
machine-learning libraries to the Electric Barometer evaluation and readiness
frameworks, enabling consistent metric computation and model comparison across
heterogeneous modeling approaches.

Conceptual definitions and evaluation philosophy are maintained in the companion
research repository:
**`eb-papers`**.

---

## Role Within Electric Barometer

Within the Electric Barometer ecosystem:

- **`eb-papers`** defines *concepts, frameworks, and meaning*
- **`eb-metrics`** implements *individual metrics*
- **`eb-evaluation`** orchestrates *evaluation workflows*
- **`eb-adapters`** standardizes *model interfaces*

This repository focuses on bridging external modeling libraries into a common,
evaluation-ready form.

---

## What This Library Provides

- **A common adapter base class** defining a unified interface for forecast models
- **Library-specific adapters** for popular forecasting and ML frameworks
- **Consistent prediction and evaluation hooks** compatible with EB metrics
- **Extensible patterns** for adding new model families without changing
  downstream evaluation logic

Current adapters include support for:
- CatBoost
- LightGBM
- Prophet
- Statsmodels-based models

---

## Scope

This repository focuses on **model adaptation**, not model training algorithms or
metric definitions.

**In scope:**
- Wrapping external model APIs behind a consistent adapter interface
- Standardizing prediction outputs for evaluation
- Providing compatibility with EB evaluation pipelines

**Out of scope:**
- Metric definitions and loss formulations (see `eb-metrics`)
- Evaluation orchestration logic (see `eb-evaluation`)
- Model training methodology or hyperparameter optimization
- Conceptual framework definitions (see `eb-papers`)

---

## Installation

Once published, the package will be installable via PyPI:

```bash
pip install eb-adapters
```

For development or local use:

```bash
pip install -e .
```

---

## Package Structure

The repository follows a clean, modern Python package layout:

```text
eb-adapters/
├── src/eb_adapters/
│   ├── base.py            # Abstract adapter base class
│   ├── catboost.py        # CatBoost model adapter
│   ├── lightgbm.py        # LightGBM model adapter
│   ├── prophet.py         # Prophet model adapter
│   └── statsmodels.py     # Statsmodels-based adapters
│
├── tests/
│   └── adapters/          # Unit tests for adapter implementations
│
├── pyproject.toml         # Build and dependency configuration
├── README.md              # Project documentation
└── LICENSE                # BSD-3-Clause license
```

---

## Relationship to Other EB Repositories

- **`eb-papers`**  
  Source of truth for conceptual definitions and evaluation philosophy.

- **`eb-metrics`**  
  Provides the metric implementations used during evaluation.

- **`eb-evaluation`**  
  Orchestrates evaluation workflows using adapted models.

- **`eb-adapters`**  
  Ensures heterogeneous models can be evaluated consistently.

When discrepancies arise, conceptual intent in `eb-papers` should be treated as
authoritative.

---

## Development and Testing

Tests are located under the `tests/` directory and mirror adapter coverage.

To run the test suite:

```bash
pytest
```

---

## Status

This package is under active development.
Public APIs may evolve prior to the first stable release.