# eb-adapters

**eb-adapters** provides a small, focused collection of adapter classes that
wrap external forecasting and regression libraries so they can be used
interchangeably inside the ElectricBarometer ecosystem.

The goal of this package is *interface normalization*, not modeling innovation.
Each adapter exposes a consistent, scikit-learn-like API:

- `fit(X, y, sample_weight=None)` → returns `self`
- `predict(X)` → returns a one-dimensional numpy array

This allows ElectricBarometer evaluation, selection, and cost-aware comparison
utilities to operate uniformly across native scikit-learn estimators and
non-sklearn forecasting engines.

---

## What this package does

- Wraps external libraries (Prophet, statsmodels, CatBoost, LightGBM, etc.)
- Normalizes their interfaces to a common `.fit / .predict` contract
- Preserves initialization parameters so models are cloneable
- Keeps optional dependencies truly optional

**eb-adapters does not**:
- Define loss functions or evaluation metrics
- Perform model selection or ranking
- Implement forecasting logic itself

Those responsibilities live in other ElectricBarometer packages.

---

## Adapter philosophy

Adapters are intentionally thin:

- Minimal logic beyond input normalization
- No hidden state outside the wrapped model
- Clear, documented assumptions about how `X` and `y` are interpreted
- Safe to import even when optional dependencies are not installed

If a model cannot reasonably conform to this contract, it likely does not belong
in this package.

---

## Available adapters

Current adapters include:

- Prophet (univariate time-series forecasting)
- statsmodels ARIMA / SARIMAX
- CatBoost regressor
- LightGBM regressor

Each adapter is documented in the API section and generated directly from
NumPy-style docstrings in the source code.

---

## Relationship to other ElectricBarometer packages

- **eb-metrics**: defines cost-aware error and loss functions
- **eb-evaluation**: performs model evaluation, comparison, and selection
- **eb-adapters**: makes external models compatible with those workflows

This separation keeps responsibilities clean and composable.

---

## API documentation

See the **API** section for detailed adapter reference material generated
automatically from the source code.