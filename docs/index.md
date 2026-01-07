# eb-adapters

`eb-adapters` provides integration layers that adapt Electric Barometer components to external libraries, frameworks, and modeling ecosystems.

This package is responsible for **bridging interfaces**, not defining core metrics, evaluation logic, or optimization policy.

## Scope

This package focuses on:

- Adapting Electric Barometer abstractions to third-party libraries
- Normalizing inputs and outputs across heterogeneous frameworks
- Encapsulating framework-specific quirks behind stable interfaces
- Allowing EB components to be reused without vendor lock-in

It intentionally avoids implementing metrics, models, or workflows.

## Contents

- **Base adapter interfaces**  
  Shared abstractions and contracts used by all adapters

- **Framework adapters**  
  Implementations for specific libraries (e.g. statsmodels, LightGBM, CatBoost, Prophet)

## API reference

- [Base adapters](api/base.md)
- [statsmodels adapter](api/statsmodels.md)
- [LightGBM adapter](api/lightgbm.md)
- [CatBoost adapter](api/catboost.md)
- [Prophet adapter](api/prophet.md)
