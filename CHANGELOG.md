# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Unblock CI after the System Release 0.2.10 PyPI publish race (`eb-metrics==0.2.9` was missing during the original audit install).

## [0.2.5] - 2026-08-24

### Added

- QSR interval adapter maps `IS_TRAINABLE` to canonical `is_observable` when present. Absent that column, `is_observable` is the Kleene AND of `IS_INTERVAL_OBSERVABLE` and `IS_DATE_OBSERVABLE` when both exist. Warehouse observability columns are retained uncoerced on the panel frame.
- QSR interval adapter treats `IS_STRUCTURAL_ZERO` as optional. Missing or `None` source columns default canonical `is_structural_zero` to `False`.

### Changed

- QSR interval adapter source defaults now match the warehouse schema (`FORECAST_ENTITY_KEY`, `BUSINESS_DATE`, `INTERVAL_INDEX`, `INTERVAL_INDEX_START_TIME`, `FORECAST_ENTITY_DEMAND_QUANTITY`, `IS_DATE_OBSERVABLE`). `HALF_HOUR_NUMBER` and `LOCAL_START_TIME` remain accepted aliases.
- `_coerce_nullable_bool` maps unrecognized boolean tokens to `<NA>` instead of raising, so invalid gate strings cannot crash `to_panel_demand_v1`.
- Pinned sibling packages to System Release 0.2.10 (`eb-metrics==0.2.9`, `eb-contracts==0.2.3`).

### Performance

- `to_panel_demand_v1` copies only spec-referenced source columns from the warehouse frame.
- `_coerce_nullable_bool` uses `Series.isin` / boolean masks instead of a Python per-cell map.
- `to_panel_demand_v1` projects source columns without a defensive deep copy; string-coercion of bool encodings runs only on values still unrecognized after `isin()`.
- `impute_zero_when_observable` uses `Series.where` instead of chained `.loc` / `.fillna`.

## [0.2.4] - 2026-08-23

### Changed

- Pinned runtime floors `numpy>=1.24` and `pandas>=2.0`.
- Removed type-checker narration comments; rephrased first-person docstrings.
- Tightened README Overview; removed cloned Role section.
- Dropped `_clone_model` from the public root `__all__`; use `clone_model`.
- Changelog version header now matches `pyproject.toml` (`0.2.4`).
- Pinned sibling Electric Barometer packages to exact System Release 0.2.9 versions.

### Fixed

- Declared the runtime `eb-contracts` dependency used by demand-panel adapters.

### Breaking Changes

- Removed runtime dependency on `eb-evaluation` to break circular installation cycle.

### Added

- `__version__` on the package root.
- Re-exported `QSRIntervalPanelDemandSpecV1` and `to_panel_demand_v1` from the package root.
- Exposed real adapter classes (`ArimaAdapter`, `SarimaxAdapter`, `XGBoostRegressorAdapter`) with `fit(X, y)` / `predict(X)` signatures.
- Added `test` and `all` extras to `pyproject.toml`.
