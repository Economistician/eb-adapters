# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `__version__` on the package root.
- Re-exported `QSRIntervalPanelDemandSpecV1` and `to_panel_demand_v1` from the package root.

### Changed

- Pinned runtime floors `numpy>=1.24` and `pandas>=2.0`.
- Removed type-checker narration comments; rephrased first-person docstrings.
- Tightened README Overview; removed cloned Role section.
- Dropped `_clone_model` from the public root `__all__`; use `clone_model`.
- Changelog version header now matches `pyproject.toml` (`0.2.4`).

### Fixed

- Declared the runtime `eb-contracts>=0.2,<0.3` dependency used by demand-panel adapters.

## [0.2.4] - 2026-08-22

### Breaking Changes

- Removed runtime dependency on `eb-evaluation` to break circular installation cycle.

### Added

- Exposed real adapter classes (`ArimaAdapter`, `SarimaxAdapter`, `XGBoostRegressorAdapter`) with `fit(X, y)` / `predict(X)` signatures.
- Added `test` and `all` extras to `pyproject.toml`.
