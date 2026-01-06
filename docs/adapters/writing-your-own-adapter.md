# Writing your Own Adapter

This guide shows how to adapt *your* dataset into **Electric Barometer** contract artifacts (for example, `PanelDemandV1`) **without** using `eb-adapters`.

The philosophy is:

- **Contracts (`eb-contracts`)** define *stable interfaces* and *validation semantics*.
- **Adapters (`eb-adapters` or your own code)** are responsible for mapping your raw data into those interfaces.

If you can produce a valid contract artifact, you can use downstream EB tooling (DQC/FPC/RAL diagnostics, evaluation, optimization) that expects that artifact.

## When you should write your own adapter

Write your own adapter when:

- your source schema is unique (internal feature store, vendor feed, etc.)
- you want custom policy (imputation rules, gating rules, filtering)
- you want to keep domain logic private while still adopting EB contracts

You do **not** need `eb-adapters` to use EB contracts.

## The minimum: adapt to `PanelDemandV1`

`PanelDemandV1` represents a normalized demand panel at some time grain.

At minimum, you provide:

- **Identity keys**: one or more columns that uniquely identify an entity (e.g., `site_id`, `forecast_entity_id`)
- **A time index**: either
  - `time_mode="timestamp"` with a `ts_col`, or
  - `time_mode="day_interval"` with `day_col` + `interval_index_col` (+ interval metadata)
- **A target column** `y` (numeric when present)
- **Governance gates** (nullable booleans: `True`, `False`, or `NA`)
  - `is_observable` (was the interval observed / trusted?)
  - `is_possible` (could demand have occurred?)
  - `is_structural_zero` (structurally impossible; must be excluded from training)

EB contracts are intentionally **agnostic**: they avoid enforcing domain-specific policy such as “observable implies y is present”.

## Governance gate semantics (tri-state)

EB treats governance gates as **nullable booleans** with domain:

- `True`
- `False`
- `NA` (unknown)

This matters because `NA` is not the same as `False`.
For example, `is_observable = NA` means “we don't know”, not “it was unobservable”.

### Minimal universal semantic enforced by the contract

The demand-panel contract enforces a minimal semantic relationship:

- If `is_structural_zero == True`:
  - `y` must be `NA`
  - `is_observable` must **not** be `True`

Everything else is up to adapters and upstream policy.

## A simple adapter pattern

Most adapters follow this structure:

1. **Pick a spec** (column mapping + fixed constants)
2. **Copy the input frame**
3. **Create canonical columns**
4. **Normalize governance gates**
5. **(Optional) apply policy** (impute, filter, enforce additional invariants)
6. **Construct the contract artifact** via `from_frame(validate=True)`

### Example: day-interval adapter

```python
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from eb_contracts.contracts.demand_panel.v1.panel_demand import PanelDemandV1


@dataclass(frozen=True, slots=True)
class MyDemandSpec:
    # Identity
    site_col: str = "site_id"
    entity_col: str = "forecast_entity_id"

    # Time (day-interval mode)
    day_col: str = "business_day"
    interval_index_col: str = "interval_index"
    interval_minutes: int = 30
    periods_per_day: int = 48

    # Target
    y_source_col: str = "usage_qty"

    # Governance
    is_observable_col: str = "is_interval_observable"
    is_possible_col: str = "is_possible"
    is_structural_zero_col: str = "is_structural_zero"


def to_nullable_bool(series: pd.Series) -> pd.Series:
    """Normalize a series to pandas nullable boolean dtype."""
    # Keep NA as NA; map common tokens to True/False.
    true_set = {"true", "t", "1", 1, True}
    false_set = {"false", "f", "0", 0, False}

    def _map(v):
        if pd.isna(v):
            return pd.NA
        v_norm = v.lower() if isinstance(v, str) else v
        if v_norm in true_set:
            return True
        if v_norm in false_set:
            return False
        raise ValueError(f"Unrecognized boolean token: {v!r}")

    return series.map(_map).astype("boolean")


def my_to_panel_demand_v1(frame: pd.DataFrame, *, spec: MyDemandSpec | None = None) -> PanelDemandV1:
    if spec is None:
        spec = MyDemandSpec()

    df = frame.copy()

    # Canonical y (do NOT blindly fill missing values)
    df["y"] = pd.to_numeric(df[spec.y_source_col], errors="coerce")

    # Canonical gates (nullable booleans)
    df["is_observable"] = to_nullable_bool(df[spec.is_observable_col])
    df["is_possible"] = to_nullable_bool(df[spec.is_possible_col])
    df["is_structural_zero"] = to_nullable_bool(df[spec.is_structural_zero_col])

    return PanelDemandV1.from_frame(
        frame=df,
        keys=[spec.site_col, spec.entity_col],
        y_col="y",
        time_mode="day_interval",
        day_col=spec.day_col,
        interval_index_col=spec.interval_index_col,
        interval_minutes=spec.interval_minutes,
        periods_per_day=spec.periods_per_day,
        is_observable_col="is_observable",
        is_possible_col="is_possible",
        is_structural_zero_col="is_structural_zero",
        validate=True,
    )
```

## Policy decisions belong in adapters

Two common policy examples:

### 1) Conditional zero-imputation

If your dataset uses scaffolds and you want to impute zero only when you know demand was truly observed:

- If `is_observable == True` and `is_structural_zero != True` and `y is NA` → set `y = 0`
- Otherwise keep `y = NA`

This policy is **domain-specific** and should live in your adapter (or upstream transform), not the contract.

### 2) Filtering structural zeros for training

Training code should typically exclude:

- `is_structural_zero == True`
- and possibly `is_observable != True` (depending on your governance policy)

Again: filter rules belong outside contracts.

## Testing your adapter

You should unit test that:

- you produce the canonical columns (`y`, `is_observable`, `is_possible`, `is_structural_zero`)
- your gates are truly tri-state (`dtype == "boolean"` is recommended)
- your policy does what you expect (imputation, filtering)
- `PanelDemandV1.from_frame(..., validate=True)` succeeds on “happy path”
- bad inputs raise loudly (unknown boolean tokens, missing required columns, etc.)

A good pattern is to maintain “fixture frames” with minimal rows that hit edge cases.

## Versioning guidance

- Treat contract versions (e.g., `V1`) as stable interfaces.
- If you need to evolve your adapter logic, keep the adapter versioned independently.
- Prefer **adding** optional fields/columns over breaking existing meaning.

## Summary

- `eb-contracts` provides the stable artifact surface.
- Anyone can write an adapter that produces a valid artifact.
- Downstream EB tooling can then operate on that artifact reliably.

If you want, share a few rows + your intended semantics and we can sketch an adapter spec + tests tailored to your source schema.
