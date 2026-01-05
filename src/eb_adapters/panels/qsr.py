"""
QSR panel demand adapter.

This module adapts QSR-style intraday demand datasets into the PanelDemandV1
contract. It is intentionally brand-agnostic and suitable for any quick-service
restaurant or similar operational environment with fixed intraday intervals.

The adapter:
- normalizes source columns into canonical contract columns
- coerces governance gates to boolean
- preserves NULL demand semantics
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from eb_contracts.demand.v1 import PanelDemandV1

# -------------------------
# Spec
# -------------------------


@dataclass(frozen=True, slots=True)
class QSRPanelDemandSpecV1:
    """Column and semantic mapping for a QSR demand panel."""

    # Identity
    store_col: str = "STORE_ID"
    entity_col: str = "FORECAST_ENTITY_ID"

    # Time
    business_day_col: str = "BUSINESS_DAY"
    interval_index_col: str = "INTERVAL_30_INDEX"

    interval_minutes: int = 30
    periods_per_day: int = 48
    business_day_start_local_minutes: int = 240  # 4:00 AM

    # Target
    demand_col: str = "ACTUAL_COMMODITY_USAGE_QTY"

    # Governance gates
    is_observable_col: str = "IS_INTERVAL_OBSERVABLE"
    is_possible_col: str = "HAS_DEMAND"
    is_structural_zero_col: str = "IS_STRUCTURAL_ZERO"


# -------------------------
# Utilities
# -------------------------

_TRUE_VALUES = {"true", "t", "1", 1}
_FALSE_VALUES = {"false", "f", "0", 0}


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a single column as a Series (guards against duplicate columns)."""
    s = df[col]
    if isinstance(s, pd.DataFrame):
        raise ValueError(
            f"Expected column {col!r} to be a Series, got DataFrame. "
            "This usually indicates duplicate column names."
        )
    return s


def _coerce_bool(series: pd.Series, name: str) -> pd.Series:
    """Coerce a series to boolean with strict, explicit semantics."""

    def _map(v):
        if pd.isna(v):
            return False
        v_norm = v.lower() if isinstance(v, str) else v
        if v_norm in _TRUE_VALUES:
            return True
        if v_norm in _FALSE_VALUES:
            return False
        raise ValueError(f"Unrecognized boolean value in {name!r}: {v!r}")

    return series.map(_map).astype(bool)


# -------------------------
# Adapter
# -------------------------


def qsr_to_panel_demand_v1(
    frame: pd.DataFrame,
    *,
    spec: QSRPanelDemandSpecV1 | None = None,
    validate: bool = True,
) -> PanelDemandV1:
    """
    Adapt a QSR-style dataframe to PanelDemandV1.

    This function:
    - creates canonical contract columns
    - preserves NULL demand values
    - enforces clean boolean governance gates
    """
    if spec is None:
        spec = QSRPanelDemandSpecV1()

    df = frame.copy()

    # --- canonical columns
    df["y"] = pd.to_numeric(_series(df, spec.demand_col), errors="coerce")

    df["is_observable"] = _coerce_bool(
        _series(df, spec.is_observable_col),
        "is_observable",
    )
    df["is_possible"] = _coerce_bool(
        _series(df, spec.is_possible_col),
        "is_possible",
    )
    df["is_structural_zero"] = _coerce_bool(
        _series(df, spec.is_structural_zero_col),
        "is_structural_zero",
    )

    return PanelDemandV1.from_frame(
        frame=df,
        # Identity
        keys=[
            spec.store_col,
            spec.entity_col,
        ],
        # Target
        y_col="y",
        # Time semantics
        time_mode="day_interval",
        day_col=spec.business_day_col,
        interval_index_col=spec.interval_index_col,
        # Interval metadata
        interval_minutes=spec.interval_minutes,
        periods_per_day=spec.periods_per_day,
        business_day_start_local_minutes=spec.business_day_start_local_minutes,
        # Governance gates (canonical)
        is_observable_col="is_observable",
        is_possible_col="is_possible",
        is_structural_zero_col="is_structural_zero",
        validate=validate,
    )
