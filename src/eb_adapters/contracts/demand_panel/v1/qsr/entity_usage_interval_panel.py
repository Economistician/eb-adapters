"""
QSR interval panel demand adapter.

This module adapts QSR-style intraday interval panels into the PanelDemandV1
contract. It is intentionally brand-agnostic and suitable for any operational
environment with fixed intraday intervals (e.g., QSR, retail, contact centers).

The adapter:
- normalizes source columns into canonical contract columns
- preserves tri-state governance semantics for gates {True, False, NA}
- preserves NULL demand semantics (no implicit imputation)
- optionally imputes zero demand only for observable, non-structural intervals
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from eb_contracts.contracts.demand_panel.v1.panel_demand import PanelDemandV1

# -------------------------
# Spec
# -------------------------


@dataclass(frozen=True, slots=True)
class QSRIntervalPanelDemandSpecV1:
    """Column and semantic mapping for a QSR-style interval demand panel."""

    # Identity (source columns)
    site_col: str = "STORE_ID"
    forecast_entity_col: str = "FORECAST_ENTITY_ID"

    # Time (source columns)
    # Default to day-interval mode (business day + interval index).
    time_mode: str = "day_interval"
    business_day_col: str | None = "BUSINESS_DAY"
    interval_index_col: str | None = "INTERVAL_30_INDEX"

    # Optional timestamp column (may exist even in day_interval mode)
    interval_start_ts_col: str | None = "INTERVAL_START_TS"

    # Interval metadata
    interval_minutes: int = 30
    periods_per_day: int = 48
    business_day_start_local_minutes: int | None = 240  # 4:00 AM

    # Target (source column)
    # Matches your DDL; override if your table uses a different name.
    y_source_col: str = "LABEL_COMMODITY_USAGE_QTY"

    # Governance gates (source columns)
    # Interval-level observability is preferred; day-level is a fallback.
    is_interval_observable_col: str | None = "IS_INTERVAL_OBSERVABLE"
    is_day_observable_col: str | None = "IS_DAY_OBSERVABLE"

    is_structural_zero_col: str = "IS_STRUCTURAL_ZERO"

    # Optional: if provided, mapped to canonical "is_possible"; otherwise derived.
    is_possible_col: str | None = None

    # Optional behavior
    impute_zero_when_observable: bool = False


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


def _coerce_nullable_bool(series: pd.Series, name: str) -> pd.Series:
    """
    Coerce a series into pandas nullable boolean dtype ("boolean").

    Preserves tri-state semantics:
      - NA stays NA (unknown)
      - recognized true/false encodings map to True/False
    """

    def _map(v):
        if pd.isna(v):
            return pd.NA
        v_norm = v.lower() if isinstance(v, str) else v
        if v_norm in _TRUE_VALUES:
            return True
        if v_norm in _FALSE_VALUES:
            return False
        raise ValueError(f"Unrecognized boolean value in {name!r}: {v!r}")

    return series.map(_map).astype("boolean")


# -------------------------
# Adapter
# -------------------------


def to_panel_demand_v1(
    frame: pd.DataFrame,
    *,
    spec: QSRIntervalPanelDemandSpecV1 | None = None,
    validate: bool = True,
) -> PanelDemandV1:
    """
    Adapt a QSR-style interval dataframe to PanelDemandV1.

    This function:
    - creates canonical contract columns: site_id, forecast_entity_id, y, gates
    - preserves NULL demand values by default
    - preserves tri-state gate semantics {True, False, NA}
    - optionally imputes y=0 only for observable, non-structural intervals
    """
    if spec is None:
        spec = QSRIntervalPanelDemandSpecV1()

    df = frame.copy()

    # --- canonical identity
    df["site_id"] = _series(df, spec.site_col)
    df["forecast_entity_id"] = _series(df, spec.forecast_entity_col)

    # --- canonical target
    y_raw = _series(df, spec.y_source_col)
    df["y"] = pd.to_numeric(y_raw, errors="coerce")

    # --- canonical governance gates (nullable booleans)
    # Observability: prefer interval-level; fallback to day-level if needed.
    src_obs_col: str | None = spec.is_interval_observable_col
    if src_obs_col is None or src_obs_col not in df.columns:
        src_obs_col = spec.is_day_observable_col

    if not src_obs_col or src_obs_col not in df.columns:
        raise ValueError(
            "No observability column found. Provide is_interval_observable_col "
            "or is_day_observable_col in the spec."
        )

    df["is_observable"] = _coerce_nullable_bool(_series(df, src_obs_col), "is_observable")
    df["is_structural_zero"] = _coerce_nullable_bool(
        _series(df, spec.is_structural_zero_col),
        "is_structural_zero",
    )

    # is_possible: optional override; otherwise default to is_observable
    if spec.is_possible_col and spec.is_possible_col in df.columns:
        df["is_possible"] = _coerce_nullable_bool(
            _series(df, spec.is_possible_col),
            "is_possible",
        )
    else:
        df["is_possible"] = df["is_observable"]

    # Optional: impute y=0 only where we *know* it's observable and not structural.
    if spec.impute_zero_when_observable:
        obs = df["is_observable"]
        structural = df["is_structural_zero"]
        mask = (obs == True) & (structural != True)  # noqa: E712
        df.loc[mask, "y"] = df.loc[mask, "y"].fillna(0)

    # --- build contract
    time_mode = spec.time_mode

    if time_mode == "day_interval":
        if not spec.business_day_col or not spec.interval_index_col:
            raise ValueError(
                "time_mode='day_interval' requires business_day_col and interval_index_col."
            )

        return PanelDemandV1.from_frame(
            frame=df,
            keys=["site_id", "forecast_entity_id"],
            y_col="y",
            time_mode="day_interval",
            day_col=spec.business_day_col,
            interval_index_col=spec.interval_index_col,
            ts_col=spec.interval_start_ts_col if spec.interval_start_ts_col in df.columns else None,
            interval_minutes=spec.interval_minutes,
            periods_per_day=spec.periods_per_day,
            business_day_start_local_minutes=spec.business_day_start_local_minutes,
            is_observable_col="is_observable",
            is_possible_col="is_possible",
            is_structural_zero_col="is_structural_zero",
            validate=validate,
        )

    if time_mode == "timestamp":
        if not spec.interval_start_ts_col or spec.interval_start_ts_col not in df.columns:
            raise ValueError("time_mode='timestamp' requires interval_start_ts_col in the frame.")

        return PanelDemandV1.from_frame(
            frame=df,
            keys=["site_id", "forecast_entity_id"],
            y_col="y",
            time_mode="timestamp",
            ts_col=spec.interval_start_ts_col,
            is_observable_col="is_observable",
            is_possible_col="is_possible",
            is_structural_zero_col="is_structural_zero",
            validate=validate,
        )

    raise ValueError(f"Unrecognized time_mode: {time_mode!r}")


# Backwards-compatible alias (if you already called the old name elsewhere).
qsr_to_panel_demand_v1 = to_panel_demand_v1
