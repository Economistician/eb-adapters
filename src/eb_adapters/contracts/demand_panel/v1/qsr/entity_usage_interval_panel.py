"""
QSR interval panel demand adapter.

This module adapts QSR-style intraday interval panels into the PanelDemandV1
contract. It is intentionally brand-agnostic and suitable for any operational
environment with fixed intraday intervals (e.g., QSR, retail, contact centers).

The adapter:
- normalizes source columns into canonical contract columns
- maps ``IS_TRAINABLE`` (when present) to canonical ``is_observable``
- otherwise derives ``is_observable`` from interval/date observability flags
- retains warehouse observability columns as uncoerced metadata when present
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


_INTERVAL_INDEX_ALIASES: tuple[str, ...] = ("HALF_HOUR_NUMBER",)
_INTERVAL_START_TIME_ALIASES: tuple[str, ...] = ("LOCAL_START_TIME",)


@dataclass(frozen=True, slots=True)
class QSRIntervalPanelDemandSpecV1:
    """Column and semantic mapping for a QSR-style interval demand panel."""

    # Identity (source columns)
    site_col: str = "STORE_ID"
    forecast_entity_col: str = "FORECAST_ENTITY_KEY"

    # Time (source columns)
    # Default to day-interval mode (business date + interval index).
    time_mode: str = "day_interval"
    business_day_col: str | None = "BUSINESS_DATE"
    interval_index_col: str | None = "INTERVAL_INDEX"

    # Optional timestamp column (may exist even in day_interval mode)
    interval_start_ts_col: str | None = "INTERVAL_INDEX_START_TIME"

    # Interval metadata
    interval_minutes: int = 30
    periods_per_day: int = 48
    business_day_start_local_minutes: int | None = 240  # 4:00 AM

    # Target (source column)
    # Matches your DDL; override if your table uses a different name.
    y_source_col: str = "FORECAST_ENTITY_DEMAND_QUANTITY"

    # Governance gates (source columns)
    # Trainable is the primary canonical observability source when present.
    # Interval- and date-level flags remain upstream metadata and a fallback.
    is_trainable_col: str | None = "IS_TRAINABLE"
    is_interval_observable_col: str | None = "IS_INTERVAL_OBSERVABLE"
    is_day_observable_col: str | None = "IS_DATE_OBSERVABLE"

    # Optional: mapped when present; otherwise canonical is_structural_zero is False.
    is_structural_zero_col: str | None = "IS_STRUCTURAL_ZERO"

    # Optional: if provided, mapped to canonical "is_possible"; otherwise derived.
    is_possible_col: str | None = None

    # Optional behavior
    impute_zero_when_observable: bool = False


# -------------------------
# Utilities
# -------------------------

_TRUE_VALUES = {"true", "t", "1", 1}
_FALSE_VALUES = {"false", "f", "0", 0}


def _resolve_source_col(
    df: pd.DataFrame,
    primary: str | None,
    aliases: tuple[str, ...],
    *,
    required: bool,
    label: str,
) -> str | None:
    """Return the first present source column among ``primary`` then ``aliases``."""
    candidates = [c for c in (primary, *aliases) if c]
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"No {label} column found. Tried: {candidates}.")
    return None


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
    if series.dtype != object and (
        str(series.dtype) == "boolean" or pd.api.types.is_bool_dtype(series.dtype)
    ):
        return series.astype("boolean")

    na_mask = series.isna()
    true_mask = series.isin(_TRUE_VALUES)
    false_mask = series.isin(_FALSE_VALUES)
    lowered = series.astype("string").str.lower()
    true_mask = true_mask | lowered.isin({"true", "t", "1"})
    false_mask = false_mask | lowered.isin({"false", "f", "0"})
    unrecognized = ~na_mask & ~true_mask & ~false_mask
    if bool(unrecognized.any()):
        v = series.loc[unrecognized].iloc[0]
        raise ValueError(f"Unrecognized boolean value in {name!r}: {v!r}")

    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out = out.mask(true_mask.fillna(False), True)
    out = out.mask(false_mask.fillna(False), False)
    return out.astype("boolean")


def _spec_source_columns(frame: pd.DataFrame, spec: QSRIntervalPanelDemandSpecV1) -> list[str]:
    """Return spec-referenced source columns present on ``frame``, in first-seen order."""
    candidates = [
        spec.site_col,
        spec.forecast_entity_col,
        spec.y_source_col,
        spec.business_day_col,
        spec.interval_index_col,
        spec.interval_start_ts_col,
        spec.is_trainable_col,
        spec.is_interval_observable_col,
        spec.is_day_observable_col,
        spec.is_structural_zero_col,
        spec.is_possible_col,
        *_INTERVAL_INDEX_ALIASES,
        *_INTERVAL_START_TIME_ALIASES,
    ]
    cols: list[str] = []
    seen: set[str] = set()
    for col in candidates:
        if col and col in frame.columns and col not in seen:
            cols.append(col)
            seen.add(col)
    return cols


def _col_present(df: pd.DataFrame, col: str | None) -> bool:
    return bool(col) and col in df.columns


def _resolve_is_observable(df: pd.DataFrame, spec: QSRIntervalPanelDemandSpecV1) -> pd.Series:
    """Resolve canonical ``is_observable`` from trainable, then observability flags.

    Precedence:
    1. ``is_trainable_col`` when present on the frame
    2. Kleene AND of interval- and date-level flags when both columns exist
    3. Whichever single observability column exists
    """
    if _col_present(df, spec.is_trainable_col):
        return _coerce_nullable_bool(_series(df, str(spec.is_trainable_col)), "is_observable")

    interval_col = spec.is_interval_observable_col
    day_col = spec.is_day_observable_col
    interval_present = _col_present(df, interval_col)
    day_present = _col_present(df, day_col)

    if interval_present and day_present:
        interval = _coerce_nullable_bool(_series(df, str(interval_col)), str(interval_col))
        day = _coerce_nullable_bool(_series(df, str(day_col)), str(day_col))
        return (interval & day).astype("boolean")

    if interval_present:
        return _coerce_nullable_bool(_series(df, str(interval_col)), "is_observable")
    if day_present:
        return _coerce_nullable_bool(_series(df, str(day_col)), "is_observable")

    raise ValueError(
        "No observability column found. Provide is_trainable_col, "
        "is_interval_observable_col, or is_day_observable_col in the spec."
    )


def _resolve_is_structural_zero(df: pd.DataFrame, spec: QSRIntervalPanelDemandSpecV1) -> pd.Series:
    """Resolve canonical ``is_structural_zero``.

    Uses ``is_structural_zero_col`` when that column is present. Otherwise fills
    False so frames without the warehouse flag still satisfy PanelDemandV1.
    Unobservable rows are not inferred as structural zeros: that would require
    null ``y`` and fail validation when demand is present.
    """
    if _col_present(df, spec.is_structural_zero_col):
        return _coerce_nullable_bool(
            _series(df, str(spec.is_structural_zero_col)),
            "is_structural_zero",
        )
    return pd.Series(False, index=df.index, dtype="boolean")


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
    - maps ``IS_TRAINABLE`` to ``is_observable`` when present; otherwise uses
      interval/date observability (AND when both exist, else the remaining flag)
    - retains ``IS_DATE_OBSERVABLE`` and ``IS_INTERVAL_OBSERVABLE`` uncoerced
    - maps ``IS_STRUCTURAL_ZERO`` when present; otherwise ``is_structural_zero`` is False
    - preserves NULL demand values by default
    - preserves tri-state gate semantics {True, False, NA}
    - optionally imputes y=0 only for observable, non-structural intervals
    """
    if spec is None:
        spec = QSRIntervalPanelDemandSpecV1()

    source_cols = _spec_source_columns(frame, spec)
    df = frame.loc[:, source_cols].copy() if source_cols else frame.iloc[:, 0:0].copy()

    # --- canonical identity
    df["site_id"] = _series(df, spec.site_col)
    df["forecast_entity_id"] = _series(df, spec.forecast_entity_col)

    # --- canonical target
    y_raw = _series(df, spec.y_source_col)
    df["y"] = pd.to_numeric(y_raw, errors="coerce")

    # --- canonical governance gates (nullable booleans)
    # Warehouse observability columns stay on the frame as uncoerced metadata.
    df["is_observable"] = _resolve_is_observable(df, spec)
    df["is_structural_zero"] = _resolve_is_structural_zero(df, spec)

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
    interval_index_col = _resolve_source_col(
        df,
        spec.interval_index_col,
        _INTERVAL_INDEX_ALIASES,
        required=time_mode == "day_interval",
        label="interval index",
    )
    interval_start_ts_col = _resolve_source_col(
        df,
        spec.interval_start_ts_col,
        _INTERVAL_START_TIME_ALIASES,
        required=time_mode == "timestamp",
        label="interval start timestamp",
    )

    if time_mode == "day_interval":
        if not spec.business_day_col or not interval_index_col:
            raise ValueError(
                "time_mode='day_interval' requires business_day_col and interval_index_col."
            )

        return PanelDemandV1.from_frame(
            frame=df,
            keys=["site_id", "forecast_entity_id"],
            y_col="y",
            time_mode="day_interval",
            day_col=spec.business_day_col,
            interval_index_col=interval_index_col,
            ts_col=interval_start_ts_col,
            interval_minutes=spec.interval_minutes,
            periods_per_day=spec.periods_per_day,
            business_day_start_local_minutes=spec.business_day_start_local_minutes,
            is_observable_col="is_observable",
            is_possible_col="is_possible",
            is_structural_zero_col="is_structural_zero",
            validate=validate,
        )

    if time_mode == "timestamp":
        if not interval_start_ts_col:
            raise ValueError("time_mode='timestamp' requires interval_start_ts_col in the frame.")

        return PanelDemandV1.from_frame(
            frame=df,
            keys=["site_id", "forecast_entity_id"],
            y_col="y",
            time_mode="timestamp",
            ts_col=interval_start_ts_col,
            is_observable_col="is_observable",
            is_possible_col="is_possible",
            is_structural_zero_col="is_structural_zero",
            validate=validate,
        )

    raise ValueError(f"Unrecognized time_mode: {time_mode!r}")


# Backwards-compatible alias (if you already called the old name elsewhere).
qsr_to_panel_demand_v1 = to_panel_demand_v1
