"""
Unit tests for the QSR interval panel adapter.

These tests validate that:
- the adapter produces canonical columns
- governance gates preserve tri-state semantics (True/False/NA)
- unknown boolean tokens raise loudly
- NULL demand is preserved (not imputed) by default
- optional imputation (when enabled) only fills observable, non-structural intervals
- custom column mapping via spec works
- IS_TRAINABLE takes precedence for canonical is_observable
- date and interval flags AND when IS_TRAINABLE is absent
- warehouse observability columns are retained uncoerced
- IS_STRUCTURAL_ZERO may be omitted; canonical is_structural_zero defaults to False
"""

from __future__ import annotations

import pandas as pd
import pytest

from eb_adapters.contracts.demand_panel.v1.qsr.entity_usage_interval_panel import (
    QSRIntervalPanelDemandSpecV1,
    to_panel_demand_v1,
)


def _make_base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STORE_ID": [101, 101, 101],
            "FORECAST_ENTITY_KEY": [1, 1, 1],
            "BUSINESS_DATE": ["2025-05-01", "2025-05-01", "2025-05-01"],
            "INTERVAL_INDEX": [0, 1, 2],
            # Use the DDL-aligned default for the adapter/spec
            "FORECAST_ENTITY_DEMAND_QUANTITY": [None, 4, 8],
            # Gate tokens include strings and nulls to verify tri-state behavior
            "IS_INTERVAL_OBSERVABLE": ["TRUE", None, "FALSE"],
            "IS_STRUCTURAL_ZERO": [0, 0, 0],
        }
    )


def test_adapter_happy_path_creates_panel_and_canonical_columns() -> None:
    df = _make_base_frame()

    panel = to_panel_demand_v1(df, validate=True)

    # Contract object basics
    assert panel.time_mode == "day_interval"
    assert panel.keys == ("site_id", "forecast_entity_id")
    assert panel.y_col == "y"

    # Canonical columns exist
    out = panel.frame
    for col in (
        "site_id",
        "forecast_entity_id",
        "y",
        "is_observable",
        "is_possible",
        "is_structural_zero",
    ):
        assert col in out.columns


def test_adapter_gates_are_nullable_booleans_and_preserve_na() -> None:
    df = _make_base_frame()

    panel = to_panel_demand_v1(df, validate=False)
    out = panel.frame

    # pandas nullable boolean dtype prints as "boolean"
    assert str(out["is_observable"].dtype) == "boolean"
    assert str(out["is_possible"].dtype) == "boolean"
    assert str(out["is_structural_zero"].dtype) == "boolean"

    # Values coerced as expected (NA preserved)
    assert out["is_observable"].tolist() == [True, pd.NA, False]

    # is_possible defaults to is_observable unless overridden
    assert out["is_possible"].tolist() == [True, pd.NA, False]

    # 0 -> False
    assert out["is_structural_zero"].tolist() == [False, False, False]


def test_adapter_raises_on_unknown_gate_token() -> None:
    df = _make_base_frame()
    df.loc[0, "IS_INTERVAL_OBSERVABLE"] = "MAYBE"

    with pytest.raises(ValueError, match="Unrecognized boolean value"):
        to_panel_demand_v1(df, validate=False)


def test_adapter_preserves_null_y_by_default() -> None:
    df = _make_base_frame()

    panel = to_panel_demand_v1(df, validate=False)
    out = panel.frame

    # First row should remain null in canonical y
    assert pd.isna(out.loc[0, "y"])


def test_adapter_optional_impute_zero_only_when_observable_and_not_structural() -> None:
    df = _make_base_frame()

    # Make row0 observable True + y null -> should impute to 0 when enabled
    df.loc[0, "IS_INTERVAL_OBSERVABLE"] = "TRUE"
    df.loc[0, "FORECAST_ENTITY_DEMAND_QUANTITY"] = None

    # Make row1 observable NA + y null -> should remain NA
    df.loc[1, "IS_INTERVAL_OBSERVABLE"] = None
    df.loc[1, "FORECAST_ENTITY_DEMAND_QUANTITY"] = None

    # Make row2 structural True + y null + observable True -> should NOT impute
    df.loc[2, "IS_STRUCTURAL_ZERO"] = 1
    df.loc[2, "IS_INTERVAL_OBSERVABLE"] = "TRUE"
    df.loc[2, "FORECAST_ENTITY_DEMAND_QUANTITY"] = None

    spec = QSRIntervalPanelDemandSpecV1(impute_zero_when_observable=True)

    panel = to_panel_demand_v1(df, spec=spec, validate=False)
    out = panel.frame

    assert out.loc[0, "y"] == 0
    assert pd.isna(out.loc[1, "y"])
    assert pd.isna(out.loc[2, "y"])


def test_adapter_supports_custom_column_mapping_via_spec() -> None:
    df = _make_base_frame().rename(
        columns={
            "STORE_ID": "site",
            "FORECAST_ENTITY_KEY": "entity",
            "BUSINESS_DATE": "day",
            "INTERVAL_INDEX": "idx",
            "FORECAST_ENTITY_DEMAND_QUANTITY": "usage",
            "IS_INTERVAL_OBSERVABLE": "obs",
            "IS_STRUCTURAL_ZERO": "struct0",
        }
    )

    spec = QSRIntervalPanelDemandSpecV1(
        site_col="site",
        forecast_entity_col="entity",
        business_day_col="day",
        interval_index_col="idx",
        y_source_col="usage",
        is_interval_observable_col="obs",
        is_structural_zero_col="struct0",
    )

    panel = to_panel_demand_v1(df, spec=spec, validate=False)

    assert panel.keys == ("site_id", "forecast_entity_id")
    assert panel.time_mode == "day_interval"
    assert panel.y_col == "y"

    out = panel.frame
    assert out["site_id"].tolist() == [101, 101, 101]
    assert out["forecast_entity_id"].tolist() == [1, 1, 1]


def test_adapter_falls_back_to_half_hour_number_alias() -> None:
    df = _make_base_frame().rename(columns={"INTERVAL_INDEX": "HALF_HOUR_NUMBER"})
    panel = to_panel_demand_v1(df, validate=True)
    assert panel.interval_index_col == "HALF_HOUR_NUMBER"
    assert panel.frame["HALF_HOUR_NUMBER"].tolist() == [0, 1, 2]


def test_adapter_falls_back_to_local_start_time_alias() -> None:
    df = _make_base_frame()
    df["LOCAL_START_TIME"] = [
        "2025-05-01 04:00:00",
        "2025-05-01 04:30:00",
        "2025-05-01 05:00:00",
    ]
    spec = QSRIntervalPanelDemandSpecV1(time_mode="timestamp")
    panel = to_panel_demand_v1(df, spec=spec, validate=True)
    assert panel.time_mode == "timestamp"
    assert panel.ts_col == "LOCAL_START_TIME"


def test_adapter_trainable_takes_precedence_over_observability_flags() -> None:
    df = _make_base_frame()
    df["IS_DATE_OBSERVABLE"] = ["TRUE", "TRUE", "FALSE"]
    df["IS_TRAINABLE"] = ["FALSE", "TRUE", "TRUE"]

    panel = to_panel_demand_v1(df, validate=False)
    out = panel.frame

    assert out["is_observable"].tolist() == [False, True, True]
    assert out["is_possible"].tolist() == [False, True, True]


def test_adapter_observable_is_conjunction_when_trainable_missing() -> None:
    df = pd.DataFrame(
        {
            "STORE_ID": [101, 101, 101, 101],
            "FORECAST_ENTITY_KEY": [1, 1, 1, 1],
            "BUSINESS_DATE": ["2025-05-01"] * 4,
            "INTERVAL_INDEX": [0, 1, 2, 3],
            "FORECAST_ENTITY_DEMAND_QUANTITY": [1, 2, 3, 4],
            "IS_INTERVAL_OBSERVABLE": ["TRUE", "TRUE", "FALSE", None],
            "IS_DATE_OBSERVABLE": ["TRUE", "FALSE", "TRUE", "TRUE"],
            "IS_STRUCTURAL_ZERO": [0, 0, 0, 0],
        }
    )

    panel = to_panel_demand_v1(df, validate=False)
    out = panel.frame

    assert out["is_observable"].tolist() == [True, False, False, pd.NA]


def test_adapter_retains_uncoerced_observability_metadata_columns() -> None:
    df = _make_base_frame()
    df["IS_DATE_OBSERVABLE"] = ["TRUE", "TRUE", "FALSE"]
    df["IS_TRAINABLE"] = ["TRUE", None, "FALSE"]
    interval_raw = df["IS_INTERVAL_OBSERVABLE"].tolist()
    date_raw = df["IS_DATE_OBSERVABLE"].tolist()

    panel = to_panel_demand_v1(df, validate=False)
    out = panel.frame

    assert "IS_INTERVAL_OBSERVABLE" in out.columns
    assert "IS_DATE_OBSERVABLE" in out.columns
    assert out["IS_INTERVAL_OBSERVABLE"].tolist() == interval_raw
    assert out["IS_DATE_OBSERVABLE"].tolist() == date_raw
    assert str(out["IS_INTERVAL_OBSERVABLE"].dtype) != "boolean"
    assert str(out["IS_DATE_OBSERVABLE"].dtype) != "boolean"
    assert out["is_observable"].tolist() == [True, pd.NA, False]


def test_adapter_defaults_structural_zero_when_source_column_missing() -> None:
    df = _make_base_frame().drop(columns=["IS_STRUCTURAL_ZERO"])

    panel = to_panel_demand_v1(df, validate=True)
    out = panel.frame

    assert "IS_STRUCTURAL_ZERO" not in out.columns
    assert str(out["is_structural_zero"].dtype) == "boolean"
    assert out["is_structural_zero"].tolist() == [False, False, False]


def test_adapter_ignores_structural_zero_column_when_spec_is_none() -> None:
    df = _make_base_frame()
    df["IS_STRUCTURAL_ZERO"] = [1, 0, 0]
    spec = QSRIntervalPanelDemandSpecV1(is_structural_zero_col=None)

    panel = to_panel_demand_v1(df, spec=spec, validate=True)
    out = panel.frame

    assert out["is_structural_zero"].tolist() == [False, False, False]
