"""
Unit tests for the QSR interval panel adapter.

These tests validate that:
- the adapter produces canonical columns
- governance gates preserve tri-state semantics (True/False/NA)
- unknown boolean tokens raise loudly
- NULL demand is preserved (not imputed) by default
- optional imputation (when enabled) only fills observable, non-structural intervals
- custom column mapping via spec works
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
            "FORECAST_ENTITY_ID": [1, 1, 1],
            "BUSINESS_DAY": ["2025-05-01", "2025-05-01", "2025-05-01"],
            "INTERVAL_30_INDEX": [0, 1, 2],
            # Use the DDL-aligned default for the adapter/spec
            "LABEL_COMMODITY_USAGE_QTY": [None, 4, 8],
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
    df.loc[0, "LABEL_COMMODITY_USAGE_QTY"] = None

    # Make row1 observable NA + y null -> should remain NA
    df.loc[1, "IS_INTERVAL_OBSERVABLE"] = None
    df.loc[1, "LABEL_COMMODITY_USAGE_QTY"] = None

    # Make row2 structural True + y null + observable True -> should NOT impute
    df.loc[2, "IS_STRUCTURAL_ZERO"] = 1
    df.loc[2, "IS_INTERVAL_OBSERVABLE"] = "TRUE"
    df.loc[2, "LABEL_COMMODITY_USAGE_QTY"] = None

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
            "FORECAST_ENTITY_ID": "entity",
            "BUSINESS_DAY": "day",
            "INTERVAL_30_INDEX": "idx",
            "LABEL_COMMODITY_USAGE_QTY": "usage",
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
