"""
Unit tests for the QSR panel adapter.

These tests validate that:
- the adapter produces canonical columns
- governance gates are coerced deterministically
- unknown boolean tokens raise loudly
- NULL demand is preserved (not imputed)
"""

from __future__ import annotations

import pandas as pd
import pytest

from eb_adapters.panels.qsr import QSRPanelDemandSpecV1, qsr_to_panel_demand_v1


def _make_base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STORE_ID": [101, 101, 101],
            "FORECAST_ENTITY_ID": [1, 1, 1],
            "BUSINESS_DAY": ["2025-05-01", "2025-05-01", "2025-05-01"],
            "INTERVAL_30_INDEX": [0, 1, 2],
            "ACTUAL_COMMODITY_USAGE_QTY": [None, 4, 8],
            "IS_INTERVAL_OBSERVABLE": ["TRUE", "FALSE", "TRUE"],
            "HAS_DEMAND": [1, 0, 1],
            "IS_STRUCTURAL_ZERO": [0, 0, 0],
        }
    )


def test_qsr_adapter_happy_path_creates_panel_and_canonical_columns() -> None:
    df = _make_base_frame()

    panel = qsr_to_panel_demand_v1(df, validate=True)

    # Contract object basics
    assert panel.time_mode == "day_interval"
    assert panel.keys == ("STORE_ID", "FORECAST_ENTITY_ID")
    assert panel.y_col == "y"

    # Canonical columns exist
    out = panel.frame
    for col in ("y", "is_observable", "is_possible", "is_structural_zero"):
        assert col in out.columns


def test_qsr_adapter_coerces_gates_to_bool() -> None:
    df = _make_base_frame()

    panel = qsr_to_panel_demand_v1(df, validate=True)
    out = panel.frame

    assert out["is_observable"].dtype == bool
    assert out["is_possible"].dtype == bool
    assert out["is_structural_zero"].dtype == bool

    # Values coerced as expected
    assert out["is_observable"].tolist() == [True, False, True]
    assert out["is_possible"].tolist() == [True, False, True]
    assert out["is_structural_zero"].tolist() == [False, False, False]


def test_qsr_adapter_raises_on_unknown_gate_token() -> None:
    df = _make_base_frame()
    df.loc[0, "IS_INTERVAL_OBSERVABLE"] = "MAYBE"

    with pytest.raises(ValueError, match="Unrecognized boolean value"):
        qsr_to_panel_demand_v1(df, validate=True)


def test_qsr_adapter_preserves_null_y() -> None:
    df = _make_base_frame()

    panel = qsr_to_panel_demand_v1(df, validate=True)
    out = panel.frame

    # First row should remain null in canonical y
    assert pd.isna(out.loc[0, "y"])


def test_qsr_adapter_supports_custom_column_mapping_via_spec() -> None:
    df = _make_base_frame().rename(
        columns={
            "STORE_ID": "store",
            "FORECAST_ENTITY_ID": "entity",
            "BUSINESS_DAY": "day",
            "INTERVAL_30_INDEX": "idx",
            "ACTUAL_COMMODITY_USAGE_QTY": "usage",
            "IS_INTERVAL_OBSERVABLE": "obs",
            "HAS_DEMAND": "possible",
            "IS_STRUCTURAL_ZERO": "struct0",
        }
    )

    spec = QSRPanelDemandSpecV1(
        store_col="store",
        entity_col="entity",
        business_day_col="day",
        interval_index_col="idx",
        demand_col="usage",
        is_observable_col="obs",
        is_possible_col="possible",
        is_structural_zero_col="struct0",
    )

    panel = qsr_to_panel_demand_v1(df, spec=spec, validate=True)
    assert panel.keys == ("store", "entity")
    assert panel.time_mode == "day_interval"
    assert panel.y_col == "y"
