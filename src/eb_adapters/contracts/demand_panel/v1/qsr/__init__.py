"""
QSR-specific contract adapters for demand-panel inputs.

This package contains adapters that transform QSR-style interval panel datasets
into EB contract artifacts (e.g., PanelDemandV1).
"""

from __future__ import annotations

from eb_adapters.contracts.demand_panel.v1.qsr.entity_usage_interval_panel import (
    QSRIntervalPanelDemandSpecV1,
    to_panel_demand_v1,
)

__all__ = [
    "QSRIntervalPanelDemandSpecV1",
    "to_panel_demand_v1",
]
