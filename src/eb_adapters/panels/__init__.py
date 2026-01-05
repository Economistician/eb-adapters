"""
Public panel adapter API.

This module provides adapters that map domain-specific panel datasets
into Electric Barometer contract artifacts (e.g., PanelDemandV1).

Panel adapters are responsible for:
- schema normalization
- semantic mapping
- governance-safe construction of contract objects

They intentionally live outside eb-contracts to preserve domain agnosticism.
"""

from __future__ import annotations

######################################
# Public API
######################################
from eb_adapters.panels.qsr import QSRPanelDemandSpecV1, qsr_to_panel_demand_v1

__all__ = [
    "QSRPanelDemandSpecV1",
    "qsr_to_panel_demand_v1",
]
