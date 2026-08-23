"""Public root-export smoke tests for ``eb_adapters``."""

from __future__ import annotations

import eb_adapters as m


def test_public_root_surface() -> None:
    assert "_clone_model" not in m.__all__
    assert "clone_model" in m.__all__
    assert hasattr(m, "clone_model")
    assert "QSRIntervalPanelDemandSpecV1" in m.__all__
    assert "to_panel_demand_v1" in m.__all__
    assert hasattr(m, "QSRIntervalPanelDemandSpecV1")
    assert hasattr(m, "to_panel_demand_v1")
    assert isinstance(m.__version__, str) and m.__version__
