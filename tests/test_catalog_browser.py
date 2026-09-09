"""Smoke tests for CatalogBrowser config (Panel UI is notebook integration)."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("panel")

from lwa_catalog.viz.browser import (  # noqa: E402
    CatalogBrowserConfig,
    default_display_columns,
    order_columns,
)


def test_catalog_browser_config_defaults() -> None:
    cfg = CatalogBrowserConfig(
        catalog_dirs=("/tmp/demo",),
        default_catalog_dir="/tmp/demo",
    )
    assert cfg.sky_fov_deg == 10.0
    assert cfg.max_table_columns == 40
    assert cfg.prefer_spectral is True
    assert "meta_id" in cfg.preferred_columns


def test_order_and_default_display_columns() -> None:
    df = pd.DataFrame({"ZZ": [1], "meta_id": [2], "RA": [3.0]})
    ordered = order_columns(df, preferred_columns=["meta_id", "RA", "DEC"], limit=2)
    assert ordered == ["meta_id", "RA"]
    shown = default_display_columns(
        df,
        preferred_columns=["meta_id", "missing"],
        max_table_columns=10,
    )
    assert shown == ["meta_id"]


def test_catalog_browser_exports() -> None:
    from lwa_catalog.viz import CatalogBrowser, CatalogBrowserConfig

    assert CatalogBrowser is not None
    assert CatalogBrowserConfig is not None
