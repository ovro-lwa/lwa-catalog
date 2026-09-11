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


def test_spectrum_figure_includes_survey_points() -> None:
    pytest.importorskip("matplotlib")
    from lwa_catalog.viz.browser import _spectrum_figure_for_row

    row = pd.Series(
        {
            "meta_id": 1,
            "Total_flux_55MHz": 1.0,
            "E_Total_flux_55MHz": 0.1,
            "Total_flux_NVSS": 0.2,
            "E_Total_flux_NVSS": 0.02,
            "spec_model_n_terms": 2,
            "spec_model_n_flux": 2,
            "spec_model_a0": 0.0,
            "spec_model_a1": -0.7,
            "spec_model_a2": float("nan"),
            "spec_model_a3": float("nan"),
            "spec_model_bic": 1.0,
            "spec_model_chi2_red": 1.0,
            "spec_model_nu0_mhz": 55.0,
        }
    )
    fig = _spectrum_figure_for_row(row)
    ax = fig.axes[0]
    labels = set(ax.get_legend_handles_labels()[1])
    assert "LWA" in labels
    assert "survey" in labels
