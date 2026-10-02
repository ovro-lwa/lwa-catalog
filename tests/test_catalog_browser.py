"""Smoke tests for CatalogBrowser config (Panel UI is notebook integration)."""

from __future__ import annotations

from pathlib import Path

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
    assert cfg.healpix_cutout_nside == 2048
    assert cfg.healpix_cutout_beam_factor == 6.0
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
            "n_confused_55MHz": 1,
            "Total_flux_NVSS": 0.2,
            "E_Total_flux_NVSS": 0.02,
            "n_confused_NVSS": 1,
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
    assert "n_flux=2" in ax.get_title()


def test_spectrum_figure_confused_uses_x_marker() -> None:
    """Channels with n_confused != 1 (or missing) use an x marker; title n_flux matches fit."""
    pytest.importorskip("matplotlib")
    from lwa_catalog.constants import SUBBAND_BANDS_MHZ
    from lwa_catalog.viz.browser import _spectrum_figure_for_row

    row = pd.Series(
        {
            "meta_id": 7,
            "spec_model_n_terms": 2,
            "spec_model_n_flux": 1,
            "spec_model_a0": 0.0,
            "spec_model_a1": -0.7,
            "spec_model_a2": float("nan"),
            "spec_model_a3": float("nan"),
            "spec_model_bic": 1.0,
            "spec_model_chi2_red": 1.0,
            "spec_model_nu0_mhz": 55.0,
            "Total_flux_55MHz": 1.0,
            "E_Total_flux_55MHz": 0.1,
            "n_confused_55MHz": 1,
            "Total_flux_82MHz": 0.8,
            "E_Total_flux_82MHz": 0.08,
            # Missing n_confused_82MHz → excluded from unconfused fit (seed-band bug).
            "Total_flux_41MHz": 0.9,
            "E_Total_flux_41MHz": 0.09,
            "n_confused_41MHz": 3,
        }
    )
    fig = _spectrum_figure_for_row(row, bands=SUBBAND_BANDS_MHZ)
    ax = fig.axes[0]
    labels = set(ax.get_legend_handles_labels()[1])
    assert "LWA" in labels
    assert "LWA (confused)" in labels
    assert "n_flux=1" in ax.get_title()
    confused_handles = [
        h
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True)
        if lab == "LWA (confused)"
    ]
    assert confused_handles
    assert confused_handles[0].lines[0].get_marker() == "x"


def test_spectrum_figure_title_n_flux_without_model_is_unconfused_count() -> None:
    """Without spec_model_* columns, title n_flux counts unconfused channels only."""
    pytest.importorskip("matplotlib")
    from lwa_catalog.constants import SUBBAND_BANDS_MHZ
    from lwa_catalog.viz.browser import _spectrum_figure_for_row

    row = pd.Series(
        {
            "meta_id": 3,
            "Total_flux_55MHz": 1.0,
            "E_Total_flux_55MHz": 0.1,
            "n_confused_55MHz": 1,
            "Total_flux_82MHz": 0.8,
            "E_Total_flux_82MHz": 0.08,
            "n_confused_82MHz": 2,
        }
    )
    fig = _spectrum_figure_for_row(row, bands=SUBBAND_BANDS_MHZ)
    title = fig.axes[0].get_title()
    assert "n_flux=1" in title
    assert "no Taylor model columns" in title


def test_spectrum_figure_ylim_follows_data_not_fit() -> None:
    """Y-limits stay near measured fluxes even when the Taylor curve diverges."""
    pytest.importorskip("matplotlib")
    from lwa_catalog.constants import SUBBAND_BANDS_MHZ
    from lwa_catalog.viz.browser import _spectrum_figure_for_row

    row = pd.Series(
        {
            "meta_id": 9,
            "Total_flux_55MHz": 1.0,
            "E_Total_flux_55MHz": 0.1,
            "n_confused_55MHz": 1,
            "Total_flux_82MHz": 0.8,
            "E_Total_flux_82MHz": 0.08,
            "n_confused_82MHz": 1,
            # Steep power law: fit will be far below the points at high freq.
            "spec_model_n_terms": 2,
            "spec_model_n_flux": 2,
            "spec_model_a0": 0.0,
            "spec_model_a1": -5.0,
            "spec_model_a2": float("nan"),
            "spec_model_a3": float("nan"),
            "spec_model_bic": 1.0,
            "spec_model_chi2_red": 1.0,
            "spec_model_nu0_mhz": 55.0,
        }
    )
    fig = _spectrum_figure_for_row(row, bands=SUBBAND_BANDS_MHZ)
    ymin, ymax = fig.axes[0].get_ylim()
    assert ymin > 0.5  # not dragged down by the steep fit
    assert ymax < 2.0


def test_spectrum_figure_unconfused_only_mask_length() -> None:
    """Regression: LWA mask must match gathered points when a confused band is skipped."""
    pytest.importorskip("matplotlib")
    from lwa_catalog.constants import SUBBAND_BANDS_MHZ
    from lwa_catalog.viz.browser import _spectrum_figure_for_row

    row_data: dict = {
        "meta_id": 42,
        "spec_model_n_terms": 2,
        "spec_model_n_flux": 14,
        "spec_model_a0": 0.0,
        "spec_model_a1": -0.7,
        "spec_model_a2": float("nan"),
        "spec_model_a3": float("nan"),
        "spec_model_bic": 1.0,
        "spec_model_chi2_red": 1.0,
        "spec_model_nu0_mhz": 55.0,
    }
    for band in SUBBAND_BANDS_MHZ:
        row_data[f"Total_flux_{band}"] = 1.0
        row_data[f"E_Total_flux_{band}"] = 0.1
        row_data[f"n_confused_{band}"] = 1
    # One positive-flux channel is confused — gather with unconfused_only would drop it.
    # Plotting must not rebuild a longer boolean mask from raw Total_flux.
    confused = SUBBAND_BANDS_MHZ[3]
    row_data[f"n_confused_{confused}"] = 3
    row = pd.Series(row_data)

    from lwa_catalog.analyze.spectral import gather_band_flux_measurements

    nu_hz, flux_jy, err_jy, point_bands = gather_band_flux_measurements(
        row,
        bands=SUBBAND_BANDS_MHZ,
        flux_kind="total",
        unconfused_only=True,
    )
    assert nu_hz.size == len(SUBBAND_BANDS_MHZ) - 1
    assert len(point_bands) == nu_hz.size
    is_lwa = [b in set(SUBBAND_BANDS_MHZ) for b in point_bands]
    assert len(is_lwa) == nu_hz.size
    # Browser path (no unconfused_only) still plots; smoke that it does not raise.
    fig = _spectrum_figure_for_row(row, bands=SUBBAND_BANDS_MHZ)
    assert fig.axes


def test_load_sky_view_centers_on_meta_id_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entering meta_id + Load sky view selects that source and recenters."""
    pytest.importorskip("ipyaladin")
    import panel as pn
    from lwa_catalog.viz import browser as browser_mod
    from lwa_catalog.viz.browser import CatalogBrowser, CatalogBrowserConfig

    class _FakeAladin:
        def __init__(self, **kwargs):
            self.target = kwargs.get("target")
            self.fov = kwargs.get("fov")
            self.survey = kwargs.get("survey")
            self._listeners: dict = {}

        def set_listener(self, event: str, callback) -> None:
            self._listeners[event] = callback

        def save_view_as_image(self, *args, **kwargs) -> None:
            return None

    monkeypatch.setattr("ipyaladin.Aladin", _FakeAladin)
    monkeypatch.setattr(
        pn.pane,
        "IPyWidget",
        lambda object, **kwargs: type("W", (), {"object": object})(),
    )
    monkeypatch.setattr(
        browser_mod,
        "fetch_catalog_hips_surveys",
        lambda *a, **k: ["CDS/P/DSS2/color"],
    )
    monkeypatch.setattr(
        browser_mod,
        "preferred_hips_survey",
        lambda *a, **k: "CDS/P/DSS2/color",
    )
    monkeypatch.setattr(browser_mod, "hips_survey_url", lambda name, base="": f"mock://{name}")
    monkeypatch.setattr(
        browser_mod,
        "DebouncedAladinViewRefresh",
        lambda *a, **k: type("R", (), {"cancel_pending": lambda self: None})(),
    )
    monkeypatch.setattr(
        browser_mod,
        "overlay_catalog_by_band",
        lambda *a, **k: type("O", (), {"drawn": 0, "in_fov": 0, "truncated": False})(),
    )
    monkeypatch.setattr(browser_mod, "clear_trace_overlays", lambda *a, **k: None)
    monkeypatch.setattr(browser_mod, "clear_catalog_overlays", lambda *a, **k: None)
    monkeypatch.setattr(browser_mod, "cancel_aladin_view_timers", lambda *a, **k: None)
    monkeypatch.setattr(browser_mod, "aladin_view_center_fov", lambda *a, **k: (None, 10.0))

    catalog_dir = tmp_path / "cat"
    catalog_dir.mkdir()
    df = pd.DataFrame(
        {
            "meta_id": [10, 20],
            "RA": [100.0, 200.0],
            "DEC": [10.0, -20.0],
            "Peak_flux": [1.0, 2.0],
            "bands_present": ["Full", "Full"],
            "origin_band": ["Full", "Full"],
        }
    )
    path = catalog_dir / "metacatalog.parquet"
    df.to_parquet(path, index=False)

    index = pd.DataFrame(
        [
            {
                "file": "metacatalog.parquet",
                "kind": "metacatalog",
                "lst_hour": None,
                "band": None,
                "n_rows": 2,
                "size_mb": 0.01,
                "path": str(path),
            }
        ]
    )
    cfg = CatalogBrowserConfig(
        catalog_dirs=(str(catalog_dir),),
        default_catalog_dir=str(catalog_dir),
        quality_flag_mask=None,
        display_column_prefs_path=tmp_path / "display_cols.json",
    )
    br = CatalogBrowser(index, config=cfg)
    assert br._selected_meta_id == 10

    br.meta_id = 20
    br._on_load_sky()

    assert br._selected_meta_id == 20
    assert br._sky_loaded is True
    assert "200" in br.coordinate
    assert br._aladin.target.ra.deg == pytest.approx(200.0)
    assert br._aladin.target.dec.deg == pytest.approx(-20.0)
    row = br._catalog_row_for_selection()
    assert row is not None
    assert int(row["meta_id"]) == 20
