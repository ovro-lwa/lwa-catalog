"""Tests for HiPS catalog overlay helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from lwa_catalog.constants import BAND_OVERLAY_COLORS
from lwa_catalog.viz.aladin import (
    _catalog_pa_to_regions_angle,
    _dataframe_to_ellipse_regions,
    catalog_to_astropy_table,
    clear_catalog_overlays,
    filter_catalog_fov,
    overlay_catalog_by_band,
    shape_complete_mask,
)
from lwa_catalog.viz.bands import band_overlay_color, resolve_band_labels


def test_resolve_band_labels_origin_band() -> None:
    df = pd.DataFrame({"origin_band": ["Red", "Blue", None]})
    labels = resolve_band_labels(df, "metacatalog")
    assert labels.tolist() == ["Red", "Blue", "unknown"]


def test_resolve_band_labels_band_column() -> None:
    df = pd.DataFrame({"band": ["Green", "Full"]})
    labels = resolve_band_labels(df, "metacatalog_lst_Green")
    assert labels.tolist() == ["Green", "Full"]


def test_resolve_band_labels_infer_from_catalog_name() -> None:
    df = pd.DataFrame({"RA": [0.0], "DEC": [0.0]})
    labels = resolve_band_labels(df, "metacatalog_lst_Blue")
    assert labels.tolist() == ["Blue"]


def test_band_overlay_color_known_and_unknown() -> None:
    assert band_overlay_color("Red") == BAND_OVERLAY_COLORS["Red"]
    assert band_overlay_color("not_a_band").startswith("#")


def test_band_overlay_color_subband_red_to_blue() -> None:
    low = band_overlay_color("18MHz")
    high = band_overlay_color("82MHz")
    assert low == BAND_OVERLAY_COLORS["Red"]
    assert high == BAND_OVERLAY_COLORS["Blue"]
    assert low != high


def test_resolve_band_labels_mhz_subband() -> None:
    df = pd.DataFrame({"band": ["55MHz", "18MHz"]})
    labels = resolve_band_labels(df, "metacatalog")
    assert labels.tolist() == ["55MHz", "18MHz"]


def test_catalog_to_astropy_table_columns() -> None:
    df = pd.DataFrame(
        {
            "RA": [10.0],
            "DEC": [20.0],
            "Peak_flux": [1.5],
            "extra": ["x"],
        }
    )
    table = catalog_to_astropy_table(df, columns=["RA", "DEC", "Peak_flux"])
    assert list(table.colnames) == ["RA", "DEC", "Peak_flux"]


def test_catalog_to_astropy_table_missing_ra_dec_raises() -> None:
    with pytest.raises(ValueError, match="position columns"):
        catalog_to_astropy_table(pd.DataFrame({"DEC": [1.0]}))


def test_shape_complete_mask() -> None:
    df = pd.DataFrame(
        {
            "Maj": [0.2, np.nan, 0.0],
            "Min": [0.1, 0.1, 0.1],
            "PA": [30.0, 30.0, 30.0],
        }
    )
    mask = shape_complete_mask(df)
    assert mask.tolist() == [True, False, False]


def test_filter_catalog_fov() -> None:
    center = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, frame="icrs")
    df = pd.DataFrame(
        {
            "RA": [0.0, 0.5, 180.0],
            "DEC": [0.0, 0.0, 0.0],
        }
    )
    in_view = filter_catalog_fov(df, center, fov_deg=2.0, margin_deg=0.0)
    assert len(in_view) == 2
    assert set(in_view["RA"].tolist()) == {0.0, 0.5}


def test_catalog_to_astropy_table_ignores_source_file_columns() -> None:
    df = pd.DataFrame(
        {
            "RA": [10.0],
            "DEC": [20.0],
            "Maj": [0.2],
            "Min": [0.1],
            "PA": [30.0],
            "source_file_Blue": [np.array(b"path/to/fits", dtype=object)],
        }
    )
    table = catalog_to_astropy_table(df, attach_beam_units=True)
    assert "source_file_Blue" not in table.colnames
    from io import BytesIO

    table.write(BytesIO(), format="votable")


def test_catalog_to_astropy_table_beam_units() -> None:
    df = pd.DataFrame(
        {
            "RA": [10.0],
            "DEC": [20.0],
            "Maj": [0.2],
            "Min": [0.1],
            "PA": [30.0],
        }
    )
    table = catalog_to_astropy_table(df, attach_beam_units=True)
    assert table["Maj"].unit == u.deg
    assert table["Min"].unit == u.deg
    assert table["PA"].unit == u.deg


def test_catalog_pa_to_regions_angle_offsets_ipyaladin() -> None:
    assert _catalog_pa_to_regions_angle(0.0).to_value(u.deg) == pytest.approx(90.0)
    assert _catalog_pa_to_regions_angle(30.0).to_value(u.deg) == pytest.approx(120.0)


def test_ellipse_regions_use_catalog_pa_offset() -> None:
    pytest.importorskip("regions")
    df = pd.DataFrame(
        {
            "RA": [10.0],
            "DEC": [20.0],
            "Maj": [0.2],
            "Min": [0.1],
            "PA": [30.0],
        }
    )
    regions = _dataframe_to_ellipse_regions(df)
    assert len(regions) == 1
    assert regions[0].angle.to_value(u.deg) == pytest.approx(120.0)


def test_overlay_catalog_by_band_mock_aladin() -> None:
    center = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, frame="icrs")
    df = pd.DataFrame(
        {
            "RA": [0.0, 0.1],
            "DEC": [0.0, 0.0],
            "Maj": [0.2, np.nan],
            "Min": [0.1, np.nan],
            "PA": [10.0, np.nan],
            "origin_band": ["Red", "Blue"],
        }
    )

    aladin = MagicMock()
    aladin.remove_overlay = MagicMock()
    aladin.overlays = []

    result = overlay_catalog_by_band(
        aladin,
        df,
        "metacatalog",
        center,
        fov_deg=5.0,
        max_rows=10,
        selection_idx=1,
    )

    assert result.drawn == 2
    assert result.in_fov == 2
    assert result.truncated is False
    assert result.per_band == {"Red": 1, "Blue": 1}
    assert aladin.remove_overlay.call_count >= 1
    assert aladin.add_graphic_overlay_from_region.call_count >= 1
    assert aladin.add_table.call_count >= 2

    ellipse_names = [
        call.kwargs.get("name")
        for call in aladin.add_graphic_overlay_from_region.call_args_list
    ]
    assert "catalog_Red" in ellipse_names

    table_names = [call.kwargs.get("name") for call in aladin.add_table.call_args_list]
    assert "catalog_Blue_cross" in table_names
    assert "catalog_selection" in table_names

    shapes = {call.kwargs.get("shape") for call in aladin.add_table.call_args_list}
    assert shapes == {"cross"}

    colors = {
        call.kwargs.get("color")
        for call in aladin.add_graphic_overlay_from_region.call_args_list
    }
    colors |= {call.kwargs.get("color") for call in aladin.add_table.call_args_list}
    assert BAND_OVERLAY_COLORS["Red"] in colors
    assert BAND_OVERLAY_COLORS["Blue"] in colors


def test_clear_catalog_overlays_removes_suffixed_layers() -> None:
    aladin = MagicMock()
    aladin.overlays = ["catalog_Red", "catalog_Red_1", "catalog_selection", "trace_lst_Full"]
    aladin.remove_overlay = MagicMock()

    clear_catalog_overlays(aladin)

    removed = {call.args[0] for call in aladin.remove_overlay.call_args_list}
    assert "catalog_Red" in removed
    assert "catalog_Red_1" in removed
    assert "catalog_selection" in removed
    assert "trace_lst_Full" not in removed
