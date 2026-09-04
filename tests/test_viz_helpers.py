"""Tests for HiPS / coordinate viz helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from lwa_catalog.viz.aladin import (
    catalog_name_from_file,
    clear_trace_overlays,
    overlay_trace_members,
)
from lwa_catalog.viz.coordinates import format_coordinate_deg, parse_coordinate
from lwa_catalog.constants import NVSS_BMAJ_DEG, VLASS_BMAJ_DEG, VLSSR_BMAJ_DEG
from lwa_catalog.viz.aladin import catalog_with_survey_beam
from lwa_catalog.viz.hips import (
    default_hips_survey,
    discover_local_hips_surveys,
    fetch_hips_surveys,
    survey_hips_url,
)


def test_parse_coordinate_decimal() -> None:
    coord = parse_coordinate("10.5 -3.2")
    assert coord.ra.deg == pytest.approx(10.5)
    assert coord.dec.deg == pytest.approx(-3.2)


def test_format_coordinate_deg_roundtrip() -> None:
    text = format_coordinate_deg(83.633, -5.391)
    coord = parse_coordinate(text)
    assert coord.ra.deg == pytest.approx(83.633)
    assert coord.dec.deg == pytest.approx(-5.391)


def test_catalog_name_from_file() -> None:
    assert catalog_name_from_file("metacatalog.parquet") == "metacatalog"
    assert catalog_name_from_file("metacatalog_lst_Blue.parquet") == "metacatalog_lst_Blue"


def test_discover_local_hips_surveys(tmp_path: Path) -> None:
    hips = tmp_path / "survey_a"
    hips.mkdir()
    (hips / "properties").write_text("hips_service_url = ./\n")
    assert discover_local_hips_surveys(tmp_path) == ["survey_a"]


def test_fetch_hips_surveys_fallback(tmp_path: Path) -> None:
    hips = tmp_path / "local_survey"
    hips.mkdir()
    (hips / "properties").write_text("hips_service_url = ./\n")
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        surveys = fetch_hips_surveys(
            "http://example.invalid",
            catalog_dir=tmp_path,
            default_survey="fallback.hips",
        )
    assert surveys == ["local_survey"]


def test_default_hips_survey_prefers_configured() -> None:
    assert default_hips_survey(
        ["other", "metacatalog_coadd2_full.hips"],
        default_survey="metacatalog_coadd2_full.hips",
    ) == "metacatalog_coadd2_full.hips"


def test_survey_hips_url_known() -> None:
    assert survey_hips_url("NVSS").startswith("https://")
    assert survey_hips_url("nvss").endswith("/")
    assert "VLSSr" in survey_hips_url("VLSSR")
    assert "VLASS" in survey_hips_url("VLASS").upper() or "vlass" in survey_hips_url("VLASS")


def test_survey_hips_url_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown survey HiPS"):
        survey_hips_url("NOT_A_SURVEY")


def test_catalog_with_survey_beam_lwa_preserves_shapes() -> None:
    df = pd.DataFrame(
        {
            "RA": [10.0],
            "DEC": [20.0],
            "Maj": [0.2],
            "Min": [0.1],
            "PA": [30.0],
        }
    )
    out = catalog_with_survey_beam(df, "LWA")
    assert out["Maj"].tolist() == [0.2]
    assert out["Min"].tolist() == [0.1]
    assert out["PA"].tolist() == [30.0]


@pytest.mark.parametrize(
    ("survey", "beam"),
    [
        ("VLSSR", VLSSR_BMAJ_DEG),
        ("NVSS", NVSS_BMAJ_DEG),
        ("VLASS", VLASS_BMAJ_DEG),
    ],
)
def test_catalog_with_survey_beam_circular(survey: str, beam: float) -> None:
    df = pd.DataFrame(
        {
            "RA": [10.0],
            "DEC": [20.0],
            "Maj": [0.2],
            "Min": [0.1],
            "PA": [30.0],
        }
    )
    out = catalog_with_survey_beam(df, survey)
    assert out["Maj"].tolist() == pytest.approx([beam])
    assert out["Min"].tolist() == pytest.approx([beam])
    assert out["PA"].tolist() == [0.0]


def test_catalog_with_survey_beam_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown survey beam"):
        catalog_with_survey_beam(pd.DataFrame({"RA": [0.0], "DEC": [0.0]}), "OTHER")


def test_overlay_trace_members_mock_aladin() -> None:
    center = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, frame="icrs")
    lst = pd.DataFrame(
        {
            "RA": [0.0],
            "DEC": [0.0],
            "Maj": [0.2],
            "Min": [0.1],
            "PA": [10.0],
            "band": ["Red"],
        }
    )
    src = pd.DataFrame(
        {
            "RA": [0.05],
            "DEC": [0.0],
            "Maj": [0.2],
            "Min": [0.1],
            "PA": [10.0],
            "band": ["Blue"],
        }
    )
    aladin = MagicMock()
    aladin.remove_overlay = MagicMock()
    results = overlay_trace_members(aladin, lst, src, center, fov_deg=5.0)
    assert set(results) == {"trace_lst", "trace_src"}
    assert aladin.remove_overlay.call_count >= 2


def test_clear_trace_overlays() -> None:
    aladin = MagicMock()
    aladin.remove_overlay = MagicMock()
    clear_trace_overlays(aladin)
    assert aladin.remove_overlay.call_count >= 2
