"""Tests for VLSSR cross-match QA."""

from __future__ import annotations

import pandas as pd
import pytest

from lwa_catalog.analyze.vlssr import (
    _catalog_match_frame,
    _footprint_filter_vlssr,
    load_vlssr_catalog,
    select_blue_associated_rows,
)
from lwa_catalog.constants import VLSSR_BMAJ_DEG


def _write_mini_vlssr(path) -> None:
    path.write_text(
        'RA(2000) DEC(2000) "PEAK INT"\n'
        "10.0 20.0 1.5\n"
        "10.01 20.0 0.8\n"
        "nan 30.0 0.5\n"
    )


def test_load_vlssr_catalog_assigns_bmaj(tmp_path) -> None:
    catalog_path = tmp_path / "vlssr.txt"
    _write_mini_vlssr(catalog_path)

    loaded = load_vlssr_catalog(catalog_path)

    assert list(loaded.columns) == ["RA", "DEC", "Peak_flux", "BMAJ", "BMIN"]
    assert len(loaded) == 2
    assert (loaded["BMAJ"] == VLSSR_BMAJ_DEG).all()
    assert (loaded["BMIN"] == VLSSR_BMAJ_DEG).all()
    assert loaded.iloc[0]["Peak_flux"] == pytest.approx(1.5)
    assert loaded.iloc[1]["RA"] == pytest.approx(10.01)


def test_load_vlssr_catalog_missing_file(tmp_path) -> None:
    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError, match="VLSSR catalog not found"):
        load_vlssr_catalog(missing)


def test_select_blue_associated_rows() -> None:
    catalog = pd.DataFrame(
        [
            {"meta_id": 0, "bands_present": "Full,Blue", "RA": 10.0, "DEC": 20.0},
            {"meta_id": 1, "bands_present": "Full", "RA": 11.0, "DEC": 21.0},
            {"meta_id": 2, "bands_present": "Blue", "RA": 12.0, "DEC": 22.0},
        ]
    )
    blue = select_blue_associated_rows(catalog)
    assert list(blue["meta_id"]) == [0, 2]
    assert len(blue) == 2


def test_select_blue_associated_rows_missing_column() -> None:
    with pytest.raises(ValueError, match="bands_present"):
        select_blue_associated_rows(pd.DataFrame({"RA": [1.0], "DEC": [2.0]}))


def test_catalog_match_frame_uses_primary_coords_and_bmaj_match() -> None:
    catalog = pd.DataFrame(
        [
            {
                "RA": 10.0,
                "DEC": 20.0,
                "RA_Blue": 10.5,
                "DEC_Blue": 20.5,
                "BMAJ_match": 0.5,
            },
            {"RA": float("nan"), "DEC": 30.0, "BMAJ_match": 0.5},
        ]
    )
    frame = _catalog_match_frame(catalog)
    assert len(frame) == 1
    assert frame.iloc[0]["RA"] == pytest.approx(10.0)
    assert frame.iloc[0]["DEC"] == pytest.approx(20.0)
    assert frame.iloc[0]["BMAJ"] == pytest.approx(0.5)


def test_footprint_filter_vlssr() -> None:
    vlssr = pd.DataFrame(
        {
            "RA": [0.0, 0.0, 0.0],
            "DEC": [-10.0, 5.0, 20.0],
            "Peak_flux": [1.0, 1.0, 1.0],
            "BMAJ": [VLSSR_BMAJ_DEG] * 3,
        }
    )
    lwa = pd.DataFrame({"RA": [1.0, 2.0], "DEC": [0.0, 10.0]})
    filtered = _footprint_filter_vlssr(vlssr, lwa)
    assert len(filtered) == 1
    assert filtered.iloc[0]["DEC"] == pytest.approx(5.0)
