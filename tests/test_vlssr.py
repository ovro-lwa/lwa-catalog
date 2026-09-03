"""Tests for VLSSR cross-match QA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.crossmatch_radius import LWA_CROSSMATCH_RADIUS_BEAM
from lwa_catalog.analyze.vlssr import (
    VlssrMatchConfig,
    _catalog_match_frame,
    _footprint_filter_vlssr,
    load_vlssr_catalog,
    match_catalog_to_vlssr,
    select_blue_associated_rows,
    summarize_vlssr_match,
)
from lwa_catalog.constants import VLSSR_BMAJ_DEG


def _write_mini_vlssr(path) -> None:
    path.write_text(
        'RA(2000) DEC(2000) "PEAK INT"\n'
        "10.0 20.0 1.5\n"
        "10.01 20.0 0.8\n"
        "nan 30.0 0.5\n"
    )


def _meta_row(
    *,
    meta_id: int = 0,
    ra: float = 10.0,
    dec: float = 20.0,
    bmaj: float = 0.5,
    bands_present: str = "Full,Blue",
) -> dict:
    return {
        "meta_id": meta_id,
        "RA": ra,
        "DEC": dec,
        "BMAJ_match": bmaj,
        "bands_present": bands_present,
        "Peak_flux": 1.0,
        "origin_band": "Full",
    }


def _vlssr_rows(rows: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "RA": ra,
                "DEC": dec,
                "Peak_flux": 1.0,
                "BMAJ": VLSSR_BMAJ_DEG,
                "BMIN": VLSSR_BMAJ_DEG,
            }
            for ra, dec in rows
        ]
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
    frame = _catalog_match_frame(catalog, LWA_CROSSMATCH_RADIUS_BEAM)
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


def test_match_completeness_single_hit() -> None:
    meta = pd.DataFrame([_meta_row(meta_id=7)])
    vlssr = _vlssr_rows([(10.0, 20.0)])
    result = match_catalog_to_vlssr(meta, vlssr=vlssr)
    assert result.summary["blue_completeness"] == pytest.approx(1.0)
    assert int(result.meta_flags.iloc[0]["n_vlssr"]) == 1
    assert int(result.meta_flags.iloc[0]["meta_id"]) == 7
    assert int(result.summary["n_vlssr_matched"]) == 1
    assert result.summary["vlssr_recovery"] == pytest.approx(1.0)


def test_match_multi_vlssr_per_meta() -> None:
    meta = pd.DataFrame([_meta_row()])
    vlssr = _vlssr_rows([(10.0, 20.0), (10.01, 20.0)])
    result = match_catalog_to_vlssr(meta, vlssr=vlssr)
    assert result.summary["blue_completeness"] == pytest.approx(1.0)
    assert int(result.meta_flags.iloc[0]["n_vlssr"]) == 2
    assert int(result.summary["n_meta_multi_vlssr"]) == 1


def test_match_oversplit_two_meta_one_vlssr() -> None:
    meta = pd.DataFrame(
        [
            _meta_row(meta_id=0, ra=10.0, dec=20.0),
            _meta_row(meta_id=1, ra=10.05, dec=20.0),
        ]
    )
    vlssr = _vlssr_rows([(10.02, 20.0)])
    result = match_catalog_to_vlssr(meta, vlssr=vlssr)
    assert int(result.summary["n_vlssr_oversplit"]) == 1
    assert bool(result.vlssr_flags.iloc[0]["oversplit"])
    assert result.vlssr_flags.iloc[0]["meta_ids"] == [0, 1]


def test_match_completeness_no_vlssr_in_beam() -> None:
    meta = pd.DataFrame([_meta_row()])
    vlssr = _vlssr_rows([(50.0, 50.0)])
    result = match_catalog_to_vlssr(meta, vlssr=vlssr)
    assert result.summary["blue_completeness"] == pytest.approx(0.0)
    assert not bool(result.meta_flags.iloc[0]["matched"])


def test_empty_lwa_catalog_returns_empty_summary() -> None:
    meta = pd.DataFrame(
        [{"meta_id": 0, "RA": 10.0, "DEC": 20.0, "bands_present": "Full"}]
    )
    vlssr = _vlssr_rows([(10.0, 20.0)])
    result = match_catalog_to_vlssr(meta, vlssr=vlssr)
    assert int(result.summary["n_lwa_target"]) == 0
    assert np.isnan(float(result.summary["blue_completeness"]))
    assert result.warnings


def test_summarize_vlssr_match_includes_key_metrics() -> None:
    meta = pd.DataFrame([_meta_row()])
    vlssr = _vlssr_rows([(10.0, 20.0)])
    result = match_catalog_to_vlssr(meta, vlssr=vlssr)
    text = summarize_vlssr_match(result)
    assert "Blue completeness:" in text
    assert "VLSSR over-split" in text
    assert "Meta multi-VLSSR" in text


def test_match_lst_merged_blue_target() -> None:
    lst_blue = pd.DataFrame(
        [
            {
                "RA": 10.0,
                "DEC": 20.0,
                "BMAJ": 0.5,
                "Peak_flux": 1.0,
                "band": "Blue",
            }
        ]
    )
    vlssr = _vlssr_rows([(10.0, 20.0)])
    config = VlssrMatchConfig(target="lst_merged_blue")
    result = match_catalog_to_vlssr(lst_blue, vlssr=vlssr, config=config)
    assert int(result.summary["n_lwa_target"]) == 1
    assert result.summary["blue_completeness"] == pytest.approx(1.0)


def test_match_full_metacatalog_includes_non_blue() -> None:
    meta = pd.DataFrame([_meta_row(bands_present="Full,Green")])
    vlssr = _vlssr_rows([(10.0, 20.0)])
    config = VlssrMatchConfig(target="metacatalog")
    result = match_catalog_to_vlssr(meta, vlssr=vlssr, config=config)
    assert int(result.summary["n_lwa_target"]) == 1
    assert bool(result.meta_flags["matched"].iloc[0]) is True
