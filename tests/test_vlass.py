"""Tests for VLASS cross-match."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.vlass import (
    VlassMatchConfig,
    _footprint_filter_vlass,
    load_vlass_catalog,
    match_catalog_to_vlass,
    select_unique_vlass_matches,
    summarize_vlass_match,
)
from lwa_catalog.constants import VLASS_BMAJ_DEG, VLASS_DEC_MIN_DEG


def _write_mini_vlass_csv(path) -> None:
    lines = [
        "Component_name,RA,DEC,Peak_flux,Total_flux,Maj,Min,PA,BMAJ,BMIN,BPA,"
        "Duplicate_flag,Quality_flag,S_Code",
        "VLASS1QLCIR J100000.00+200000.0,1,10.0,20.0,0.05,0.08,1.0,0.8,45.0,"
        "2.5,2.0,0.0,0.0,0,S",
        "VLASS1QLCIR J100003.60+200000.0,2,10.01,20.0,0.04,0.07,1.0,0.8,45.0,"
        "2.5,2.0,0.0,0.0,0,S",
        "VLASS1QLCIR J100000.00+200000.0,3,10.0,20.0,0.03,0.06,1.0,0.8,45.0,"
        "2.5,2.0,0.0,5.0,0,E",
    ]
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(lines))


def _meta_row(
    *,
    meta_id: int = 0,
    ra: float = 10.0,
    dec: float = 20.0,
    bmaj: float = 0.5,
) -> dict:
    return {
        "meta_id": meta_id,
        "RA": ra,
        "DEC": dec,
        "BMAJ_match": bmaj,
        "bands_present": "Full,Blue",
        "Peak_flux": 1.0,
        "origin_band": "Full",
    }


def test_load_vlass_catalog_applies_quality_filter(tmp_path) -> None:
    catalog_path = tmp_path / "vlass.csv.gz"
    _write_mini_vlass_csv(catalog_path)

    loaded = load_vlass_catalog(catalog_path)

    assert len(loaded) == 2
    assert (loaded["BMAJ"] > 0).all()
    assert loaded.iloc[0]["Peak_flux"] == pytest.approx(0.05)


def test_load_vlass_catalog_missing_file(tmp_path) -> None:
    missing = tmp_path / "missing.csv.gz"
    with pytest.raises(FileNotFoundError, match="VLASS catalog not found"):
        load_vlass_catalog(missing)


def test_footprint_filter_vlass_applies_dec_limit() -> None:
    vlass = pd.DataFrame(
        {
            "RA": [0.0, 0.0, 0.0],
            "DEC": [-45.0, -35.0, 5.0],
            "BMAJ": [VLASS_BMAJ_DEG] * 3,
        }
    )
    lwa = pd.DataFrame({"RA": [1.0, 2.0], "DEC": [-38.0, 6.0]})
    filtered = _footprint_filter_vlass(vlass, lwa, dec_min_deg=VLASS_DEC_MIN_DEG)
    assert len(filtered) == 2
    assert set(filtered["DEC"].tolist()) == {-35.0, 5.0}


def test_match_completeness_single_hit(tmp_path) -> None:
    catalog_path = tmp_path / "vlass.csv.gz"
    _write_mini_vlass_csv(catalog_path)
    vlass = load_vlass_catalog(catalog_path).iloc[:1]
    meta = pd.DataFrame([_meta_row(meta_id=7)])
    result = match_catalog_to_vlass(meta, vlass=vlass)
    assert result.summary["match_completeness"] == pytest.approx(1.0)
    assert int(result.meta_flags.iloc[0]["n_vlass"]) == 1


def test_select_unique_vlass_matches() -> None:
    flags = pd.DataFrame({"meta_id": [0, 1, 2], "n_vlass": [0, 1, 2]})
    unique = select_unique_vlass_matches(flags)
    assert list(unique["meta_id"]) == [1]


def test_summarize_vlass_match_includes_key_metrics(tmp_path) -> None:
    catalog_path = tmp_path / "vlass.csv.gz"
    _write_mini_vlass_csv(catalog_path)
    vlass = load_vlass_catalog(catalog_path)
    meta = pd.DataFrame([_meta_row()])
    result = match_catalog_to_vlass(meta, vlass=vlass)
    text = summarize_vlass_match(result)
    assert "Match completeness:" in text
    assert "3.00 GHz" in text
