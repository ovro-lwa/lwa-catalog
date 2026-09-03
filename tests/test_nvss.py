"""Tests for NVSS cross-match."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from lwa_catalog.analyze.nvss import (
    _catalog_match_frame,
    _footprint_filter_nvss,
    load_nvss_catalog,
    match_catalog_to_nvss,
    predict_flux_at_frequency_hz,
    select_unique_nvss_matches,
    summarize_nvss_match,
)
from lwa_catalog.analyze.crossmatch_radius import LWA_CROSSMATCH_RADIUS_BEAM
from lwa_catalog.constants import NVSS_BMAJ_DEG, NVSS_DEC_MIN_DEG


def _write_mini_nvss_fits(path) -> None:
    table = Table(
        {
            "RA(2000)": [10.0, 10.01],
            "DEC(2000)": [20.0, 20.0],
            "PEAK INT": [0.05, 0.03],
            "MAJOR AX": [NVSS_BMAJ_DEG, NVSS_BMAJ_DEG],
            "MINOR AX": [NVSS_BMAJ_DEG, NVSS_BMAJ_DEG],
            "POSANGLE": [0.0, 0.0],
            "P FLUX": [0.001, 0.002],
            "FIELD": ["C0100P20", "C0100P20"],
        }
    )
    table.write(path, overwrite=True)


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


def _nvss_rows(rows: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "RA": ra,
                "DEC": dec,
                "Peak_intensity": 0.05,
                "Maj": NVSS_BMAJ_DEG,
                "Min": NVSS_BMAJ_DEG,
                "PA": 0.0,
                "Pol_flux": 0.001,
                "Field": "C0100P20",
                "BMAJ": NVSS_BMAJ_DEG,
                "BMIN": NVSS_BMAJ_DEG,
            }
            for ra, dec in rows
        ]
    )


def test_load_nvss_catalog_assigns_bmaj(tmp_path) -> None:
    catalog_path = tmp_path / "CATALOG.FIT"
    _write_mini_nvss_fits(catalog_path)

    loaded = load_nvss_catalog(catalog_path)

    assert "Peak_intensity" in loaded.columns
    assert len(loaded) == 2
    assert (loaded["BMAJ"] == NVSS_BMAJ_DEG).all()
    assert loaded.iloc[0]["RA"] == pytest.approx(10.0)


def test_load_nvss_catalog_missing_file(tmp_path) -> None:
    missing = tmp_path / "missing.FIT"
    with pytest.raises(FileNotFoundError, match="NVSS catalog not found"):
        load_nvss_catalog(missing)


def test_footprint_filter_nvss_applies_dec_limit() -> None:
    nvss = pd.DataFrame(
        {
            "RA": [0.0, 0.0, 0.0],
            "DEC": [-45.0, -35.0, 5.0],
            "BMAJ": [NVSS_BMAJ_DEG] * 3,
        }
    )
    lwa = pd.DataFrame({"RA": [1.0, 2.0], "DEC": [-38.0, 6.0]})
    filtered = _footprint_filter_nvss(nvss, lwa, dec_min_deg=NVSS_DEC_MIN_DEG)
    assert len(filtered) == 2
    assert set(filtered["DEC"].tolist()) == {-35.0, 5.0}


def test_match_completeness_single_hit() -> None:
    meta = pd.DataFrame([_meta_row(meta_id=7)])
    nvss = _nvss_rows([(10.0, 20.0)])
    result = match_catalog_to_nvss(meta, nvss=nvss)
    assert result.summary["match_completeness"] == pytest.approx(1.0)
    assert int(result.meta_flags.iloc[0]["n_nvss"]) == 1
    assert int(result.meta_flags.iloc[0]["meta_id"]) == 7


def test_match_multi_nvss_per_meta() -> None:
    meta = pd.DataFrame([_meta_row()])
    nvss = _nvss_rows([(10.0, 20.0), (10.01, 20.0)])
    result = match_catalog_to_nvss(meta, nvss=nvss)
    assert result.summary["match_completeness"] == pytest.approx(1.0)
    assert int(result.meta_flags.iloc[0]["n_nvss"]) == 2
    assert int(result.summary["n_meta_multi_nvss"]) == 1


def test_match_oversplit_two_meta_one_nvss() -> None:
    meta = pd.DataFrame(
        [
            _meta_row(meta_id=0, ra=10.0, dec=20.0),
            _meta_row(meta_id=1, ra=10.05, dec=20.0),
        ]
    )
    nvss = _nvss_rows([(10.02, 20.0)])
    result = match_catalog_to_nvss(meta, nvss=nvss)
    assert int(result.summary["n_nvss_oversplit"]) == 1
    assert bool(result.nvss_flags.iloc[0]["oversplit"])
    assert result.nvss_flags.iloc[0]["meta_ids"] == [0, 1]


def test_select_unique_nvss_matches() -> None:
    flags = pd.DataFrame({"meta_id": [0, 1, 2], "n_nvss": [0, 1, 2]})
    unique = select_unique_nvss_matches(flags)
    assert list(unique["meta_id"]) == [1]


def test_predict_flux_at_frequency_hz() -> None:
    row = pd.Series(
        {
            "spec_peak_n_terms": 2,
            "spec_peak_nu0_mhz": 55.0,
            "spec_peak_a0": np.log(1.0),
            "spec_peak_a1": -0.7,
        }
    )
    flux = predict_flux_at_frequency_hz(row, 1.4e9, flux_kind="peak")
    assert flux == pytest.approx(1.0 * (1.4e9 / 55e6) ** -0.7, rel=1e-6)


def test_summarize_nvss_match_includes_key_metrics() -> None:
    meta = pd.DataFrame([_meta_row()])
    nvss = _nvss_rows([(10.0, 20.0)])
    result = match_catalog_to_nvss(meta, nvss=nvss)
    text = summarize_nvss_match(result)
    assert "Match completeness:" in text
    assert "NVSS over-split" in text
    assert "1.40 GHz" in text


def test_catalog_match_frame_uses_primary_coords() -> None:
    catalog = pd.DataFrame(
        [
            {
                "RA": 10.0,
                "DEC": 20.0,
                "RA_Blue": 10.5,
                "DEC_Blue": 20.5,
                "BMAJ_match": 0.5,
            }
        ]
    )
    frame = _catalog_match_frame(catalog, LWA_CROSSMATCH_RADIUS_BEAM)
    assert frame.iloc[0]["RA"] == pytest.approx(10.0)
    assert frame.iloc[0]["BMAJ"] == pytest.approx(0.5)
