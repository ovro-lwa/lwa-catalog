"""Tests for NVSS cross-match."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.crossmatch_radius import (
    LWA_CROSSMATCH_RADIUS_BEAM,
    NVSS_REFERENCE_RADIUS_LOCALIZATION,
    match_radius_deg,
)
from lwa_catalog.analyze.nvss import (
    NvssMatchConfig,
    _catalog_match_frame,
    _footprint_filter_nvss,
    load_nvss_catalog,
    match_catalog_to_nvss,
    predict_flux_at_frequency_hz,
    select_unique_nvss_matches,
    summarize_nvss_match,
)
from lwa_catalog.constants import NVSS_BMAJ_DEG, NVSS_DEC_MIN_DEG


def _vizier_line(
    *,
    ra_h: int = 0,
    ra_m: int = 40,
    ra_s: float = 0.0,
    dec_sign: str = "+",
    dec_d: int = 20,
    dec_m: int = 0,
    dec_s: float = 0.0,
    e_ras: float = 0.1,
    e_des: float = 1.5,
    s14_mjy: float = 50.0,
) -> str:
    """Build one 158-char VizieR VIII/65 fixed-width row."""
    # Positions from ReadMe (1-indexed inclusive).
    buf = [" "] * 158
    def put(start: int, end: int, text: str) -> None:
        s = text[: end - start + 1].rjust(end - start + 1)
        buf[start - 1 : end] = list(s)

    put(1, 8, "C0100P20")
    put(10, 16, f"{100.0:7.2f}")
    put(18, 24, f"{200.0:7.2f}")
    put(26, 39, "004000+200000")
    put(41, 42, f"{ra_h:2d}")
    put(44, 45, f"{ra_m:2d}")
    put(47, 51, f"{ra_s:5.2f}")
    put(53, 53, dec_sign)
    put(54, 55, f"{dec_d:2d}")
    put(57, 58, f"{dec_m:2d}")
    put(60, 63, f"{dec_s:4.1f}")
    put(65, 69, f"{e_ras:5.2f}")
    put(71, 74, f"{e_des:4.1f}")
    put(76, 83, f"{s14_mjy:8.1f}")
    put(85, 91, f"{1.0:7.1f}")
    put(93, 93, "<")
    put(94, 98, f"{45.0:5.1f}")
    put(100, 100, "<")
    put(101, 105, f"{45.0:5.1f}")
    return "".join(buf)


def _write_mini_nvss_vizier(path, *, gzipped: bool = True) -> None:
    lines = [
        _vizier_line(ra_h=0, ra_m=40, ra_s=0.0, dec_d=20, e_ras=0.10, e_des=1.5, s14_mjy=50.0),
        _vizier_line(ra_h=0, ra_m=40, ra_s=2.4, dec_d=20, e_ras=0.20, e_des=2.0, s14_mjy=30.0),
    ]
    text = "\n".join(lines) + "\n"
    if gzipped:
        with gzip.open(path, "wt") as handle:
            handle.write(text)
    else:
        path.write_text(text)


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
                "E_RA": 1.5 / 3600.0,
                "E_DEC": 1.5 / 3600.0,
                "Peak_intensity": 0.05,
                "Total_flux": 0.05,
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


def test_load_nvss_catalog_from_vizier_gzip(tmp_path) -> None:
    catalog_path = tmp_path / "nvss_vizier.dat.gz"
    _write_mini_nvss_vizier(catalog_path)

    loaded = load_nvss_catalog(catalog_path)

    assert len(loaded) == 2
    assert "E_RA" in loaded.columns
    assert "E_DEC" in loaded.columns
    assert "Peak_intensity" in loaded.columns
    assert "Total_flux" in loaded.columns
    assert (loaded["BMAJ"] == NVSS_BMAJ_DEG).all()
    # 0h40m0s = 10 deg
    assert loaded.iloc[0]["RA"] == pytest.approx(10.0)
    assert loaded.iloc[0]["DEC"] == pytest.approx(20.0)
    assert loaded.iloc[0]["Peak_intensity"] == pytest.approx(0.05)
    # e_RAs=0.10s * 15 * cos(20°) / 3600 degrees
    expected_e_ra = 0.10 * 15.0 * np.cos(np.deg2rad(20.0)) / 3600.0
    assert loaded.iloc[0]["E_RA"] == pytest.approx(expected_e_ra)
    assert loaded.iloc[0]["E_DEC"] == pytest.approx(1.5 / 3600.0)


def test_load_nvss_catalog_from_parquet(tmp_path) -> None:
    catalog_path = tmp_path / "nvss.parquet"
    frame = _nvss_rows([(10.0, 20.0)])
    frame.to_parquet(catalog_path, index=False)
    loaded = load_nvss_catalog(catalog_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["RA"] == pytest.approx(10.0)


def test_nvss_default_radius_uses_localization() -> None:
    row = pd.Series({"E_RA": 0.001, "E_DEC": 0.0005})
    radius = match_radius_deg(row, NVSS_REFERENCE_RADIUS_LOCALIZATION)
    assert radius == pytest.approx(float(np.hypot(0.001, 0.0005)))
    assert NvssMatchConfig().reference_radius is NVSS_REFERENCE_RADIUS_LOCALIZATION


def test_load_nvss_catalog_missing_file(tmp_path) -> None:
    missing = tmp_path / "missing.parquet"
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


def test_predict_flux_prefers_total_and_falls_back_to_peak() -> None:
    peak_only = pd.Series(
        {
            "spec_peak_n_terms": 2,
            "spec_peak_nu0_mhz": 55.0,
            "spec_peak_a0": np.log(1.0),
            "spec_peak_a1": -0.7,
        }
    )
    # Default prefer_total should fall back to peak when only peak fit exists.
    flux = predict_flux_at_frequency_hz(peak_only, 1.4e9)
    assert flux == pytest.approx(1.0 * (1.4e9 / 55e6) ** -0.7, rel=1e-6)

    both = pd.Series(
        {
            "spec_total_n_terms": 2,
            "spec_total_nu0_mhz": 55.0,
            "spec_total_a0": np.log(2.0),
            "spec_total_a1": -0.5,
            "spec_peak_n_terms": 2,
            "spec_peak_nu0_mhz": 55.0,
            "spec_peak_a0": np.log(1.0),
            "spec_peak_a1": -0.7,
        }
    )
    flux_total = predict_flux_at_frequency_hz(both, 1.4e9, flux_kind="total")
    assert flux_total == pytest.approx(2.0 * (1.4e9 / 55e6) ** -0.5, rel=1e-6)


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
