"""Tests for ASKAP RACS Stokes I source-catalogue cross-match."""

from __future__ import annotations

import pandas as pd
import pytest

from lwa_catalog.analyze.crossmatch_radius import (
    LWA_CROSSMATCH_RADIUS_BEAM,
    CrossmatchRadiusSpec,
)
from lwa_catalog.analyze.racs import (
    RacsMatchConfig,
    _finalize_racs_frame,
    _footprint_filter_racs,
    load_racs_catalog,
    match_catalog_to_racs,
    select_unique_racs_matches,
    summarize_racs_match,
)
from lwa_catalog.constants import (
    RACS_BMAJ_DEG,
    RACS_DEC_MAX_DEG,
    RACS_FREQ_HZ,
)


def _raw_tap_like_row(
    *,
    ra: float = 10.0,
    dec: float = 20.0,
    dec_corr: float | None = None,
    peak_mjy: float = 100.0,
    total_mjy: float = 120.0,
    flag: float = 0.0,
    name: str = "J004000+200000",
    psf_maj_arcsec: float = 25.0,
) -> dict:
    return {
        "ra": ra,
        "dec": dec,
        "dec_corr": dec if dec_corr is None else dec_corr,
        "e_ra": 0.0003,
        "e_dec": 0.0003,
        "peak_flux": peak_mjy,
        "total_flux": total_mjy,
        "e_peak_flux": 5.0,
        "e_total_flux": 6.0,
        "maj_axis": 30.0,
        "min_axis": 20.0,
        "pa": 45.0,
        "psf_maj": psf_maj_arcsec,
        "psf_min": psf_maj_arcsec,
        "flag": flag,
        "s_code": "S",
        "name": name,
        "source_id": "src0",
    }


def _write_mini_racs_parquet(path, rows: list[dict] | None = None) -> None:
    frame = pd.DataFrame(rows or [_raw_tap_like_row()])
    frame.to_parquet(path, index=False)


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
        "bands_present": "82MHz,73MHz",
        "Peak_flux": 1.0,
        "origin_band": "82MHz",
    }


def test_load_racs_catalog_mjy_to_jy_and_dec_corr(tmp_path) -> None:
    path = tmp_path / "racs_mid.parquet"
    _write_mini_racs_parquet(
        path,
        [
            _raw_tap_like_row(dec=19.0, dec_corr=20.5, peak_mjy=100.0, total_mjy=150.0),
            _raw_tap_like_row(ra=11.0, dec=21.0, flag=2.0, name="spurious"),
        ],
    )
    df = load_racs_catalog(path, band="RACSmid")
    assert len(df) == 1
    row = df.iloc[0]
    assert float(row["DEC"]) == pytest.approx(20.5)
    assert float(row["Peak_flux"]) == pytest.approx(0.1)
    assert float(row["Total_flux"]) == pytest.approx(0.15)
    assert float(row["BMAJ"]) == pytest.approx(25.0 / 3600.0)
    assert row["source_file"] == "J004000+200000"
    assert row["Name"] == "J004000+200000"


def test_load_racs_missing_file_mentions_download(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Download with"):
        load_racs_catalog(tmp_path / "missing.parquet", band="RACSlow2")


def test_finalize_prefers_band_default_bmaj_when_psf_missing() -> None:
    raw = pd.DataFrame(
        {
            "RA": [10.0],
            "DEC": [20.0],
            "Peak_flux": [0.05],
            "Total_flux": [0.06],
            "Name": ["n0"],
        }
    )
    out = _finalize_racs_frame(raw, band="RACShigh")
    assert float(out.iloc[0]["BMAJ"]) == pytest.approx(RACS_BMAJ_DEG["RACShigh"])


def test_footprint_filter_respects_mid_dec_max() -> None:
    racs = pd.DataFrame(
        {
            "RA": [1.0, 2.0, 3.0],
            "DEC": [10.0, 25.0, 35.0],
            "Peak_flux": [1.0, 1.0, 1.0],
            "BMAJ": [RACS_BMAJ_DEG["RACSmid"]] * 3,
        }
    )
    lwa = pd.DataFrame({"DEC": [5.0, 40.0]})
    out = _footprint_filter_racs(
        racs, lwa, dec_max_deg=RACS_DEC_MAX_DEG["RACSmid"]
    )
    assert list(out["DEC"]) == [10.0, 25.0]


def test_match_catalog_to_racs_unique_hit(tmp_path) -> None:
    path = tmp_path / "racs_mid.parquet"
    _write_mini_racs_parquet(
        path,
        [
            _raw_tap_like_row(ra=10.0, dec=20.0, peak_mjy=80.0, total_mjy=90.0),
            _raw_tap_like_row(ra=50.0, dec=10.0, name="far"),
        ],
    )
    racs = load_racs_catalog(path, band="RACSmid")
    meta = pd.DataFrame([_meta_row(meta_id=7, ra=10.0, dec=20.0, bmaj=0.5)])
    result = match_catalog_to_racs(
        meta,
        racs=racs,
        config=RacsMatchConfig(
            band="RACSmid",
            catalog_path=path,
            lwa_radius=LWA_CROSSMATCH_RADIUS_BEAM,
            reference_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=25.0),
        ),
    )
    assert result.band == "RACSmid"
    assert int(result.summary["n_meta_matched"]) == 1
    assert int(result.meta_flags.iloc[0]["n_racs"]) == 1
    unique = select_unique_racs_matches(result.meta_flags)
    assert len(unique) == 1
    text = summarize_racs_match(result)
    assert "RACSmid" in text
    assert f"{RACS_FREQ_HZ['RACSmid'] / 1e6:.1f}" in text or "1367" in text


def test_match_multi_and_oversplit(tmp_path) -> None:
    path = tmp_path / "racs_low2.parquet"
    _write_mini_racs_parquet(
        path,
        [
            _raw_tap_like_row(ra=10.0, dec=20.0, name="a"),
            _raw_tap_like_row(ra=10.01, dec=20.0, name="b"),
        ],
    )
    racs = load_racs_catalog(path, band="RACSlow2")
    meta = pd.DataFrame(
        [
            _meta_row(meta_id=0, ra=10.0, dec=20.0, bmaj=1.0),
            _meta_row(meta_id=1, ra=10.005, dec=20.0, bmaj=1.0),
        ]
    )
    result = match_catalog_to_racs(
        meta,
        racs=racs,
        config=RacsMatchConfig(
            band="RACSlow2",
            catalog_path=path,
            lwa_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=3600.0),
            reference_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=3600.0),
        ),
    )
    assert int(result.summary["n_meta_multi_racs"]) >= 1
    assert int(result.summary["n_racs_oversplit"]) >= 1


def test_match_respects_lwa_match_frame(tmp_path) -> None:
    path = tmp_path / "racs_high.parquet"
    _write_mini_racs_parquet(path, [_raw_tap_like_row(ra=10.0, dec=20.0)])
    racs = load_racs_catalog(path, band="RACShigh")
    meta = pd.DataFrame([_meta_row(meta_id=1, ra=10.0, dec=20.0, bmaj=0.5)])
    lwa_match = pd.DataFrame(
        {"RA": [180.0], "DEC": [20.0], "BMAJ": [0.01]},
        index=meta.index,
    )
    result = match_catalog_to_racs(
        meta,
        racs=racs,
        config=RacsMatchConfig(band="RACShigh", catalog_path=path),
        lwa_match=lwa_match,
    )
    assert int(result.summary["n_meta_matched"]) == 0
