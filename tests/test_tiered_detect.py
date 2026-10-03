"""Tests for two-tier PyBDSF GAUL fusion."""

from __future__ import annotations

import pandas as pd

from lwa_catalog.create.tiered_detect import (
    DEFAULT_TIER2_BDSF_KW,
    fuse_gaul_m_with_tier2_s,
    merge_tier2_bdsf_kw,
)


def _row(
    *,
    ra: float,
    dec: float,
    peak: float,
    s_code: str,
    bmaj: float = 0.5,
    source_id: int = 0,
    isl_id: int = 0,
) -> dict:
    return {
        "RA": ra,
        "DEC": dec,
        "Peak_flux": peak,
        "Total_flux": peak,
        "E_Peak_flux": 0.1,
        "E_Total_flux": 0.1,
        "Maj": bmaj,
        "Min": bmaj,
        "PA": 0.0,
        "DC_Maj": bmaj,
        "DC_Min": bmaj,
        "DC_PA": 0.0,
        "Resid_Isl_rms": 0.01,
        "Resid_Isl_mean": 0.0,
        "S_Code": s_code,
        "Source_id": source_id,
        "Isl_id": isl_id,
        "BMAJ": bmaj,
        "BMIN": bmaj,
        "BPA": 0.0,
    }


def test_merge_tier2_bdsf_kw_defaults_and_overrides() -> None:
    merged = merge_tier2_bdsf_kw(
        {"thresh": "hard", "thresh_isl": 2.0, "thresh_pix": 3.0, "ncores": 1},
        {},
    )
    assert merged["thresh"] == "hard"
    assert merged["thresh_isl"] == DEFAULT_TIER2_BDSF_KW["thresh_isl"]
    assert merged["thresh_pix"] == DEFAULT_TIER2_BDSF_KW["thresh_pix"]
    assert merged["ncores"] == 1

    overridden = merge_tier2_bdsf_kw(
        {"thresh_isl": 2.0},
        {"thresh_isl": 8.0, "quiet": True},
    )
    assert overridden["thresh_isl"] == 8.0
    assert overridden["thresh_pix"] == 4.0
    assert overridden["quiet"] is True


def test_fuse_empty_tier2_returns_tier1() -> None:
    t1 = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=1.0, s_code="M")])
    out = fuse_gaul_m_with_tier2_s(t1, pd.DataFrame())
    assert len(out) == 1
    assert out.iloc[0]["S_Code"] == "M"


def test_fuse_replaces_m_clump_with_single_s() -> None:
    # Three over-decomposed M Gaussians near one tier-2 S.
    t1 = pd.DataFrame(
        [
            _row(ra=10.00, dec=20.0, peak=1.0, s_code="M", source_id=0),
            _row(ra=10.05, dec=20.0, peak=0.8, s_code="M", source_id=1),
            _row(ra=10.10, dec=20.0, peak=0.7, s_code="M", source_id=2),
            _row(ra=30.00, dec=20.0, peak=0.5, s_code="S", source_id=3),
        ]
    )
    t2 = pd.DataFrame(
        [
            _row(ra=10.05, dec=20.0, peak=2.5, s_code="S", source_id=100),
            _row(ra=30.00, dec=20.0, peak=0.4, s_code="S", source_id=101),
        ]
    )
    out = fuse_gaul_m_with_tier2_s(t1, t2)
    codes = out["S_Code"].astype(str).tolist()
    assert codes.count("M") == 0
    assert (out["Source_id"] == 100).sum() == 1
    # Distant tier-1 S kept; its tier-2 twin not inserted unless replacing M.
    assert (out["Source_id"] == 3).sum() == 1
    assert len(out) == 2


def test_fuse_keeps_unmatched_m() -> None:
    t1 = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=1.0, s_code="M")])
    t2 = pd.DataFrame([_row(ra=50.0, dec=20.0, peak=2.0, s_code="S")])
    out = fuse_gaul_m_with_tier2_s(t1, t2)
    assert len(out) == 1
    assert out.iloc[0]["S_Code"] == "M"


def test_fuse_picks_nearest_then_brightest_s() -> None:
    t1 = pd.DataFrame([_row(ra=10.0, dec=20.0, peak=1.0, s_code="M")])
    t2 = pd.DataFrame(
        [
            _row(ra=10.20, dec=20.0, peak=5.0, s_code="S", source_id=1),  # farther, brighter
            _row(ra=10.05, dec=20.0, peak=1.0, s_code="S", source_id=2),  # nearer
        ]
    )
    out = fuse_gaul_m_with_tier2_s(t1, t2)
    assert len(out) == 1
    assert int(out.iloc[0]["Source_id"]) == 2


def test_fuse_collapses_many_m_to_one_s() -> None:
    t1 = pd.DataFrame(
        [
            _row(ra=10.00, dec=20.0, peak=1.0, s_code="M", source_id=0),
            _row(ra=10.05, dec=20.0, peak=0.9, s_code="M", source_id=1),
        ]
    )
    t2 = pd.DataFrame([_row(ra=10.02, dec=20.0, peak=3.0, s_code="S", source_id=99)])
    out = fuse_gaul_m_with_tier2_s(t1, t2)
    assert len(out) == 1
    assert int(out.iloc[0]["Source_id"]) == 99
    assert out.iloc[0]["S_Code"] == "S"
