"""Tests for LoDeSS cross-match."""

from __future__ import annotations

import pandas as pd
import pytest
from astropy.table import Table

from lwa_catalog.analyze.crossmatch_radius import LWA_CROSSMATCH_RADIUS_BEAM
from lwa_catalog.analyze.lodess import (
    LodesMatchConfig,
    _footprint_filter_lodess,
    load_lodess_catalog,
    match_catalog_to_lodess,
    select_unique_lodess_matches,
    summarize_lodess_match,
)
from lwa_catalog.constants import LODES_BMAJ_DEG, LODES_DEC_MIN_DEG, LODES_FREQ_HZ


def _write_mini_lodess_fits(path) -> None:
    table = Table(
        {
            "RA": [10.0, 10.01, 50.0],
            "DEC": [25.0, 25.0, 10.0],  # last row below Dec cut
            "E_RA": [0.0005, 0.0005, 0.0005],
            "E_DEC": [0.0004, 0.0004, 0.0004],
            "Total_flux": [1.0, 0.5, 2.0],
            "E_Total_flux": [0.1, 0.05, 0.2],
            "Peak_flux": [0.8, 0.4, 1.5],
            "E_Peak_flux": [0.08, 0.04, 0.15],
            "Maj": [0.02, 0.02, 0.02],
            "Min": [0.015, 0.015, 0.015],
            "PA": [45.0, 10.0, 0.0],
            "S_Code": ["S", "S", "S"],
            "Source_id": [0, 1, 2],
            "Isl_id": [0, 1, 2],
            "Gaus_id": [0, 1, 2],
        }
    )
    table.write(path, format="fits", overwrite=True)


def _meta_row(
    *,
    meta_id: int = 0,
    ra: float = 10.0,
    dec: float = 25.0,
    bmaj: float = 0.5,
) -> dict:
    return {
        "meta_id": meta_id,
        "RA": ra,
        "DEC": dec,
        "BMAJ_match": bmaj,
        "bands_present": "23MHz,27MHz",
        "Peak_flux": 1.0,
        "origin_band": "23MHz",
    }


def test_load_lodess_catalog_fits(tmp_path) -> None:
    path = tmp_path / "lodess_gaul.fits"
    _write_mini_lodess_fits(path)
    df = load_lodess_catalog(path)
    assert len(df) == 3
    assert set(["RA", "DEC", "Peak_flux", "Total_flux", "BMAJ", "BMIN"]).issubset(
        df.columns
    )
    assert float(df["BMAJ"].iloc[0]) == pytest.approx(LODES_BMAJ_DEG)


def test_load_lodess_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="LoDeSS catalog not found"):
        load_lodess_catalog(tmp_path / "missing.fits")


def test_footprint_filter_applies_dec_min() -> None:
    lodess = pd.DataFrame(
        {
            "RA": [1.0, 2.0, 3.0],
            "DEC": [10.0, 25.0, 40.0],
            "Peak_flux": [1.0, 1.0, 1.0],
            "BMAJ": [LODES_BMAJ_DEG] * 3,
        }
    )
    lwa = pd.DataFrame({"DEC": [15.0, 50.0]})
    out = _footprint_filter_lodess(lodess, lwa, dec_min_deg=LODES_DEC_MIN_DEG)
    assert list(out["DEC"]) == [25.0, 40.0]


def test_match_catalog_to_lodess_unique_hit(tmp_path) -> None:
    path = tmp_path / "lodess_gaul.fits"
    _write_mini_lodess_fits(path)
    # Keep only the Dec≥20° source nearest (10, 25); drop the 10.01 neighbor.
    lodess = load_lodess_catalog(path)
    lodess = lodess.loc[lodess["RA"] == 10.0].reset_index(drop=True)
    meta = pd.DataFrame([_meta_row(meta_id=7, ra=10.0, dec=25.0, bmaj=0.5)])
    result = match_catalog_to_lodess(
        meta,
        lodess=lodess,
        config=LodesMatchConfig(
            catalog_path=path,
            lwa_radius=LWA_CROSSMATCH_RADIUS_BEAM,
        ),
    )
    assert int(result.summary["n_meta_matched"]) == 1
    assert int(result.meta_flags.iloc[0]["n_lodess"]) == 1
    unique = select_unique_lodess_matches(result.meta_flags)
    assert len(unique) == 1
    text = summarize_lodess_match(result)
    assert "LoDeSS" in text
    assert f"{LODES_FREQ_HZ / 1e6:.0f} MHz" in text


def test_match_respects_lwa_match_frame(tmp_path) -> None:
    path = tmp_path / "lodess_gaul.fits"
    _write_mini_lodess_fits(path)
    lodess = load_lodess_catalog(path)
    meta = pd.DataFrame([_meta_row(meta_id=1, ra=10.0, dec=25.0, bmaj=0.5)])
    # Shift the match frame far from both LoDeSS sources near (10, 25).
    lwa_match = pd.DataFrame(
        {"RA": [180.0], "DEC": [25.0], "BMAJ": [0.01]},
        index=meta.index,
    )
    result = match_catalog_to_lodess(
        meta,
        lodess=lodess,
        config=LodesMatchConfig(catalog_path=path),
        lwa_match=lwa_match,
    )
    assert int(result.summary["n_meta_matched"]) == 0
