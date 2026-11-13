"""Tests for GLEAM EG catalog cross-match and multi-band attach."""

from __future__ import annotations

import pandas as pd
import pytest
from astropy.table import Table

from lwa_catalog.analyze.associations import (
    association_rows_from_survey_attach,
    harmonize_class_native,
)
from lwa_catalog.analyze.crossmatch_radius import LWA_CROSSMATCH_RADIUS_BEAM
from lwa_catalog.analyze.gleam import (
    GLEAM_ASSOC_BAND,
    GleamMatchConfig,
    _footprint_filter_gleam,
    attach_gleam_to_metacatalog,
    load_gleam_catalog,
    match_catalog_to_gleam,
    select_unique_gleam_matches,
    summarize_gleam_match,
)
from lwa_catalog.analyze.spectral import resolve_sed_bands
from lwa_catalog.constants import (
    GLEAM_BANDS,
    GLEAM_BMAJ_DEG,
    GLEAM_DEC_MAX_DEG,
)


def _write_mini_gleam_fits(path) -> None:
    """CDS-like mini GLEAM table (Jy; err_* in degrees; psf in arcsec)."""
    table = Table(
        {
            "Name": ["GLEAM J004000+250000", "GLEAM J004005+250000", "GLEAM J120000+400000"],
            "RAJ2000": [10.0, 10.01, 180.0],
            "DEJ2000": [25.0, 25.0, 40.0],  # last row above Dec=+30
            "err_RAJ2000": [0.0005, 0.0005, 0.0005],
            "err_DEJ2000": [0.0004, 0.0004, 0.0004],
            "peak_flux_wide": [1.0, 0.5, 2.0],
            "err_peak_flux_wide": [0.1, 0.05, 0.2],
            "int_flux_wide": [1.2, 0.6, 2.5],
            "err_int_flux_wide": [0.12, 0.06, 0.25],
            "peak_flux_076": [1.5, 0.7, 3.0],
            "err_peak_flux_076": [0.15, 0.07, 0.3],
            "int_flux_076": [1.8, 0.8, 3.5],
            "err_int_flux_076": [0.18, 0.08, 0.35],
            "peak_flux_227": [0.4, 0.2, 0.9],
            "err_peak_flux_227": [0.04, 0.02, 0.09],
            "int_flux_227": [0.5, 0.25, 1.0],
            "err_int_flux_227": [0.05, 0.025, 0.1],
            "psf_a_wide": [120.0, 120.0, 120.0],
            "psf_b_wide": [110.0, 110.0, 110.0],
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
        "bands_present": "78MHz,73MHz",
        "Peak_flux": 1.0,
        "origin_band": "78MHz",
        "astrometry_band": "78MHz",
    }


def test_load_gleam_catalog_fits(tmp_path) -> None:
    path = tmp_path / "gleam.fits"
    _write_mini_gleam_fits(path)
    df = load_gleam_catalog(path)
    assert len(df) == 3
    assert {"RA", "DEC", "Peak_flux", "Total_flux", "BMAJ", "BMIN", "Name"}.issubset(
        df.columns
    )
    assert "Peak_flux_GLEAM76" in df.columns
    assert "Total_flux_GLEAM227" in df.columns
    assert float(df["Peak_flux_GLEAM76"].iloc[0]) == pytest.approx(1.5)
    assert float(df["BMAJ"].iloc[0]) == pytest.approx(120.0 / 3600.0)


def test_load_gleam_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="GLEAM catalog not found"):
        load_gleam_catalog(tmp_path / "missing.fits")


def test_footprint_filter_applies_dec_max() -> None:
    gleam = pd.DataFrame(
        {
            "RA": [1.0, 2.0, 3.0],
            "DEC": [10.0, 25.0, 40.0],
            "Peak_flux": [1.0, 1.0, 1.0],
            "BMAJ": [GLEAM_BMAJ_DEG] * 3,
        }
    )
    lwa = pd.DataFrame({"DEC": [-10.0, 50.0]})
    out = _footprint_filter_gleam(gleam, lwa, dec_max_deg=GLEAM_DEC_MAX_DEG)
    assert list(out["DEC"]) == [10.0, 25.0]


def test_match_catalog_to_gleam_unique_hit(tmp_path) -> None:
    path = tmp_path / "gleam.fits"
    _write_mini_gleam_fits(path)
    gleam = load_gleam_catalog(path)
    gleam = gleam.loc[gleam["RA"] == 10.0].reset_index(drop=True)
    meta = pd.DataFrame([_meta_row(meta_id=7, ra=10.0, dec=25.0, bmaj=0.5)])
    result = match_catalog_to_gleam(
        meta,
        gleam=gleam,
        config=GleamMatchConfig(
            catalog_path=path,
            lwa_radius=LWA_CROSSMATCH_RADIUS_BEAM,
        ),
    )
    assert int(result.summary["n_meta_matched"]) == 1
    assert int(result.meta_flags.iloc[0]["n_gleam"]) == 1
    unique = select_unique_gleam_matches(result.meta_flags)
    assert len(unique) == 1
    text = summarize_gleam_match(result)
    assert "GLEAM" in text
    assert "76" in text


def test_match_respects_lwa_match_frame(tmp_path) -> None:
    path = tmp_path / "gleam.fits"
    _write_mini_gleam_fits(path)
    gleam = load_gleam_catalog(path)
    meta = pd.DataFrame([_meta_row(meta_id=1, ra=10.0, dec=25.0, bmaj=0.5)])
    lwa_match = pd.DataFrame(
        {"RA": [180.0], "DEC": [25.0], "BMAJ": [0.01]},
        index=meta.index,
    )
    result = match_catalog_to_gleam(
        meta,
        gleam=gleam,
        config=GleamMatchConfig(catalog_path=path),
        lwa_match=lwa_match,
    )
    assert int(result.summary["n_meta_matched"]) == 0


def test_attach_gleam_writes_narrowband_fluxes(tmp_path) -> None:
    path = tmp_path / "gleam.fits"
    _write_mini_gleam_fits(path)
    gleam = load_gleam_catalog(path)
    gleam = gleam.loc[gleam["RA"] == 10.0].reset_index(drop=True)
    meta = pd.DataFrame([_meta_row(meta_id=3, ra=10.0, dec=25.0, bmaj=0.5)])
    out = attach_gleam_to_metacatalog(
        meta,
        gleam,
        lwa_radius=LWA_CROSSMATCH_RADIUS_BEAM,
        footprint_filter=False,
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert float(row["RA"]) == 10.0
    assert row["origin_band"] == "78MHz"
    assert int(row["n_assoc_GLEAM"]) == 1
    assert float(row["Peak_flux_GLEAM76"]) == pytest.approx(1.5)
    assert float(row["Total_flux_GLEAM76"]) == pytest.approx(1.8)
    assert float(row["Peak_flux_GLEAM227"]) == pytest.approx(0.4)
    assert "GLEAM" in str(row["bands_present"]).split(",")

    assoc = association_rows_from_survey_attach(out, GLEAM_ASSOC_BAND)
    assert len(assoc) == 1
    assert assoc.iloc[0]["catalog"] == "GLEAM"
    assert assoc.iloc[0]["ref_id"].startswith("GLEAM")
    assert harmonize_class_native("GLEAM", "GLEAM") == "radio"

    bands = resolve_sed_bands(out.columns)
    assert "GLEAM76" in bands
    assert "GLEAM227" in bands
    assert set(GLEAM_BANDS).issuperset({"GLEAM76", "GLEAM227"})
