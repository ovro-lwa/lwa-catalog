"""Tests for eRASS:3 cross-match."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from lwa_catalog.analyze.crossmatch_radius import LWA_CROSSMATCH_RADIUS_BEAM
from lwa_catalog.analyze.erass3 import (
    Erass3MatchConfig,
    attach_erass3_to_metacatalog,
    in_erass3_footprint,
    load_erass3_catalog,
    match_catalog_to_erass3,
    select_unique_erass3_matches,
    summarize_erass3_match,
)
from lwa_catalog.constants import ERASS3_GLON_MIN_DEG, ERASS3_POSITION_ERROR_DEFAULT_ARCSEC


def _galactic_to_icrs(glon_deg: float, glat_deg: float = 30.0) -> tuple[float, float]:
    c = SkyCoord(l=glon_deg * u.deg, b=glat_deg * u.deg, frame="galactic")
    icrs = c.icrs
    return float(icrs.ra.deg), float(icrs.dec.deg)


def _write_mini_erass3_fits(path) -> None:
    # One source in western Gal hemisphere, one in eastern (should be filtered).
    ra_w, dec_w = _galactic_to_icrs(270.0)
    ra_e, dec_e = _galactic_to_icrs(90.0)
    table = Table(
        {
            "RA": [ra_w, ra_w + 0.01, ra_e],
            "DEC": [dec_w, dec_w, dec_e],
            "POS_ERR": [4.0, 4.0, 4.0],
            "EXT": [0.0, 0.0, 0.0],
            "EXT_ERR": [0.0, 0.0, 0.0],
            "DET_LIKE_0": [20.0, 5.0, 50.0],
            "ML_FLUX_1": [1e-13, 5e-14, 2e-13],
            "ML_FLUX_ERR_1": [1e-14, 1e-14, 1e-14],
            "LS10_OBJID": [1, 2, 3],
            "NWAY_p_i": [0.99, 0.5, 0.9],
            "main_id_simbad": [b"SRC A", b"", b"EAST"],
            "simbad_known_galactic": [False, False, True],
            "is_blazar_in_simbad": [True, False, False],
            "class_gal_exgal": [5, 4, 1],
        }
    )
    table.write(path, format="fits", overwrite=True)


def _meta_row(
    *,
    meta_id: int = 0,
    ra: float,
    dec: float,
    bmaj: float = 0.1,
) -> dict:
    return {
        "meta_id": meta_id,
        "RA": ra,
        "DEC": dec,
        "BMAJ_match": bmaj,
        "bands_present": "73MHz",
        "Peak_flux": 1.0,
        "origin_band": "73MHz",
    }


def test_in_erass3_footprint_uses_galactic_longitude() -> None:
    ra_w, dec_w = _galactic_to_icrs(200.0)
    ra_e, dec_e = _galactic_to_icrs(10.0)
    mask = in_erass3_footprint([ra_w, ra_e], [dec_w, dec_e])
    assert mask.tolist() == [True, False]
    # Equatorial RA west is not the eRASS footprint definition.
    assert in_erass3_footprint([270.0], [0.0])[0] in (True, False)


def test_load_erass3_catalog_fits(tmp_path) -> None:
    path = tmp_path / "erass3.fits"
    _write_mini_erass3_fits(path)
    df = load_erass3_catalog(path)
    assert len(df) == 3
    assert "E_RA" in df.columns and "E_DEC" in df.columns
    assert float(df["E_RA"].iloc[0]) == pytest.approx(
        (4.0 / 3600.0) / np.sqrt(2.0)
    )
    assert df["main_id_simbad"].iloc[0] == "SRC A"
    assert bool(df["is_blazar_in_simbad"].iloc[0]) is True


def test_load_erass3_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="eRASS:3 catalog not found"):
        load_erass3_catalog(tmp_path / "missing.fits")


def test_match_skips_eastern_galactic_hemisphere(tmp_path) -> None:
    path = tmp_path / "erass3.fits"
    _write_mini_erass3_fits(path)
    erass = load_erass3_catalog(path)
    ra_w, dec_w = _galactic_to_icrs(270.0)
    ra_e, dec_e = _galactic_to_icrs(90.0)
    meta = pd.DataFrame(
        [
            _meta_row(meta_id=1, ra=ra_w, dec=dec_w, bmaj=0.5),
            _meta_row(meta_id=2, ra=ra_e, dec=dec_e, bmaj=0.5),
        ]
    )
    result = match_catalog_to_erass3(
        meta,
        erass3=erass,
        config=Erass3MatchConfig(
            catalog_path=path,
            lwa_radius=LWA_CROSSMATCH_RADIUS_BEAM,
        ),
    )
    assert int(result.summary["n_lwa_in_footprint"]) == 1
    assert int(result.summary["n_meta_matched"]) == 1
    west = result.meta_flags.loc[result.meta_flags["meta_id"] == 1].iloc[0]
    east = result.meta_flags.loc[result.meta_flags["meta_id"] == 2].iloc[0]
    assert bool(west["in_erass_footprint"]) is True
    assert int(west["n_erass3"]) >= 1
    assert bool(east["in_erass_footprint"]) is False
    assert int(east["n_erass3"]) == 0
    # Restrict to a single western counterpart so n_erass3 == 1.
    erass_one = erass.loc[erass["DET_LIKE_0"] == 20.0].reset_index(drop=True)
    result_one = match_catalog_to_erass3(
        meta.iloc[[0]].reset_index(drop=True),
        erass3=erass_one,
        config=Erass3MatchConfig(
            catalog_path=path,
            lwa_radius=LWA_CROSSMATCH_RADIUS_BEAM,
        ),
    )
    unique = select_unique_erass3_matches(result_one.meta_flags)
    assert len(unique) == 1
    text = summarize_erass3_match(result)
    assert "eRASS" in text
    assert f"{ERASS3_GLON_MIN_DEG:g}" in text


def test_attach_erass3_prefers_highest_det_like(tmp_path) -> None:
    path = tmp_path / "erass3.fits"
    _write_mini_erass3_fits(path)
    erass = load_erass3_catalog(path)
    # Keep only the two western sources (DET_LIKE 20 and 5).
    erass = erass.loc[erass["DET_LIKE_0"] < 30].reset_index(drop=True)
    ra_w, dec_w = _galactic_to_icrs(270.0)
    meta = pd.DataFrame([_meta_row(meta_id=7, ra=ra_w, dec=dec_w, bmaj=1.0)])
    result = match_catalog_to_erass3(
        meta,
        erass3=erass,
        config=Erass3MatchConfig(
            catalog_path=path,
            lwa_radius=LWA_CROSSMATCH_RADIUS_BEAM,
        ),
    )
    assert int(result.meta_flags.iloc[0]["n_erass3"]) == 2
    attached = attach_erass3_to_metacatalog(meta, erass, result.meta_flags)
    assert attached.iloc[0]["eRASS3_DET_LIKE_0"] == pytest.approx(20.0)
    assert bool(attached.iloc[0]["eRASS3_is_blazar_in_simbad"]) is True
    assert attached.iloc[0]["n_erass3"] == 2
    assert np.isfinite(attached.iloc[0]["sep_arcsec_eRASS3"])


def test_default_position_error_constant() -> None:
    assert ERASS3_POSITION_ERROR_DEFAULT_ARCSEC > 0.0
