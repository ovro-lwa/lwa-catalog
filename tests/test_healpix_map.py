"""Tests for catalog HEALPix / HiPS helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

healpy = pytest.importorskip("healpy")
pytest.importorskip("lwa_healpix")

from lwa_catalog.analyze.healpix_map import (  # noqa: E402
    _FWHM_TO_SIGMA,
    metacatalog_to_healpix,
    metacatalog_to_hips,
    write_healpix_hips,
)


def test_metacatalog_to_healpix_point_weighted() -> None:
    nside = 8
    ra0, dec0 = 0.0, 0.0
    cat = pd.DataFrame(
        {
            "RA": [ra0, ra0],
            "DEC": [dec0, dec0],
            "Peak_flux": [1.0, 2.0],
        }
    )
    m = metacatalog_to_healpix(cat, nside=nside, weight_col="Peak_flux", profile="point")
    assert m.shape == (healpy.nside2npix(nside),)
    assert m.sum() == pytest.approx(3.0)
    pix = healpy.ang2pix(nside, np.deg2rad(90.0 - dec0), np.deg2rad(ra0))
    assert m[pix] == pytest.approx(3.0)


def test_metacatalog_to_healpix_point_counts() -> None:
    cat = pd.DataFrame({"RA": [10.0, 20.0], "DEC": [0.0, 0.0], "Peak_flux": [5.0, 7.0]})
    m = metacatalog_to_healpix(cat, nside=16, weight_col=None, profile="point")
    assert m.sum() == pytest.approx(2.0)


def _pixel_center_radec(nside: int, pix: int) -> tuple[float, float]:
    th, ph = healpy.pix2ang(nside, pix)
    return float(np.rad2deg(ph)), float(90.0 - np.rad2deg(th))


def test_metacatalog_to_healpix_gaussian_peak_and_extent() -> None:
    nside = 64
    pix0 = healpy.ang2pix(nside, np.deg2rad(90.0), np.deg2rad(180.0))
    ra0, dec0 = _pixel_center_radec(nside, pix0)
    peak = 10.0
    maj = 1.0  # deg FWHM
    cat = pd.DataFrame(
        {
            "RA": [ra0],
            "DEC": [dec0],
            "Peak_flux": [peak],
            "Maj": [maj],
            "Min": [maj],
            "PA": [0.0],
        }
    )
    m = metacatalog_to_healpix(cat, nside=nside, profile="gaussian")
    assert m[pix0] == pytest.approx(peak, rel=1e-6)
    # Flux is spread: map sum ≫ Peak_flux
    assert m.sum() > peak
    # Far from source should be near zero
    pix_far = healpy.ang2pix(nside, np.deg2rad(90.0), np.deg2rad(0.0))
    assert m[pix_far] == pytest.approx(0.0)


def test_metacatalog_to_healpix_gaussian_pa_orientation() -> None:
    """PA=0 (N→E): major axis along North; pixels north of center brighter than east."""
    nside = 256
    pix0 = healpy.ang2pix(nside, np.deg2rad(70.0), np.deg2rad(45.0))
    ra0, dec0 = _pixel_center_radec(nside, pix0)
    maj, minor = 2.0, 0.4  # deg FWHM
    cat = pd.DataFrame(
        {
            "RA": [ra0],
            "DEC": [dec0],
            "Peak_flux": [1.0],
            "Maj": [maj],
            "Min": [minor],
            "PA": [0.0],
        }
    )
    m = metacatalog_to_healpix(cat, nside=nside, profile="gaussian")
    # Offset ~0.5 deg along North vs East (within major, outside minor σ scale)
    d = 0.5
    pix_n = healpy.ang2pix(nside, np.deg2rad(90.0 - (dec0 + d)), np.deg2rad(ra0))
    pix_e = healpy.ang2pix(
        nside,
        np.deg2rad(90.0 - dec0),
        np.deg2rad(ra0 + d / np.cos(np.deg2rad(dec0))),
    )
    assert m[pix_n] > 4.0 * m[pix_e]
    # Analytic check using each pixel's true local offset (PA=0: u=d_north, v=d_east)
    sig_maj = maj * _FWHM_TO_SIGMA
    sig_min = minor * _FWHM_TO_SIGMA

    def _expected(pix: int) -> float:
        th, ph = healpy.pix2ang(nside, pix)
        ra_p = float(np.rad2deg(ph))
        dec_p = float(90.0 - np.rad2deg(th))
        d_east = (ra_p - ra0) * np.cos(np.deg2rad(dec0))
        d_north = dec_p - dec0
        return float(np.exp(-0.5 * ((d_north / sig_maj) ** 2 + (d_east / sig_min) ** 2)))

    assert m[pix_n] == pytest.approx(_expected(pix_n), rel=1e-6)
    assert m[pix_e] == pytest.approx(_expected(pix_e), rel=1e-6)


def test_metacatalog_to_healpix_gaussian_missing_shape_uses_pixel_floor() -> None:
    nside = 32
    pix0 = healpy.ang2pix(nside, np.deg2rad(85.0), np.deg2rad(10.0))
    ra0, dec0 = _pixel_center_radec(nside, pix0)
    cat = pd.DataFrame({"RA": [ra0], "DEC": [dec0], "Peak_flux": [3.0]})
    m = metacatalog_to_healpix(cat, nside=nside, profile="gaussian")
    assert m[pix0] == pytest.approx(3.0, rel=1e-6)
    assert (m > 0).sum() > 1


def test_metacatalog_to_healpix_rejects_bad_profile() -> None:
    cat = pd.DataFrame({"RA": [0.0], "DEC": [0.0], "Peak_flux": [1.0]})
    with pytest.raises(ValueError, match="profile"):
        metacatalog_to_healpix(cat, nside=8, profile="box")


def test_write_healpix_hips(tmp_path: Path) -> None:
    nside = 8
    cat = pd.DataFrame(
        {
            "RA": [45.0],
            "DEC": [30.0],
            "Peak_flux": [4.0],
            "Maj": [0.5],
            "Min": [0.3],
            "PA": [30.0],
        }
    )
    m = metacatalog_to_healpix(cat, nside=nside, profile="gaussian")
    out = write_healpix_hips(
        m,
        tmp_path / "hips_map",
        nest=False,
        coord_frame="equatorial",
        threads=False,
    )
    assert out.is_dir()
    assert (out / "properties").is_file()
    assert (out / "index.html").is_file()
    assert list(out.glob("Norder*"))


def test_metacatalog_to_hips(tmp_path: Path) -> None:
    cat = pd.DataFrame(
        {
            "RA": [10.0, 20.0],
            "DEC": [5.0, -5.0],
            "Peak_flux": [1.0, 2.0],
            "Maj": [0.2, 0.3],
            "Min": [0.1, 0.2],
            "PA": [0.0, 90.0],
        }
    )
    out = metacatalog_to_hips(
        cat,
        tmp_path / "hips_cat",
        nside=8,
        threads=False,
        properties={"obs_title": "test catalog HiPS"},
        profile="gaussian",
    )
    assert (out / "properties").is_file()
    props = (out / "properties").read_text()
    assert "hips_pixel_cut" in props or "obs_title" in props
