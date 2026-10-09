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
    catalog_with_peak_from_total_flux,
    lst_merged_catalog_for_healpix,
    metacatalog_to_healpix,
    metacatalog_to_hips,
    peak_jy_beam_from_total_flux,
    write_healpix_hips,
)


def test_peak_jy_beam_from_total_flux_unresolved() -> None:
    """When source FWHM equals the beam, Peak == Total."""
    peak = peak_jy_beam_from_total_flux(
        total_flux=np.array([2.0]),
        maj_fwhm=np.array([0.5]),
        min_fwhm=np.array([0.4]),
        bmaj=np.array([0.5]),
        bmin=np.array([0.4]),
    )
    assert peak[0] == pytest.approx(2.0)


def test_peak_jy_beam_from_total_flux_resolved() -> None:
    """Resolved source: Peak = Total * Ω_beam / Ω_src."""
    peak = peak_jy_beam_from_total_flux(
        total_flux=np.array([4.0]),
        maj_fwhm=np.array([1.0]),
        min_fwhm=np.array([0.5]),
        bmaj=np.array([0.5]),
        bmin=np.array([0.25]),
    )
    assert peak[0] == pytest.approx(4.0 * (0.5 * 0.25) / (1.0 * 0.5))


def test_catalog_with_peak_from_total_flux_uses_bmaj_match() -> None:
    cat = pd.DataFrame(
        {
            "Total_flux": [3.0],
            "Maj": [0.6],
            "Min": [0.3],
            "BMAJ_match": [0.2],
        }
    )
    out = catalog_with_peak_from_total_flux(cat)
    assert float(out.iloc[0]["Peak_flux"]) == pytest.approx(3.0 * (0.2 * 0.2) / (0.6 * 0.3))


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


def test_lst_merged_catalog_for_healpix_uses_band_native_shape() -> None:
    """Rematch pulls LST Maj/Min/PA; Peak from Total_flux × Ω_beam/Ω_src."""
    # Fused row: astrometry at 82 MHz (wrong shape), flux at 78 MHz.
    meta = pd.DataFrame(
        {
            "meta_id": [1],
            "RA": [10.0],
            "DEC": [5.0],
            "BMAJ_match": [0.5],
            "Maj": [0.4],
            "Min": [0.3],
            "PA": [10.0],
            "Total_flux_78MHz": [2.5],
            "Peak_flux_78MHz": [2.5],
        }
    )
    lst = pd.DataFrame(
        {
            "RA": [10.01, 10.2],
            "DEC": [5.01, 5.2],
            "BMAJ": [0.2, 0.2],
            "BMIN": [0.18, 0.18],
            "Peak_flux": [2.5, 9.0],
            "Total_flux": [2.5, 9.0],
            "Maj": [0.12, 0.5],
            "Min": [0.11, 0.4],
            "PA": [120.0, 0.0],
        }
    )
    out = lst_merged_catalog_for_healpix(meta, band="78MHz", lst_merged=lst)
    assert len(out) == 1
    expected = 2.5 * (0.2 * 0.18) / (0.12 * 0.11)
    assert float(out.iloc[0]["Peak_flux"]) == pytest.approx(expected)
    assert float(out.iloc[0]["Maj"]) == pytest.approx(0.12)
    assert float(out.iloc[0]["PA"]) == pytest.approx(120.0)


def test_lst_merged_catalog_for_healpix_keeps_peak_without_total() -> None:
    """Legacy LST rows without Total_flux keep catalog Peak_flux."""
    meta = pd.DataFrame(
        {
            "meta_id": [1],
            "RA": [10.0],
            "DEC": [5.0],
            "BMAJ_match": [0.5],
            "Peak_flux_78MHz": [2.5],
        }
    )
    lst = pd.DataFrame(
        {
            "RA": [10.01],
            "DEC": [5.01],
            "BMAJ": [0.2],
            "Peak_flux": [2.5],
            "Maj": [0.12],
            "Min": [0.11],
            "PA": [120.0],
        }
    )
    out = lst_merged_catalog_for_healpix(meta, band="78MHz", lst_merged=lst)
    assert float(out.iloc[0]["Peak_flux"]) == pytest.approx(2.5)


def test_lst_merged_catalog_for_healpix_dedups_confused_lst_row() -> None:
    """Two meta rows claiming one LST Gaussian → paint once."""
    meta = pd.DataFrame(
        {
            "meta_id": [1, 2],
            "RA": [10.0, 10.05],
            "DEC": [5.0, 5.02],
            "BMAJ_match": [0.5, 0.5],
            "Peak_flux_78MHz": [3.0, 3.0],
        }
    )
    lst = pd.DataFrame(
        {
            "RA": [10.02],
            "DEC": [5.01],
            "BMAJ": [0.25],
            "Peak_flux": [3.0],
            "Maj": [0.15],
            "Min": [0.14],
            "PA": [45.0],
        }
    )
    out = lst_merged_catalog_for_healpix(meta, band="78MHz", lst_merged=lst)
    assert len(out) == 1
    assert float(out.iloc[0]["Maj"]) == pytest.approx(0.15)


def test_lst_merged_catalog_for_healpix_skips_missing_band_flux() -> None:
    meta = pd.DataFrame(
        {
            "RA": [1.0],
            "DEC": [2.0],
            "BMAJ_match": [0.2],
            "Peak_flux_78MHz": [np.nan],
        }
    )
    lst = pd.DataFrame(
        {
            "RA": [1.0],
            "DEC": [2.0],
            "BMAJ": [0.2],
            "Peak_flux": [1.0],
            "Maj": [0.1],
            "Min": [0.1],
            "PA": [0.0],
        }
    )
    out = lst_merged_catalog_for_healpix(meta, band="78MHz", lst_merged=lst)
    assert out.empty
