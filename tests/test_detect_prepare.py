"""Tests for FITS prep before PyBDSF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from lwa_catalog.create.detect import blank_below_elevation, prepare_hdu, upsample_hdu


def _zenith_sin_hdu(ny: int = 21, nx: int = 21, cdelt_deg: float = 5.0) -> fits.PrimaryHDU:
    """Small SIN image with CRVAL at array center (zenith)."""
    data = np.ones((ny, nx), dtype=np.float32)
    header = fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": nx,
            "NAXIS2": ny,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CRVAL1": 180.0,
            "CRVAL2": 37.0,
            "CRPIX1": (nx + 1) / 2.0,
            "CRPIX2": (ny + 1) / 2.0,
            "CDELT1": -cdelt_deg,
            "CDELT2": cdelt_deg,
            "BMAJ": 0.1,
            "BMIN": 0.1,
            "BPA": 0.0,
        }
    )
    return fits.PrimaryHDU(data=data, header=header)


def test_prepare_hdu_preserves_nan_blanks(tmp_path: Path) -> None:
    data = np.ones((8, 8), dtype=np.float32)
    data[:2, :] = np.nan
    data[4, 4] = np.inf
    path = tmp_path / "blanked.fits"
    fits.PrimaryHDU(
        data=data,
        header=fits.Header(
            {
                "BMAJ": 0.1,
                "BMIN": 0.1,
                "BPA": 0.0,
                "RESTFRQ": 1.8e7,
                "CDELT1": -0.01,
                "CDELT2": 0.01,
            }
        ),
    ).writeto(path)

    hdu = prepare_hdu(path)
    assert hdu.data.shape == (8, 8)
    assert np.isnan(hdu.data[:2, :]).all()
    assert np.isnan(hdu.data[4, 4])
    assert np.isfinite(hdu.data[2:, :]).sum() == 8 * 6 - 1
    assert float(hdu.header["RESTFREQ"]) == 1.8e7


def test_blank_below_elevation_sets_nan_not_zero() -> None:
    hdu = _zenith_sin_hdu()
    blank_below_elevation(hdu, min_elevation_deg=10.0)

    wcs_2d = WCS(hdu.header).celestial
    ny, nx = hdu.data.shape
    y, x = np.mgrid[:ny, :nx]
    center = SkyCoord(
        wcs_2d.wcs.crval[0], wcs_2d.wcs.crval[1], unit="deg", frame="icrs"
    )
    elev = 90.0 - wcs_2d.pixel_to_world(x, y).separation(center).deg

    low = elev < 10.0
    high = elev >= 10.0
    assert low.any()
    assert high.any()
    assert np.isnan(hdu.data[low]).all()
    assert np.isfinite(hdu.data[high]).all()
    assert float(np.nanmin(hdu.data[high])) == 1.0


def test_blank_below_elevation_preserves_prior_nans() -> None:
    hdu = _zenith_sin_hdu()
    hdu.data[10, 10] = np.nan  # zenith pixel
    blank_below_elevation(hdu, min_elevation_deg=10.0)
    assert np.isnan(hdu.data[10, 10])


def test_upsample_hdu_doubles_shape_and_wcs(tmp_path: Path) -> None:
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    data[0, :] = np.nan
    path = tmp_path / "small.fits"
    fits.PrimaryHDU(
        data=data,
        header=fits.Header(
            {
                "BMAJ": 0.2,
                "BMIN": 0.15,
                "BPA": 45.0,
                "CDELT1": -0.02,
                "CDELT2": 0.02,
                "CRPIX1": 2.5,
                "CRPIX2": 3.0,
            }
        ),
    ).writeto(path)

    hdu = upsample_hdu(prepare_hdu(path), factor=2)
    assert hdu.data.shape == (8, 8)
    assert np.isnan(hdu.data[0, :]).all()
    assert float(hdu.header["CDELT1"]) == -0.01
    assert float(hdu.header["CDELT2"]) == 0.01
    assert float(hdu.header["CRPIX1"]) == 4.0
    assert float(hdu.header["CRPIX2"]) == 5.0
    assert float(hdu.header["BMAJ"]) == 0.2
