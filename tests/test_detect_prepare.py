"""Tests for FITS prep before PyBDSF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from lwa_catalog.create.detect import prepare_hdu, upsample_hdu


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
