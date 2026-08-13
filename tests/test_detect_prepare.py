"""Tests for FITS prep before PyBDSF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from lwa_catalog.create.detect import prepare_hdu


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
