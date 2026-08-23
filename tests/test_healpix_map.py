"""Tests for catalog HEALPix map helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

healpy = pytest.importorskip("healpy")

from lwa_catalog.analyze.healpix_map import metacatalog_to_healpix, write_healpix_fits  # noqa: E402


def test_metacatalog_to_healpix_weighted() -> None:
    nside = 8
    # Two sources in same pixel → weights sum
    ra0, dec0 = 0.0, 0.0
    cat = pd.DataFrame(
        {
            "RA": [ra0, ra0],
            "DEC": [dec0, dec0],
            "Peak_flux": [1.0, 2.0],
        }
    )
    m = metacatalog_to_healpix(cat, nside=nside, weight_col="Peak_flux")
    assert m.shape == (healpy.nside2npix(nside),)
    assert m.sum() == pytest.approx(3.0)
    pix = healpy.ang2pix(nside, np.deg2rad(90.0 - dec0), np.deg2rad(ra0))
    assert m[pix] == pytest.approx(3.0)


def test_metacatalog_to_healpix_counts() -> None:
    cat = pd.DataFrame({"RA": [10.0, 20.0], "DEC": [0.0, 0.0], "Peak_flux": [5.0, 7.0]})
    m = metacatalog_to_healpix(cat, nside=16, weight_col=None)
    assert m.sum() == pytest.approx(2.0)


def test_write_healpix_fits_roundtrip(tmp_path: Path) -> None:
    nside = 4
    cat = pd.DataFrame({"RA": [45.0], "DEC": [30.0], "Peak_flux": [4.0]})
    m = metacatalog_to_healpix(cat, nside=nside)
    path = write_healpix_fits(m, tmp_path / "map.fits", nside=nside)
    assert path.is_file()
    m2 = healpy.read_map(str(path))
    assert np.allclose(m, m2)
