"""Tests for catalog HEALPix / HiPS helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

healpy = pytest.importorskip("healpy")
pytest.importorskip("lwa_healpix")

from lwa_catalog.analyze.healpix_map import (  # noqa: E402
    metacatalog_to_healpix,
    metacatalog_to_hips,
    write_healpix_hips,
)


def test_metacatalog_to_healpix_weighted() -> None:
    nside = 8
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


def test_write_healpix_hips(tmp_path: Path) -> None:
    nside = 8
    cat = pd.DataFrame({"RA": [45.0], "DEC": [30.0], "Peak_flux": [4.0]})
    m = metacatalog_to_healpix(cat, nside=nside)
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
    cat = pd.DataFrame({"RA": [10.0, 20.0], "DEC": [5.0, -5.0], "Peak_flux": [1.0, 2.0]})
    out = metacatalog_to_hips(
        cat,
        tmp_path / "hips_cat",
        nside=8,
        threads=False,
        properties={"obs_title": "test catalog HiPS"},
    )
    assert (out / "properties").is_file()
    props = (out / "properties").read_text()
    assert "hips_pixel_cut" in props or "obs_title" in props
