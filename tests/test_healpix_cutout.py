"""Tests for HEALPix coadd cutout helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS

pytest.importorskip("lwa_healpix")
pytest.importorskip("healpy")

from lwa_healpix import write_healpix_fits  # noqa: E402

from lwa_catalog.analyze.healpix_cutout import (  # noqa: E402
    HealpixMapCache,
    cutout_grid_shape,
    healpix_coadd_path,
    healpix_cutout,
    plot_band_cutouts,
    tan_cutout_header,
)


def test_healpix_coadd_path() -> None:
    assert healpix_coadd_path("/tmp/out", "82MHz", nside=2048) == Path(
        "/tmp/out/healpix_82MHz_nside2048.fits"
    )


def test_tan_cutout_header_centered() -> None:
    hdr = tan_cutout_header(83.6, -5.4, size_deg=1.0, pixscale_deg=0.05)
    assert hdr["NAXIS1"] == hdr["NAXIS2"]
    assert hdr["NAXIS1"] % 2 == 1
    assert hdr["CRVAL1"] == pytest.approx(83.6)
    assert hdr["CRVAL2"] == pytest.approx(-5.4)
    assert hdr["CRPIX1"] == pytest.approx((hdr["NAXIS1"] + 1) / 2)
    wcs = WCS(hdr)
    x, y = wcs.world_to_pixel_values(83.6, -5.4)
    assert x == pytest.approx(hdr["CRPIX1"] - 1, abs=1e-6)
    assert y == pytest.approx(hdr["CRPIX2"] - 1, abs=1e-6)


def test_healpix_cutout_roundtrip(tmp_path: Path) -> None:
    nside = 8
    npix = 12 * nside**2
    healpix_map = np.full(npix, 0.5, dtype=np.float32)
    weight = np.ones(npix, dtype=np.float32)
    # Put a bright pixel near CRVAL by filling all
    healpix_map[:] = 1.0
    path = write_healpix_fits(
        tmp_path / "healpix_82MHz_nside8.fits",
        healpix_map,
        weight,
        nside=nside,
        nested=True,
        overwrite=True,
    )
    assert path.is_file()

    cache = HealpixMapCache(tmp_path, nside=nside)
    m, w, meta = cache.get("82MHz")
    assert meta["nside"] == nside
    assert m.shape == (npix,)

    hdu = healpix_cutout(
        m,
        w,
        45.0,
        10.0,
        size_deg=5.0,
        nside=nside,
        nested=True,
    )
    assert hdu.data.ndim == 2
    assert np.isfinite(hdu.data).any()


def test_plot_band_cutouts_handles_missing(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    nside = 8
    npix = 12 * nside**2
    write_healpix_fits(
        tmp_path / f"healpix_82MHz_nside{nside}.fits",
        np.ones(npix, dtype=np.float32),
        np.ones(npix, dtype=np.float32),
        nside=nside,
        nested=True,
        overwrite=True,
    )
    cache = HealpixMapCache(tmp_path, nside=nside)
    meta = pd.Series(
        {
            "RA": 45.0,
            "DEC": 10.0,
            "bands_present": "82MHz,55MHz",
        }
    )
    lst = pd.DataFrame(
        {
            "band": ["82MHz", "55MHz"],
            "RA": [45.05, 44.95],
            "DEC": [10.02, 9.98],
            "Peak_flux": [1.0, 0.8],
            "BMAJ": [1.0, 1.2],
            "Maj": [0.5, 0.6],
            "Min": [0.4, 0.5],
            "PA": [0.0, 30.0],
        }
    )
    fig = plot_band_cutouts(lst, meta, cache, beam_factor=3.0)
    assert fig is not None
    assert len(fig.axes) == 2


def test_plot_band_cutouts_confused_yellow_box(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib
    from matplotlib.patches import Rectangle

    matplotlib.use("Agg")

    nside = 8
    npix = 12 * nside**2
    for band in ("82MHz", "55MHz"):
        write_healpix_fits(
            tmp_path / f"healpix_{band}_nside{nside}.fits",
            np.ones(npix, dtype=np.float32),
            np.ones(npix, dtype=np.float32),
            nside=nside,
            nested=True,
            overwrite=True,
        )
    cache = HealpixMapCache(tmp_path, nside=nside)
    meta = pd.Series(
        {
            "RA": 45.0,
            "DEC": 10.0,
            "bands_present": "82MHz,55MHz",
            "n_confused_82MHz": 3,
            "n_confused_55MHz": 1,
        }
    )
    lst = pd.DataFrame(
        {
            "band": ["82MHz", "55MHz"],
            "RA": [45.05, 44.95],
            "DEC": [10.02, 9.98],
            "Peak_flux": [1.0, 0.8],
            "BMAJ": [1.0, 1.2],
            "Maj": [0.5, 0.6],
            "Min": [0.4, 0.5],
            "PA": [0.0, 30.0],
        }
    )
    fig = plot_band_cutouts(lst, meta, cache, beam_factor=3.0)
    yellow_boxes = [
        p
        for ax in fig.axes
        for p in ax.patches
        if isinstance(p, Rectangle)
        and p.get_edgecolor()[:3] == (1.0, 1.0, 0.0)
        and p.get_linestyle() == "--"
    ]
    assert len(yellow_boxes) == 1
    titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
    assert any("confused" in t for t in titles)


def test_cutout_grid_shape_aspect() -> None:
    assert cutout_grid_shape(1) == (1, 1)
    assert cutout_grid_shape(2) == (1, 2)
    assert cutout_grid_shape(8) == (2, 4)
    nrows, ncols = cutout_grid_shape(15)
    assert nrows * ncols >= 15
    assert ncols >= nrows
    # Prefer roughly 2∶1 width∶height for mid-sized n.
    assert abs(ncols / nrows - 2.0) < 0.6


def test_plot_band_cutouts_uses_grid(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    nside = 8
    npix = 12 * nside**2
    bands = [f"{f}MHz" for f in (18, 34, 55, 74, 82)]
    for band in bands:
        write_healpix_fits(
            tmp_path / f"healpix_{band}_nside{nside}.fits",
            np.ones(npix, dtype=np.float32),
            np.ones(npix, dtype=np.float32),
            nside=nside,
            nested=True,
            overwrite=True,
        )
    cache = HealpixMapCache(tmp_path, nside=nside)
    meta = pd.Series({"RA": 45.0, "DEC": 10.0, "bands_present": ",".join(bands)})
    lst = pd.DataFrame(
        {
            "band": bands,
            "RA": [45.0] * len(bands),
            "DEC": [10.0] * len(bands),
            "Peak_flux": [1.0] * len(bands),
            "BMAJ": [1.0] * len(bands),
            "Maj": [0.5] * len(bands),
            "Min": [0.4] * len(bands),
            "PA": [0.0] * len(bands),
        }
    )
    fig = plot_band_cutouts(lst, meta, cache, beam_factor=3.0)
    nrows, ncols = cutout_grid_shape(len(bands))
    assert len(fig.axes) == nrows * ncols
    visible = sum(1 for ax in fig.axes if ax.get_visible())
    assert visible == len(bands)
