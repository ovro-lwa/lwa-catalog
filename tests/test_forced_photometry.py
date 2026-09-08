"""Tests for constrained Gaussian forced photometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from lwa_catalog.analyze.forced_photometry import (
    ForcedPhotometryConfig,
    discover_and_index_fits,
    filter_fits_by_elevation,
    fit_gaussian2d_cutout,
    index_discovered_fits,
    resolve_forced_fits_path,
    seed_row_from_meta,
    select_forced_seed_row,
    summarize_forced_photometry,
)
from lwa_catalog.create.discover import FitsMetadata
from lwa_catalog.create.merge import source_elevation_deg


def _write_gaussian_fits(
    path: Path,
    *,
    peak: float = 2.5,
    ra0: float = 180.0,
    dec0: float = 37.24,
    npix: int = 64,
    cdelt_deg: float = 0.05,
    bmaj_deg: float = 0.2,
) -> Path:
    """Write a simple TAN FITS with a circular Gaussian at (*ra0*, *dec0*)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [npix / 2 + 0.5, npix / 2 + 0.5]
    w.wcs.cdelt = [-cdelt_deg, cdelt_deg]
    w.wcs.cunit = ["deg", "deg"]

    yy, xx = np.mgrid[0:npix, 0:npix]
    x0 = npix / 2
    y0 = npix / 2
    sigma = (bmaj_deg / cdelt_deg) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    data = peak * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sigma**2)
    data = data.astype(np.float32)

    header = w.to_header()
    header["BMAJ"] = bmaj_deg
    header["BMIN"] = bmaj_deg
    header["BPA"] = 0.0
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)
    return path


def test_seed_row_from_meta_uses_band_suffix() -> None:
    meta = pd.Series(
        {
            "RA": 10.0,
            "DEC": 20.0,
            "Peak_flux_18MHz": 1.5,
            "Maj_18MHz": 0.3,
            "Min_18MHz": 0.2,
            "PA_18MHz": 45.0,
        }
    )
    seed = seed_row_from_meta(
        meta, source_file="img.fits", lst_hour="01h", band="18MHz"
    )
    assert float(seed["Peak_flux"]) == pytest.approx(1.5)
    assert float(seed["Maj"]) == pytest.approx(0.3)
    assert float(seed["Min"]) == pytest.approx(0.2)
    assert float(seed["PA"]) == pytest.approx(45.0)
    assert seed["lst_hour"] == "01h"
    assert seed["band"] == "18MHz"


def test_select_forced_seed_row_prefers_rematch() -> None:
    meta = pd.Series({"RA": 1.0, "DEC": 2.0, "Peak_flux": 0.1, "Maj": 0.1, "Min": 0.1, "PA": 0.0})
    rematch = pd.DataFrame(
        [
            {
                "source_file": "a.fits",
                "RA": 1.1,
                "DEC": 2.1,
                "Peak_flux": 9.0,
                "Maj": 0.4,
                "Min": 0.3,
                "PA": 10.0,
            }
        ]
    )
    seed = select_forced_seed_row(
        source_file="a.fits",
        rematch=rematch,
        meta_row=meta,
        lst_hour="02h",
        band="Blue",
    )
    assert float(seed["Peak_flux"]) == pytest.approx(9.0)

    seed_miss = select_forced_seed_row(
        source_file="missing.fits",
        rematch=rematch,
        meta_row=meta,
        lst_hour="02h",
        band="Blue",
    )
    assert float(seed_miss["Peak_flux"]) == pytest.approx(0.1)
    assert seed_miss["source_file"] == "missing.fits"


def test_filter_fits_by_elevation_respects_min_cut() -> None:
    # Source at transit for 12h LST near OVRO latitude → high elevation.
    ra = 180.0
    dec = 37.24
    discovered = [
        FitsMetadata(path=Path("low.fits"), lst_hour="04h", band="Blue"),
        FitsMetadata(path=Path("high.fits"), lst_hour="12h", band="Blue"),
    ]
    assert source_elevation_deg(ra, dec, "12h") > 10.0
    assert 0.0 < source_elevation_deg(ra, dec, "04h") < 10.0

    kept = filter_fits_by_elevation(discovered, ra, dec, min_elevation_deg=10.0)
    assert [m.path.name for m in kept] == ["high.fits"]

    kept0 = filter_fits_by_elevation(discovered, ra, dec, min_elevation_deg=0.0)
    assert {m.path.name for m in kept0} == {"low.fits", "high.fits"}


def test_index_and_resolve_fits(tmp_path: Path) -> None:
    fits_dir = tmp_path / "01h_Blue"
    fits_dir.mkdir()
    path = fits_dir / "Blue_I_deep_LST01h_t0001.fits"
    _write_gaussian_fits(path)
    discovered, by_name, meta_by_name = discover_and_index_fits(
        tmp_path, patterns="**/*.fits"
    )
    assert len(discovered) == 1
    assert path.name in by_name
    assert meta_by_name[path.name].lst_hour == "01h"
    assert resolve_forced_fits_path(path.name, by_name=by_name) == by_name[path.name]
    with pytest.raises(FileNotFoundError):
        resolve_forced_fits_path("nope.fits", by_name=by_name)


def test_fit_gaussian2d_recovers_peak_and_fixed_position(tmp_path: Path) -> None:
    path = tmp_path / "src.fits"
    ra0, dec0, peak = 180.0, 37.24, 2.5
    bmaj = 0.2
    _write_gaussian_fits(path, peak=peak, ra0=ra0, dec0=dec0, bmaj_deg=bmaj)

    free = fit_gaussian2d_cutout(
        path,
        ra_deg=ra0,
        dec_deg=dec0,
        maj_deg=bmaj,
        min_deg=bmaj,
        pa_deg=0.0,
        peak_guess=peak,
        bmaj_deg=bmaj,
        bmin_deg=bmaj,
        fixed_position=False,
        meta_id=42,
        config=ForcedPhotometryConfig(cutout_size_pix=48),
    )
    assert free.forced_Peak_flux == pytest.approx(peak, rel=0.15)
    assert free.forced_RA == pytest.approx(ra0, abs=0.05)
    assert free.forced_DEC == pytest.approx(dec0, abs=0.05)
    assert free.summary["fixed_position"] is False
    assert free.summary["meta_id"] == 42
    assert "Peak=" in summarize_forced_photometry(free)

    fixed = fit_gaussian2d_cutout(
        path,
        ra_deg=ra0,
        dec_deg=dec0,
        maj_deg=bmaj,
        min_deg=bmaj,
        pa_deg=0.0,
        peak_guess=peak,
        bmaj_deg=bmaj,
        bmin_deg=bmaj,
        fixed_position=True,
        config=ForcedPhotometryConfig(cutout_size_pix=48),
    )
    assert fixed.summary["fixed_position"] is True
    assert fixed.forced_RA == pytest.approx(ra0, abs=1e-5)
    assert fixed.forced_DEC == pytest.approx(dec0, abs=1e-5)
    assert fixed.forced_Peak_flux == pytest.approx(peak, rel=0.15)


def test_index_discovered_fits_empty() -> None:
    by_name, meta_by_name = index_discovered_fits([])
    assert by_name == {}
    assert meta_by_name == {}
