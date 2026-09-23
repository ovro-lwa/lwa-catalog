"""Tests for HEALPix tile detection glue."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

pytest.importorskip("lwa_healpix")

from lwa_catalog.create.healpix_detect import (  # noqa: E402
    attach_beam_and_freq,
    detect_sources_on_healpix_tiles,
    median_beam_from_paths,
    restfreq_hz_from_header,
)
from lwa_healpix import iter_nested_tile_headers  # noqa: E402


def _tiny_fits(path: Path, *, bmaj: float = 0.1, bmin: float = 0.08, bpa: float = 1.0) -> Path:
    hdr = fits.Header(
        {
            "BMAJ": bmaj,
            "BMIN": bmin,
            "BPA": bpa,
            "RESTFREQ": 5.5e7,
            "BUNIT": "JY/BEAM",
        }
    )
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32), header=hdr).writeto(path)
    return path


def test_median_beam_from_paths(tmp_path: Path) -> None:
    p1 = _tiny_fits(tmp_path / "a.fits", bmaj=0.10, bmin=0.08, bpa=0.0)
    p2 = _tiny_fits(tmp_path / "b.fits", bmaj=0.20, bmin=0.12, bpa=10.0)
    bmaj, bmin, bpa = median_beam_from_paths([p1, p2])
    assert bmaj == pytest.approx(0.15)
    assert bmin == pytest.approx(0.10)
    assert bpa == pytest.approx(5.0)


def test_restfreq_hz_from_header() -> None:
    hdr = fits.Header({"RESTFRQ": 7.4e7})
    assert restfreq_hz_from_header(hdr) == pytest.approx(7.4e7)


def test_attach_beam_and_freq() -> None:
    hdu = fits.PrimaryHDU(
        np.ones((8, 8), dtype=np.float32),
        header=fits.Header({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"}),
    )
    out = attach_beam_and_freq(
        hdu, bmaj=0.1, bmin=0.09, bpa=2.0, restfreq_hz=5e7, bunit="JY/BEAM",
    )
    assert out.header["BMAJ"] == 0.1
    assert out.header["BMIN"] == 0.09
    assert out.header["BPA"] == 2.0
    assert out.header["RESTFREQ"] == 5e7
    assert out.header["RESTFRQ"] == 5e7
    assert out.header["BUNIT"] == "JY/BEAM"


def test_iter_nested_tile_count() -> None:
    tiles = list(iter_nested_tile_headers(4, 16, overlap=0.0))
    assert len(tiles) == 12 * 4**2


def test_detect_sources_on_healpix_tiles_tags_ipix(monkeypatch: pytest.MonkeyPatch) -> None:
    nside_map = 8
    nside_tile = 2
    npix = 12 * nside_map**2
    ratio = (nside_map // nside_tile) ** 2
    healpix_map = np.ones(npix, dtype=np.float64)
    weight = np.zeros(npix, dtype=np.float64)
    weight[0:ratio] = 1.0  # only nested parent 0

    def _fake_pybdsf(hdu, *, bdsf_kw=None, **kwargs):
        return Table(
            {
                "RA": [float(hdu.header["CRVAL1"])],
                "DEC": [float(hdu.header["CRVAL2"])],
                "Peak_flux": [1.0],
                "Total_flux": [1.0],
                "E_Peak_flux": [0.1],
                "E_Total_flux": [0.1],
                "Maj": [0.1],
                "Min": [0.1],
                "PA": [0.0],
                "DC_Maj": [0.1],
                "DC_Min": [0.1],
                "DC_PA": [0.0],
                "Resid_Isl_rms": [0.01],
                "Resid_Isl_mean": [0.0],
                "S_Code": ["S"],
            }
        )

    monkeypatch.setattr(
        "lwa_catalog.create.healpix_detect.run_pybdsf_on_hdu",
        _fake_pybdsf,
    )

    df = detect_sources_on_healpix_tiles(
        healpix_map,
        weight,
        nside_map=nside_map,
        nside_tile=nside_tile,
        overlap=0.0,
        bmaj=0.1,
        bmin=0.1,
        bpa=0.0,
        restfreq_hz=5.5e7,
        band="Full",
        skip_empty=True,
    )
    assert len(df) == 1
    assert int(df.iloc[0]["tile_ipix"]) == 0
    assert int(df.iloc[0]["nside_tile"]) == nside_tile
    assert df.iloc[0]["band"] == "Full"
    assert df.iloc[0]["BMAJ"] == 0.1
