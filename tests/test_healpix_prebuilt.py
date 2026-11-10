"""Tests for prebuilt HEALPix mosaic discovery / load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lwa_catalog.create.healpix_prebuilt import (
    DEFAULT_PREBUILT_HEALPIX_GLOB,
    beam_and_restfreq_from_healpix_fits,
    discover_prebuilt_healpix_maps,
    list_mhz_subband_dirs,
    load_prebuilt_healpix_map,
    resolve_prebuilt_healpix_for_band,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _jybeam_name(band: str, *, ssub: bool) -> str:
    mid = "coadd_ssub-rst8_mosaic" if ssub else "coadd_mosaic"
    return (
        f"{band}_I_deep_Taper_Robust-0.75_{mid}_jybeam_feather_allsky_"
        f"20241218-20251029_N024.fits"
    )


def _band_healpix_dir(root: Path, band: str) -> Path:
    return root / band / "I" / "healpix"


def test_list_mhz_subband_dirs_high_to_low(tmp_path: Path) -> None:
    for name in ("Blue", "Full", "tables", "18MHz", "82MHz", "55MHz"):
        (tmp_path / name).mkdir()
    assert list_mhz_subband_dirs(tmp_path) == ["82MHz", "55MHz", "18MHz"]


def test_list_mhz_subband_dirs_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_mhz_subband_dirs(tmp_path / "nope")


def test_resolve_prefers_ssub_when_both_present(tmp_path: Path) -> None:
    band = "82MHz"
    healpix = _band_healpix_dir(tmp_path, band)
    plain = _touch(healpix / _jybeam_name(band, ssub=False))
    ssub = _touch(healpix / _jybeam_name(band, ssub=True))
    assert plain != ssub
    got = resolve_prebuilt_healpix_for_band(tmp_path, band)
    assert got == ssub.resolve()


def test_resolve_unique_without_ssub(tmp_path: Path) -> None:
    band = "36MHz"
    healpix = _band_healpix_dir(tmp_path, band)
    only = _touch(healpix / _jybeam_name(band, ssub=False))
    got = resolve_prebuilt_healpix_for_band(tmp_path, band)
    assert got == only.resolve()


def test_resolve_ambiguous_without_unique_prefer(tmp_path: Path) -> None:
    band = "82MHz"
    healpix = _band_healpix_dir(tmp_path, band)
    _touch(healpix / _jybeam_name(band, ssub=False))
    _touch(
        healpix
        / (
            f"{band}_I_deep_Taper_Robust-0.75_coadd_mosaic_jybeam_feather_allsky_"
            f"OTHER.fits"
        )
    )
    with pytest.raises(FileNotFoundError, match="Ambiguous"):
        resolve_prebuilt_healpix_for_band(tmp_path, band)


def test_resolve_missing(tmp_path: Path) -> None:
    (tmp_path / "82MHz" / "I" / "healpix").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No prebuilt"):
        resolve_prebuilt_healpix_for_band(tmp_path, "82MHz")


def test_discover_omits_missing_raises_ambiguous(tmp_path: Path) -> None:
    for band in ("82MHz", "36MHz"):
        healpix = _band_healpix_dir(tmp_path, band)
        _touch(healpix / _jybeam_name(band, ssub=False))
        if band == "82MHz":
            _touch(healpix / _jybeam_name(band, ssub=True))
    (tmp_path / "18MHz" / "I" / "healpix").mkdir(parents=True)
    (tmp_path / "Blue").mkdir()

    found = discover_prebuilt_healpix_maps(tmp_path)
    assert set(found) == {"82MHz", "36MHz"}
    assert "ssub-rst8" in found["82MHz"].name
    assert "ssub-rst8" not in found["36MHz"].name

    # Ambiguous: two non-ssub for a third band
    bad = "59MHz"
    healpix = _band_healpix_dir(tmp_path, bad)
    _touch(healpix / _jybeam_name(bad, ssub=False))
    _touch(
        healpix
        / (
            f"{bad}_I_deep_Taper_Robust-0.75_coadd_mosaic_jybeam_feather_allsky_"
            f"ALT.fits"
        )
    )
    with pytest.raises(FileNotFoundError, match="Ambiguous"):
        discover_prebuilt_healpix_maps(tmp_path)


def test_default_glob_format() -> None:
    assert "{band}" in DEFAULT_PREBUILT_HEALPIX_GLOB
    assert "jybeam" in DEFAULT_PREBUILT_HEALPIX_GLOB


def test_beam_and_restfreq_from_healpix_primary(tmp_path: Path) -> None:
    from astropy.io import fits

    path = tmp_path / "beam_hdr.fits"
    primary = fits.PrimaryHDU()
    primary.header["BMAJ"] = 0.1
    primary.header["BMIN"] = 0.08
    primary.header["BPA"] = 12.5
    primary.header["RESTFRQ"] = 7.3e7
    fits.HDUList([primary]).writeto(path)

    bmaj, bmin, bpa, restfreq = beam_and_restfreq_from_healpix_fits(path)
    assert bmaj == pytest.approx(0.1)
    assert bmin == pytest.approx(0.08)
    assert bpa == pytest.approx(12.5)
    assert restfreq == pytest.approx(7.3e7)


def test_beam_and_restfreq_missing_bmaj(tmp_path: Path) -> None:
    from astropy.io import fits

    path = tmp_path / "nobbeam.fits"
    primary = fits.PrimaryHDU()
    primary.header["RESTFRQ"] = 1.0e8
    fits.HDUList([primary]).writeto(path)
    with pytest.raises(ValueError, match="BMAJ/BMIN"):
        beam_and_restfreq_from_healpix_fits(path)


def test_load_prebuilt_ring_to_nested(tmp_path: Path) -> None:
    hp = pytest.importorskip("healpy")
    from astropy.io import fits

    nside = 8
    npix = hp.nside2npix(nside)
    ring = np.arange(npix, dtype=np.float32)
    ring[0] = np.nan
    path = tmp_path / "toy_ring.fits"
    hp.write_map(
        str(path),
        ring,
        nest=False,
        coord="C",
        column_names=["SIGNAL"],
        dtype=np.float32,
        overwrite=True,
    )
    # Ensure ORDERING present (healpy write_map sets it).
    with fits.open(path) as hdul:
        assert any("ORDERING" in hdu.header for hdu in hdul)

    healpix_map, weight, meta = load_prebuilt_healpix_map(path)
    assert meta["nside"] == nside
    assert meta["nested"] is True
    assert meta["source_ordering"] == "RING"
    assert meta["coord_frame"] == "icrs"
    assert healpix_map.shape == (npix,)
    assert weight.dtype == np.float32
    assert weight.sum() == npix - 1
    expected = hp.reorder(ring, r2n=True).astype(np.float32)
    np.testing.assert_allclose(healpix_map, expected, equal_nan=True)
