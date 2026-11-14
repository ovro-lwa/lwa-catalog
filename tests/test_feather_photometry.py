"""Tests for FEATHER3 companion-map photometric calibration."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.beam_alpha_bootstrap import (
    load_beam_scale,
    read_beam_scale_json,
    write_beam_scale_json,
)
from lwa_catalog.analyze.feather_photometry import (
    FeatherPhotometryConfig,
    PhotometryCalibration,
    PhotometryKModel,
    calibrate_layout_photometry,
    calibrate_lst_photometry,
    estimate_photometry_k_model,
    load_photometry_calibration,
    lst_bands_in_layout,
    sample_healpix_at,
    summarize_feather_photometry,
    verify_lst_photometry,
    write_photometry_calibration,
)
from lwa_catalog.constants import OVRO_LATITUDE_DEG
from lwa_catalog.create.healpix_prebuilt import FeatherCompanions
from lwa_catalog.io import read_lst_merged, write_lst_merged
from lwa_catalog.paths import CatalogLayout

hp = pytest.importorskip("healpy")

NSIDE = 8
OMEGA_EFF = 4.0e-6
GAUSS_AREA = np.pi / (4.0 * np.log(2.0))
DEC_CENTRES = np.arange(-35.0, 86.0, 10.0)


def _fwhm_deg_for_ratio(ratio: np.ndarray) -> np.ndarray:
    """Circular FWHM (deg) whose Gaussian solid angle is ``ratio · OMEGA_EFF``."""
    return np.degrees(np.sqrt(np.asarray(ratio) * OMEGA_EFF / GAUSS_AREA))


def _write_map(path: Path, values: np.ndarray, *, nest: bool = False) -> Path:
    hp.write_map(
        str(path),
        values,
        nest=nest,
        coord="C",
        column_names=["SIGNAL"],
        dtype=np.float32,
        overwrite=True,
    )
    return path


def _constant_companions(tmp_path: Path, *, pb: float = 0.5, pk: float = 0.8) -> FeatherCompanions:
    npix = hp.nside2npix(NSIDE)
    return FeatherCompanions(
        pbmap=_write_map(tmp_path / "pbmap.fits", np.full(npix, pb, np.float32)),
        psfmap=_write_map(tmp_path / "psfmap.fits", np.full(npix, OMEGA_EFF, np.float32)),
        pkcorr=_write_map(tmp_path / "pkcorr.fits", np.full(npix, pk, np.float32)),
    )


def _lst(dec: np.ndarray, ratio: np.ndarray, *, s_code: str = "S") -> pd.DataFrame:
    """LST-merged-shaped rows with fitted size ``Ω_fit = ratio · Ω_eff``."""
    n = len(dec)
    fwhm = _fwhm_deg_for_ratio(ratio)
    peak = np.linspace(2.0, 20.0, n)
    total = peak * 1.3
    return pd.DataFrame(
        {
            "RA": np.linspace(10.0, 350.0, n),
            "DEC": np.asarray(dec, dtype=float),
            "Peak_flux": peak,
            "E_Peak_flux": peak / 50.0,
            "Total_flux": total,
            "E_Total_flux": total / 40.0,
            "Maj": fwhm,
            "Min": fwhm,
            "S_Code": [s_code] * n,
        }
    )


def _true_ln_k(band_mhz: float, dec: np.ndarray) -> np.ndarray:
    """Quartic-in-x, linear-in-u truth inside the joint model space."""
    x = (np.asarray(dec) - OVRO_LATITUDE_DEG) / 60.0
    u = np.log(band_mhz / 55.0)
    offset = 0.30 + 0.05 * u
    return offset + (-0.03 - 0.04 * u) * x + 0.06 * x**2 + (0.01 + 0.02 * u) * x**3 - 0.02 * x**4


def _flat_model(bands: tuple[str, ...], k: float) -> PhotometryKModel:
    return PhotometryKModel(
        band_offset={b: float(np.log(k)) for b in bands},
        band_freq_mhz={b: float(b.removesuffix("MHz")) for b in bands},
        shape=((0.0,),),
        dec_range=(-90.0, 90.0),
        phi_deg=OVRO_LATITUDE_DEG,
        dec_scale_deg=60.0,
        ref_freq_mhz=55.0,
    )


# --------------------------------------------------------------------------- sampling


@pytest.mark.parametrize("nest", [False, True])
def test_sample_healpix_at_matches_ang2pix(tmp_path: Path, nest: bool) -> None:
    values = np.arange(hp.nside2npix(NSIDE), dtype=np.float32)
    path = _write_map(tmp_path / f"toy_{nest}.fits", values, nest=nest)
    ra = np.array([0.0, 45.0, 181.0, 300.0, -60.0])
    dec = np.array([0.0, 37.2, -30.0, 80.0, 10.0])
    got = sample_healpix_at(path, ra, dec)
    expected = values[hp.ang2pix(NSIDE, np.mod(ra, 360.0), dec, nest=nest, lonlat=True)]
    np.testing.assert_array_equal(got, expected)


def test_sample_healpix_at_unseen_and_bad_coords(tmp_path: Path) -> None:
    values = np.full(hp.nside2npix(NSIDE), 0.5, dtype=np.float32)
    values[hp.ang2pix(NSIDE, 10.0, 20.0, lonlat=True)] = hp.UNSEEN
    path = _write_map(tmp_path / "toy_unseen.fits", values)
    got = sample_healpix_at(
        path,
        np.array([10.0, 100.0, np.nan, 50.0]),
        np.array([20.0, 0.0, 0.0, 95.0]),
    )
    assert np.isnan(got[0])
    assert got[1] == pytest.approx(0.5)
    assert np.isnan(got[2])
    assert np.isnan(got[3])


def test_sample_healpix_at_reloads_after_rewrite(tmp_path: Path) -> None:
    npix = hp.nside2npix(NSIDE)
    path = _write_map(tmp_path / "toy_rewrite.fits", np.ones(npix, dtype=np.float32))
    assert sample_healpix_at(path, [0.0], [0.0])[0] == pytest.approx(1.0)
    _write_map(path, np.full(npix, 2.0, dtype=np.float32))
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert sample_healpix_at(path, [0.0], [0.0])[0] == pytest.approx(2.0)


# --------------------------------------------------------------------------- joint k model


def _band_tables(
    bands_mhz: dict[str, float], centres: dict[str, np.ndarray], per_bin: int = 12
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    lst_by_band, omega_by_band = {}, {}
    for band, mhz in bands_mhz.items():
        dec = np.repeat(centres[band], per_bin)
        lst_by_band[band] = _lst(dec, np.exp(_true_ln_k(mhz, dec)))
        omega_by_band[band] = np.full(len(dec), OMEGA_EFF)
    return lst_by_band, omega_by_band


def test_joint_model_recovers_k_dec_nu() -> None:
    bands = {"82MHz": 82.0, "55MHz": 55.0, "23MHz": 23.0}
    lst_by_band, omega = _band_tables(bands, dict.fromkeys(bands, DEC_CENTRES))
    model = estimate_photometry_k_model(lst_by_band, omega)

    assert model.bands == tuple(bands)
    assert model.dec_range == pytest.approx((-35.0, 85.0))
    for band, mhz in bands.items():
        np.testing.assert_allclose(
            np.log(model.k_at(band, DEC_CENTRES)), _true_ln_k(mhz, DEC_CENTRES), atol=1e-6
        )
        diag = model.diagnostics[band]
        assert diag.n_used == len(DEC_CENTRES) * 12
        assert diag.dec_centres == pytest.approx(tuple(DEC_CENTRES))
    # Clamped outside the fitted Dec range; NaN Dec stays NaN.
    assert model.k_at("82MHz", -60.0) == pytest.approx(model.k_at("82MHz", -35.0))
    assert np.isnan(model.k_at("82MHz", np.nan))
    with pytest.raises(KeyError, match="18MHz"):
        model.k_at("18MHz", 0.0)


def test_joint_model_sparse_band_borrows_dec_shape() -> None:
    """A band without southern rows gets the shared shape, not a constant."""
    bands = {"82MHz": 82.0, "55MHz": 55.0, "23MHz": 23.0}
    centres = {"82MHz": DEC_CENTRES, "55MHz": DEC_CENTRES, "23MHz": DEC_CENTRES[3:]}
    lst_by_band, omega = _band_tables(bands, centres)
    model = estimate_photometry_k_model(lst_by_band, omega)
    south = np.array([-35.0, -25.0, -15.0])
    np.testing.assert_allclose(
        np.log(model.k_at("23MHz", south)), _true_ln_k(23.0, south), atol=1e-6
    )
    assert model.diagnostics["23MHz"].dec_centres[0] == pytest.approx(-5.0)


def test_joint_model_selects_bright_compact_only() -> None:
    dec = np.repeat(DEC_CENTRES, 12)
    bright = _lst(dec, np.full(len(dec), 1.4))
    extended = _lst(dec, np.full(len(dec), 4.0), s_code="M")
    faint = _lst(dec, np.full(len(dec), 4.0))
    faint["E_Peak_flux"] = faint["Peak_flux"]  # SNR 1
    lst = pd.concat([bright, extended, faint], ignore_index=True)
    model = estimate_photometry_k_model({"78MHz": lst}, {"78MHz": np.full(len(lst), OMEGA_EFF)})
    assert model.diagnostics["78MHz"].n_used == len(bright)
    np.testing.assert_allclose(model.k_at("78MHz", DEC_CENTRES), 1.4, rtol=1e-6)
    # Single frequency: no u terms.
    assert all(len(row) == 1 for row in model.shape)


@pytest.mark.parametrize("column", ["S_Code", "E_Peak_flux"])
def test_joint_model_requires_inputs(column: str) -> None:
    lst = _lst(np.repeat(DEC_CENTRES, 12), np.full(len(DEC_CENTRES) * 12, 1.4))
    with pytest.raises(ValueError, match=column):
        estimate_photometry_k_model(
            {"78MHz": lst.drop(columns=[column])}, {"78MHz": np.full(len(lst), OMEGA_EFF)}
        )


def test_joint_model_rejects_sparse_band() -> None:
    bands = {"82MHz": 82.0, "18MHz": 18.0}
    lst_by_band, omega = _band_tables(
        bands, {"82MHz": DEC_CENTRES, "18MHz": DEC_CENTRES[:4]}, per_bin=12
    )
    with pytest.raises(ValueError, match=r"18MHz \(48\)"):
        estimate_photometry_k_model(lst_by_band, omega)


def test_joint_model_rejects_too_few_dec_bins() -> None:
    dec = np.repeat(DEC_CENTRES[:3], 50)
    lst = _lst(dec, np.full(len(dec), 1.4))
    with pytest.raises(ValueError, match="Dec bins"):
        estimate_photometry_k_model({"78MHz": lst}, {"78MHz": np.full(len(lst), OMEGA_EFF)})


# --------------------------------------------------------------------------- calibration


def test_calibrate_synthetic_pb_pk_k(tmp_path: Path) -> None:
    companions = _constant_companions(tmp_path)
    lst = _lst(np.linspace(-20.0, 60.0, 60), np.full(60, 1.4))
    raw = lst.copy()
    result = calibrate_lst_photometry(lst, "78MHz", companions, _flat_model(("78MHz",), 1.4))
    table = result.table

    np.testing.assert_allclose(table["Peak_flux_cal"], raw["Peak_flux"] / 0.4, rtol=1e-6)
    np.testing.assert_allclose(table["Total_flux_cal"], table["Peak_flux_cal"], rtol=1e-4)
    np.testing.assert_allclose(table["E_Peak_flux_cal"], raw["E_Peak_flux"] / 0.4, rtol=1e-6)
    np.testing.assert_allclose(
        table["E_Total_flux_cal"],
        raw["E_Total_flux"] * table["Total_flux_cal"] / raw["Total_flux"],
        rtol=1e-9,
    )
    np.testing.assert_allclose(table["PB_eff"], 0.5, rtol=1e-6)
    np.testing.assert_allclose(table["PKCORR"], 0.8, rtol=1e-6)
    np.testing.assert_allclose(table["Omega_eff"], OMEGA_EFF, rtol=1e-6)
    np.testing.assert_allclose(table["k_phot"], 1.4, rtol=1e-12)
    pd.testing.assert_frame_equal(table[list(raw.columns)], raw)
    pd.testing.assert_frame_equal(lst, raw)
    assert result.summary["n_total_cal"] == 60
    assert result.summary["median_total_over_peak_cal_bright"] == pytest.approx(1.0, rel=1e-4)
    assert result.warnings == ()
    assert verify_lst_photometry(table) == []
    assert "k(zen)=1.4" in summarize_feather_photometry(result)


def test_calibrate_is_idempotent(tmp_path: Path) -> None:
    companions = _constant_companions(tmp_path)
    model = _flat_model(("78MHz",), 1.3)
    lst = _lst(np.linspace(-20.0, 60.0, 40), np.full(40, 1.3))
    once = calibrate_lst_photometry(lst, "78MHz", companions, model).table
    twice = calibrate_lst_photometry(once, "78MHz", companions, model).table
    pd.testing.assert_frame_equal(once, twice)


def test_calibrate_nan_maps(tmp_path: Path) -> None:
    npix = hp.nside2npix(NSIDE)
    pb = np.full(npix, 0.5, np.float32)
    lst = _lst(np.linspace(-20.0, 60.0, 30), np.full(30, 1.4))
    pb[hp.ang2pix(NSIDE, lst["RA"].iloc[0], lst["DEC"].iloc[0], lonlat=True)] = hp.UNSEEN
    companions = FeatherCompanions(
        pbmap=_write_map(tmp_path / "pb.fits", pb),
        psfmap=_write_map(tmp_path / "psf.fits", np.full(npix, OMEGA_EFF, np.float32)),
        pkcorr=_write_map(tmp_path / "pk.fits", np.full(npix, 1.0, np.float32)),
    )
    result = calibrate_lst_photometry(lst, "78MHz", companions, _flat_model(("78MHz",), 1.4))
    table = result.table
    assert np.isnan(table["Peak_flux_cal"].iloc[0])
    assert np.isnan(table["Total_flux_cal"].iloc[0])
    np.testing.assert_allclose(table["Peak_flux_cal"].iloc[1:], lst["Peak_flux"].iloc[1:] / 0.5)
    assert result.summary["n_invalid_maps"] >= 1
    assert "*_cal NaN" in " ".join(result.warnings)
    assert verify_lst_photometry(table) == []


def test_calibrate_without_shape_keeps_peak(tmp_path: Path) -> None:
    companions = _constant_companions(tmp_path)
    lst = _lst(np.linspace(-20.0, 60.0, 30), np.full(30, 1.4))
    lst.loc[0, ["Maj", "Min"]] = np.nan
    table = calibrate_lst_photometry(lst, "78MHz", companions, _flat_model(("78MHz",), 1.4)).table
    assert np.isnan(table["Total_flux_cal"].iloc[0])
    assert table["Peak_flux_cal"].iloc[0] == pytest.approx(lst["Peak_flux"].iloc[0] / 0.4)


def test_verify_lst_photometry_detects_stale_rows(tmp_path: Path) -> None:
    companions = _constant_companions(tmp_path)
    lst = _lst(np.linspace(-20.0, 60.0, 30), np.full(30, 1.4))
    assert verify_lst_photometry(lst) == []  # uncalibrated
    table = calibrate_lst_photometry(lst, "78MHz", companions, _flat_model(("78MHz",), 1.4)).table

    appended = pd.concat([table, lst.iloc[[0]]], ignore_index=True)
    assert any("without k_phot" in p for p in verify_lst_photometry(appended))

    replaced = table.copy()
    replaced.loc[3, "Peak_flux"] *= 1.5
    problems = verify_lst_photometry(replaced)
    assert any("Peak_flux_cal inconsistent" in p for p in problems)
    assert any("Total_flux_cal inconsistent" in p for p in problems)

    reshaped = table.copy()
    reshaped.loc[4, "Maj"] *= 1.2
    assert verify_lst_photometry(reshaped) == [
        "1 rows with Total_flux_cal inconsistent with raw columns"
    ]

    assert "missing calibration columns: k_phot" in verify_lst_photometry(
        table.drop(columns=["k_phot"])
    )


# --------------------------------------------------------------------------- JSON + layout


def test_photometry_json_round_trip(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    companions = _constant_companions(tmp_path)
    bands = {"82MHz": 82.0, "55MHz": 55.0}
    lst_by_band, omega = _band_tables(bands, dict.fromkeys(bands, DEC_CENTRES))
    calibration = PhotometryCalibration(
        k_model=estimate_photometry_k_model(lst_by_band, omega),
        companions=dict.fromkeys(bands, companions),
        config=FeatherPhotometryConfig(),
    )

    assert load_photometry_calibration(layout) is None  # no file yet
    path = write_photometry_calibration(layout, calibration)
    assert set(read_beam_scale_json(path)) == {"PHOTOMETRY"}

    write_beam_scale_json({"82MHz": 1.2, "55MHz": 1.3}, layout)
    payload = read_beam_scale_json(layout)
    assert set(payload) >= {"BEAM_SCALE", "PHOTOMETRY"}
    assert load_beam_scale(layout)["55MHz"].alpha == pytest.approx(1.3)

    loaded = load_photometry_calibration(layout)
    assert loaded == calibration
    assert loaded.method == "feather3_psfmap_joint_k"
    np.testing.assert_allclose(
        loaded.k_model.k_at("55MHz", DEC_CENTRES), calibration.k_model.k_at("55MHz", DEC_CENTRES)
    )


def test_load_photometry_calibration_absent_key(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path)
    write_beam_scale_json(1.2, layout)
    assert load_photometry_calibration(layout) is None


def _write_tree(layout: CatalogLayout, bands: dict[str, float]) -> None:
    dec = np.repeat(DEC_CENTRES, 12)
    for band, mhz in bands.items():
        write_lst_merged(_lst(dec, np.exp(_true_ln_k(mhz, dec))), layout, band)


def test_calibrate_layout_photometry(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path / "out")
    companions = _constant_companions(tmp_path)
    bands = {"82MHz": 82.0, "55MHz": 55.0, "23MHz": 23.0}
    _write_tree(layout, bands)
    assert lst_bands_in_layout(layout) == ("82MHz", "55MHz", "23MHz")

    batch = calibrate_layout_photometry(layout, dict.fromkeys(bands, companions))
    assert set(batch.results) == set(bands)
    assert list(batch.summary["band"]) == list(bands)
    assert batch.warnings == ()
    for band, mhz in bands.items():
        on_disk = read_lst_merged(layout, band)
        assert verify_lst_photometry(on_disk) == []
        np.testing.assert_allclose(
            np.log(on_disk["k_phot"]), _true_ln_k(mhz, on_disk["DEC"]), atol=1e-6
        )
        np.testing.assert_allclose(on_disk["Total_flux_cal"], on_disk["Peak_flux_cal"], rtol=1e-6)
    stored = load_photometry_calibration(layout)
    assert stored == batch.calibration
    text = summarize_feather_photometry(batch)
    assert "3 band(s)" in text
    assert "joint k over Dec -35..85" in text

    # Re-run with the stored model (reliability path after forced fill): unchanged.
    again = calibrate_layout_photometry(
        layout, stored.companions, k_model=stored.k_model, write=False
    )
    for band in bands:
        pd.testing.assert_frame_equal(again.results[band].table, read_lst_merged(layout, band))


def test_calibrate_layout_photometry_requires_every_band(tmp_path: Path) -> None:
    layout = CatalogLayout(tmp_path / "out")
    companions = _constant_companions(tmp_path)
    bands = {"82MHz": 82.0, "55MHz": 55.0}
    _write_tree(layout, bands)

    with pytest.raises(ValueError, match="every LST band"):
        calibrate_layout_photometry(layout, dict.fromkeys(bands, companions), bands=("82MHz",))
    with pytest.raises(ValueError, match="companions missing for: 55MHz"):
        calibrate_layout_photometry(layout, {"82MHz": companions})
    with pytest.raises(ValueError, match="no entry for: 55MHz"):
        calibrate_layout_photometry(
            layout, dict.fromkeys(bands, companions), k_model=_flat_model(("82MHz",), 1.4)
        )
    with pytest.raises(FileNotFoundError):
        calibrate_layout_photometry(CatalogLayout(tmp_path / "empty"), {})
    assert "Total_flux_cal" not in read_lst_merged(layout, "82MHz").columns
