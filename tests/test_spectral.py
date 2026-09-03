"""Tests for post-hoc Taylor spectral modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.spectral import (
    SpectralFitConfig,
    evaluate_taylor_spectrum,
    fit_metacatalog_spectra,
    fit_single_spectrum,
    gather_band_flux_measurements,
    summarize_spectral_fit,
)
from lwa_catalog.constants import SUBBAND_BANDS_MHZ, SUBBAND_REF_FREQ_MHZ


def _power_law_flux(nu_hz: np.ndarray, *, alpha: float, s_ref: float, nu_ref_hz: float) -> np.ndarray:
    return s_ref * (nu_hz / nu_ref_hz) ** alpha


def test_fit_power_law_selects_two_terms() -> None:
    bands = ("18MHz", "32MHz", "46MHz", "59MHz", "82MHz")
    nu_hz = np.array([float(b.removesuffix("MHz")) * 1e6 for b in bands])
    nu_ref = SUBBAND_REF_FREQ_MHZ * 1e6
    alpha_true = -0.7
    flux = _power_law_flux(nu_hz, alpha=alpha_true, s_ref=1.0, nu_ref_hz=nu_ref)
    err = 0.05 * flux

    fit = fit_single_spectrum(
        nu_hz,
        flux,
        err,
        config=SpectralFitConfig(bands=bands, use_flux_errors=False),
    )
    assert fit.n_terms == 2
    assert fit.n_flux == 5
    assert fit.coeffs[1] == pytest.approx(alpha_true, abs=1e-6)


def test_fit_single_point_returns_empty() -> None:
    nu_hz = np.array([55e6])
    flux = np.array([2.5])
    err = np.array([0.1])

    fit = fit_single_spectrum(nu_hz, flux, err)
    assert fit.n_terms == 0
    assert fit.n_flux == 0
    assert np.isnan(fit.bic)


def test_fit_two_points_selects_two_terms() -> None:
    nu_hz = np.array([18e6, 82e6])
    nu_ref = SUBBAND_REF_FREQ_MHZ * 1e6
    flux = _power_law_flux(nu_hz, alpha=-0.8, s_ref=1.0, nu_ref_hz=nu_ref)
    fit = fit_single_spectrum(
        nu_hz,
        flux,
        0.05 * flux,
        config=SpectralFitConfig(use_flux_errors=False),
    )
    assert fit.n_terms == 2
    assert fit.n_flux == 2
    assert fit.coeffs[1] == pytest.approx(-0.8, abs=1e-6)


def test_fit_insufficient_points() -> None:
    fit = fit_single_spectrum(np.array([]), np.array([]), np.array([]))
    assert fit.n_flux == 0
    assert fit.n_terms == 0
    assert np.isnan(fit.bic)
    assert all(np.isnan(c) for c in fit.coeffs)


def test_fit_curvature_selects_three_terms() -> None:
    nu0_hz = SUBBAND_REF_FREQ_MHZ * 1e6
    bands = SUBBAND_BANDS_MHZ
    nu_hz = np.array([float(b.removesuffix("MHz")) * 1e6 for b in bands])
    x = np.log(nu_hz / nu0_hz)
    a0, a1, a2 = 0.5, -0.3, 0.15
    log_flux = a0 + a1 * x + a2 * x**2
    flux = np.exp(log_flux)

    fit = fit_single_spectrum(
        nu_hz,
        flux,
        np.zeros_like(flux),
        config=SpectralFitConfig(use_flux_errors=False),
    )
    assert fit.n_terms == 3
    assert fit.coeffs[0] == pytest.approx(a0, abs=1e-6)
    assert fit.coeffs[1] == pytest.approx(a1, abs=1e-6)
    assert fit.coeffs[2] == pytest.approx(a2, abs=1e-6)


def test_gather_band_flux_origin_fallback() -> None:
    row = pd.Series(
        {
            "origin_band": "55MHz",
            "Total_flux": 3.2,
            "E_Total_flux": 0.2,
            "Total_flux_18MHz": 1.0,
            "E_Total_flux_18MHz": 0.1,
        }
    )
    nu_hz, flux, err = gather_band_flux_measurements(
        row,
        bands=("18MHz", "55MHz"),
        flux_kind="total",
    )
    assert nu_hz.size == 2
    assert 18e6 in nu_hz
    assert 55e6 in nu_hz
    assert 3.2 in flux
    assert 1.0 in flux
    assert 0.2 in err


def test_gather_includes_radio_survey_bands() -> None:
    row = pd.Series(
        {
            "Total_flux_55MHz": 1.0,
            "Total_flux_VLSSR": 0.8,
            "Total_flux_NVSS": 0.15,
            "Total_flux_VLASS": 0.05,
        }
    )
    nu_hz, flux, _ = gather_band_flux_measurements(
        row,
        bands=("55MHz", "VLSSR", "NVSS", "VLASS"),
        flux_kind="total",
    )
    assert list(nu_hz) == pytest.approx([55e6, 74e6, 1.4e9, 3e9])
    assert list(flux) == pytest.approx([1.0, 0.8, 0.15, 0.05])


def test_bic_prefers_simpler_model() -> None:
    rng = np.random.default_rng(42)
    bands = SUBBAND_BANDS_MHZ
    nu_hz = np.array([float(b.removesuffix("MHz")) * 1e6 for b in bands])
    nu_ref = SUBBAND_REF_FREQ_MHZ * 1e6
    flux = _power_law_flux(nu_hz, alpha=-0.6, s_ref=1.0, nu_ref_hz=nu_ref)
    flux_noisy = flux * (1.0 + rng.normal(0.0, 0.02, size=flux.shape))
    err = 0.05 * flux_noisy

    fit = fit_single_spectrum(
        nu_hz,
        flux_noisy,
        err,
        config=SpectralFitConfig(use_flux_errors=False, max_terms=4),
    )
    assert fit.n_terms <= 2


def test_evaluate_taylor_spectrum_roundtrip() -> None:
    nu_hz = np.array([18e6, 55e6, 82e6])
    flux = _power_law_flux(
        nu_hz,
        alpha=-0.5,
        s_ref=1.0,
        nu_ref_hz=SUBBAND_REF_FREQ_MHZ * 1e6,
    )
    fit = fit_single_spectrum(
        nu_hz,
        flux,
        np.zeros_like(flux),
        config=SpectralFitConfig(use_flux_errors=False),
    )
    recovered = evaluate_taylor_spectrum(nu_hz, fit)
    np.testing.assert_allclose(recovered, flux, rtol=1e-6)


def _mini_metacatalog_rows() -> pd.DataFrame:
    bands = ("18MHz", "23MHz", "27MHz")
    nu_ref = SUBBAND_REF_FREQ_MHZ * 1e6
    rows = []
    for idx in range(3):
        nu_hz = np.array([float(b.removesuffix("MHz")) * 1e6 for b in bands])
        flux = _power_law_flux(nu_hz, alpha=-0.5 - 0.1 * idx, s_ref=1.0, nu_ref_hz=nu_ref)
        row = {
            "meta_id": idx,
            "origin_band": bands[0],
            "bands_present": ",".join(bands),
        }
        for band, value in zip(bands, flux, strict=True):
            row[f"Total_flux_{band}"] = float(value)
            row[f"E_Total_flux_{band}"] = 0.05 * float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def test_fit_metacatalog_spectra_columns() -> None:
    meta = _mini_metacatalog_rows()
    result = fit_metacatalog_spectra(
        meta,
        config=SpectralFitConfig(bands=("18MHz", "23MHz", "27MHz"), use_flux_errors=False),
    )
    expected = {
        "spec_model_n_terms",
        "spec_model_bic",
        "spec_model_chi2_red",
        "spec_model_n_flux",
        "spec_model_nu0_mhz",
        "spec_model_a0",
        "spec_model_a1",
        "spec_model_a2",
        "spec_model_a3",
    }
    assert expected.issubset(result.metacatalog.columns)
    assert len(result.metacatalog) == 3
    n_flux = result.metacatalog["spec_model_n_flux"]
    assert n_flux.notna().all()
    assert (n_flux >= 2).all()
    assert not (n_flux == 0).any()
    assert not (n_flux == 1).any()


def test_fit_metacatalog_skips_single_band_rows() -> None:
    meta = pd.DataFrame(
        [
            {
                "meta_id": 0,
                "origin_band": "18MHz",
                "bands_present": "18MHz",
                "Total_flux_18MHz": 1.0,
                "E_Total_flux_18MHz": 0.05,
            },
            {
                "meta_id": 1,
                "origin_band": "18MHz",
                "bands_present": "18MHz,23MHz",
                "Total_flux_18MHz": 1.0,
                "E_Total_flux_18MHz": 0.05,
                "Total_flux_23MHz": 0.9,
                "E_Total_flux_23MHz": 0.05,
            },
        ]
    )
    result = fit_metacatalog_spectra(
        meta,
        config=SpectralFitConfig(bands=("18MHz", "23MHz"), use_flux_errors=False),
    )
    assert result.summary["n_sources"] == 2
    assert result.summary["n_fitted"] == 1
    assert np.isnan(result.metacatalog.loc[0, "spec_model_n_flux"])
    assert np.isnan(result.metacatalog.loc[0, "spec_model_n_terms"])
    assert int(result.metacatalog.loc[1, "spec_model_n_flux"]) == 2
    assert int(result.metacatalog.loc[1, "spec_model_n_terms"]) == 2
    assert not (result.metacatalog["spec_model_n_flux"] == 0).any()
    assert not (result.metacatalog["spec_model_n_flux"] == 1).any()


def test_fit_metacatalog_spectra_summary() -> None:
    meta = _mini_metacatalog_rows()
    result = fit_metacatalog_spectra(
        meta,
        config=SpectralFitConfig(bands=("18MHz", "23MHz", "27MHz"), use_flux_errors=False),
    )
    assert result.summary["n_sources"] == 3
    assert result.summary["n_fitted"] == 3
    assert sum(result.summary["n_terms_hist"].values()) == 3
    text = summarize_spectral_fit(result)
    assert "Sources:" in text
    assert "Fitted" in text


def test_fit_metacatalog_empty() -> None:
    result = fit_metacatalog_spectra(pd.DataFrame())
    assert result.metacatalog.empty
    assert result.summary["n_sources"] == 0
    assert result.summary["n_fitted"] == 0
    assert result.warnings
    assert "empty" in result.warnings[0].lower()


def test_fit_metacatalog_spectra_ten_rows_under_one_second() -> None:
    import time

    rows = [_mini_metacatalog_rows().iloc[0].copy() for _ in range(10)]
    meta = pd.DataFrame(rows)
    start = time.perf_counter()
    result = fit_metacatalog_spectra(
        meta,
        config=SpectralFitConfig(bands=("18MHz", "23MHz", "27MHz"), use_flux_errors=False),
    )
    elapsed = time.perf_counter() - start
    assert result.summary["n_sources"] == 10
    assert elapsed < 1.0
