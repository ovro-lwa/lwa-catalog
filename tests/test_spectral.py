"""Tests for post-hoc Taylor spectral modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.spectral import (
    SpectralFitConfig,
    evaluate_taylor_spectrum,
    fit_single_spectrum,
    gather_band_flux_measurements,
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


def test_fit_constant_selects_one_term() -> None:
    nu_hz = np.array([55e6])
    flux = np.array([2.5])
    err = np.array([0.1])

    fit = fit_single_spectrum(nu_hz, flux, err)
    assert fit.n_terms == 1
    assert fit.n_flux == 1
    assert fit.coeffs[0] == pytest.approx(np.log(2.5), rel=1e-9)


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
