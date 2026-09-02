"""Post-hoc Taylor spectral modeling for multi-band metacatalog rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from lwa_catalog.constants import (
    SUBBAND_BANDS_MHZ,
    SUBBAND_REF_FREQ_MHZ,
    band_frequency_hz,
)

FluxKind = Literal["total", "peak"]
_MIN_TAYLOR_TERMS = 2
_MAX_TAYLOR_TERMS = 4
_NAN_COEFFS = (float("nan"),) * _MAX_TAYLOR_TERMS


@dataclass(frozen=True)
class SpectralFitConfig:
    """Configuration for :func:`fit_single_spectrum` and batch fits."""

    bands: tuple[str, ...] = SUBBAND_BANDS_MHZ
    ref_freq_mhz: float = SUBBAND_REF_FREQ_MHZ
    flux_kind: FluxKind = "total"
    max_terms: int = _MAX_TAYLOR_TERMS
    use_flux_errors: bool = True
    column_prefix: str = "spec_"


@dataclass(frozen=True)
class SingleSpectrumFit:
    """Taylor fit result for one source."""

    n_terms: int
    bic: float
    chi2_red: float
    n_flux: int
    coeffs: tuple[float, float, float, float]
    nu0_mhz: float


@dataclass
class SpectralFitResult:
    """Batch Taylor spectral fit for a metacatalog table."""

    summary: dict[str, float | int | dict[int, int]]
    metacatalog: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _flux_column_names(flux_kind: FluxKind) -> tuple[str, str]:
    if flux_kind == "peak":
        return "Peak_flux", "E_Peak_flux"
    return "Total_flux", "E_Total_flux"


def gather_band_flux_measurements(
    row: pd.Series,
    *,
    bands: tuple[str, ...] = SUBBAND_BANDS_MHZ,
    flux_kind: FluxKind = "total",
    origin_band_key: str = "origin_band",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(nu_hz, flux_jy, err_jy)`` for positive finite per-band fluxes.

  Checks ``{flux}_{band}`` columns in *bands* order. When the suffixed column is
  missing or invalid and *band* equals ``row[origin_band_key]``, falls back to the
  primary unsuffixed flux column (same pattern as
  :func:`~lwa_catalog.analyze.nedlvs.resolve_highest_frequency_peak_flux`).
    """
    flux_prefix, err_prefix = _flux_column_names(flux_kind)
    origin_band = str(row.get(origin_band_key, "") or "")

    nu_list: list[float] = []
    flux_list: list[float] = []
    err_list: list[float] = []

    for band in bands:
        freq = band_frequency_hz(band)
        if not np.isfinite(freq) or freq <= 0.0:
            continue

        flux_col = f"{flux_prefix}_{band}"
        err_col = f"{err_prefix}_{band}"
        flux = np.nan
        err = np.nan

        if flux_col in row.index:
            flux = pd.to_numeric(row[flux_col], errors="coerce")
        if band == origin_band and (not np.isfinite(flux) or float(flux) <= 0.0):
            if flux_prefix in row.index:
                flux = pd.to_numeric(row[flux_prefix], errors="coerce")
        if not np.isfinite(flux) or float(flux) <= 0.0:
            continue

        if err_col in row.index:
            err = pd.to_numeric(row[err_col], errors="coerce")
        if band == origin_band and (not np.isfinite(err) or float(err) < 0.0):
            primary_err = f"E_{flux_prefix}"
            if primary_err in row.index:
                err = pd.to_numeric(row[primary_err], errors="coerce")

        nu_list.append(float(freq))
        flux_list.append(float(flux))
        err_list.append(float(err) if np.isfinite(err) and float(err) >= 0.0 else np.nan)

    if not nu_list:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )
    return (
        np.asarray(nu_list, dtype=float),
        np.asarray(flux_list, dtype=float),
        np.asarray(err_list, dtype=float),
    )


def _design_matrix(x: np.ndarray, n_terms: int) -> np.ndarray:
    """Vandermonde matrix with columns ``[1, x, x^2, ...]`` up to ``n_terms``."""
    return np.vstack([x**power for power in range(n_terms)]).T


def _weighted_lstsq(
    design: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
) -> np.ndarray:
    """Solve weighted least squares; unweighted when *weights* is ``None``."""
    if weights is None:
        coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        return coeffs
    w_sqrt = np.sqrt(weights)
    coeffs, _, _, _ = np.linalg.lstsq(design * w_sqrt[:, None], y * w_sqrt, rcond=None)
    return coeffs


def _bic(n_points: int, n_terms: int, chi2: float) -> float:
    """Bayesian Information Criterion for Gaussian residuals."""
    if n_points < n_terms or chi2 <= 0.0 or n_points <= 0:
        return float("nan")
    return float(n_points * np.log(chi2 / n_points) + n_terms * np.log(n_points))


def _log_flux_weights(flux_jy: np.ndarray, err_jy: np.ndarray) -> np.ndarray | None:
    """Return ``1 / sigma_lnS^2`` with ``sigma_lnS = E_S / S`` when all errors exist."""
    if err_jy.shape != flux_jy.shape:
        return None
    valid = (
        np.isfinite(err_jy)
        & (err_jy > 0.0)
        & np.isfinite(flux_jy)
        & (flux_jy > 0.0)
    )
    if not valid.all():
        return None
    sigma_ln = err_jy / flux_jy
    return 1.0 / sigma_ln**2


def _empty_fit(nu0_mhz: float) -> SingleSpectrumFit:
    return SingleSpectrumFit(
        n_terms=0,
        bic=float("nan"),
        chi2_red=float("nan"),
        n_flux=0,
        coeffs=_NAN_COEFFS,
        nu0_mhz=nu0_mhz,
    )


def _chi2_red_near_best(chi2_red: float, min_chi2_red: float) -> bool:
    """True when *chi2_red* is statistically indistinguishable from the best fit."""
    if min_chi2_red <= 1e-8:
        return chi2_red <= max(1e-6, min_chi2_red + 1e-8)
    return chi2_red <= min_chi2_red + max(0.05, 0.1 * min_chi2_red)


def _select_best_fit(candidates: list[SingleSpectrumFit], *, nu0_mhz: float) -> SingleSpectrumFit:
    """Pick the simplest model near the best χ²_red, else minimum BIC."""
    if not candidates:
        return _empty_fit(nu0_mhz)

    finite_chi2 = [candidate for candidate in candidates if np.isfinite(candidate.chi2_red)]
    if finite_chi2:
        min_chi2_red = min(candidate.chi2_red for candidate in finite_chi2)
        adequate = [
            candidate
            for candidate in finite_chi2
            if _chi2_red_near_best(candidate.chi2_red, min_chi2_red)
        ]
        if adequate:
            return min(adequate, key=lambda candidate: candidate.n_terms)

    finite_bic = [candidate for candidate in candidates if np.isfinite(candidate.bic)]
    if finite_bic:
        return min(finite_bic, key=lambda candidate: (candidate.bic, candidate.n_terms))

    return candidates[-1]


def _pad_coeffs(coeffs: np.ndarray) -> tuple[float, float, float, float]:
    out = list(_NAN_COEFFS)
    for idx, value in enumerate(coeffs[:_MAX_TAYLOR_TERMS]):
        out[idx] = float(value)
    return tuple(out)  # type: ignore[return-value]


def fit_single_spectrum(
    nu_hz: np.ndarray,
    flux_jy: np.ndarray,
    err_jy: np.ndarray,
    *,
    config: SpectralFitConfig | None = None,
) -> SingleSpectrumFit:
    """Fit nested Taylor models in log-flux and select the simplest adequate model.

  Models use ``ln S = sum_j a_j [ln(nu/nu0)]^j`` for ``j = 0 .. n_terms-1`` with
  ``n_terms`` in ``2 .. min(max_terms, n_points)``. Among models with reduced
  chi-squared within 5% (or 0.05 absolute) of the best reduced chi-squared, the fewest
  terms wins; otherwise the lowest BIC wins.
    """
    cfg = SpectralFitConfig() if config is None else config
    nu0_mhz = float(cfg.ref_freq_mhz)
    nu0_hz = nu0_mhz * 1e6
    max_terms = max(_MIN_TAYLOR_TERMS, min(int(cfg.max_terms), _MAX_TAYLOR_TERMS))

    nu = np.asarray(nu_hz, dtype=float)
    flux = np.asarray(flux_jy, dtype=float)
    err = np.asarray(err_jy, dtype=float)
    valid = np.isfinite(nu) & (nu > 0.0) & np.isfinite(flux) & (flux > 0.0)
    nu = nu[valid]
    flux = flux[valid]
    err = err[valid] if err.shape == valid.shape else np.full(nu.shape, np.nan)

    n_points = int(nu.size)
    if n_points == 0:
        return _empty_fit(nu0_mhz)

    x = np.log(nu / nu0_hz)
    y = np.log(flux)
    weights: np.ndarray | None
    if cfg.use_flux_errors:
        weights = _log_flux_weights(flux, err)
    else:
        weights = None

    candidates: list[SingleSpectrumFit] = []

    for n_terms in range(_MIN_TAYLOR_TERMS, min(max_terms, n_points) + 1):
        design = _design_matrix(x, n_terms)
        coeffs = _weighted_lstsq(design, y, weights)
        y_hat = design @ coeffs
        if weights is None:
            residuals = y - y_hat
            chi2 = float(np.dot(residuals, residuals))
        else:
            residuals = y - y_hat
            chi2 = float(np.dot(residuals**2, weights))

        bic = _bic(n_points, n_terms, chi2)
        dof = n_points - n_terms
        chi2_red = float(chi2 / dof) if dof > 0 else float("nan")

        candidates.append(
            SingleSpectrumFit(
                n_terms=n_terms,
                bic=bic,
                chi2_red=chi2_red,
                n_flux=n_points,
                coeffs=_pad_coeffs(coeffs),
                nu0_mhz=nu0_mhz,
            )
        )

    return _select_best_fit(candidates, nu0_mhz=nu0_mhz)


def evaluate_taylor_spectrum(
    nu_hz: np.ndarray | float,
    fit: SingleSpectrumFit,
) -> np.ndarray:
    """Evaluate ``S(nu)`` from a :class:`SingleSpectrumFit` on the Taylor basis."""
    nu0_hz = fit.nu0_mhz * 1e6
    nu_arr = np.asarray(nu_hz, dtype=float)
    x = np.log(nu_arr / nu0_hz)
    log_s = np.zeros_like(x, dtype=float)
    for idx in range(fit.n_terms):
        log_s += fit.coeffs[idx] * x**idx
    return np.exp(log_s)


def _output_column_names(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}model_n_terms",
        f"{prefix}model_bic",
        f"{prefix}model_chi2_red",
        f"{prefix}model_n_flux",
        f"{prefix}model_nu0_mhz",
        f"{prefix}model_a0",
        f"{prefix}model_a1",
        f"{prefix}model_a2",
        f"{prefix}model_a3",
    )


def _single_fit_to_row(fit: SingleSpectrumFit, prefix: str) -> dict[str, float | int]:
    return {
        f"{prefix}model_n_terms": int(fit.n_terms),
        f"{prefix}model_bic": float(fit.bic),
        f"{prefix}model_chi2_red": float(fit.chi2_red),
        f"{prefix}model_n_flux": int(fit.n_flux),
        f"{prefix}model_nu0_mhz": float(fit.nu0_mhz),
        f"{prefix}model_a0": float(fit.coeffs[0]),
        f"{prefix}model_a1": float(fit.coeffs[1]),
        f"{prefix}model_a2": float(fit.coeffs[2]),
        f"{prefix}model_a3": float(fit.coeffs[3]),
    }


def _has_flux_columns(metacatalog: pd.DataFrame, config: SpectralFitConfig) -> bool:
    flux_prefix, _ = _flux_column_names(config.flux_kind)
    if flux_prefix in metacatalog.columns:
        return True
    return any(f"{flux_prefix}_{band}" in metacatalog.columns for band in config.bands)


def fit_metacatalog_spectra(
    metacatalog: pd.DataFrame,
    config: SpectralFitConfig | None = None,
) -> SpectralFitResult:
    """Fit Taylor spectra for every row in a metacatalog table.

    Gathers per-band flux measurements, runs :func:`fit_single_spectrum` per row,
    and appends ``{prefix}model_*`` columns to a copy of the input table.
    """
    cfg = SpectralFitConfig() if config is None else config
    warnings: list[str] = []
    prefix = cfg.column_prefix

    if metacatalog.empty:
        warnings.append("metacatalog is empty; no spectral fits computed")
        return SpectralFitResult(
            summary={
                "n_sources": 0,
                "n_fitted": 0,
                "n_terms_hist": {2: 0, 3: 0, 4: 0},
                "median_bic": float("nan"),
                "median_n_flux": float("nan"),
                "n_warnings": len(warnings),
            },
            metacatalog=metacatalog.copy(),
            warnings=warnings,
        )

    if not _has_flux_columns(metacatalog, cfg):
        flux_prefix, _ = _flux_column_names(cfg.flux_kind)
        warnings.append(
            f"no {flux_prefix} or {flux_prefix}_{{band}} columns found; fits will be empty"
        )

    out = metacatalog.copy()
    fit_rows: list[dict[str, float | int]] = []
    n_terms_hist = {2: 0, 3: 0, 4: 0}
    bic_values: list[float] = []
    n_flux_values: list[int] = []
    n_fitted = 0

    for _, row in out.iterrows():
        nu_hz, flux_jy, err_jy = gather_band_flux_measurements(
            row,
            bands=cfg.bands,
            flux_kind=cfg.flux_kind,
        )
        fit = fit_single_spectrum(nu_hz, flux_jy, err_jy, config=cfg)
        fit_rows.append(_single_fit_to_row(fit, prefix))
        if fit.n_flux > 0:
            n_fitted += 1
            n_flux_values.append(fit.n_flux)
            if fit.n_terms in n_terms_hist:
                n_terms_hist[fit.n_terms] += 1
            if np.isfinite(fit.bic):
                bic_values.append(float(fit.bic))

    fit_df = pd.DataFrame(fit_rows, index=out.index)
    for col in _output_column_names(prefix):
        out[col] = fit_df[col]

    summary: dict[str, float | int | dict[int, int]] = {
        "n_sources": int(len(out)),
        "n_fitted": int(n_fitted),
        "n_terms_hist": n_terms_hist,
        "median_bic": float(np.median(bic_values)) if bic_values else float("nan"),
        "median_n_flux": float(np.median(n_flux_values)) if n_flux_values else float("nan"),
        "n_warnings": len(warnings),
    }
    return SpectralFitResult(summary=summary, metacatalog=out, warnings=warnings)


def summarize_spectral_fit(result: SpectralFitResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    summary = result.summary
    hist = summary["n_terms_hist"]
    median_bic = float(summary["median_bic"])
    median_n_flux = float(summary["median_n_flux"])
    lines = [
        f"Sources:                      {int(summary['n_sources']):6d}",
        f"Fitted (>=1 flux):            {int(summary['n_fitted']):6d}",
        (
            f"Median BIC:                   {median_bic:.3f}"
            if np.isfinite(median_bic)
            else "Median BIC:                        nan"
        ),
        (
            f"Median flux channels:         {median_n_flux:.1f}"
            if np.isfinite(median_n_flux)
            else "Median flux channels:              nan"
        ),
        "Model terms selected:",
        f"  2-term: {int(hist[2]):6d}",
        f"  3-term: {int(hist[3]):6d}",
        f"  4-term: {int(hist[4]):6d}",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in result.warnings)
    return "\n".join(lines)
