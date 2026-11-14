"""Per-source photometric calibration on FEATHER3 apparent-flux mosaics.

FEATHER3 ``jybeam`` mosaics are apparent flux (primary beam used as a weight,
never divided out) in units of the per-pixel effective beam ``Ω_eff``. The
companion maps (``pbmap``, ``psfmap``, ``pkcorr``; see
:func:`lwa_catalog.create.healpix_prebuilt.companion_maps_for_jybeam`) let a
catalog-side step turn raw PyBDSF ``Peak_flux`` / ``Total_flux`` into
PB-corrected, ``Ω_eff``-normalised ``*_cal`` columns without touching the
detection product::

    Peak_flux_cal  = Peak_flux / (PB_eff · PKCORR)
    Total_flux_cal = Peak_flux · Ω_fit / (k(δ, ν) · Ω_eff) / (PB_eff · PKCORR)
    Ω_fit          = π / (4 ln 2) · Maj · Min

``k(δ, ν)`` (the median ``Ω_fit / Ω_eff`` of bright compact sources) is one
joint model fitted across **all** bands of a tree::

    ln k = a_band + Σ_p (c_p0 + c_p1 · u) · x^p,   x = (δ − φ) / 60°,  u = ln(ν / 55 MHz)

so every frequency measures Total the same way; sparse bands borrow the Dec
shape from their neighbours instead of falling back to a constant. Inputs are
required up front (all three companion maps, ``S_Code``, ``E_Peak_flux``,
enough bright compact sources per band); anything missing raises. Raw columns
are never modified, so calibration is idempotent. The joint model and the
resolved map paths persist under the tree-level ``PHOTOMETRY`` key of
``beam_scale.json``.

Derived in ``.agents/experiment-pybdsf-flux-reliability-psfmap.md``.
Requires ``healpy`` (``lwa-catalog[analyze]``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lwa_catalog.analyze.beam_alpha_bootstrap import PHOTOMETRY_KEY, read_beam_scale_json
from lwa_catalog.constants import (
    BAND_CAL_FLUX_FIELDS,
    CAL_DIAGNOSTIC_COLUMNS,
    OVRO_LATITUDE_DEG,
    SUBBAND_REF_FREQ_MHZ,
    band_frequency_hz,
)
from lwa_catalog.create.healpix_prebuilt import FeatherCompanions
from lwa_catalog.io import read_lst_merged, write_lst_merged
from lwa_catalog.paths import CatalogLayout

__all__ = [
    "FEATHER_PHOTOMETRY_METHOD",
    "BandKDiagnostics",
    "FeatherPhotometryBatchResult",
    "FeatherPhotometryConfig",
    "FeatherPhotometryResult",
    "PhotometryCalibration",
    "PhotometryKModel",
    "calibrate_layout_photometry",
    "calibrate_lst_photometry",
    "estimate_photometry_k_model",
    "load_photometry_calibration",
    "lst_bands_in_layout",
    "sample_healpix_at",
    "summarize_feather_photometry",
    "verify_lst_photometry",
    "write_photometry_calibration",
]

FEATHER_PHOTOMETRY_METHOD = "feather3_psfmap_joint_k"
_GAUSS_AREA = np.pi / (4.0 * np.log(2.0))
_HEALPY_UNSEEN = -1.6375e30
_LST_PREFIX = "metacatalog_lst_"


def _require_healpy():
    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - exercised when healpy absent
        msg = (
            "healpy is required to sample FEATHER3 companion maps; "
            "install with: pip install 'lwa-catalog[analyze]'"
        )
        raise ImportError(msg) from exc
    return hp


@lru_cache(maxsize=3)
def _load_healpix_signal(path: str, mtime_ns: int) -> tuple[np.ndarray, int, bool]:
    """Return ``(signal, nside, nested)`` for a full-sky healpy BinTable map."""
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        table = next((hdu for hdu in hdul[1:] if isinstance(hdu, fits.BinTableHDU)), None)
        if table is None:
            msg = f"No HEALPix BinTable extension in {path}"
            raise ValueError(msg)
        header = table.header
        if str(header.get("INDXSCHM", "IMPLICIT")).upper() != "IMPLICIT":
            msg = f"Partial-sky (explicit index) HEALPix maps are not supported: {path}"
            raise ValueError(msg)
        names = table.columns.names
        column = "SIGNAL" if "SIGNAL" in names else names[0]
        signal = np.array(table.data[column], dtype=np.float32).ravel()
    signal[signal <= _HEALPY_UNSEEN * 0.99] = np.nan
    nested = str(header.get("ORDERING", "RING")).upper().startswith("NEST")
    hp = _require_healpy()
    nside = int(hp.npix2nside(signal.size))
    return signal, nside, nested


def sample_healpix_at(
    path: Path | str,
    ra: np.ndarray,
    dec: np.ndarray,
) -> np.ndarray:
    """Sample a full-sky healpy ``SIGNAL`` map at ``(RA, Dec)`` in degrees.

    Nearest-pixel lookup honouring the file's ``ORDERING`` (RING or NESTED).
    ``UNSEEN`` pixels and non-finite coordinates give ``NaN``. The last three
    maps read are cached (keyed by path and modification time), so sampling
    ``pbmap`` / ``psfmap`` / ``pkcorr`` for one band reads each file once.
    """
    hp = _require_healpy()
    path = Path(path).resolve()
    signal, nside, nested = _load_healpix_signal(str(path), path.stat().st_mtime_ns)

    ra = np.atleast_1d(np.asarray(ra, dtype=np.float64))
    dec = np.atleast_1d(np.asarray(dec, dtype=np.float64))
    out = np.full(ra.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(ra) & np.isfinite(dec) & (np.abs(dec) <= 90.0)
    if ok.any():
        ipix = hp.ang2pix(nside, np.mod(ra[ok], 360.0), dec[ok], nest=nested, lonlat=True)
        out[ok] = signal[ipix]
    return out


@dataclass(frozen=True, slots=True)
class FeatherPhotometryConfig:
    """Configuration for the joint ``k(δ, ν)`` model and calibration.

    ``k`` uses rows with ``S_Code == s_code`` and
    ``Peak_flux / E_Peak_flux >= min_snr``. Per band, the median
    ``Ω_fit / Ω_eff`` in ``dec_bin_deg`` bins with at least
    ``min_rows_per_bin`` rows enters a count-weighted least-squares fit of
    ``ln k`` (``dec_degree`` polynomial in ``x = (δ − φ) / dec_scale_deg``,
    coefficients polynomial of ``freq_degree`` in ``u = ln(ν / ref_freq_mhz)``,
    plus one offset per band). A band with fewer than ``min_rows_per_band``
    usable rows raises instead of borrowing a fallback.
    """

    min_snr: float = 20.0
    s_code: str = "S"
    dec_bin_deg: float = 10.0
    min_rows_per_bin: int = 5
    min_rows_per_band: int = 100
    dec_degree: int = 4
    freq_degree: int = 1
    dec_scale_deg: float = 60.0
    ref_freq_mhz: float = SUBBAND_REF_FREQ_MHZ

    def as_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> FeatherPhotometryConfig:
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__ if name in raw})


@dataclass(frozen=True, slots=True)
class BandKDiagnostics:
    """Per-band inputs to the joint ``k`` fit (bin medians are not used directly)."""

    n_used: int
    k_all: float
    dec_centres: tuple[float, ...]
    k_dec_median: tuple[float, ...]
    n_dec: tuple[int, ...]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "n_used": int(self.n_used),
            "k_all": float(self.k_all),
            "dec_centres": [float(d) for d in self.dec_centres],
            "k_dec_median": [float(k) for k in self.k_dec_median],
            "n_dec": [int(n) for n in self.n_dec],
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BandKDiagnostics:
        return cls(
            n_used=int(raw["n_used"]),
            k_all=float(raw["k_all"]),
            dec_centres=tuple(float(d) for d in raw["dec_centres"]),
            k_dec_median=tuple(float(k) for k in raw["k_dec_median"]),
            n_dec=tuple(int(n) for n in raw["n_dec"]),
        )


@dataclass(frozen=True)
class PhotometryKModel:
    """Joint ``k(δ, ν) = median Ω_fit / Ω_eff`` model shared by every band of a tree.

    ``shape[p - 1][q]`` multiplies ``u^q · x^p``. ``δ`` is clamped to
    ``dec_range`` (the span of fitted bin centres) before evaluation.
    """

    band_offset: Mapping[str, float]
    band_freq_mhz: Mapping[str, float]
    shape: tuple[tuple[float, ...], ...]
    dec_range: tuple[float, float]
    phi_deg: float
    dec_scale_deg: float
    ref_freq_mhz: float
    diagnostics: Mapping[str, BandKDiagnostics] = field(default_factory=dict)

    @property
    def bands(self) -> tuple[str, ...]:
        return tuple(self.band_offset)

    def k_at(self, band: str, dec: np.ndarray | float) -> np.ndarray:
        """``k`` for *band* at declination(s) *dec* (deg); NaN where *dec* is NaN."""
        if band not in self.band_offset:
            msg = f"band {band!r} is not in the photometry k model ({', '.join(self.bands)})"
            raise KeyError(msg)
        dec = np.asarray(dec, dtype=np.float64)
        x = (np.clip(dec, *self.dec_range) - self.phi_deg) / self.dec_scale_deg
        u = np.log(self.band_freq_mhz[band] / self.ref_freq_mhz)
        ln_k = np.full(dec.shape, float(self.band_offset[band]))
        for p, coeffs in enumerate(self.shape, start=1):
            ln_k = ln_k + sum(c * u**q for q, c in enumerate(coeffs)) * x**p
        return np.exp(ln_k)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "band_offset": {b: float(a) for b, a in self.band_offset.items()},
            "band_freq_mhz": {b: float(f) for b, f in self.band_freq_mhz.items()},
            "shape": [[float(c) for c in row] for row in self.shape],
            "dec_range": [float(d) for d in self.dec_range],
            "phi_deg": float(self.phi_deg),
            "dec_scale_deg": float(self.dec_scale_deg),
            "ref_freq_mhz": float(self.ref_freq_mhz),
            "diagnostics": {b: d.as_mapping() for b, d in self.diagnostics.items()},
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PhotometryKModel:
        lo, hi = (float(d) for d in raw["dec_range"])
        return cls(
            band_offset={str(b): float(a) for b, a in raw["band_offset"].items()},
            band_freq_mhz={str(b): float(f) for b, f in raw["band_freq_mhz"].items()},
            shape=tuple(tuple(float(c) for c in row) for row in raw["shape"]),
            dec_range=(lo, hi),
            phi_deg=float(raw["phi_deg"]),
            dec_scale_deg=float(raw["dec_scale_deg"]),
            ref_freq_mhz=float(raw["ref_freq_mhz"]),
            diagnostics={
                str(b): BandKDiagnostics.from_mapping(d)
                for b, d in (raw.get("diagnostics") or {}).items()
            },
        )


@dataclass(frozen=True)
class PhotometryCalibration:
    """Tree-level ``PHOTOMETRY`` entry of ``beam_scale.json``."""

    k_model: PhotometryKModel
    companions: Mapping[str, FeatherCompanions]
    config: FeatherPhotometryConfig
    method: str = FEATHER_PHOTOMETRY_METHOD

    def as_mapping(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "config": self.config.as_mapping(),
            "k_model": self.k_model.as_mapping(),
            "bands": {
                band: {
                    "maps": {
                        "pbmap": str(c.pbmap),
                        "psfmap": str(c.psfmap),
                        "pkcorr": str(c.pkcorr),
                    }
                }
                for band, c in self.companions.items()
            },
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PhotometryCalibration:
        companions = {
            str(band): FeatherCompanions(
                pbmap=Path(entry["maps"]["pbmap"]),
                psfmap=Path(entry["maps"]["psfmap"]),
                pkcorr=Path(entry["maps"]["pkcorr"]),
            )
            for band, entry in raw["bands"].items()
        }
        return cls(
            k_model=PhotometryKModel.from_mapping(raw["k_model"]),
            companions=companions,
            config=FeatherPhotometryConfig.from_mapping(raw["config"]),
            method=str(raw["method"]),
        )


@dataclass(frozen=True, slots=True)
class FeatherPhotometryResult:
    """Outcome of :func:`calibrate_lst_photometry` for one band."""

    table: pd.DataFrame
    summary: dict[str, Any]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FeatherPhotometryBatchResult:
    """Outcome of :func:`calibrate_layout_photometry` (every band of the tree)."""

    results: dict[str, FeatherPhotometryResult]
    calibration: PhotometryCalibration
    summary: pd.DataFrame
    json_path: Path | None = None
    warnings: tuple[str, ...] = ()


def _numeric(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.full(len(df), np.nan, dtype=np.float64)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)


def _omega_fit_sr(df: pd.DataFrame) -> np.ndarray:
    """Fitted Gaussian solid angle from ``Maj`` / ``Min`` FWHM in degrees."""
    return _GAUSS_AREA * np.radians(_numeric(df, "Maj")) * np.radians(_numeric(df, "Min"))


def _bright_compact_mask(df: pd.DataFrame, config: FeatherPhotometryConfig) -> np.ndarray:
    missing = [c for c in ("S_Code", "E_Peak_flux") if c not in df.columns]
    if missing:
        msg = f"FEATHER3 photometry requires {', '.join(missing)} on the LST table"
        raise ValueError(msg)
    peak = _numeric(df, "Peak_flux")
    e_peak = _numeric(df, "E_Peak_flux")
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = peak / e_peak
    return (
        np.isfinite(peak)
        & (peak > 0.0)
        & (df["S_Code"].astype(str).to_numpy() == config.s_code)
        & (snr >= config.min_snr)
    )


def _band_freq_mhz(band: str, override: Mapping[str, float] | None) -> float:
    freq = float(override[band]) if override and band in override else band_frequency_hz(band) / 1e6
    if not np.isfinite(freq) or freq <= 0.0:
        msg = f"no frequency for band {band!r}; pass band_freq_mhz"
        raise ValueError(msg)
    return freq


def estimate_photometry_k_model(
    lst_by_band: Mapping[str, pd.DataFrame],
    omega_eff_by_band: Mapping[str, np.ndarray],
    *,
    band_freq_mhz: Mapping[str, float] | None = None,
    config: FeatherPhotometryConfig | None = None,
) -> PhotometryKModel:
    """Fit the joint ``k(δ, ν)`` model over every band in *lst_by_band*.

    *omega_eff_by_band* holds ``Ω_eff`` (sr) sampled at each row of the
    matching table. Frequencies come from the band label unless
    *band_freq_mhz* overrides them. Frequency terms are limited to the number
    of distinct frequencies minus one (no ``u`` terms for a single band).

    Raises
    ------
    ValueError
        Missing ``S_Code`` / ``E_Peak_flux``, a band with fewer than
        ``config.min_rows_per_band`` bright compact rows, or too few populated
        Dec bins for ``config.dec_degree``.
    """
    config = config or FeatherPhotometryConfig()
    if not lst_by_band:
        msg = "no bands to fit"
        raise ValueError(msg)
    if 180.0 % config.dec_bin_deg:
        msg = f"dec_bin_deg must divide 180 (got {config.dec_bin_deg})"
        raise ValueError(msg)
    edges = np.arange(-90.0, 90.0 + 0.5 * config.dec_bin_deg, config.dec_bin_deg)
    centres = 0.5 * (edges[:-1] + edges[1:])

    bands = tuple(lst_by_band)
    freqs = {b: _band_freq_mhz(b, band_freq_mhz) for b in bands}
    diagnostics: dict[str, BandKDiagnostics] = {}
    band_idx, x_rows, u_rows, y_rows, w_rows = [], [], [], [], []
    too_few: list[str] = []
    for j, band in enumerate(bands):
        lst = lst_by_band[band]
        mask = _bright_compact_mask(lst, config)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = _omega_fit_sr(lst) / np.asarray(omega_eff_by_band[band], dtype=np.float64)
        mask &= np.isfinite(ratio) & (ratio > 0.0)
        ratio, dec = ratio[mask], _numeric(lst, "DEC")[mask]
        if ratio.size < config.min_rows_per_band:
            too_few.append(f"{band} ({ratio.size})")
            continue
        used_c, used_k, used_n = [], [], []
        for lo, hi, c in zip(edges[:-1], edges[1:], centres, strict=True):
            in_bin = (dec > lo) & (dec <= hi)
            n = int(in_bin.sum())
            if n < config.min_rows_per_bin:
                continue
            med = float(np.median(ratio[in_bin]))
            used_c.append(float(c))
            used_k.append(med)
            used_n.append(n)
            band_idx.append(j)
            x_rows.append((c - OVRO_LATITUDE_DEG) / config.dec_scale_deg)
            u_rows.append(np.log(freqs[band] / config.ref_freq_mhz))
            y_rows.append(np.log(med))
            w_rows.append(n)
        diagnostics[band] = BandKDiagnostics(
            n_used=int(ratio.size),
            k_all=float(np.median(ratio)),
            dec_centres=tuple(used_c),
            k_dec_median=tuple(used_k),
            n_dec=tuple(used_n),
        )
    if too_few:
        msg = (
            f"bands with fewer than {config.min_rows_per_band} bright compact rows "
            f"(S_Code={config.s_code!r}, SNR>={config.min_snr:g}): {', '.join(too_few)}"
        )
        raise ValueError(msg)

    band_idx_a = np.asarray(band_idx)
    x = np.asarray(x_rows)
    u = np.asarray(u_rows)
    freq_degree = min(config.freq_degree, len(set(freqs.values())) - 1)
    if len(np.unique(x)) <= config.dec_degree:
        msg = (
            f"{len(np.unique(x))} populated Dec bins cannot constrain a degree-"
            f"{config.dec_degree} Dec shape"
        )
        raise ValueError(msg)
    columns = [(band_idx_a == j).astype(float) for j in range(len(bands))]
    for p in range(1, config.dec_degree + 1):
        columns.extend(u**q * x**p for q in range(freq_degree + 1))
    design = np.column_stack(columns)
    sqrt_w = np.sqrt(np.asarray(w_rows, dtype=np.float64))
    coef, _, rank, _ = np.linalg.lstsq(
        design * sqrt_w[:, None], np.asarray(y_rows) * sqrt_w, rcond=None
    )
    if rank < design.shape[1]:
        msg = "joint k(δ, ν) fit is rank deficient; check Dec coverage per band"
        raise ValueError(msg)

    n_q = freq_degree + 1
    shape_coef = coef[len(bands) :]
    shape = tuple(
        tuple(float(c) for c in shape_coef[(p - 1) * n_q : p * n_q])
        for p in range(1, config.dec_degree + 1)
    )
    fitted_centres = [c for d in diagnostics.values() for c in d.dec_centres]
    dec_lo, dec_hi = float(min(fitted_centres)), float(max(fitted_centres))
    return PhotometryKModel(
        band_offset={b: float(coef[j]) for j, b in enumerate(bands)},
        band_freq_mhz=freqs,
        shape=shape,
        dec_range=(dec_lo, dec_hi),
        phi_deg=OVRO_LATITUDE_DEG,
        dec_scale_deg=float(config.dec_scale_deg),
        ref_freq_mhz=float(config.ref_freq_mhz),
        diagnostics=diagnostics,
    )


def _calibrated_fluxes(
    table: pd.DataFrame,
    pb: np.ndarray,
    pk: np.ndarray,
    omega_eff: np.ndarray,
    k_phot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(peak_factor, Peak_flux_cal, Total_flux_cal)``; NaN outside valid maps."""
    valid = (
        np.isfinite(pb)
        & (pb > 0.0)
        & np.isfinite(pk)
        & (pk > 0.0)
        & np.isfinite(omega_eff)
        & (omega_eff > 0.0)
    )
    peak = _numeric(table, "Peak_flux")
    with np.errstate(divide="ignore", invalid="ignore"):
        peak_factor = np.where(valid, 1.0 / (pb * pk), np.nan)
        peak_cal = peak * peak_factor
        total_cal = peak * _omega_fit_sr(table) / (k_phot * omega_eff) * peak_factor
    return peak_factor, peak_cal, total_cal


def _sample_companions(
    table: pd.DataFrame, companions: FeatherCompanions
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra = _numeric(table, "RA")
    dec = _numeric(table, "DEC")
    return (
        sample_healpix_at(companions.pbmap, ra, dec),
        sample_healpix_at(companions.pkcorr, ra, dec),
        sample_healpix_at(companions.psfmap, ra, dec),
    )


def _calibrate_sampled(
    lst: pd.DataFrame,
    band: str,
    k_model: PhotometryKModel,
    pb: np.ndarray,
    pk: np.ndarray,
    omega_eff: np.ndarray,
    config: FeatherPhotometryConfig,
) -> FeatherPhotometryResult:
    warnings: list[str] = []
    drop = [c for c in (*BAND_CAL_FLUX_FIELDS, *CAL_DIAGNOSTIC_COLUMNS) if c in lst.columns]
    table = lst.drop(columns=drop).copy()
    k_phot = k_model.k_at(band, _numeric(table, "DEC"))
    peak_factor, peak_cal, total_cal = _calibrated_fluxes(table, pb, pk, omega_eff, k_phot)
    n_invalid = int(np.isnan(peak_factor).sum())
    if n_invalid:
        warnings.append(f"{n_invalid} rows outside valid PB_eff/PKCORR/Omega_eff: *_cal NaN")

    total = _numeric(table, "Total_flux")
    with np.errstate(divide="ignore", invalid="ignore"):
        total_factor = np.where(np.isfinite(total) & (total > 0.0), total_cal / total, np.nan)
    table["Peak_flux_cal"] = peak_cal
    table["E_Peak_flux_cal"] = _numeric(table, "E_Peak_flux") * peak_factor
    table["Total_flux_cal"] = total_cal
    table["E_Total_flux_cal"] = _numeric(table, "E_Total_flux") * total_factor
    table["PB_eff"] = pb
    table["PKCORR"] = pk
    table["Omega_eff"] = omega_eff
    table["k_phot"] = k_phot

    bright = _bright_compact_mask(table, config)
    with np.errstate(divide="ignore", invalid="ignore"):
        tp_bright = (total_cal / peak_cal)[bright]
    tp_bright = tp_bright[np.isfinite(tp_bright)]
    diag = k_model.diagnostics.get(band)
    summary: dict[str, Any] = {
        "n_rows": int(len(table)),
        "n_peak_cal": int(np.isfinite(peak_cal).sum()),
        "n_total_cal": int(np.isfinite(total_cal).sum()),
        "n_invalid_maps": n_invalid,
        "n_k": int(diag.n_used) if diag else 0,
        "k_all": float(diag.k_all) if diag else float("nan"),
        "k_south": float(k_model.k_at(band, -30.0)),
        "k_zenith": float(k_model.k_at(band, OVRO_LATITUDE_DEG)),
        "k_fit_max_dev": _k_fit_max_dev(k_model, band),
        "median_pb_eff": float(np.nanmedian(pb)) if np.isfinite(pb).any() else float("nan"),
        "median_pkcorr": float(np.nanmedian(pk)) if np.isfinite(pk).any() else float("nan"),
        "median_total_over_peak_cal_bright": (
            float(np.median(tp_bright)) if tp_bright.size else float("nan")
        ),
    }
    return FeatherPhotometryResult(table=table, summary=summary, warnings=tuple(warnings))


def _k_fit_max_dev(k_model: PhotometryKModel, band: str, min_n: int = 20) -> float:
    """Largest ``|model / bin median − 1|`` over bins with ``n >= min_n``."""
    diag = k_model.diagnostics.get(band)
    if diag is None:
        return float("nan")
    keep = np.asarray(diag.n_dec) >= min_n
    if not keep.any():
        return float("nan")
    model = k_model.k_at(band, np.asarray(diag.dec_centres)[keep])
    return float(np.max(np.abs(model / np.asarray(diag.k_dec_median)[keep] - 1.0)))


def calibrate_lst_photometry(
    lst: pd.DataFrame,
    band: str,
    companions: FeatherCompanions,
    k_model: PhotometryKModel,
    *,
    config: FeatherPhotometryConfig | None = None,
) -> FeatherPhotometryResult:
    """Add ``*_cal`` flux and diagnostic columns to one LST-merged band table.

    Reads only raw ``Peak_flux``, ``E_Peak_flux``, ``Total_flux``,
    ``E_Total_flux``, ``Maj``, ``Min``, ``RA``, ``DEC`` (and ``S_Code``);
    previous ``*_cal`` / diagnostic columns are replaced, so repeated calls
    give identical tables. *k_model* must contain *band* (see
    :func:`estimate_photometry_k_model`). Rows with non-finite or
    non-positive ``PB_eff`` / ``PKCORR`` / ``Omega_eff`` get NaN ``*_cal``;
    rows without ``Maj`` / ``Min`` get NaN ``Total_flux_cal``.
    """
    config = config or FeatherPhotometryConfig()
    pb, pk, omega_eff = _sample_companions(lst, companions)
    return _calibrate_sampled(lst, band, k_model, pb, pk, omega_eff, config)


def verify_lst_photometry(table: pd.DataFrame, *, rtol: float = 1e-5) -> list[str]:
    """Problems that make a calibrated LST table unusable (empty list = OK).

    Checks that every ``*_cal`` / diagnostic column is present, that every row
    with a finite position carries ``k_phot`` (catches rows appended after
    calibration, e.g. forced fill), and that ``Peak_flux_cal`` /
    ``Total_flux_cal`` still match the raw ``Peak_flux`` / ``Maj`` / ``Min``
    (catches rows replaced after calibration). Tables without any ``*_cal``
    column are uncalibrated and return no problems.
    """
    present = [c for c in (*BAND_CAL_FLUX_FIELDS, *CAL_DIAGNOSTIC_COLUMNS) if c in table.columns]
    if not present:
        return []
    missing = [c for c in (*BAND_CAL_FLUX_FIELDS, *CAL_DIAGNOSTIC_COLUMNS) if c not in present]
    if missing:
        return [f"missing calibration columns: {', '.join(missing)}"]
    problems: list[str] = []
    k_phot = _numeric(table, "k_phot")
    positioned = np.isfinite(_numeric(table, "RA")) & np.isfinite(_numeric(table, "DEC"))
    n_uncal = int((positioned & ~np.isfinite(k_phot)).sum())
    if n_uncal:
        problems.append(f"{n_uncal} rows without k_phot (added after calibration)")
    _, peak_cal, total_cal = _calibrated_fluxes(
        table,
        _numeric(table, "PB_eff"),
        _numeric(table, "PKCORR"),
        _numeric(table, "Omega_eff"),
        k_phot,
    )
    for column, expected in (("Peak_flux_cal", peak_cal), ("Total_flux_cal", total_cal)):
        stale = ~np.isclose(_numeric(table, column), expected, rtol=rtol, equal_nan=True)
        n_stale = int((stale & positioned & np.isfinite(k_phot)).sum())
        if n_stale:
            problems.append(f"{n_stale} rows with {column} inconsistent with raw columns")
    return problems


def lst_bands_in_layout(layout: CatalogLayout) -> tuple[str, ...]:
    """Bands with a ``metacatalog_lst_{band}.parquet`` in *layout* (high → low MHz first)."""
    bands = [p.stem[len(_LST_PREFIX) :] for p in layout.root.glob(f"{_LST_PREFIX}*.parquet")]
    return tuple(sorted(bands, key=lambda b: (-np.nan_to_num(band_frequency_hz(b), nan=-1.0), b)))


def _beam_scale_json_path(path: Path | str | CatalogLayout) -> Path:
    if isinstance(path, CatalogLayout):
        return path.beam_scale_json()
    out = Path(path)
    if out.suffix.lower() != ".json":
        out = out / "beam_scale.json"
    return out


def write_photometry_calibration(
    path: Path | str | CatalogLayout,
    calibration: PhotometryCalibration,
) -> Path:
    """Replace ``PHOTOMETRY`` in ``beam_scale.json``, keeping every other key.

    Creates the file with only ``PHOTOMETRY`` when it does not exist yet.
    """
    out = _beam_scale_json_path(path)
    payload = read_beam_scale_json(out) if out.is_file() else {}
    payload[PHOTOMETRY_KEY] = calibration.as_mapping()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


def load_photometry_calibration(
    path: Path | str | CatalogLayout,
) -> PhotometryCalibration | None:
    """Read ``PHOTOMETRY`` from ``beam_scale.json`` (``None`` when absent)."""
    out = _beam_scale_json_path(path)
    if not out.is_file():
        return None
    raw = read_beam_scale_json(out).get(PHOTOMETRY_KEY)
    return None if raw is None else PhotometryCalibration.from_mapping(raw)


def calibrate_layout_photometry(
    layout: CatalogLayout,
    companions_by_band: Mapping[str, FeatherCompanions],
    *,
    bands: Iterable[str] | None = None,
    k_model: PhotometryKModel | None = None,
    band_freq_mhz: Mapping[str, float] | None = None,
    config: FeatherPhotometryConfig | None = None,
    write: bool = True,
) -> FeatherPhotometryBatchResult:
    """Calibrate every ``metacatalog_lst_{band}`` of *layout* with one joint ``k``.

    *bands* defaults to every LST band in the tree and must equal that set
    (partial calibration raises). Every band needs companions. The joint
    ``k(δ, ν)`` model is fitted from all bands unless *k_model* is given
    (e.g. ``load_photometry_calibration(layout).k_model`` after forced fill),
    in which case it must cover every band. With *write*, each LST table is
    rewritten and ``PHOTOMETRY`` is stored in ``beam_scale.json``. Use
    ``results[band].table`` to refresh an in-memory LST dict before fusion.
    """
    config = config or FeatherPhotometryConfig()
    tree_bands = lst_bands_in_layout(layout)
    if not tree_bands:
        msg = f"no {_LST_PREFIX}*.parquet under {layout.root}"
        raise FileNotFoundError(msg)
    bands = tree_bands if bands is None else tuple(bands)
    if set(bands) != set(tree_bands):
        msg = (
            "FEATHER3 photometry must calibrate every LST band of the tree: "
            f"tree has {', '.join(tree_bands)}; requested {', '.join(bands)}"
        )
        raise ValueError(msg)
    no_maps = [b for b in bands if companions_by_band.get(b) is None]
    if no_maps:
        msg = f"FEATHER3 companions missing for: {', '.join(no_maps)}"
        raise ValueError(msg)
    if k_model is not None:
        not_modelled = [b for b in bands if b not in k_model.band_offset]
        if not_modelled:
            msg = f"k model has no entry for: {', '.join(not_modelled)}"
            raise ValueError(msg)

    lst_by_band = {b: read_lst_merged(layout, b) for b in bands}
    sampled = {b: _sample_companions(lst_by_band[b], companions_by_band[b]) for b in bands}
    if k_model is None:
        k_model = estimate_photometry_k_model(
            lst_by_band,
            {b: sampled[b][2] for b in bands},
            band_freq_mhz=band_freq_mhz,
            config=config,
        )
    calibration = PhotometryCalibration(
        k_model=k_model,
        companions={b: companions_by_band[b] for b in bands},
        config=config,
    )

    warnings: list[str] = []
    results: dict[str, FeatherPhotometryResult] = {}
    for band in bands:
        pb, pk, omega_eff = sampled[band]
        result = _calibrate_sampled(lst_by_band[band], band, k_model, pb, pk, omega_eff, config)
        results[band] = result
        warnings.extend(f"{band}: {w}" for w in result.warnings)
        if write:
            write_lst_merged(result.table, layout, band)
    json_path = write_photometry_calibration(layout, calibration) if write else None

    summary = pd.DataFrame([{"band": band, **result.summary} for band, result in results.items()])
    return FeatherPhotometryBatchResult(
        results=results,
        calibration=calibration,
        summary=summary,
        json_path=json_path,
        warnings=tuple(warnings),
    )


def summarize_feather_photometry(
    result: FeatherPhotometryResult | FeatherPhotometryBatchResult,
) -> str:
    """Human-readable summary of a single-band or layout calibration."""
    if isinstance(result, FeatherPhotometryResult):
        items = [("band", result.summary)]
        warnings = result.warnings
        header = f"FEATHER3 photometry ({FEATHER_PHOTOMETRY_METHOD}): 1 band"
    else:
        items = [(band, r.summary) for band, r in result.results.items()]
        warnings = result.warnings
        k_model = result.calibration.k_model
        lo, hi = k_model.dec_range
        header = (
            f"FEATHER3 photometry ({FEATHER_PHOTOMETRY_METHOD}): {len(items)} band(s), "
            f"joint k over Dec {lo:.0f}..{hi:.0f} "
            f"(dec degree {len(k_model.shape)}, freq degree {len(k_model.shape[0]) - 1})"
        )
    lines = [header]
    for band, s in items:
        lines.append(
            f"  {band}: rows={s['n_rows']} peak_cal={s['n_peak_cal']} "
            f"total_cal={s['n_total_cal']}  n_k={s['n_k']} k_all={s['k_all']:.4g} "
            f"k(-30)={s['k_south']:.4g} k(zen)={s['k_zenith']:.4g} "
            f"max fit dev={100 * s['k_fit_max_dev']:.1f}%  "
            f"med PB_eff={s['median_pb_eff']:.3g} PKCORR={s['median_pkcorr']:.3g}  "
            f"med T/P cal (bright S)={s['median_total_over_peak_cal_bright']:.3g}"
        )
    if warnings:
        lines.append(f"warnings: {len(warnings)}")
        lines.extend(f"  - {w}" for w in warnings)
    return "\n".join(lines)
