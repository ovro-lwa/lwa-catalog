"""Metacatalog reliability tiers and per-source 32-bit quality flags."""

from __future__ import annotations

import warnings as py_warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import IntFlag, unique
from pathlib import Path
from typing import Literal

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from lwa_catalog.analyze.trace import (
    _ensure_bmaj_column,
    _load_lst_band,
    _match_bmaj,
    _one_row_base,
    _parse_bands_present,
    _parse_lst_hours,
    _pick_rematch_lst_row,
)
from lwa_catalog.constants import (
    CLUSTER_JITTER_RMS_COL,
    DEFAULT_QUALITY_FLAG_MASK,
    SUBBAND_QUALITY_NAN_COLUMNS,
)
from lwa_catalog.create.merge import associate_catalogs, catalog_elevation_deg
from lwa_catalog.gaul import cast_s_code_value
from lwa_catalog.io import read_sources_catalog
from lwa_catalog.paths import CatalogLayout

_EXCLUDE_TIERS = ("E0", "E1", "E2", "E3", "E4", "E5")
_INCLUDE_TIERS = ("I0", "I1", "I2", "I3", "I4", "I5")

# Primary measurement columns that should be populated on every merged source.
# Per-band association fields (e.g. Peak_flux_Blue) are omitted: those NaNs are
# expected when a band is absent.
QUALITY_NAN_COLUMNS: tuple[str, ...] = (
    "RA",
    "DEC",
    "Peak_flux",
    "Total_flux",
    "Maj",
    "Min",
    "PA",
    "DC_Maj",
    "DC_Min",
    "DC_PA",
    "n_lst_contributions",
    "origin_band",
    "bands_present",
    "Resid_Isl_rms",
    "Resid_Isl_mean",
    "E_Peak_flux",
    "E_Total_flux",
)

_FLUX_QA_FIELDS: tuple[str, ...] = (
    "Peak_flux",
    "Total_flux",
    "E_Peak_flux",
    "E_Total_flux",
)


def is_subband_metacatalog(df: pd.DataFrame) -> bool:
    """True when *df* uses per-subband flux columns without top-level ``Peak_flux``."""
    if "Peak_flux" in df.columns:
        return False
    return any(str(col).startswith("Peak_flux_") for col in df.columns)


def qa_band_for_row(row: pd.Series | Mapping) -> str:
    """Band used for LST rematch and flux QA (``astrometry_band`` or ``origin_band``)."""
    series = row if isinstance(row, pd.Series) else pd.Series(row)
    astro = series.get("astrometry_band")
    if astro is not None and str(astro).strip().lower() not in {"", "nan", "none"}:
        return str(astro).strip()
    return str(series.get("origin_band", "")).strip()


def quality_nan_columns(df: pd.DataFrame) -> tuple[str, ...]:
    """Core columns checked by :func:`flag_has_nan` for this metacatalog schema."""
    if is_subband_metacatalog(df):
        return SUBBAND_QUALITY_NAN_COLUMNS
    return QUALITY_NAN_COLUMNS


def flux_qa_frame(
    meta_row: pd.Series,
    seed_row: pd.Series | None,
    band: str,
) -> pd.DataFrame:
    """One-row frame with Peak/Total flux for unphysical-flux checks."""
    if "Peak_flux" in meta_row.index and "Total_flux" in meta_row.index:
        try:
            peak = float(meta_row["Peak_flux"])
        except (TypeError, ValueError):
            peak = float("nan")
        if np.isfinite(peak):
            return pd.DataFrame([meta_row])

    entry: dict[str, object] = {}
    for field in _FLUX_QA_FIELDS:
        col = f"{field}_{band}"
        if col in meta_row.index:
            entry[field] = meta_row[col]
    if "Peak_flux" in entry and pd.notna(entry["Peak_flux"]):
        return pd.DataFrame([entry])
    if seed_row is not None:
        return pd.DataFrame([seed_row])
    return pd.DataFrame()


def representative_peak_flux(meta: pd.DataFrame) -> pd.Series:
    """Peak flux for display / HiPS weighting (top-level or per-subband)."""
    if meta.empty:
        return pd.Series(dtype=float)
    if "Peak_flux" in meta.columns:
        return pd.to_numeric(meta["Peak_flux"], errors="coerce")
    from lwa_catalog.analyze.nedlvs import resolve_highest_frequency_peak_flux

    values = [
        resolve_highest_frequency_peak_flux(row if isinstance(row, pd.Series) else pd.Series(row))[0]
        for _, row in meta.iterrows()
    ]
    return pd.Series(values, index=meta.index, dtype=float, name="Peak_flux")


@unique
class SourceQualityFlag(IntFlag):
    """32-bit per-source quality mask.

    A bit is **0 when that check is good/reliable** and **1 when the property
    is a quality concern**. ``quality_flag == 0`` means every implemented check
    passed. Bits 14–31 are reserved (stay 0).

    ====== ===================== =================================================
    Bit    Name                  Set (1) when
    ====== ===================== =================================================
    0      HAS_NAN               a core measurement column is NaN
    1      INVALID_ASTROMETRY    RA, DEC, or Peak_flux is non-finite (was E0)
    2      SINGLE_LST            ``n_lst_contributions == 1``
    3      SINGLE_UNIQUE_BAND    uniquely associated in exactly one band
    4      UNPHYSICAL_FLUX       ``(Total-Peak)/hypot(E) < -3`` (was E2)
    5      RESID_ABS_FAIL        island RMS or |mean| above absolute Jy/beam (E3)
    6      RESID_PCTL_RMS        ``Resid_Isl_rms`` outside the catalog 1–99%
    7      RESID_PCTL_MEAN       ``Resid_Isl_mean`` outside the catalog 1–99%
    8      JITTER_FAIL           cluster RMS ``> 0.3 × BMAJ`` (was E4)
    9      CONFUSED_ASSOC        any ``n_assoc_* > 1`` (was E5)
    10     NO_VLSSR              no positional match to the VLSSR catalog
    11     SCODE_COMPLEX         PyBDSF ``S_Code`` is ``C`` or ``M``
    12     LOW_ELEVATION         elevation at ``representative_lst`` < 10°
    13     HIGH_ELLIPTICITY      ``Maj / Min > 3`` (source FWHM axes)
    ====== ===================== =================================================
    """

    HAS_NAN = 1 << 0
    INVALID_ASTROMETRY = 1 << 1
    SINGLE_LST = 1 << 2
    SINGLE_UNIQUE_BAND = 1 << 3
    UNPHYSICAL_FLUX = 1 << 4
    RESID_ABS_FAIL = 1 << 5
    RESID_PCTL_RMS = 1 << 6
    RESID_PCTL_MEAN = 1 << 7
    JITTER_FAIL = 1 << 8
    CONFUSED_ASSOC = 1 << 9
    NO_VLSSR = 1 << 10
    SCODE_COMPLEX = 1 << 11
    LOW_ELEVATION = 1 << 12
    HIGH_ELLIPTICITY = 1 << 13


_QUALITY_FLAG_COLUMNS: tuple[tuple[str, SourceQualityFlag], ...] = (
    ("has_nan", SourceQualityFlag.HAS_NAN),
    ("invalid", SourceQualityFlag.INVALID_ASTROMETRY),
    ("single_lst", SourceQualityFlag.SINGLE_LST),
    ("single_unique_band", SourceQualityFlag.SINGLE_UNIQUE_BAND),
    ("unphysical_soft", SourceQualityFlag.UNPHYSICAL_FLUX),
    ("resid_fail_soft", SourceQualityFlag.RESID_ABS_FAIL),
    ("resid_pctl_rms", SourceQualityFlag.RESID_PCTL_RMS),
    ("resid_pctl_mean", SourceQualityFlag.RESID_PCTL_MEAN),
    ("jitter_fail_soft", SourceQualityFlag.JITTER_FAIL),
    ("confused_assoc", SourceQualityFlag.CONFUSED_ASSOC),
    ("no_vlssr", SourceQualityFlag.NO_VLSSR),
    ("scode_complex", SourceQualityFlag.SCODE_COMPLEX),
    ("low_elevation", SourceQualityFlag.LOW_ELEVATION),
    ("high_ellipticity", SourceQualityFlag.HIGH_ELLIPTICITY),
)

_QUALITY_FLAG_HELP: dict[SourceQualityFlag, str] = {
    SourceQualityFlag.HAS_NAN: "core measurement column is NaN",
    SourceQualityFlag.INVALID_ASTROMETRY: "RA, DEC, or Peak_flux is non-finite",
    SourceQualityFlag.SINGLE_LST: "n_lst_contributions == 1",
    SourceQualityFlag.SINGLE_UNIQUE_BAND: "uniquely associated in exactly one band",
    SourceQualityFlag.UNPHYSICAL_FLUX: "(Total-Peak)/hypot(E) < -3",
    SourceQualityFlag.RESID_ABS_FAIL: (
        "Resid_Isl_rms or |Resid_Isl_mean| above absolute Jy/beam cut"
    ),
    SourceQualityFlag.RESID_PCTL_RMS: "Resid_Isl_rms outside catalog 1–99 percentile",
    SourceQualityFlag.RESID_PCTL_MEAN: "Resid_Isl_mean outside catalog 1–99 percentile",
    SourceQualityFlag.JITTER_FAIL: "cluster RA/Dec RMS > 0.3 × BMAJ",
    SourceQualityFlag.CONFUSED_ASSOC: "any n_assoc_* > 1",
    SourceQualityFlag.NO_VLSSR: "not associated with the VLSSR catalog",
    SourceQualityFlag.SCODE_COMPLEX: "S_Code is C or M",
    SourceQualityFlag.LOW_ELEVATION: "elevation at representative_lst below min_elevation_deg",
    SourceQualityFlag.HIGH_ELLIPTICITY: "Maj / Min exceeds max_source_ellipticity",
}


@dataclass(frozen=True)
class ReliabilityConfig:
    """Thresholds and tier depth for reliability filters and quality flags."""

    resid_rms_thresh_jy: float = 1.0
    resid_mean_thresh_jy: float = 1.0
    resid_percentile_lo: float = 1.0
    resid_percentile_hi: float = 99.0
    flux_unphysical_nsigma: float = 3.0
    jitter_bmaj_frac: float = 0.3
    min_elevation_deg: float = 10.0
    max_source_ellipticity: float = 3.0
    min_lst_contributions: int = 2
    require_unique_assoc_include: bool = True
    require_unique_assoc_exclude: bool = False
    strict: bool = False
    include_through: str = "I5"
    exclude_through: str = "E4"


@dataclass
class ReliabilityResult:
    """Filtered metacatalog subset plus tier yields and per-row flags."""

    catalog: pd.DataFrame
    meta_ids: np.ndarray
    tier_counts: pd.DataFrame
    flags: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


@dataclass
class QualityFlagResult:
    """Full metacatalog with a uint32 ``quality_flag`` column (no row drops)."""

    catalog: pd.DataFrame
    flags: pd.DataFrame
    bit_counts: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def flux_sigma_total_minus_peak(df: pd.DataFrame) -> pd.Series:
    """Return ``(Total - Peak) / hypot(E_Total, E_Peak)``; NaN when not calculable."""
    need = ("Total_flux", "Peak_flux", "E_Total_flux", "E_Peak_flux")
    if any(c not in df.columns for c in need):
        return pd.Series(np.nan, index=df.index, dtype=float, name="flux_sigma")
    total = pd.to_numeric(df["Total_flux"], errors="coerce")
    peak = pd.to_numeric(df["Peak_flux"], errors="coerce")
    e_tot = pd.to_numeric(df["E_Total_flux"], errors="coerce")
    e_peak = pd.to_numeric(df["E_Peak_flux"], errors="coerce")
    denom = np.hypot(e_tot.to_numpy(dtype=float), e_peak.to_numpy(dtype=float))
    numer = (total - peak).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.where(
            np.isfinite(numer) & np.isfinite(denom) & (denom > 0.0),
            numer / denom,
            np.nan,
        )
    return pd.Series(sigma, index=df.index, dtype=float, name="flux_sigma")


def flag_unphysical_flux(
    df: pd.DataFrame,
    *,
    nsigma: float = 3.0,
) -> pd.Series:
    """True when finite ``σ < -nsigma`` (positive unphysical failure)."""
    sigma = flux_sigma_total_minus_peak(df)
    return (sigma < -float(nsigma)).fillna(False).rename("unphysical_flux")


def flag_residual_absolute(
    df: pd.DataFrame,
    *,
    rms_thresh_jy: float = 1.0,
    mean_thresh_jy: float = 1.0,
) -> pd.DataFrame:
    """Boolean residual flags using absolute Jy/beam thresholds.

    Columns: ``resid_rms_fail``, ``resid_mean_fail``, ``resid_fail`` (either).
    Failures require finite values above threshold (soft-exclude semantics).
    """
    n = len(df)
    if "Resid_Isl_rms" in df.columns:
        rms = pd.to_numeric(df["Resid_Isl_rms"], errors="coerce")
        rms_fail = (rms > float(rms_thresh_jy)).fillna(False)
    else:
        rms_fail = pd.Series(False, index=df.index)
    if "Resid_Isl_mean" in df.columns:
        mean = pd.to_numeric(df["Resid_Isl_mean"], errors="coerce").abs()
        mean_fail = (mean > float(mean_thresh_jy)).fillna(False)
    else:
        mean_fail = pd.Series(False, index=df.index)
    out = pd.DataFrame(
        {
            "resid_rms_fail": rms_fail,
            "resid_mean_fail": mean_fail,
            "resid_fail": rms_fail | mean_fail,
        },
        index=df.index,
    )
    if n == 0:
        return out
    return out


def parse_bands_present(row: pd.Series | Mapping) -> list[str]:
    """Parse comma-separated ``bands_present`` into a band list."""
    if isinstance(row, pd.Series):
        return _parse_bands_present(row)
    return _parse_bands_present(pd.Series(row))


def unique_assoc_band_count(row: pd.Series, *, min_lst: int = 2) -> tuple[int, int]:
    """Return ``(n_lst, n_unique_assoc_bands)``.

    ``Full`` in ``bands_present`` counts as uniquely associated (no ``n_assoc_Full``).
    The association seed band (``origin_band`` when ``n_assoc_{band}`` is absent)
    counts the same way for MHz subband catalogs.
    Color bands count only when ``n_assoc_{band} == 1``.
    """
    del min_lst  # reserved for API symmetry with passes_multi_image
    raw = row.get("n_lst_contributions", np.nan)
    try:
        n_lst = int(raw) if pd.notna(raw) else 0
    except (TypeError, ValueError):
        n_lst = 0

    origin = str(row.get("origin_band", "")).strip()
    bands = parse_bands_present(row)
    n_unique = 0
    for band in bands:
        if band == "Full":
            n_unique += 1
            continue
        col = f"n_assoc_{band}"
        if col not in row.index:
            if band == origin:
                n_unique += 1
            continue
        try:
            n_assoc = int(row[col])
        except (TypeError, ValueError):
            continue
        if n_assoc == 1:
            n_unique += 1
    return n_lst, n_unique


def passes_multi_image(
    row: pd.Series,
    *,
    min_lst: int = 2,
) -> bool:
    """True if ``n_lst >= min_lst`` or ≥2 uniquely associated bands."""
    n_lst, n_unique = unique_assoc_band_count(row, min_lst=min_lst)
    return n_lst >= int(min_lst) or n_unique >= 2


def flag_invalid_astrometry_flux(df: pd.DataFrame) -> pd.Series:
    """True when RA/DEC or representative peak flux is non-finite."""
    ok = pd.Series(True, index=df.index)
    for col in ("RA", "DEC"):
        if col not in df.columns:
            ok &= False
            continue
        ok &= pd.to_numeric(df[col], errors="coerce").apply(np.isfinite)

    if is_subband_metacatalog(df):
        flux_ok = []
        for _, row in df.iterrows():
            band = qa_band_for_row(row)
            col = f"Peak_flux_{band}"
            if col not in row.index:
                flux_ok.append(False)
                continue
            flux_ok.append(
                bool(np.isfinite(float(pd.to_numeric(row[col], errors="coerce"))))
            )
        ok &= pd.Series(flux_ok, index=df.index, dtype=bool)
    else:
        if "Peak_flux" not in df.columns:
            ok &= False
        else:
            ok &= pd.to_numeric(df["Peak_flux"], errors="coerce").apply(np.isfinite)
    return (~ok).rename("invalid_astrometry_flux")


def flag_confused_assoc(df: pd.DataFrame) -> pd.Series:
    """True when any band in ``bands_present`` has ``n_assoc_* > 1``."""
    flags = []
    for _idx, row in df.iterrows():
        confused = False
        for band in parse_bands_present(row):
            if band == "Full":
                continue
            col = f"n_assoc_{band}"
            if col not in row.index or pd.isna(row[col]):
                continue
            try:
                if int(row[col]) > 1:
                    confused = True
                    break
            except (TypeError, ValueError):
                continue
        flags.append(confused)
    return pd.Series(flags, index=df.index, dtype=bool, name="confused_assoc")


def all_bands_unique_assoc(row: pd.Series) -> bool:
    """True when every band in ``bands_present`` is uniquely associated (or Full/seed)."""
    bands = parse_bands_present(row)
    if not bands:
        return False
    origin = str(row.get("origin_band", "")).strip()
    for band in bands:
        if band == "Full":
            continue
        col = f"n_assoc_{band}"
        if col not in row.index:
            if band == origin:
                continue
            return False
        try:
            if int(row[col]) != 1:
                return False
        except (TypeError, ValueError):
            return False
    return True


def flag_has_nan(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> pd.Series:
    """True when any listed core column that exists on *df* is NA."""
    cols = quality_nan_columns(df) if columns is None else columns
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(False, index=df.index, dtype=bool, name="has_nan")
    return df[present].isna().any(axis=1).rename("has_nan")


def flag_single_lst(df: pd.DataFrame) -> pd.Series:
    """True when ``n_lst_contributions == 1``."""
    if "n_lst_contributions" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool, name="single_lst")
    n_lst = pd.to_numeric(df["n_lst_contributions"], errors="coerce")
    return (n_lst == 1).fillna(False).rename("single_lst")


def flag_single_unique_band(df: pd.DataFrame) -> pd.Series:
    """True when exactly one band is uniquely associated (Full counts)."""
    flags = [unique_assoc_band_count(row)[1] == 1 for _, row in df.iterrows()]
    return pd.Series(flags, index=df.index, dtype=bool, name="single_unique_band")


def flag_scode_complex(codes: pd.Series | np.ndarray) -> pd.Series:
    """True when PyBDSF ``S_Code`` is ``C`` (in-island) or ``M`` (multi-Gaussian)."""
    series = pd.Series(codes, dtype=object)
    text = series.astype("string").str.strip().str.upper()
    return text.isin(["C", "M"]).fillna(False).rename("scode_complex")


def flag_low_elevation(
    df: pd.DataFrame,
    *,
    min_deg: float = 10.0,
) -> pd.Series:
    """True when source elevation at ``representative_lst`` is below *min_deg*."""
    name = "low_elevation"
    if df.empty:
        return pd.Series(dtype=bool, name=name)
    need = {"RA", "DEC", "representative_lst"}
    if not need <= set(df.columns):
        return pd.Series(False, index=df.index, dtype=bool, name=name)
    try:
        elev = catalog_elevation_deg(df)
    except KeyError:
        return pd.Series(False, index=df.index, dtype=bool, name=name)
    low = np.asarray(elev, dtype=float) < float(min_deg)
    return pd.Series(low & np.isfinite(elev), index=df.index, dtype=bool, name=name)


def flag_high_ellipticity(
    df: pd.DataFrame,
    *,
    max_ratio: float = 3.0,
) -> pd.Series:
    """True when ``Maj / Min > max_ratio`` with finite positive minor axis."""
    name = "high_ellipticity"
    if "Maj" not in df.columns or "Min" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool, name=name)
    maj = pd.to_numeric(df["Maj"], errors="coerce")
    min_ = pd.to_numeric(df["Min"], errors="coerce")
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = maj / min_
    ok = maj.notna() & min_.notna() & (min_ > 0.0)
    high = ok & (ratio > float(max_ratio))
    return high.fillna(False).rename(name)


def flag_residual_percentile(
    rms: np.ndarray | pd.Series,
    mean: np.ndarray | pd.Series,
    *,
    lo: float = 1.0,
    hi: float = 99.0,
) -> pd.DataFrame:
    """Flag finite residuals outside the ``[lo, hi]`` percentile range.

    Percentiles are computed independently for RMS and mean among finite values.
    NaN residuals do not set these bits (use :func:`flag_has_nan` for missingness).
    """
    rms_arr = np.asarray(rms, dtype=float)
    mean_arr = np.asarray(mean, dtype=float)
    n = rms_arr.size
    rms_out = np.zeros(n, dtype=bool)
    mean_out = np.zeros(n, dtype=bool)

    finite_rms = np.isfinite(rms_arr)
    if int(finite_rms.sum()) >= 2:
        p_lo, p_hi = np.percentile(rms_arr[finite_rms], [float(lo), float(hi)])
        rms_out = finite_rms & ((rms_arr < p_lo) | (rms_arr > p_hi))

    finite_mean = np.isfinite(mean_arr)
    if int(finite_mean.sum()) >= 2:
        p_lo, p_hi = np.percentile(mean_arr[finite_mean], [float(lo), float(hi)])
        mean_out = finite_mean & ((mean_arr < p_lo) | (mean_arr > p_hi))

    return pd.DataFrame(
        {"resid_pctl_rms": rms_out, "resid_pctl_mean": mean_out},
    )


def quality_flag_legend() -> pd.DataFrame:
    """Return the bit layout as a table (bit index, name, meaning)."""
    rows = []
    for flag in SourceQualityFlag:
        rows.append(
            {
                "bit": int(flag.bit_length() - 1),
                "value": int(flag),
                "name": flag.name,
                "meaning": _QUALITY_FLAG_HELP[flag],
            }
        )
    return pd.DataFrame(rows)


def decode_quality_flag(value: int) -> list[str]:
    """Return names of bits set in *value* (empty if the mask is 0)."""
    mask = SourceQualityFlag(int(value) & 0xFFFFFFFF)
    return [flag.name for flag in SourceQualityFlag if flag & mask and flag.name]


QualityMatchMode = Literal["any", "all", "none"]


def quality_flag_names() -> list[str]:
    """Return defined ``SourceQualityFlag`` member names in bit order."""
    return [flag.name for flag in SourceQualityFlag if flag.name]


def quality_flag_mask_from_names(names: Sequence[str]) -> int:
    """OR named ``SourceQualityFlag`` members into an integer mask."""
    mask = 0
    known = {flag.name: int(flag) for flag in SourceQualityFlag if flag.name}
    for name in names:
        key = str(name).strip().upper()
        if key not in known:
            msg = f"Unknown quality flag {name!r}; expected one of {sorted(known)}"
            raise ValueError(msg)
        mask |= known[key]
    return mask


def filter_by_quality_mask(
    df: pd.DataFrame,
    mask: int = DEFAULT_QUALITY_FLAG_MASK,
    *,
    flag_col: str = "quality_flag",
) -> pd.DataFrame:
    """Keep rows with no quality concerns among the bits set in *mask*.

    A set bit in ``quality_flag`` marks a failed check. Rows are kept when
    ``(quality_flag & mask) == 0``. When *flag_col* is missing, *df* is
    returned unchanged.
    """
    if df.empty or flag_col not in df.columns:
        return df
    quality = pd.to_numeric(df[flag_col], errors="coerce").to_numpy(dtype=np.uint32)
    keep = (quality & np.uint32(mask)) == 0
    return df.loc[keep]


def filter_by_quality_flags(
    df: pd.DataFrame,
    names: Sequence[str],
    *,
    match: QualityMatchMode = "any",
    flag_col: str = "quality_flag",
) -> pd.DataFrame:
    """Return rows matching *names* under *match* (any / all / none).

    An empty *names* list returns *df* unchanged (no bit filter). Missing
    ``quality_flag`` also returns *df* unchanged.
    """
    if df.empty or not names:
        return df
    if flag_col not in df.columns:
        return df
    if match not in {"any", "all", "none"}:
        msg = f"Unknown match mode {match!r}; expected 'any', 'all', or 'none'"
        raise ValueError(msg)

    mask = np.uint32(quality_flag_mask_from_names(names))
    quality = pd.to_numeric(df[flag_col], errors="coerce").to_numpy(dtype=np.uint32)
    bits = quality & mask
    if match == "any":
        keep = bits != 0
    elif match == "all":
        keep = bits == mask
    else:
        keep = bits == 0
    return df.loc[keep]


OR_HESL_EXCLUDE_FLAGS: tuple[str, ...] = (
    "LOW_ELEVATION",
    "HAS_NAN",
    "INVALID_ASTROMETRY",
    "UNPHYSICAL_FLUX",
    "RESID_ABS_FAIL",
    "RESID_PCTL_RMS",
    "JITTER_FAIL",
    "RESID_PCTL_MEAN",
)

OR_HESL_EXCLUDE_MASK: int = quality_flag_mask_from_names(OR_HESL_EXCLUDE_FLAGS)


def filter_or_hesl(
    df: pd.DataFrame,
    *,
    flag_col: str = "quality_flag",
    or_exclude_mask: int | None = None,
) -> pd.DataFrame:
    """Keep rows passing the OR+HE∧SL exclusion rule.

    Excludes a row when any bit in :data:`OR_HESL_EXCLUDE_FLAGS` is set, or when
    both ``HIGH_ELLIPTICITY`` and ``SINGLE_LST`` are set. When *flag_col* is
    missing, *df* is returned unchanged.
    """
    if df.empty or flag_col not in df.columns:
        return df.copy()
    qf = pd.to_numeric(df[flag_col], errors="coerce").to_numpy(dtype=np.uint32)
    single_lst = np.uint32(SourceQualityFlag.SINGLE_LST)
    high_ellipticity = np.uint32(SourceQualityFlag.HIGH_ELLIPTICITY)
    mask = np.uint32(
        OR_HESL_EXCLUDE_MASK if or_exclude_mask is None else int(or_exclude_mask)
    )
    exclude = (
        (((qf & single_lst) != 0) & ((qf & high_ellipticity) != 0))
        | ((qf & mask) != 0)
    )
    return df.loc[~exclude].copy()


def pack_quality_flags(flags: pd.DataFrame) -> np.ndarray:
    """Pack boolean flag columns into a uint32 ``quality_flag`` array."""
    n = len(flags)
    out = np.zeros(n, dtype=np.uint32)
    for col, bit in _QUALITY_FLAG_COLUMNS:
        if col not in flags.columns:
            continue
        out |= np.where(
            flags[col].to_numpy(dtype=bool),
            np.uint32(bit),
            np.uint32(0),
        )
    return out


def quality_flag_bit_counts(flags: pd.DataFrame) -> pd.DataFrame:
    """Per-bit set counts for a flags table that includes ``quality_flag``."""
    n = len(flags)
    rows = []
    quality = (
        flags["quality_flag"].to_numpy(dtype=np.uint32)
        if "quality_flag" in flags.columns
        else pack_quality_flags(flags)
    )
    for col, bit in _QUALITY_FLAG_COLUMNS:
        if col in flags.columns:
            n_set = int(flags[col].to_numpy(dtype=bool).sum())
        elif n:
            n_set = int(((quality & np.uint32(bit)) != 0).sum())
        else:
            n_set = 0
        rows.append(
            {
                "bit": int(bit.bit_length() - 1),
                "name": bit.name,
                "n_set": n_set,
                "fraction": (n_set / n) if n else float("nan"),
                "meaning": _QUALITY_FLAG_HELP[bit],
            }
        )
    n_clear = int((quality == 0).sum()) if n else 0
    rows.append(
        {
            "bit": -1,
            "name": "ALL_CLEAR",
            "n_set": n_clear,
            "fraction": (n_clear / n) if n else float("nan"),
            "meaning": "quality_flag == 0 (every check passed)",
        }
    )
    return pd.DataFrame(rows)


def cluster_radec_jitter_rms(source_matches: pd.DataFrame) -> float:
    """RMS of member–centroid great-circle offsets in degrees.

    Returns 0.0 for fewer than two finite-coordinate members? Plan: single
    member → RMS 0 (trivial pass). Empty → NaN (not calculable).
    """
    if source_matches is None or source_matches.empty:
        return float("nan")
    if "RA" not in source_matches.columns or "DEC" not in source_matches.columns:
        return float("nan")
    ra = pd.to_numeric(source_matches["RA"], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(source_matches["DEC"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(ra) & np.isfinite(dec)
    if not mask.any():
        return float("nan")
    ra = ra[mask]
    dec = dec[mask]
    if len(ra) == 1:
        return 0.0
    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    # Cartesian mean on the unit sphere for a wrap-safe centroid
    xyz = sc.cartesian.xyz.value  # shape (3, N)
    mean_xyz = np.mean(xyz, axis=1)
    norm = np.linalg.norm(mean_xyz)
    if norm <= 0.0 or not np.isfinite(norm):
        return float("nan")
    mean_xyz = mean_xyz / norm
    centroid = SkyCoord(
        x=mean_xyz[0],
        y=mean_xyz[1],
        z=mean_xyz[2],
        representation_type="cartesian",
        frame="icrs",
    )
    seps = sc.separation(centroid).deg
    return float(np.sqrt(np.mean(seps**2)))


def resolve_bmaj(row: pd.Series) -> float:
    """Finite BMAJ from ``BMAJ_match`` / ``BMAJ_full`` / ``BMAJ``, else NaN."""
    for key in ("BMAJ_match", "BMAJ_full", "BMAJ"):
        if key not in row.index:
            continue
        try:
            val = float(row[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(val) and val > 0.0:
            return val
    return float("nan")


def flag_jitter_exceeds(
    rms_deg: float,
    bmaj_deg: float,
    *,
    frac: float = 0.3,
) -> bool:
    """True when finite RMS exceeds ``frac * BMAJ``."""
    if not np.isfinite(rms_deg) or not np.isfinite(bmaj_deg) or bmaj_deg <= 0.0:
        return False
    return bool(rms_deg > float(frac) * bmaj_deg)


def seed_lst_rows(
    metacatalog: pd.DataFrame,
    layout: CatalogLayout,
    *,
    lst_merged: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Rematch each row's QA band (``astrometry_band`` or ``origin_band``) to one LST seed.

    Returns a DataFrame indexed by ``meta_id`` with LST columns plus
    ``seed_band``, ``seed_matched``. Missing matches yield empty/NaN rows with
    ``seed_matched=False``.
    """
    warn: list[str] = []
    if metacatalog.empty:
        return pd.DataFrame(), warn
    work = metacatalog.copy()
    if "meta_id" not in work.columns:
        msg = "metacatalog missing meta_id"
        raise ValueError(msg)

    records: list[dict] = []
    work = work.copy()
    work["_qa_band"] = work.apply(qa_band_for_row, axis=1)
    for qa_band, group in work.groupby(work["_qa_band"].astype(str), sort=False):
        band = str(qa_band)
        if band.lower() in {"", "nan", "none"}:
            for _, row in group.iterrows():
                records.append(
                    {
                        "meta_id": int(row["meta_id"]),
                        "seed_band": band,
                        "seed_matched": False,
                    }
                )
            warn.append(f"{len(group)} rows have empty QA band")
            continue

        lst_df = _load_lst_band(layout, band, lst_merged, warn)
        if lst_df is None:
            for _, row in group.iterrows():
                records.append(
                    {
                        "meta_id": int(row["meta_id"]),
                        "seed_band": band,
                        "seed_matched": False,
                    }
                )
            continue

        bases = []
        meta_rows: list[pd.Series] = []
        for _, row in group.iterrows():
            bases.append(
                {
                    "RA": float(row["RA"]) if np.isfinite(float(row["RA"])) else np.nan,
                    "DEC": float(row["DEC"]) if np.isfinite(float(row["DEC"])) else np.nan,
                    "BMAJ": _match_bmaj(row),
                }
            )
            meta_rows.append(row)
        base = pd.DataFrame(bases)
        # Drop non-finite coords for association; mark unmatched later
        finite = np.isfinite(base["RA"]) & np.isfinite(base["DEC"])
        hits_map: dict[int, list[int]] = {}
        if finite.any():
            sub_base = base.loc[finite].reset_index(drop=True)
            hits, _ = associate_catalogs(sub_base, lst_df)
            finite_idxs = np.flatnonzero(finite.to_numpy())
            for local_i, meta_i in enumerate(finite_idxs):
                hits_map[int(meta_i)] = hits.get(local_i, [])

        for i, row in enumerate(meta_rows):
            mid = int(row["meta_id"])
            idxs = hits_map.get(i, [])
            if not idxs:
                records.append({"meta_id": mid, "seed_band": band, "seed_matched": False})
                continue
            best = _pick_rematch_lst_row(lst_df.iloc[idxs], row, band)
            entry = best.to_dict()
            entry["meta_id"] = mid
            entry["seed_band"] = band
            entry["seed_matched"] = True
            records.append(entry)

    seed = pd.DataFrame.from_records(records)
    if seed.empty:
        return seed, warn
    return seed.set_index("meta_id", drop=False), warn


def expand_seed_source_matches(
    seed_lst_row: pd.Series,
    meta_row: pd.Series,
    layout: CatalogLayout,
) -> tuple[pd.DataFrame, list[str]]:
    """Per-hour source rematch for one seed LST row (jitter membership)."""
    warn: list[str] = []
    band = str(
        seed_lst_row.get("seed_band")
        or seed_lst_row.get("band")
        or meta_row.get("origin_band")
    )
    hours = _parse_lst_hours(seed_lst_row.get("lst_hours", ""))
    if not hours:
        hours = _parse_lst_hours(meta_row.get("lst_hours", ""))
    if not hours:
        warn.append(f"No lst_hours for seed band {band}")
        return pd.DataFrame(), warn

    base_bmaj = float(seed_lst_row["BMAJ"]) if "BMAJ" in seed_lst_row.index and np.isfinite(
        float(seed_lst_row.get("BMAJ", np.nan))
    ) else _match_bmaj(meta_row)
    lst_base = _one_row_base(float(seed_lst_row["RA"]), float(seed_lst_row["DEC"]), base_bmaj)

    frames: list[pd.DataFrame] = []
    for hour in hours:
        path = layout.sources(hour, band)
        if not path.is_file():
            warn.append(f"Missing sources catalog: {path.name}")
            continue
        try:
            sources = read_sources_catalog(layout, hour, band, as_pandas=True)
        except FileNotFoundError:
            warn.append(f"Missing sources catalog: {path.name}")
            continue
        assert isinstance(sources, pd.DataFrame)
        if sources.empty:
            continue
        sources = _ensure_bmaj_column(sources)
        hits, _ = associate_catalogs(lst_base, sources)
        idxs = hits.get(0, [])
        if not idxs:
            continue
        hit = sources.iloc[idxs].copy()
        hit["band"] = band
        if "lst_hour" not in hit.columns:
            hit["lst_hour"] = hour
        frames.append(hit)
    if not frames:
        return pd.DataFrame(), warn
    return pd.concat(frames, ignore_index=True), warn


def _tier_cutoff(through: str, order: Sequence[str]) -> list[str]:
    through = str(through).upper()
    if through not in order:
        msg = f"Unknown tier {through!r}; expected one of {list(order)}"
        raise ValueError(msg)
    return list(order[: order.index(through) + 1])


def assert_gold_subset_of_cleaned(
    cleaned: ReliabilityResult | pd.DataFrame,
    gold: ReliabilityResult | pd.DataFrame,
    *,
    strict: bool = False,
) -> None:
    """Enforce ``gold ⊆ cleaned`` by ``meta_id``; warn or raise."""
    def _ids(obj: ReliabilityResult | pd.DataFrame) -> set[int]:
        if isinstance(obj, ReliabilityResult):
            return set(int(x) for x in obj.meta_ids)
        if "meta_id" not in obj.columns:
            return set()
        return set(int(x) for x in obj["meta_id"].tolist())

    extra = _ids(gold) - _ids(cleaned)
    if not extra:
        return
    msg = (
        f"gold ⊆ cleaned violated: {len(extra)} meta_id(s) in gold not in cleaned "
        f"(e.g. {sorted(extra)[:5]})"
    )
    if strict:
        raise ValueError(msg)
    py_warnings.warn(msg, UserWarning, stacklevel=2)


def _empty_result(metacatalog: pd.DataFrame) -> ReliabilityResult:
    return ReliabilityResult(
        catalog=metacatalog.iloc[0:0].copy(),
        meta_ids=np.array([], dtype=int),
        tier_counts=pd.DataFrame(columns=["tier", "n_in", "n_out", "n_removed"]),
        flags=pd.DataFrame(),
    )


def _build_context(
    metacatalog: pd.DataFrame,
    layout: CatalogLayout,
    *,
    config: ReliabilityConfig,
    lst_merged: Mapping[str, pd.DataFrame] | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach seed LST, flux/resid, jitter columns keyed by metacatalog index."""
    warn: list[str] = []
    meta = metacatalog.reset_index(drop=True).copy()
    if "meta_id" not in meta.columns:
        meta["meta_id"] = np.arange(len(meta), dtype=int)

    seed, seed_warn = seed_lst_rows(meta, layout, lst_merged=lst_merged)
    warn.extend(seed_warn)

    n = len(meta)
    use_merge_jitter = CLUSTER_JITTER_RMS_COL in meta.columns
    if use_merge_jitter:
        warn.append("using merge-time cluster jitter (skipping per-hour rematch)")

    seed_matched = np.zeros(n, dtype=bool)
    flux_sigma = np.full(n, np.nan)
    unphys_soft = np.zeros(n, dtype=bool)
    resid_fail_soft = np.zeros(n, dtype=bool)
    resid_rms = np.full(n, np.nan)
    resid_mean = np.full(n, np.nan)
    sigma_finite = np.zeros(n, dtype=bool)
    resid_finite = np.zeros(n, dtype=bool)
    jitter_rms = np.full(n, np.nan)
    n_rematch = np.zeros(n, dtype=int)
    n_lst_seed = np.full(n, np.nan)
    bmaj = np.full(n, np.nan)
    if "S_Code" in meta.columns:
        s_code = meta["S_Code"].map(cast_s_code_value).astype(object).to_numpy()
    else:
        s_code = np.array([pd.NA] * n, dtype=object)

    mid_to_i = {int(m): i for i, m in enumerate(meta["meta_id"].tolist())}

    if use_merge_jitter:
        jitter_vals = pd.to_numeric(meta[CLUSTER_JITTER_RMS_COL], errors="coerce")
        if "n_lst_contributions" in meta.columns:
            n_lst_vals = pd.to_numeric(meta["n_lst_contributions"], errors="coerce")
        else:
            n_lst_vals = pd.Series(np.nan, index=meta.index)
        for i in range(n):
            jitter_rms[i] = float(jitter_vals.iloc[i])
            if pd.notna(n_lst_vals.iloc[i]):
                n_rematch[i] = int(n_lst_vals.iloc[i])
                n_lst_seed[i] = float(n_lst_vals.iloc[i])

    if not seed.empty:
        for mid, srow in seed.iterrows():
            mid_i = int(mid) if not isinstance(mid, (int, np.integer)) else int(mid)
            # seed index may be meta_id
            if "meta_id" in srow.index:
                mid_i = int(srow["meta_id"])
            i = mid_to_i.get(mid_i)
            if i is None:
                continue
            matched = bool(srow.get("seed_matched", False))
            seed_matched[i] = matched
            if not matched:
                continue
            qa_row = meta.iloc[i]
            if "Resid_Isl_rms" not in qa_row.index or not np.isfinite(
                float(pd.to_numeric(qa_row.get("Resid_Isl_rms"), errors="coerce"))
            ):
                qa_row = srow
            qa_band = qa_band_for_row(meta.iloc[i])
            s_df = flux_qa_frame(qa_row, srow, qa_band)
            if s_df.empty:
                sigma_finite[i] = False
            else:
                sig = flux_sigma_total_minus_peak(s_df).iloc[0]
                flux_sigma[i] = sig
                sigma_finite[i] = bool(np.isfinite(sig))
                unphys_soft[i] = bool(
                    flag_unphysical_flux(
                        s_df, nsigma=config.flux_unphysical_nsigma
                    ).iloc[0]
                )
            rf = flag_residual_absolute(
                pd.DataFrame([qa_row]),
                rms_thresh_jy=config.resid_rms_thresh_jy,
                mean_thresh_jy=config.resid_mean_thresh_jy,
            )
            resid_fail_soft[i] = bool(rf["resid_fail"].iloc[0])
            if "Resid_Isl_rms" in qa_row.index:
                resid_rms[i] = float(pd.to_numeric(qa_row["Resid_Isl_rms"], errors="coerce"))
            if "Resid_Isl_mean" in qa_row.index:
                resid_mean[i] = float(pd.to_numeric(qa_row["Resid_Isl_mean"], errors="coerce"))
            resid_finite[i] = np.isfinite(resid_rms[i]) and np.isfinite(resid_mean[i])
            if not np.isfinite(n_lst_seed[i]):
                try:
                    n_lst_seed[i] = float(srow.get("n_lst_contributions", np.nan))
                except (TypeError, ValueError):
                    n_lst_seed[i] = np.nan

            if not use_merge_jitter:
                sources, src_warn = expand_seed_source_matches(srow, meta.iloc[i], layout)
                warn.extend(src_warn)
                n_rematch[i] = len(sources)
                jitter_rms[i] = cluster_radec_jitter_rms(sources)

            bmaj[i] = resolve_bmaj(meta.iloc[i])
            if not np.isfinite(bmaj[i]) and "BMAJ" in srow.index:
                try:
                    bmaj[i] = float(srow["BMAJ"])
                except (TypeError, ValueError):
                    pass
            if pd.isna(s_code[i]) and "S_Code" in srow.index:
                s_code[i] = cast_s_code_value(srow["S_Code"])

    # BMAJ from meta even without seed
    for i, row in meta.iterrows():
        if not np.isfinite(bmaj[i]):
            bmaj[i] = resolve_bmaj(row)

    invalid = flag_invalid_astrometry_flux(meta).to_numpy()
    multi = meta.apply(
        lambda r: passes_multi_image(r, min_lst=config.min_lst_contributions),
        axis=1,
    ).to_numpy()
    confused = flag_confused_assoc(meta).to_numpy()
    unique_ok = meta.apply(all_bands_unique_assoc, axis=1).to_numpy()

    jitter_fail_soft = np.array(
        [
            flag_jitter_exceeds(jitter_rms[i], bmaj[i], frac=config.jitter_bmaj_frac)
            for i in range(n)
        ],
        dtype=bool,
    )

    flags = pd.DataFrame(
        {
            "meta_id": meta["meta_id"].to_numpy(),
            "invalid": invalid,
            "passes_multi_image": multi,
            "seed_matched": seed_matched,
            "flux_sigma": flux_sigma,
            "sigma_finite": sigma_finite,
            "unphysical_soft": unphys_soft,
            "resid_rms": resid_rms,
            "resid_mean": resid_mean,
            "resid_finite": resid_finite,
            "resid_fail_soft": resid_fail_soft,
            "jitter_rms_deg": jitter_rms,
            "bmaj_deg": bmaj,
            "jitter_fail_soft": jitter_fail_soft,
            "n_rematch": n_rematch,
            "n_lst_seed": n_lst_seed,
            "confused_assoc": confused,
            "all_bands_unique": unique_ok,
            "s_code": s_code,
        }
    )
    return flags, warn


def _exclude_from_flags(
    metacatalog: pd.DataFrame,
    flags: pd.DataFrame,
    cfg: ReliabilityConfig,
    warn: list[str],
) -> ReliabilityResult:
    meta = metacatalog.reset_index(drop=True).copy()
    if "meta_id" not in meta.columns:
        meta["meta_id"] = flags["meta_id"]
    active = _tier_cutoff(cfg.exclude_through, _EXCLUDE_TIERS)
    if cfg.require_unique_assoc_exclude and "E5" not in active:
        active = [*active, "E5"]
    keep = np.ones(len(meta), dtype=bool)
    tier_rows = []
    n_in = int(keep.sum())

    def _apply(name: str, remove: np.ndarray) -> None:
        nonlocal n_in, keep
        if name not in active:
            return
        removed = keep & remove
        keep = keep & ~remove
        tier_rows.append(
            {"tier": name, "n_in": n_in, "n_out": int(keep.sum()), "n_removed": int(removed.sum())}
        )
        n_in = int(keep.sum())

    _apply("E0", flags["invalid"].to_numpy())
    _apply("E1", ~flags["passes_multi_image"].to_numpy())
    _apply("E2", flags["unphysical_soft"].to_numpy())
    _apply("E3", flags["resid_fail_soft"].to_numpy())
    _apply("E4", flags["jitter_fail_soft"].to_numpy())
    _apply("E5", flags["confused_assoc"].to_numpy())
    catalog = meta.loc[keep].copy()
    return ReliabilityResult(
        catalog=catalog,
        meta_ids=catalog["meta_id"].to_numpy(dtype=int),
        tier_counts=pd.DataFrame(tier_rows),
        flags=flags,
        warnings=list(warn),
    )


def _include_from_flags(
    metacatalog: pd.DataFrame,
    flags: pd.DataFrame,
    cfg: ReliabilityConfig,
    warn: list[str],
) -> ReliabilityResult:
    meta = metacatalog.reset_index(drop=True).copy()
    if "meta_id" not in meta.columns:
        meta["meta_id"] = flags["meta_id"]

    active = _tier_cutoff(cfg.include_through, _INCLUDE_TIERS)
    if cfg.require_unique_assoc_include and "I5" not in active:
        active = list(dict.fromkeys([*active, "I5"]))
    if not cfg.require_unique_assoc_include:
        active = [t for t in active if t != "I5"]

    keep = np.ones(len(meta), dtype=bool)
    tier_rows = []
    n_in = int(keep.sum())

    def _require(name: str, ok: np.ndarray) -> None:
        nonlocal n_in, keep
        if name not in active:
            return
        fail = keep & ~ok
        keep = keep & ok
        tier_rows.append(
            {"tier": name, "n_in": n_in, "n_out": int(keep.sum()), "n_removed": int(fail.sum())}
        )
        n_in = int(keep.sum())

    _require("I0", ~flags["invalid"].to_numpy())
    _require("I1", flags["passes_multi_image"].to_numpy())
    i2_ok = (
        flags["seed_matched"].to_numpy()
        & flags["sigma_finite"].to_numpy()
        & (flags["flux_sigma"].to_numpy(dtype=float) >= -float(cfg.flux_unphysical_nsigma))
    )
    _require("I2", i2_ok)
    i3_ok = (
        flags["seed_matched"].to_numpy()
        & flags["resid_finite"].to_numpy()
        & ~flags["resid_fail_soft"].to_numpy()
    )
    _require("I3", i3_ok)
    n_lst_meta = pd.to_numeric(
        meta["n_lst_contributions"] if "n_lst_contributions" in meta.columns else np.nan,
        errors="coerce",
    ).to_numpy(dtype=float)
    need_rematch = n_lst_meta >= float(cfg.min_lst_contributions)
    rematch_ok = (~need_rematch) | (flags["n_rematch"].to_numpy(dtype=int) >= 2)
    jitter_ok = (
        np.isfinite(flags["bmaj_deg"].to_numpy(dtype=float))
        & np.isfinite(flags["jitter_rms_deg"].to_numpy(dtype=float))
        & ~flags["jitter_fail_soft"].to_numpy()
        & rematch_ok
    )
    _require("I4", jitter_ok)
    _require("I5", flags["all_bands_unique"].to_numpy())

    catalog = meta.loc[keep].copy()
    return ReliabilityResult(
        catalog=catalog,
        meta_ids=catalog["meta_id"].to_numpy(dtype=int),
        tier_counts=pd.DataFrame(tier_rows),
        flags=flags,
        warnings=list(warn),
    )


def filter_metacatalog_exclude(
    metacatalog: pd.DataFrame,
    layout: CatalogLayout,
    *,
    config: ReliabilityConfig | None = None,
    lst_merged: Mapping[str, pd.DataFrame] | None = None,
    flags: pd.DataFrame | None = None,
) -> ReliabilityResult:
    """Soft-exclude demonstrably bad sources → cleaned catalog."""
    cfg = config or ReliabilityConfig()
    if metacatalog is None or metacatalog.empty:
        return _empty_result(metacatalog if metacatalog is not None else pd.DataFrame())
    if flags is None or len(flags) != len(metacatalog.reset_index(drop=True)):
        flags, warn = _build_context(metacatalog, layout, config=cfg, lst_merged=lst_merged)
    else:
        warn = []
    return _exclude_from_flags(metacatalog, flags, cfg, warn)


def filter_metacatalog_include(
    metacatalog: pd.DataFrame,
    layout: CatalogLayout,
    *,
    config: ReliabilityConfig | None = None,
    lst_merged: Mapping[str, pd.DataFrame] | None = None,
    flags: pd.DataFrame | None = None,
) -> ReliabilityResult:
    """Hard-include only demonstrably good sources → gold catalog."""
    cfg = config or ReliabilityConfig()
    if metacatalog is None or metacatalog.empty:
        return _empty_result(metacatalog if metacatalog is not None else pd.DataFrame())
    if flags is None or len(flags) != len(metacatalog.reset_index(drop=True)):
        flags, warn = _build_context(metacatalog, layout, config=cfg, lst_merged=lst_merged)
    else:
        warn = []
    return _include_from_flags(metacatalog, flags, cfg, warn)


def filter_metacatalog_reliability(
    metacatalog: pd.DataFrame,
    layout: CatalogLayout,
    *,
    config: ReliabilityConfig | None = None,
    lst_merged: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[ReliabilityResult, ReliabilityResult]:
    """Run cleaned (exclude) and gold (include) once; check nesting contract."""
    cfg = config or ReliabilityConfig()
    if metacatalog is None or metacatalog.empty:
        empty = _empty_result(metacatalog if metacatalog is not None else pd.DataFrame())
        return empty, empty
    flags, warn = _build_context(metacatalog, layout, config=cfg, lst_merged=lst_merged)
    cleaned = _exclude_from_flags(metacatalog, flags, cfg, warn)
    gold = _include_from_flags(metacatalog, flags, cfg, warn)
    assert_gold_subset_of_cleaned(cleaned, gold, strict=cfg.strict)
    return cleaned, gold


def _match_vlssr_unmatched(
    meta: pd.DataFrame,
    *,
    vlssr: pd.DataFrame | None,
    vlssr_config: object | None,
    warn: list[str],
) -> np.ndarray:
    """Return a boolean array: True when the row has no VLSSR match.

    If the VLSSR catalog cannot be loaded, the check is skipped (all False)
    so missing reference data does not flag every source.
    """
    from lwa_catalog.analyze.vlssr import (
        VlssrMatchConfig,
        load_vlssr_catalog,
        match_catalog_to_vlssr,
    )
    from lwa_catalog.constants import VLSSR_DEFAULT_PATH

    n = len(meta)
    none_unmatched = np.zeros(n, dtype=bool)
    cfg = vlssr_config if vlssr_config is not None else VlssrMatchConfig(target="metacatalog")
    cfg = replace(cfg, target="metacatalog")

    catalog = vlssr
    if catalog is None:
        path = Path(getattr(cfg, "catalog_path", VLSSR_DEFAULT_PATH))
        if not path.is_file():
            warn.append(f"VLSSR catalog not found at {path}; skipping NO_VLSSR bit")
            return none_unmatched
        try:
            catalog = load_vlssr_catalog(path)
        except FileNotFoundError:
            warn.append(f"VLSSR catalog not found at {path}; skipping NO_VLSSR bit")
            return none_unmatched

    result = match_catalog_to_vlssr(meta, catalog, config=cfg)
    warn.extend(result.warnings)
    matched_flags = result.meta_flags
    unmatched = np.ones(n, dtype=bool)
    if matched_flags.empty:
        return unmatched
    if "meta_id" in matched_flags.columns and "meta_id" in meta.columns:
        id_to_matched = {
            int(mid): bool(is_match)
            for mid, is_match in zip(
                matched_flags["meta_id"],
                matched_flags["matched"],
                strict=True,
            )
            if pd.notna(mid)
        }
        for i, mid in enumerate(meta["meta_id"].tolist()):
            try:
                unmatched[i] = not id_to_matched.get(int(mid), False)
            except (TypeError, ValueError):
                unmatched[i] = True
        return unmatched
    n_copy = min(n, len(matched_flags))
    unmatched[:n_copy] = ~matched_flags["matched"].to_numpy(dtype=bool)[:n_copy]
    return unmatched


def assign_source_quality_flags(
    metacatalog: pd.DataFrame,
    layout: CatalogLayout,
    *,
    config: ReliabilityConfig | None = None,
    lst_merged: Mapping[str, pd.DataFrame] | None = None,
    vlssr: pd.DataFrame | None = None,
    vlssr_config: object | None = None,
) -> QualityFlagResult:
    """Attach a uint32 ``quality_flag`` to every metacatalog row (no filtering).

    Reuses the reliability context (seed LST residuals, unphysical flux, jitter,
    confused association) and adds NaN, single-LST, single-band association,
    residual-percentile, VLSSR, and ``S_Code`` bits. Bit 0 on each flag means
    that check is good/reliable; ``quality_flag == 0`` means all checks passed.
    """
    cfg = config or ReliabilityConfig()
    if metacatalog is None or metacatalog.empty:
        empty = metacatalog if metacatalog is not None else pd.DataFrame()
        empty = empty.copy()
        if "quality_flag" not in empty.columns:
            empty["quality_flag"] = pd.Series(dtype=np.uint32)
        return QualityFlagResult(
            catalog=empty,
            flags=pd.DataFrame(),
            bit_counts=quality_flag_bit_counts(pd.DataFrame()),
        )

    flags, warn = _build_context(metacatalog, layout, config=cfg, lst_merged=lst_merged)
    meta = metacatalog.reset_index(drop=True).copy()
    if "meta_id" not in meta.columns:
        meta["meta_id"] = flags["meta_id"]

    flags = flags.copy()
    flags["has_nan"] = flag_has_nan(meta).to_numpy()
    flags["single_lst"] = flag_single_lst(meta).to_numpy()
    flags["single_unique_band"] = flag_single_unique_band(meta).to_numpy()
    flags["scode_complex"] = flag_scode_complex(flags["s_code"]).to_numpy()

    pctl = flag_residual_percentile(
        flags["resid_rms"].to_numpy(dtype=float),
        flags["resid_mean"].to_numpy(dtype=float),
        lo=cfg.resid_percentile_lo,
        hi=cfg.resid_percentile_hi,
    )
    flags["resid_pctl_rms"] = pctl["resid_pctl_rms"].to_numpy()
    flags["resid_pctl_mean"] = pctl["resid_pctl_mean"].to_numpy()
    flags["no_vlssr"] = _match_vlssr_unmatched(
        meta, vlssr=vlssr, vlssr_config=vlssr_config, warn=warn
    )
    flags["low_elevation"] = flag_low_elevation(meta, min_deg=cfg.min_elevation_deg).to_numpy()
    flags["high_ellipticity"] = flag_high_ellipticity(
        meta, max_ratio=cfg.max_source_ellipticity
    ).to_numpy()

    quality = pack_quality_flags(flags)
    flags["quality_flag"] = quality
    meta["quality_flag"] = quality
    return QualityFlagResult(
        catalog=meta,
        flags=flags,
        bit_counts=quality_flag_bit_counts(flags),
        warnings=list(warn),
    )

