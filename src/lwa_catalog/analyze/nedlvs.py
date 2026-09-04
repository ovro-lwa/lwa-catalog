"""NED-LVS cross-match QA against LWA metacatalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.table import Table

from lwa_catalog.analyze.vlssr import select_blue_associated_rows
from lwa_catalog.constants import (
    NEDLVS_CATALOG_POSITION_SIGMA_DEG,
    NEDLVS_DEFAULT_CENTROID_SIGMA_DEG,
    NEDLVS_DEFAULT_DIAM_SCALE,
    NEDLVS_DEFAULT_MAX_REDSHIFT,
    NEDLVS_DEFAULT_PATH,
    NEDLVS_DEFAULT_POSITION_SIGMA_SCALE,
    NVSS_POSITION_ERROR_DEFAULT_ARCSEC,
    VLASS_BMAJ_ARCSEC,
    VLSSR_POSITION_ERROR_ARCSEC,
    band_frequency_hz,
)
from lwa_catalog.io import read_table
from lwa_catalog.paths import CatalogLayout

NedlvsTarget = Literal["metacatalog", "metacatalog_blue", "lst_merged_blue"]
MetacatalogSelection = Literal[
    "full",
    "blue",
    "quality_all_clear",
    "query",
]

NEDLVS_LOAD_COLUMNS: tuple[str, ...] = (
    "objname",
    "ra",
    "dec",
    "objtype",
    "z",
    "DistMpc",
    "Diam",
    "Mstar",
    "SFR_hybrid",
    "SFR_W4",
)

# 1σ defaults (degrees) when ``match_source`` is a cascaded radio survey and
# ``match_sigma_deg`` is absent (e.g. older metacatalog_radio.parquet).
_MATCH_SOURCE_SIGMA_DEG: dict[str, float] = {
    "VLSSR": VLSSR_POSITION_ERROR_ARCSEC / 3600.0,
    "NVSS": NVSS_POSITION_ERROR_DEFAULT_ARCSEC / 3600.0,
    "VLASS": VLASS_BMAJ_ARCSEC / 3600.0,
}


@dataclass(frozen=True)
class NedlvsMatchConfig:
    """Configuration for :func:`match_catalog_to_nedlvs`."""

    catalog_path: Path = NEDLVS_DEFAULT_PATH
    target: NedlvsTarget = "metacatalog"
    position_sigma_scale: float = NEDLVS_DEFAULT_POSITION_SIGMA_SCALE
    max_redshift: float | None = NEDLVS_DEFAULT_MAX_REDSHIFT
    default_centroid_sigma_deg: float = NEDLVS_DEFAULT_CENTROID_SIGMA_DEG
    nedlvs_position_sigma_deg: float = NEDLVS_CATALOG_POSITION_SIGMA_DEG
    # Floor search radius as ``diam_scale * Diam_arcsec`` (degrees). ``None`` disables.
    diam_scale: float | None = NEDLVS_DEFAULT_DIAM_SCALE


@dataclass
class NedlvsMatchResult:
    """NED-LVS cross-match QA metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    nedlvs_flags: pd.DataFrame
    nedlvs_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def select_metacatalog(
    metacatalog: pd.DataFrame,
    *,
    selection: MetacatalogSelection = "full",
    layout: CatalogLayout | None = None,
    query: str | None = None,
) -> pd.DataFrame:
    """Return a metacatalog subset before NED-LVS cross-matching."""
    if selection == "full":
        out = metacatalog
    elif selection == "blue":
        out = select_blue_associated_rows(metacatalog)
    elif selection == "quality_all_clear":
        if layout is None:
            msg = "layout is required for selection='quality_all_clear'"
            raise ValueError(msg)
        quality_path = layout.metacatalog_quality()
        if not quality_path.is_file():
            msg = f"quality catalog not found: {quality_path}"
            raise FileNotFoundError(msg)
        quality = read_table(quality_path)
        if "quality_flag" not in quality.columns:
            msg = f"{quality_path} missing quality_flag column"
            raise ValueError(msg)
        clear_ids = quality.loc[quality["quality_flag"] == 0, "meta_id"]
        if "meta_id" not in metacatalog.columns:
            msg = "metacatalog missing meta_id column for quality join"
            raise ValueError(msg)
        out = metacatalog.loc[metacatalog["meta_id"].isin(clear_ids)]
    elif selection == "query":
        if not query:
            msg = "query is required for selection='query'"
            raise ValueError(msg)
        out = metacatalog.query(query, engine="python")
    else:
        msg = f"unsupported selection: {selection!r}"
        raise ValueError(msg)

    return out.reset_index(drop=True)


def resolve_centroid_sigma_deg(
    row: pd.Series,
    *,
    default_centroid_sigma_deg: float = NEDLVS_DEFAULT_CENTROID_SIGMA_DEG,
) -> float:
    """Return a 1σ sky-position radius in degrees for cross-matching.

    Preference order:

    1. ``match_sigma_deg`` from cascaded radio attach (bootstrap frame).
    2. Survey default for ``match_source`` ``VLSSR`` / ``NVSS`` / ``VLASS``.
    3. PyBDSF ``E_RA``/``E_DEC``.
    4. Merge-time ``cluster_jitter_rms_deg``.
    5. *default_centroid_sigma_deg*.
    """
    match_sigma = pd.to_numeric(row.get("match_sigma_deg"), errors="coerce")
    if np.isfinite(match_sigma) and float(match_sigma) > 0.0:
        return float(match_sigma)

    source = str(row.get("match_source", "") or "").strip().upper()
    if source in _MATCH_SOURCE_SIGMA_DEG:
        return float(_MATCH_SOURCE_SIGMA_DEG[source])

    e_ra = pd.to_numeric(row.get("E_RA"), errors="coerce")
    e_dec = pd.to_numeric(row.get("E_DEC"), errors="coerce")
    if (
        np.isfinite(e_ra)
        and np.isfinite(e_dec)
        and float(e_ra) > 0.0
        and float(e_dec) > 0.0
    ):
        return float(np.hypot(float(e_ra), float(e_dec)))

    jitter = pd.to_numeric(row.get("cluster_jitter_rms_deg"), errors="coerce")
    if np.isfinite(jitter) and float(jitter) > 0.0:
        return float(jitter)

    return float(default_centroid_sigma_deg)


def resolve_match_coordinates(row: pd.Series) -> tuple[float, float]:
    """Return ``(RA, DEC)`` degrees, preferring ``match_RA``/``match_DEC``.

    Cascaded radio cross-match writes those columns when a bijective survey
    position is available. Falls back to top-level ``RA``/``DEC``.
    """
    for ra_key, dec_key in (("match_RA", "match_DEC"), ("RA", "DEC")):
        try:
            ra = float(pd.to_numeric(row.get(ra_key), errors="coerce"))
            dec = float(pd.to_numeric(row.get(dec_key), errors="coerce"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(ra) and np.isfinite(dec):
            return ra, dec
    return float("nan"), float("nan")


def _positive_flux(row: pd.Series, *columns: str) -> float:
    """Return the first positive finite flux among *columns*, else NaN."""
    for col in columns:
        if col not in row.index:
            continue
        flux = pd.to_numeric(row[col], errors="coerce")
        if np.isfinite(flux) and float(flux) > 0.0:
            return float(flux)
    return float("nan")


def resolve_row_flux(row: pd.Series, *, prefer_total: bool = True) -> float:
    """Return a positive flux from a catalog row.

    When *prefer_total* is true (default), try ``Total_flux`` before
    ``Peak_flux`` / ``Peak_intensity``. Catalogs that only publish peak
    intensity (VLSSR, NVSS ``CATALOG.FIT``) still resolve successfully.
    """
    if prefer_total:
        return _positive_flux(row, "Total_flux", "Peak_flux", "Peak_intensity")
    return _positive_flux(row, "Peak_flux", "Peak_intensity", "Total_flux")


def resolve_highest_frequency_flux(
    row: pd.Series,
    *,
    prefer_total: bool = True,
) -> tuple[float, float, str]:
    """Return ``(flux_jy, frequency_hz, band_name)`` at the highest-frequency band.

    Prefers ``Total_flux_{band}`` (and unsuffixed ``Total_flux`` on the origin
    band) when *prefer_total* is true, falling back to peak columns when total
    flux is missing.
    """
    from lwa_catalog.analyze.reliability import parse_bands_present

    bands = parse_bands_present(row)
    if not bands:
        return float("nan"), float("nan"), ""

    ranked = sorted(bands, key=band_frequency_hz, reverse=True)
    for band in ranked:
        freq = band_frequency_hz(band)
        if not np.isfinite(freq) or freq <= 0.0:
            continue

        candidates: list[str] = []
        if prefer_total:
            candidates.append(f"Total_flux_{band}")
        candidates.append(f"Peak_flux_{band}")
        if not prefer_total:
            candidates.append(f"Total_flux_{band}")

        if band == row.get("origin_band"):
            if prefer_total:
                candidates.extend(["Total_flux", "Peak_flux"])
            else:
                candidates.extend(["Peak_flux", "Total_flux"])

        flux = _positive_flux(row, *candidates)
        if np.isfinite(flux):
            return flux, float(freq), band

    return float("nan"), float("nan"), ""


def resolve_highest_frequency_peak_flux(row: pd.Series) -> tuple[float, float, str]:
    """Return ``(peak_flux_jy, frequency_hz, band_name)`` for the highest-frequency band."""
    return resolve_highest_frequency_flux(row, prefer_total=False)


def resolve_band_flux(
    row: pd.Series,
    band: str,
    *,
    prefer_total: bool = True,
) -> float:
    """Return positive flux for a named band, preferring total when available."""
    candidates: list[str] = []
    if prefer_total:
        candidates.append(f"Total_flux_{band}")
    candidates.append(f"Peak_flux_{band}")
    if not prefer_total:
        candidates.append(f"Total_flux_{band}")
    if band == row.get("origin_band"):
        if prefer_total:
            candidates.extend(["Total_flux", "Peak_flux"])
        else:
            candidates.extend(["Peak_flux", "Total_flux"])
    return _positive_flux(row, *candidates)


def radio_luminosity_nu(
    flux_jy: float | np.ndarray,
    z: float | np.ndarray,
    frequency_hz: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute monochromatic and ``nu*L_nu`` radio luminosities."""
    flux = np.asarray(flux_jy, dtype=float)
    redshift = np.asarray(z, dtype=float)
    freq = np.asarray(frequency_hz, dtype=float)

    d_l = Planck18.luminosity_distance(redshift)
    f_nu = flux * u.Jy
    l_nu_q = (4 * np.pi * d_l**2 * f_nu).to(u.erg / u.s / u.Hz)
    nu_l_nu_q = (freq * u.Hz * l_nu_q).to(u.erg / u.s)
    l_nu = np.atleast_1d(l_nu_q.value)
    nu_l_nu = np.atleast_1d(nu_l_nu_q.value)
    if l_nu.size == 1:
        return float(l_nu[0]), float(nu_l_nu[0])
    return l_nu, nu_l_nu


def load_nedlvs_catalog(path: Path | str | None = None) -> pd.DataFrame:
    """Load the NED-LVS FITS table for cross-matching.

    Returns columns ``RA``, ``DEC``, ``objname``, ``objtype``, ``z``,
    ``DistMpc``, ``Diam_arcsec``, ``Mstar``, ``SFR_hybrid``, and ``SFR_W4``.
    Rows with non-finite ``RA``/``DEC`` are dropped.
    """
    catalog_path = Path(NEDLVS_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = f"NED-LVS catalog not found: {catalog_path}"
        raise FileNotFoundError(msg)

    table = Table.read(catalog_path)
    missing = [col for col in NEDLVS_LOAD_COLUMNS if col not in table.colnames]
    if missing:
        msg = f"NED-LVS FITS missing expected columns: {missing}"
        raise ValueError(msg)

    sub = table[NEDLVS_LOAD_COLUMNS]
    df = sub.to_pandas()
    df = df.rename(columns={"ra": "RA", "dec": "DEC"})

    ra = pd.to_numeric(df["RA"], errors="coerce")
    dec = pd.to_numeric(df["DEC"], errors="coerce")
    diam = pd.to_numeric(df["Diam"], errors="coerce").to_numpy(dtype=float)
    df["Diam_arcsec"] = diam
    df = df.drop(columns=["Diam"])

    ok = np.isfinite(ra.to_numpy(dtype=float)) & np.isfinite(dec.to_numpy(dtype=float))
    return df.loc[ok].reset_index(drop=True)


def select_unique_nedlvs_matches(meta_flags: pd.DataFrame) -> pd.DataFrame:
    """Return meta flags with exactly one NED-LVS association."""
    if meta_flags.empty:
        return meta_flags.copy()
    return meta_flags.loc[meta_flags["n_nedlvs"] == 1].copy()


def select_bijective_nedlvs_flags(
    meta_flags: pd.DataFrame,
    nedlvs_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Return NED-LVS flags participating in a 1↔1 meta association."""
    if meta_flags.empty or nedlvs_flags.empty or "meta_ids" not in nedlvs_flags.columns:
        return nedlvs_flags.iloc[0:0].copy()

    unique_meta_ids = set(select_unique_nedlvs_matches(meta_flags)["meta_id"].tolist())

    def _is_bijective(row: pd.Series) -> bool:
        if int(row.get("n_meta", 0)) != 1:
            return False
        meta_ids = row.get("meta_ids", [])
        return len(meta_ids) == 1 and meta_ids[0] in unique_meta_ids

    return nedlvs_flags.loc[nedlvs_flags.apply(_is_bijective, axis=1)].copy()


def _footprint_filter_nedlvs(nedlvs: pd.DataFrame, lwa: pd.DataFrame) -> pd.DataFrame:
    """Keep NED-LVS rows whose Dec lies within the finite Dec range of *lwa*.

    Uses ``match_DEC`` when present and finite, else ``DEC``.
    """
    if nedlvs.empty:
        return nedlvs.copy()
    if lwa.empty:
        return nedlvs.iloc[0:0].copy()

    if "match_DEC" in lwa.columns:
        match_dec = pd.to_numeric(lwa["match_DEC"], errors="coerce")
        native_dec = (
            pd.to_numeric(lwa["DEC"], errors="coerce")
            if "DEC" in lwa.columns
            else pd.Series(np.nan, index=lwa.index)
        )
        lwa_dec = match_dec.where(np.isfinite(match_dec.to_numpy(dtype=float)), native_dec)
    elif "DEC" in lwa.columns:
        lwa_dec = pd.to_numeric(lwa["DEC"], errors="coerce")
    else:
        return nedlvs.iloc[0:0].copy()

    finite_lwa = lwa_dec[np.isfinite(lwa_dec.to_numpy(dtype=float))]
    if finite_lwa.empty:
        return nedlvs.iloc[0:0].copy()

    dec_min = float(finite_lwa.min())
    dec_max = float(finite_lwa.max())
    nedlvs_dec = pd.to_numeric(nedlvs["DEC"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(nedlvs_dec) & (nedlvs_dec >= dec_min) & (nedlvs_dec <= dec_max)
    return nedlvs.loc[keep].copy()


def _filter_nedlvs_redshift(
    nedlvs: pd.DataFrame,
    max_redshift: float | None,
) -> pd.DataFrame:
    if max_redshift is None or nedlvs.empty:
        return nedlvs.copy()
    z = pd.to_numeric(nedlvs["z"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(z) & (z >= 0.0) & (z <= float(max_redshift))
    return nedlvs.loc[keep].copy()


def _select_lwa_target(lwa_catalog: pd.DataFrame, target: NedlvsTarget) -> pd.DataFrame:
    if target in {"metacatalog", "lst_merged_blue"}:
        return lwa_catalog
    if target == "metacatalog_blue":
        return select_blue_associated_rows(lwa_catalog)
    msg = f"unsupported target: {target!r}"
    raise ValueError(msg)


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_nedlvs_footprint": 0,
        "n_meta_matched": 0,
        "meta_nedlvs_fraction": float("nan"),
        "n_meta_unique_nedlvs": 0,
        "meta_unique_nedlvs_fraction": float("nan"),
        "n_nedlvs_matched": 0,
        "nedlvs_recovery": float("nan"),
        "n_nedlvs_oversplit": 0,
        "n_meta_multi_nedlvs": 0,
        "meta_nedlvs_hits_max": 0,
    }


def _skycoord_from_columns(df: pd.DataFrame) -> SkyCoord:
    return SkyCoord(
        ra=df["RA"].to_numpy(dtype=float) * u.deg,
        dec=df["DEC"].to_numpy(dtype=float) * u.deg,
    )


def _associate_by_centroid_sigma(
    base_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    *,
    position_sigma_scale: float,
    diam_scale: float | None = None,
) -> tuple[dict[int, list[int]], set[int]]:
    """Match catalogs using scaled centroid σ, with optional galaxy Diam floor.

    Pairwise limit (degrees)::

        max(
            position_sigma_scale * hypot(σ_base, σ_ref),
            diam_scale * Diam_arcsec / 3600   # when Diam present and diam_scale set
        )

    ``DIAM_ARCSEC`` may appear on either frame (typically the NED-LVS side).
    """
    if base_df.empty or ref_df.empty:
        return {}, set()

    base_ra = base_df["RA"].to_numpy(dtype=float)
    base_dec = base_df["DEC"].to_numpy(dtype=float)
    ref_ra = ref_df["RA"].to_numpy(dtype=float)
    ref_dec = ref_df["DEC"].to_numpy(dtype=float)
    base_ok = np.isfinite(base_ra) & np.isfinite(base_dec)
    ref_ok = np.isfinite(ref_ra) & np.isfinite(ref_dec)
    if not base_ok.any() or not ref_ok.any():
        return {}, set()

    base_idx = np.flatnonzero(base_ok)
    ref_idx = np.flatnonzero(ref_ok)
    base_work = base_df.iloc[base_idx]
    ref_work = ref_df.iloc[ref_idx]

    base_sc = _skycoord_from_columns(base_work)
    ref_sc = _skycoord_from_columns(ref_work)
    sigma_base = np.nan_to_num(
        base_work["SIGMA"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    sigma_ref = np.nan_to_num(
        ref_work["SIGMA"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )

    def _diam_arcsec(work: pd.DataFrame) -> np.ndarray:
        if "DIAM_ARCSEC" not in work.columns:
            return np.zeros(len(work), dtype=float)
        return np.nan_to_num(
            pd.to_numeric(work["DIAM_ARCSEC"], errors="coerce").to_numpy(dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    diam_base = _diam_arcsec(base_work)
    diam_ref = _diam_arcsec(ref_work)

    pos_cap = float(
        position_sigma_scale
        * max(float(sigma_base.max()), float(sigma_ref.max()), 1e-12)
    )
    diam_cap = 0.0
    if diam_scale is not None and float(diam_scale) > 0.0:
        diam_cap = float(diam_scale) * max(
            float(diam_base.max()) if len(diam_base) else 0.0,
            float(diam_ref.max()) if len(diam_ref) else 0.0,
            0.0,
        ) / 3600.0
    search_radius = max(pos_cap, diam_cap, 1e-12) * u.deg
    idx_ref_w, idx_base_w, sep2d, _ = base_sc.search_around_sky(ref_sc, search_radius)
    if len(idx_base_w) == 0:
        return {}, set()

    sep_deg = sep2d.to(u.deg).value
    limits = position_sigma_scale * np.hypot(
        sigma_base[idx_base_w],
        sigma_ref[idx_ref_w],
    )
    if diam_scale is not None and float(diam_scale) > 0.0:
        diam_floor = (
            float(diam_scale)
            * np.maximum(diam_base[idx_base_w], diam_ref[idx_ref_w])
            / 3600.0
        )
        limits = np.maximum(limits, diam_floor)
    keep = sep_deg <= limits
    idx_base_w = idx_base_w[keep]
    idx_ref_w = idx_ref_w[keep]

    hits_by_base: dict[int, list[int]] = {}
    matched: set[int] = set()
    for iw, jw in zip(idx_base_w.tolist(), idx_ref_w.tolist(), strict=True):
        i = int(base_idx[iw])
        j = int(ref_idx[jw])
        hits_by_base.setdefault(i, []).append(j)
        matched.add(j)
    return hits_by_base, matched


def _resolve_sfr(row: pd.Series) -> tuple[float, str]:
    for col in ("SFR_hybrid", "SFR_W4"):
        if col not in row.index:
            continue
        val = pd.to_numeric(row[col], errors="coerce")
        if np.isfinite(val) and float(val) > 0.0:
            return float(val), col
    return float("nan"), ""


def build_sfr_radio_luminosity_table(
    metacatalog: pd.DataFrame,
    nedlvs_footprint: pd.DataFrame,
    meta_flags: pd.DataFrame,
    *,
    unique_only: bool = True,
) -> pd.DataFrame:
    """Pair cross-matched rows and compute radio ``nu*L_nu`` vs NED-LVS SFR.

    When *unique_only* is ``True`` (default), only meta rows with exactly one
    NED-LVS match are included.
    """
    empty_cols = [
        "meta_id",
        "objname",
        "z",
        "SFR",
        "SFR_column",
        "radio_band",
        "radio_freq_hz",
        "Flux_jy",
        "L_nu_erg_s_hz",
        "nuL_nu_erg_s",
    ]
    if meta_flags.empty or "meta_id" not in meta_flags.columns:
        return pd.DataFrame(columns=empty_cols)

    flags = select_unique_nedlvs_matches(meta_flags) if unique_only else meta_flags
    meta_by_id = metacatalog.set_index("meta_id", drop=False)
    records: list[dict] = []
    for _, flag in flags.iterrows():
        if not bool(flag.get("matched", False)):
            continue
        positions = flag.get("nedlvs_positions", [])
        if not positions:
            continue
        if unique_only and len(positions) != 1:
            continue

        meta_id = flag["meta_id"]
        if meta_id not in meta_by_id.index:
            continue
        meta_row = meta_by_id.loc[meta_id]
        if isinstance(meta_row, pd.DataFrame):
            meta_row = meta_row.iloc[0]

        ned_pos = int(positions[0])
        ned_row = nedlvs_footprint.iloc[ned_pos]

        flux, freq_hz, band = resolve_highest_frequency_flux(meta_row, prefer_total=True)
        z = pd.to_numeric(ned_row.get("z"), errors="coerce")
        sfr, sfr_col = _resolve_sfr(ned_row)
        if not np.isfinite(flux) or not np.isfinite(freq_hz) or not np.isfinite(z) or z <= 0:
            continue
        if not np.isfinite(sfr) or sfr <= 0:
            continue

        l_nu, nu_l_nu = radio_luminosity_nu(flux, float(z), freq_hz)
        records.append(
            {
                "meta_id": meta_id,
                "objname": ned_row.get("objname", ""),
                "z": float(z),
                "SFR": sfr,
                "SFR_column": sfr_col,
                "radio_band": band,
                "radio_freq_hz": freq_hz,
                "Flux_jy": flux,
                "L_nu_erg_s_hz": float(l_nu),
                "nuL_nu_erg_s": float(nu_l_nu),
            }
        )

    return pd.DataFrame.from_records(records)


def match_catalog_to_nedlvs(
    lwa_catalog: pd.DataFrame,
    nedlvs: pd.DataFrame | None = None,
    *,
    config: NedlvsMatchConfig | None = None,
) -> NedlvsMatchResult:
    """Cross-match an LWA catalog against NED-LVS and compute QA metrics.

    Matching uses metacatalog centroid uncertainty scaled by
    ``config.position_sigma_scale``, combined in quadrature with a small NED
    catalog position sigma. When ``config.diam_scale`` is set, the pairwise
    limit is also floored at ``diam_scale * Diam_arcsec``. Coordinates prefer
    ``match_RA``/``match_DEC`` when present (cascaded radio best position),
    else top-level ``RA``/``DEC``. Uncertainties prefer ``match_sigma_deg`` or
    survey defaults for ``match_source``, else LWA ``E_RA``/``E_DEC`` /
    ``cluster_jitter_rms_deg``. NED-LVS rows above ``config.max_redshift`` are
    excluded when that limit is set.
    """
    cfg = config or NedlvsMatchConfig()
    warnings: list[str] = []

    if nedlvs is None:
        nedlvs = load_nedlvs_catalog(cfg.catalog_path)

    target = _select_lwa_target(lwa_catalog, cfg.target)
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty after band selection")
        return NedlvsMatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=[
                    "meta_id",
                    "RA",
                    "DEC",
                    "match_RA",
                    "match_DEC",
                    "centroid_sigma_deg",
                    "n_nedlvs",
                    "nedlvs_positions",
                    "matched",
                ]
            ),
            nedlvs_flags=pd.DataFrame(
                columns=[
                    "nedlvs_pos",
                    "RA",
                    "DEC",
                    "objname",
                    "DistMpc",
                    "z",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            nedlvs_footprint=pd.DataFrame(),
            warnings=warnings,
        )

    nedlvs_footprint = _footprint_filter_nedlvs(nedlvs, target)
    n_before_z = len(nedlvs_footprint)
    nedlvs_footprint = _filter_nedlvs_redshift(nedlvs_footprint, cfg.max_redshift)
    if cfg.max_redshift is not None and len(nedlvs_footprint) < n_before_z:
        warnings.append(
            f"redshift filter z <= {cfg.max_redshift}: "
            f"{n_before_z - len(nedlvs_footprint)} NED-LVS rows removed from footprint"
        )

    lwa_match = _centroid_match_frame(target, default_centroid_sigma_deg=cfg.default_centroid_sigma_deg)
    ned_match = _nedlvs_match_frame(nedlvs_footprint, position_sigma_deg=cfg.nedlvs_position_sigma_deg)
    n_nedlvs_footprint = len(nedlvs_footprint)

    meta_hits: dict[int, list[int]] = {}
    nedlvs_hits: dict[int, list[int]] = {}
    if not lwa_match.empty and not ned_match.empty:
        meta_hits, _ = _associate_by_centroid_sigma(
            lwa_match,
            ned_match,
            position_sigma_scale=cfg.position_sigma_scale,
            diam_scale=cfg.diam_scale,
        )
        nedlvs_hits, _ = _associate_by_centroid_sigma(
            ned_match,
            lwa_match,
            position_sigma_scale=cfg.position_sigma_scale,
            diam_scale=cfg.diam_scale,
        )

    index_to_match_pos = {idx: pos for pos, idx in enumerate(lwa_match.index.tolist())}
    match_pos_to_index = {pos: idx for idx, pos in index_to_match_pos.items()}
    has_meta_id = "meta_id" in target.columns

    meta_records: list[dict] = []
    for idx, row in target.iterrows():
        match_pos = index_to_match_pos.get(idx)
        hit_nedlvs = meta_hits.get(match_pos, []) if match_pos is not None else []
        n_nedlvs = len(hit_nedlvs)
        match_ra, match_dec = resolve_match_coordinates(row)
        sigma = (
            float(lwa_match.loc[idx, "SIGMA"])
            if match_pos is not None and idx in lwa_match.index
            else resolve_centroid_sigma_deg(
                row, default_centroid_sigma_deg=cfg.default_centroid_sigma_deg
            )
        )
        record: dict = {
            "RA": row.get("RA", np.nan),
            "DEC": row.get("DEC", np.nan),
            "match_RA": match_ra,
            "match_DEC": match_dec,
            "centroid_sigma_deg": sigma,
            "n_nedlvs": n_nedlvs,
            "nedlvs_positions": list(hit_nedlvs),
            "matched": n_nedlvs >= 1,
        }
        if "match_source" in target.columns:
            record["match_source"] = row.get("match_source", "LWA")
        if "meta_id" in target.columns:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    if "meta_id" in meta_flags.columns:
        cols = [
            "meta_id",
            "RA",
            "DEC",
            "match_RA",
            "match_DEC",
            "centroid_sigma_deg",
            "n_nedlvs",
            "nedlvs_positions",
            "matched",
        ]
        if "match_source" in meta_flags.columns:
            cols.insert(6, "match_source")
        meta_flags = meta_flags[cols]

    nedlvs_records: list[dict] = []
    for pos in range(len(nedlvs_footprint)):
        row = nedlvs_footprint.iloc[pos]
        hit_meta = nedlvs_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                idx = match_pos_to_index.get(match_pos)
                if idx is not None:
                    meta_ids.append(target.loc[idx, "meta_id"])
        nedlvs_records.append(
            {
                "nedlvs_pos": pos,
                "RA": row["RA"],
                "DEC": row["DEC"],
                "objname": row.get("objname", ""),
                "DistMpc": row.get("DistMpc", np.nan),
                "z": row.get("z", np.nan),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    nedlvs_cols = [
        "nedlvs_pos",
        "RA",
        "DEC",
        "objname",
        "DistMpc",
        "z",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        nedlvs_cols = [c for c in nedlvs_cols if c != "meta_ids"]
    if nedlvs_records:
        nedlvs_flags = pd.DataFrame(nedlvs_records)[nedlvs_cols]
    else:
        nedlvs_flags = pd.DataFrame(columns=nedlvs_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_meta_unique = int((meta_flags["n_nedlvs"] == 1).sum()) if not meta_flags.empty else 0
    n_nedlvs_matched = int((nedlvs_flags["n_meta"] > 0).sum()) if not nedlvs_flags.empty else 0
    n_nedlvs_oversplit = int(nedlvs_flags["oversplit"].sum()) if not nedlvs_flags.empty else 0
    n_meta_multi_nedlvs = int((meta_flags["n_nedlvs"] > 1).sum()) if not meta_flags.empty else 0
    meta_nedlvs_hits_max = int(meta_flags["n_nedlvs"].max()) if not meta_flags.empty else 0

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_nedlvs_footprint": n_nedlvs_footprint,
        "n_meta_matched": n_meta_matched,
        "meta_nedlvs_fraction": n_meta_matched / n_lwa_target,
        "n_meta_unique_nedlvs": n_meta_unique,
        "meta_unique_nedlvs_fraction": n_meta_unique / n_lwa_target,
        "n_nedlvs_matched": n_nedlvs_matched,
        "nedlvs_recovery": (
            n_nedlvs_matched / n_nedlvs_footprint if n_nedlvs_footprint else float("nan")
        ),
        "n_nedlvs_oversplit": n_nedlvs_oversplit,
        "n_meta_multi_nedlvs": n_meta_multi_nedlvs,
        "meta_nedlvs_hits_max": meta_nedlvs_hits_max,
    }

    return NedlvsMatchResult(
        summary=summary,
        meta_flags=meta_flags,
        nedlvs_flags=nedlvs_flags,
        nedlvs_footprint=nedlvs_footprint.reset_index(drop=True),
        warnings=warnings,
    )


def _centroid_match_frame(
    catalog: pd.DataFrame,
    *,
    default_centroid_sigma_deg: float,
) -> pd.DataFrame:
    """Build ``RA`` / ``DEC`` / ``SIGMA`` (1σ degrees) for centroid matching.

    Prefers ``match_RA``/``match_DEC`` when present and finite.
    """
    records: list[dict[str, float]] = []
    indices: list[object] = []
    for idx, row in catalog.iterrows():
        ra, dec = resolve_match_coordinates(row)
        if not np.isfinite(ra) or not np.isfinite(dec):
            continue
        sigma = resolve_centroid_sigma_deg(
            row, default_centroid_sigma_deg=default_centroid_sigma_deg
        )
        records.append({"RA": ra, "DEC": dec, "SIGMA": sigma})
        indices.append(idx)

    if not records:
        return pd.DataFrame(columns=["RA", "DEC", "SIGMA"])
    return pd.DataFrame(records, index=indices)


def _nedlvs_match_frame(
    nedlvs: pd.DataFrame,
    *,
    position_sigma_deg: float,
) -> pd.DataFrame:
    if nedlvs.empty:
        return pd.DataFrame(columns=["RA", "DEC", "SIGMA", "DIAM_ARCSEC"])
    diam = (
        pd.to_numeric(nedlvs["Diam_arcsec"], errors="coerce")
        if "Diam_arcsec" in nedlvs.columns
        else pd.Series(np.nan, index=nedlvs.index)
    )
    return pd.DataFrame(
        {
            "RA": pd.to_numeric(nedlvs["RA"], errors="coerce"),
            "DEC": pd.to_numeric(nedlvs["DEC"], errors="coerce"),
            "SIGMA": float(position_sigma_deg),
            "DIAM_ARCSEC": diam,
        }
    )


def summarize_nedlvs_match(result: NedlvsMatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:                {int(s['n_lwa_target']):6d}",
        f"NED-LVS footprint (Dec box):    {int(s['n_nedlvs_footprint']):6d}",
        f"Meta matched (>=1 NED-LVS):     {int(s['n_meta_matched']):6d}",
        f"Meta with NED-LVS host:         {s['meta_nedlvs_fraction']:.3f}",
        f"Meta unique match (n=1):        {int(s['n_meta_unique_nedlvs']):6d}",
        f"Meta unique fraction:           {s['meta_unique_nedlvs_fraction']:.3f}",
        f"NED-LVS matched (>=1 meta):     {int(s['n_nedlvs_matched']):6d}",
        f"NED-LVS recovery:               {s['nedlvs_recovery']:.3f}",
        f"NED-LVS over-split (n_meta>1):  {int(s['n_nedlvs_oversplit']):6d}",
        f"Meta multi-NED-LVS (n>1):       {int(s['n_meta_multi_nedlvs']):6d}",
        f"Max NED-LVS hits per meta:      {int(s['meta_nedlvs_hits_max']):6d}",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)
