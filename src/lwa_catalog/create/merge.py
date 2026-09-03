"""Fuse per-image catalogs into LST-band and global metacatalogs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from lwa_catalog.constants import (
    ASSOC_BANDS,
    BAND_FIELDS,
    BAND_FREQ_HZ,
    CLUSTER_JITTER_RMS_COL,
    COLOR_BANDS,
    GAUL_STRING_COLUMNS,
    LST_MERGE_QA_COLUMNS,
    OVRO_LATITUDE_DEG,
    SPECTRAL_INDEX_PAIRS,
    SUBBAND_METACATALOG_FLUX_FIELDS,
    band_frequency_hz,
)
from lwa_catalog.coords import normalize_ra_columns
from lwa_catalog.gaul import cast_s_code_value


def _lst_hour_to_deg(lst_hour: str | float | int) -> float:
    """Convert an ``NNh`` (or numeric hour) label to RA degrees at that LST."""
    if isinstance(lst_hour, str):
        text = lst_hour.strip().lower().rstrip("h")
        hour = float(text)
    else:
        hour = float(lst_hour)
    return hour * 15.0


def _elevation_deg(
    ra_deg: float,
    dec_deg: float,
    lst_hour: str | float | int,
    *,
    latitude_deg: float = OVRO_LATITUDE_DEG,
) -> float:
    """Source elevation (degrees) for a zenith-pointed observation at *lst_hour*."""
    zenith = SkyCoord(
        ra=_lst_hour_to_deg(lst_hour) * u.deg,
        dec=latitude_deg * u.deg,
    )
    source = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    return float(90.0 - zenith.separation(source).deg)


def _lst_label_column(df: pd.DataFrame) -> str:
    """Column name for LST hour labels (``lst_hour`` or ``representative_lst``)."""
    if "lst_hour" in df.columns:
        return "lst_hour"
    if "representative_lst" in df.columns:
        return "representative_lst"
    msg = "DataFrame needs lst_hour or representative_lst for elevation ranking"
    raise KeyError(msg)


def _pick_highest_elevation_row(
    df: pd.DataFrame,
    *,
    latitude_deg: float = OVRO_LATITUDE_DEG,
) -> pd.Series:
    """Return the row where the source is highest above the OVRO horizon.

    Elevation is evaluated at each detection's ``lst_hour`` (or
    ``representative_lst``) assuming a zenith-centered pointing (zenith at
    RA = LST, Dec = *latitude_deg*). Cluster median RA/DEC are used so
    position jitter does not flip the pick.
    """
    if len(df) == 1:
        return df.iloc[0]
    ra = float(np.nanmedian(df["RA"].to_numpy(dtype=float)))
    dec = float(np.nanmedian(df["DEC"].to_numpy(dtype=float)))
    lst_col = _lst_label_column(df)
    elev = np.asarray(
        [_elevation_deg(ra, dec, lst, latitude_deg=latitude_deg) for lst in df[lst_col].to_numpy()],
        dtype=float,
    )
    return df.iloc[int(np.nanargmax(elev))]


def _pick_peak_flux_row(df: pd.DataFrame) -> pd.Series:
    """Return the row with the highest finite flux.

    Prefers ``Total_flux`` when any positive values exist, otherwise
    ``Peak_flux`` / ``Peak_intensity``.
    """
    if len(df) == 1:
        return df.iloc[0]

    for col in ("Total_flux", "Peak_flux", "Peak_intensity"):
        if col not in df.columns:
            continue
        flux = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(flux).any() and np.nanmax(flux) > 0.0:
            return df.iloc[int(np.nanargmax(flux))]
    return df.iloc[0]


def _pick_associated_row(
    df: pd.DataFrame,
    representative: Literal["elevation", "peak_flux"],
) -> pd.Series:
    if representative == "peak_flux":
        return _pick_peak_flux_row(df)
    return _pick_highest_elevation_row(df)


def source_elevation_deg(
    ra_deg: float,
    dec_deg: float,
    lst_hour: str | float | int,
    *,
    latitude_deg: float = OVRO_LATITUDE_DEG,
) -> float:
    """Source elevation (degrees) for a zenith-centered observation at *lst_hour*."""
    return _elevation_deg(ra_deg, dec_deg, lst_hour, latitude_deg=latitude_deg)


def catalog_elevation_deg(
    catalog: pd.DataFrame,
    *,
    latitude_deg: float = OVRO_LATITUDE_DEG,
) -> np.ndarray:
    """Return source elevation (degrees) for each row in a per-LST catalog."""
    lst_col = _lst_label_column(catalog)
    ra = catalog["RA"].to_numpy(dtype=float)
    dec = catalog["DEC"].to_numpy(dtype=float)
    lst = catalog[lst_col].to_numpy()
    zenith_ra = np.array([_lst_hour_to_deg(lh) for lh in lst]) * u.deg
    zenith_dec = np.full(len(catalog), latitude_deg) * u.deg
    sources = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    zenith = SkyCoord(ra=zenith_ra, dec=zenith_dec)
    return 90.0 - zenith.separation(sources).deg


def pick_highest_elevation_row(
    df: pd.DataFrame,
    *,
    latitude_deg: float = OVRO_LATITUDE_DEG,
) -> pd.Series:
    """Public alias of :func:`_pick_highest_elevation_row` for rematch/analyze."""
    return _pick_highest_elevation_row(df, latitude_deg=latitude_deg)


def associate_catalogs(
    base_df: pd.DataFrame,
    band_df: pd.DataFrame,
) -> tuple[dict[int, list[int]], set[int]]:
    """Public alias of :func:`_associate_catalogs` for rematch/analyze."""
    return _associate_catalogs(base_df, band_df)


def _lst_cluster_qa_fields(members: pd.DataFrame, rep: pd.Series) -> dict[str, float | str]:
    """QA metrics from true LST cluster members and the representative row."""
    from lwa_catalog.analyze.reliability import cluster_radec_jitter_rms

    out: dict[str, float | str] = {CLUSTER_JITTER_RMS_COL: cluster_radec_jitter_rms(members)}
    for col in LST_MERGE_QA_COLUMNS:
        if col == CLUSTER_JITTER_RMS_COL:
            continue
        if col not in rep.index:
            continue
        if col in GAUL_STRING_COLUMNS:
            out[col] = cast_s_code_value(rep[col])
            continue
        try:
            val = float(rep[col])
        except (TypeError, ValueError):
            val = float("nan")
        out[col] = val
    return out


def _copy_lst_merge_qa(entry: dict, band_row: pd.Series) -> None:
    """Copy merge-time QA columns from an LST-merged band row into *entry*."""
    for col in LST_MERGE_QA_COLUMNS:
        if col not in band_row.index:
            continue
        if col in GAUL_STRING_COLUMNS:
            entry[col] = cast_s_code_value(band_row[col])
        else:
            entry[col] = band_row[col]


def _copy_lst_astrometry_qa(entry: dict, band_row: pd.Series) -> None:
    """Copy non-flux LST QA from the astrometry band (residuals, jitter)."""
    for col in (CLUSTER_JITTER_RMS_COL, "Resid_Isl_rms", "Resid_Isl_mean"):
        if col in band_row.index:
            entry[col] = band_row[col]


def _flux_std(df: pd.DataFrame, flux_col: str = "Peak_flux") -> float:
    """Sample standard deviation of *flux_col* over cluster members (NaN if <2)."""
    flux = df[flux_col].to_numpy(dtype=float)
    finite = flux[np.isfinite(flux)]
    if finite.size < 2:
        return float("nan")
    return float(np.std(finite, ddof=1))


def _skycoord_from_columns(df: pd.DataFrame) -> SkyCoord:
    return SkyCoord(
        ra=df["RA"].to_numpy(dtype=float) * u.deg,
        dec=df["DEC"].to_numpy(dtype=float) * u.deg,
    )


def _associate_catalogs(
    base_df: pd.DataFrame,
    band_df: pd.DataFrame,
) -> tuple[dict[int, list[int]], set[int]]:
    """Vectorized base↔band matching within per-pair beam radii.

    Rows with non-finite ``RA``/``DEC`` are skipped. Returned indices refer to
    the original *base_df* / *band_df* row positions (``iloc``).

    Returns ``{base_iloc: [band_iloc, ...]}`` and the set of matched band ilocs.
    """
    if base_df.empty or band_df.empty:
        return {}, set()

    base_ra = base_df["RA"].to_numpy(dtype=float)
    base_dec = base_df["DEC"].to_numpy(dtype=float)
    band_ra = band_df["RA"].to_numpy(dtype=float)
    band_dec = band_df["DEC"].to_numpy(dtype=float)
    base_ok = np.isfinite(base_ra) & np.isfinite(base_dec)
    band_ok = np.isfinite(band_ra) & np.isfinite(band_dec)
    if not base_ok.any() or not band_ok.any():
        return {}, set()

    base_idx = np.flatnonzero(base_ok)
    band_idx = np.flatnonzero(band_ok)
    base_work = base_df.iloc[base_idx]
    band_work = band_df.iloc[band_idx]

    base_sc = _skycoord_from_columns(base_work)
    band_sc = _skycoord_from_columns(band_work)
    bmaj_base = np.nan_to_num(
        base_work["BMAJ"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    bmaj_band = np.nan_to_num(
        band_work["BMAJ"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )

    search_radius = float(max(float(bmaj_base.max()), float(bmaj_band.max()), 0.0)) * u.deg
    # Astropy returns (idx_searcharound, idx_self) = (band_work, base_work)
    idx_band_w, idx_base_w, sep2d, _ = base_sc.search_around_sky(band_sc, search_radius)
    if len(idx_base_w) == 0:
        return {}, set()

    sep_deg = sep2d.to(u.deg).value
    limits = np.maximum(bmaj_base[idx_base_w], bmaj_band[idx_band_w])
    keep = sep_deg <= limits
    idx_base_w = idx_base_w[keep]
    idx_band_w = idx_band_w[keep]

    hits_by_base: dict[int, list[int]] = {}
    matched: set[int] = set()
    for iw, jw in zip(idx_base_w.tolist(), idx_band_w.tolist(), strict=True):
        i = int(base_idx[iw])
        j = int(band_idx[jw])
        hits_by_base.setdefault(i, []).append(j)
        matched.add(j)
    return hits_by_base, matched


def _union_find_roots(n: int, pairs_a: np.ndarray, pairs_b: np.ndarray) -> np.ndarray:
    """Return root id for each node after unioning undirected *pairs_* edges."""
    parent = np.arange(n, dtype=np.intp)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for a, b in zip(pairs_a.tolist(), pairs_b.tolist(), strict=True):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    return np.asarray([find(i) for i in range(n)], dtype=np.intp)


def _cluster_by_sky_position(
    df: pd.DataFrame,
    *,
    bmaj_col: str = "BMAJ",
) -> list[pd.DataFrame]:
    """Beam-sized clustering via ``search_around_sky`` + connected components.

    Detections are linked when their separation is within ``max(BMAJ_i, BMAJ_j)``.
    Transitive links form one cluster (order-independent; no running centroid).
    """
    if df.empty:
        return []

    work = df[np.isfinite(df["RA"]) & np.isfinite(df["DEC"])].reset_index(drop=True)
    if work.empty:
        return []
    if len(work) == 1:
        return [work]

    if bmaj_col in work.columns:
        bmaj = np.nan_to_num(work[bmaj_col].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        bmaj = np.zeros(len(work), dtype=float)
    coords = _skycoord_from_columns(work)
    search_radius = float(max(float(bmaj.max()), 0.0)) * u.deg
    idx_a, idx_b, sep2d, _ = coords.search_around_sky(coords, search_radius)

    if len(idx_a) == 0:
        return [work.iloc[[i]].copy() for i in range(len(work))]

    sep_deg = sep2d.to(u.deg).value
    limits = np.maximum(bmaj[idx_a], bmaj[idx_b])
    keep = (idx_a < idx_b) & (sep_deg <= limits)
    idx_a = idx_a[keep]
    idx_b = idx_b[keep]

    roots = _union_find_roots(len(work), idx_a, idx_b)
    clusters: list[pd.DataFrame] = []
    for root in np.unique(roots):
        members = np.flatnonzero(roots == root)
        clusters.append(work.iloc[members].copy())
    return clusters


def merge_lst_metacatalog(catalogs: Iterable[pd.DataFrame], *, band: str) -> pd.DataFrame:
    """Fuse per-LST detections within one band → one row per source."""
    frames = list(catalogs)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    combined = normalize_ra_columns(combined)

    rows: list[dict] = []
    for members in _cluster_by_sky_position(combined):
        rep = _pick_highest_elevation_row(members)
        entry = rep.to_dict()
        entry["band"] = band
        entry["n_lst_contributions"] = len(members)
        entry["lst_hours"] = ",".join(sorted(members["lst_hour"].astype(str).unique()))
        entry["representative_lst"] = rep["lst_hour"]
        entry["Peak_flux_std"] = _flux_std(members)
        entry.update(_lst_cluster_qa_fields(members, rep))
        rows.append(entry)

    meta = pd.DataFrame(rows)
    return meta.sort_values("Peak_flux", ascending=False, na_position="last").reset_index(drop=True)


def _empty_band_cols(
    *,
    assoc_bands: Sequence[str] = ASSOC_BANDS,
    band_fields: Sequence[str] = BAND_FIELDS,
) -> dict:
    out: dict = {}
    for band in assoc_bands:
        for field in band_fields:
            out[f"{field}_{band}"] = np.nan
        out[f"n_assoc_{band}"] = 0
    return out


def _lst_meta_from_band_row(row: pd.Series) -> tuple[int, str, str, float]:
    peak_std = row.get("Peak_flux_std", np.nan)
    try:
        peak_std_f = float(peak_std)
    except (TypeError, ValueError):
        peak_std_f = float("nan")
    return (
        int(row.get("n_lst_contributions", 1)),
        str(row.get("lst_hours", row.get("lst_hour", ""))),
        str(row.get("representative_lst", row.get("lst_hour", ""))),
        peak_std_f,
    )


def _shape_fields_from_band(row: pd.Series) -> dict:
    return {
        "RA": row["RA"],
        "DEC": row["DEC"],
        "Maj": row["Maj"],
        "Min": row["Min"],
        "PA": row["PA"],
        "DC_Maj": row.get("DC_Maj", np.nan),
        "DC_Min": row.get("DC_Min", np.nan),
        "DC_PA": row.get("DC_PA", np.nan),
    }


def _primary_fields_from_band(row: pd.Series, *, include_flux: bool = True) -> dict:
    out = _shape_fields_from_band(row)
    if include_flux:
        out["Peak_flux"] = row["Peak_flux"]
        out["Total_flux"] = row["Total_flux"]
    return out


def _lookup_band_frequency_hz(
    band: str,
    band_freq_hz: Mapping[str, float],
) -> float:
    if band in band_freq_hz:
        return float(band_freq_hz[band])
    return band_frequency_hz(band)


def _maybe_update_astrometry_from_band(
    entry: dict,
    band_row: pd.Series,
    band: str,
    *,
    band_freq_hz: Mapping[str, float],
) -> None:
    """Set top-level astrometry from *band_row* when *band* is the highest frequency so far."""
    new_hz = _lookup_band_frequency_hz(band, band_freq_hz)
    current_band = str(entry.get("astrometry_band", "") or "")
    current_hz = (
        _lookup_band_frequency_hz(current_band, band_freq_hz) if current_band else float("-inf")
    )
    if new_hz >= current_hz:
        entry.update(_shape_fields_from_band(band_row))
        entry["astrometry_band"] = band
        _copy_lst_astrometry_qa(entry, band_row)


def _representative_peak_flux(
    meta: pd.DataFrame,
    bands: Sequence[str],
) -> pd.Series:
    """Max ``Peak_flux_{band}`` across *bands* for sorting subband metacatalogs."""
    cols = [f"Peak_flux_{band}" for band in bands if f"Peak_flux_{band}" in meta.columns]
    if not cols:
        return pd.Series(np.nan, index=meta.index, dtype=float)
    return meta[cols].max(axis=1)


def _attach_band_columns(
    entry: dict,
    band_row: pd.Series,
    band: str,
    n_assoc: int,
    *,
    band_fields: Sequence[str] = BAND_FIELDS,
) -> None:
    entry[f"n_assoc_{band}"] = n_assoc
    for field in band_fields:
        entry[f"{field}_{band}"] = band_row.get(field, np.nan)
    entry[f"source_file_{band}"] = band_row.get("source_file", "")


def add_spectral_indices(
    meta: pd.DataFrame,
    *,
    band_freq_hz: Mapping[str, float] = BAND_FREQ_HZ,
    flux_prefix: str = "Total_flux",
    err_prefix: str = "E_Total_flux",
    pairs: Sequence[tuple[str, str, str]] = SPECTRAL_INDEX_PAIRS,
) -> pd.DataFrame:
    """Add two-point spectral indices from associated band ``Total_flux`` columns.

    For each ``(label, band_a, band_b)`` in *pairs*::

        alpha = log(S_a / S_b) / log(ν_a / ν_b)

    with optional Gaussian error propagation into ``E_alpha_{label}`` when both
    ``E_Total_flux_{band}`` columns are present. Rows lacking two positive finite
    fluxes get NaN.
    """
    if meta.empty:
        out = meta.copy()
        for label, _, _ in pairs:
            out[f"alpha_{label}"] = pd.Series(dtype=float)
            out[f"E_alpha_{label}"] = pd.Series(dtype=float)
        return out

    out = meta.copy()
    n = len(out)
    for label, band_a, band_b in pairs:
        col_a = f"{flux_prefix}_{band_a}"
        col_b = f"{flux_prefix}_{band_b}"
        alpha = np.full(n, np.nan, dtype=float)
        e_alpha = np.full(n, np.nan, dtype=float)

        if col_a not in out.columns or col_b not in out.columns:
            out[f"alpha_{label}"] = alpha
            out[f"E_alpha_{label}"] = e_alpha
            continue

        s_a = out[col_a].to_numpy(dtype=float)
        s_b = out[col_b].to_numpy(dtype=float)
        nu_a = float(band_freq_hz[band_a])
        nu_b = float(band_freq_hz[band_b])
        valid = np.isfinite(s_a) & np.isfinite(s_b) & (s_a > 0.0) & (s_b > 0.0)
        if valid.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha[valid] = np.log(s_a[valid] / s_b[valid]) / np.log(nu_a / nu_b)

        err_a_col = f"{err_prefix}_{band_a}"
        err_b_col = f"{err_prefix}_{band_b}"
        if err_a_col in out.columns and err_b_col in out.columns:
            e_a = out[err_a_col].to_numpy(dtype=float)
            e_b = out[err_b_col].to_numpy(dtype=float)
            err_ok = valid & np.isfinite(e_a) & np.isfinite(e_b) & (e_a >= 0.0) & (e_b >= 0.0)
            ln_nu = abs(np.log(nu_a / nu_b))
            if err_ok.any() and ln_nu > 0.0:
                e_alpha[err_ok] = (
                    np.sqrt((e_a[err_ok] / s_a[err_ok]) ** 2 + (e_b[err_ok] / s_b[err_ok]) ** 2)
                    / ln_nu
                )

        out[f"alpha_{label}"] = alpha
        out[f"E_alpha_{label}"] = e_alpha
    return out


def _update_bands_present(
    entry: dict,
    *bands: str,
    color_bands: Sequence[str] = COLOR_BANDS,
) -> None:
    original = [b for b in str(entry.get("bands_present", "")).split(",") if b]
    present = set(original)
    present.update(bands)
    ordered: list[str] = []
    seen: set[str] = set()

    def _append(seq: Sequence[str]) -> None:
        for band in seq:
            if band in present and band not in seen:
                ordered.append(band)
                seen.add(band)

    _append(color_bands)
    _append(original)
    _append(bands)
    entry["bands_present"] = ",".join(ordered)


def _seed_row_from_band(
    band_row: pd.Series,
    band: str,
    *,
    assoc_bands: Sequence[str] = ASSOC_BANDS,
    band_fields: Sequence[str] = BAND_FIELDS,
    seed_band: str = "Full",
    primary_flux: bool = True,
    astrometry_from_highest_frequency: bool = False,
    band_freq_hz: Mapping[str, float] | None = None,
) -> dict:
    """One metacatalog row seeded from a single-band LST-merged detection."""
    n_lst, lst_hours, rep_lst, peak_std = _lst_meta_from_band_row(band_row)
    entry = {
        "origin_band": band,
        "bands_present": band,
        **_primary_fields_from_band(band_row, include_flux=primary_flux),
        "BMAJ_match": float(band_row["BMAJ"]),
        "n_lst_contributions": n_lst,
        "lst_hours": lst_hours,
        "representative_lst": rep_lst,
    }
    if primary_flux:
        entry["Peak_flux_std"] = peak_std
    entry.update(_empty_band_cols(assoc_bands=assoc_bands, band_fields=band_fields))
    if band == seed_band:
        entry["BMAJ_full"] = float(band_row["BMAJ"])
        entry[f"source_file_{band}"] = band_row.get("source_file", "")
        for field in band_fields:
            entry[f"{field}_{band}"] = band_row.get(field, np.nan)
    else:
        entry["BMAJ_full"] = np.nan
        _attach_band_columns(entry, band_row, band, 1, band_fields=band_fields)
    if astrometry_from_highest_frequency:
        if band_freq_hz is None:
            msg = "band_freq_hz is required when astrometry_from_highest_frequency=True"
            raise ValueError(msg)
        _maybe_update_astrometry_from_band(entry, band_row, band, band_freq_hz=band_freq_hz)
    else:
        _copy_lst_merge_qa(entry, band_row)
    return entry


def _merge_build_kwargs(
    *,
    assoc_bands: Sequence[str],
    band_fields: Sequence[str],
    seed_band: str,
    primary_flux: bool,
    astrometry_from_highest_frequency: bool,
    band_freq_hz: Mapping[str, float],
) -> dict:
    return {
        "assoc_bands": assoc_bands,
        "band_fields": band_fields,
        "seed_band": seed_band,
        "primary_flux": primary_flux,
        "astrometry_from_highest_frequency": astrometry_from_highest_frequency,
        "band_freq_hz": band_freq_hz,
    }


def merge_full_and_blue(
    seed_df: pd.DataFrame,
    assoc_df: pd.DataFrame,
    *,
    seed_band: str = "Full",
    assoc_band: str = "Blue",
    assoc_bands: Sequence[str] = ASSOC_BANDS,
    band_fields: Sequence[str] = BAND_FIELDS,
    color_bands: Sequence[str] = COLOR_BANDS,
    primary_flux: bool = True,
    astrometry_from_highest_frequency: bool = False,
    band_freq_hz: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Cross-match *assoc_band* onto *seed_band*; one row per deduplicated sky position."""
    seed_df = seed_df.reset_index(drop=True)
    assoc_df = assoc_df.reset_index(drop=True)
    hits, matched_assoc = _associate_catalogs(seed_df, assoc_df)
    if band_freq_hz is None:
        band_freq_hz = BAND_FREQ_HZ
    seed_kw = _merge_build_kwargs(
        assoc_bands=assoc_bands,
        band_fields=band_fields,
        seed_band=seed_band,
        primary_flux=primary_flux,
        astrometry_from_highest_frequency=astrometry_from_highest_frequency,
        band_freq_hz=band_freq_hz,
    )

    rows: list[dict] = []
    for i, srow in seed_df.iterrows():
        entry = _seed_row_from_band(srow, seed_band, **seed_kw)
        assoc_hits = hits.get(i, [])
        if assoc_hits:
            sub = assoc_df.iloc[assoc_hits]
            best = _pick_highest_elevation_row(sub)
            _attach_band_columns(entry, best, assoc_band, len(assoc_hits), band_fields=band_fields)
            if astrometry_from_highest_frequency:
                _maybe_update_astrometry_from_band(
                    entry, best, assoc_band, band_freq_hz=band_freq_hz
                )
            _update_bands_present(entry, seed_band, assoc_band, color_bands=color_bands)
            entry["BMAJ_match"] = max(float(entry["BMAJ_match"]), float(best["BMAJ"]))
        rows.append(entry)

    for j, arow in assoc_df.iterrows():
        if j not in matched_assoc:
            rows.append(_seed_row_from_band(arow, assoc_band, **seed_kw))
    return pd.DataFrame(rows)


def associate_band_into_metacatalog(
    meta_df: pd.DataFrame,
    band_df: pd.DataFrame,
    band: str,
    *,
    assoc_bands: Sequence[str] = ASSOC_BANDS,
    band_fields: Sequence[str] = BAND_FIELDS,
    color_bands: Sequence[str] = COLOR_BANDS,
    seed_band: str = "Full",
    primary_flux: bool = True,
    astrometry_from_highest_frequency: bool = False,
    band_freq_hz: Mapping[str, float] | None = None,
    append_unmatched: bool = True,
    update_bmaj_match: bool = True,
    representative: Literal["elevation", "peak_flux"] = "elevation",
    base_bmaj: np.ndarray | None = None,
    base_ra: np.ndarray | None = None,
    base_dec: np.ndarray | None = None,
) -> pd.DataFrame:
    """Cross-match one band onto the current metacatalog.

    By default unmatched *band_df* rows are appended as new metacatalog seeds.
    Photometric survey attach uses ``append_unmatched=False``,
    ``update_bmaj_match=False``, and ``representative="peak_flux"``.

    Optional *base_ra* / *base_dec* / *base_bmaj* override the match-frame
    coordinates and radii without mutating stored metacatalog astrometry.
    """
    band_df = band_df.reset_index(drop=True)
    if band_freq_hz is None:
        band_freq_hz = BAND_FREQ_HZ
    seed_kw = _merge_build_kwargs(
        assoc_bands=assoc_bands,
        band_fields=band_fields,
        seed_band=seed_band,
        primary_flux=primary_flux,
        astrometry_from_highest_frequency=astrometry_from_highest_frequency,
        band_freq_hz=band_freq_hz,
    )
    empty_cols = _empty_band_cols(assoc_bands=(band,), band_fields=band_fields)
    if meta_df.empty:
        if append_unmatched:
            return pd.DataFrame(
                [_seed_row_from_band(brow, band, **seed_kw) for _, brow in band_df.iterrows()]
            )
        out = meta_df.copy()
        for col, value in empty_cols.items():
            out[col] = pd.Series(dtype=type(value) if not isinstance(value, float) else float)
        return out

    meta_df = meta_df.reset_index(drop=True)
    n_meta = len(meta_df)
    if base_ra is not None:
        ra = np.asarray(base_ra, dtype=float)
        if ra.shape != (n_meta,):
            msg = f"base_ra length {ra.size} does not match metacatalog rows {n_meta}"
            raise ValueError(msg)
    else:
        ra = meta_df["RA"].to_numpy(dtype=float)
    if base_dec is not None:
        dec = np.asarray(base_dec, dtype=float)
        if dec.shape != (n_meta,):
            msg = f"base_dec length {dec.size} does not match metacatalog rows {n_meta}"
            raise ValueError(msg)
    else:
        dec = meta_df["DEC"].to_numpy(dtype=float)
    match_base = pd.DataFrame({"RA": ra, "DEC": dec})
    if base_bmaj is not None:
        radii = np.asarray(base_bmaj, dtype=float)
        if radii.shape != (n_meta,):
            msg = f"base_bmaj length {radii.size} does not match metacatalog rows {n_meta}"
            raise ValueError(msg)
        match_base["BMAJ"] = radii
    else:
        match_base["BMAJ"] = meta_df["BMAJ_match"].to_numpy(dtype=float)
    hits, matched_band = _associate_catalogs(match_base, band_df)

    rows: list[dict] = []
    for i, mrow in meta_df.iterrows():
        entry = mrow.to_dict()
        for col, value in empty_cols.items():
            entry.setdefault(col, value)
        band_hits = hits.get(i, [])
        if band_hits:
            sub = band_df.iloc[band_hits]
            best = _pick_associated_row(sub, representative)
            _attach_band_columns(entry, best, band, len(band_hits), band_fields=band_fields)
            if astrometry_from_highest_frequency:
                _maybe_update_astrometry_from_band(entry, best, band, band_freq_hz=band_freq_hz)
            _update_bands_present(entry, band, color_bands=color_bands)
            if update_bmaj_match:
                entry["BMAJ_match"] = max(float(entry["BMAJ_match"]), float(best["BMAJ"]))
        rows.append(entry)

    if append_unmatched:
        for j, brow in band_df.iterrows():
            if j not in matched_band:
                rows.append(_seed_row_from_band(brow, band, **seed_kw))
    return pd.DataFrame(rows)


def build_global_metacatalog(
    lst_merged: dict[str, pd.DataFrame],
    *,
    seed_band: str = "Full",
    assoc_bands: Sequence[str] = ASSOC_BANDS,
    band_fields: Sequence[str] = BAND_FIELDS,
    color_bands: Sequence[str] = COLOR_BANDS,
    band_freq_hz: Mapping[str, float] = BAND_FREQ_HZ,
    spectral_index_pairs: Sequence[tuple[str, str, str]] = SPECTRAL_INDEX_PAIRS,
    primary_flux: bool = True,
    astrometry_from_highest_frequency: bool = False,
) -> pd.DataFrame:
    """Fuse LST-merged per-band catalogs via sequential cross-matching.

    Parameters
    ----------
    lst_merged
        Mapping of band name → LST-merged catalog DataFrame. Must include
        *seed_band* and each name in *assoc_bands*.
    seed_band
        Band that seeds metacatalog rows (default ``Full``). Remaining bands
        in *assoc_bands* are associated onto these rows in order.
    assoc_bands
        Bands associated onto the seed (default Blue, Green, Red).
    band_fields
        Per-band measurement columns copied into ``{field}_{band}``.
    color_bands
        Canonical band order for ``bands_present``.
    band_freq_hz
        Rest-frame center frequencies (Hz) keyed by band name. Override for
        frequency-labeled subbands (e.g. ``18MHz``).
    spectral_index_pairs
        ``(label, band_a, band_b)`` triples forwarded to
        :func:`add_spectral_indices`.
    primary_flux
        When true (default), copy seed-band ``Peak_flux`` / ``Total_flux`` to
        top-level columns. Subband metacatalogs set this false.
    astrometry_from_highest_frequency
        When true, set top-level astrometry from the highest-frequency band
        present on each row (requires *band_freq_hz*).
    """
    merge_kw = _merge_build_kwargs(
        assoc_bands=assoc_bands,
        band_fields=band_fields,
        seed_band=seed_band,
        primary_flux=primary_flux,
        astrometry_from_highest_frequency=astrometry_from_highest_frequency,
        band_freq_hz=band_freq_hz,
    )
    build_kw = {
        **merge_kw,
        "color_bands": color_bands,
    }
    if not assoc_bands:
        temp = pd.DataFrame(
            [
                _seed_row_from_band(row, seed_band, **merge_kw)
                for _, row in lst_merged[seed_band].iterrows()
            ]
        )
    else:
        first_assoc = assoc_bands[0]
        temp = merge_full_and_blue(
            lst_merged[seed_band],
            lst_merged[first_assoc],
            assoc_band=first_assoc,
            **build_kw,
        )
        for band in assoc_bands[1:]:
            temp = associate_band_into_metacatalog(
                temp,
                lst_merged[band],
                band,
                **build_kw,
            )
    meta = add_spectral_indices(temp, band_freq_hz=band_freq_hz, pairs=spectral_index_pairs)
    meta = normalize_ra_columns(meta)
    meta.insert(0, "meta_id", range(len(meta)))
    if primary_flux:
        sort_col = "Peak_flux"
    else:
        meta["_sort_peak_flux"] = _representative_peak_flux(meta, color_bands)
        sort_col = "_sort_peak_flux"
    meta = meta.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    if not primary_flux:
        meta = meta.drop(columns=["_sort_peak_flux"])
    return meta


def build_subband_metacatalog(
    lst_merged: dict[str, pd.DataFrame],
    *,
    seed_band: str,
    assoc_bands: Sequence[str],
    color_bands: Sequence[str],
    band_freq_hz: Mapping[str, float],
    band_fields: Sequence[str] = SUBBAND_METACATALOG_FLUX_FIELDS,
) -> pd.DataFrame:
    """Fuse MHz-subband LST catalogs into a flux-only, wide metacatalog.

    Top-level ``RA``/``DEC``/shape come from the highest-frequency subband
    present on each row. Flux is stored only in ``{field}_{subband}`` columns.
    No merge-time spectral indices (``alpha_*``); use post-hoc spectral modeling
    in ``lwa_catalog.analyze`` instead.
    """
    return build_global_metacatalog(
        lst_merged,
        seed_band=seed_band,
        assoc_bands=assoc_bands,
        band_fields=band_fields,
        color_bands=color_bands,
        band_freq_hz=band_freq_hz,
        spectral_index_pairs=(),
        primary_flux=False,
        astrometry_from_highest_frequency=True,
    )
