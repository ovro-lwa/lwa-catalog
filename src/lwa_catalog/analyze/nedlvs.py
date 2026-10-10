"""NED-LVS cross-match QA against LWA metacatalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck18
from astropy.table import Table

from lwa_catalog.analyze.crossmatch_radius import (
    NEDLVS_LWA_RADIUS,
    NEDLVS_REFERENCE_RADIUS_FIXED,
    CrossmatchRadiusSpec,
    apply_match_radius,
    catalog_match_frame,
    describe_crossmatch_radius,
    resolve_localization_sigma_deg,
    resolve_match_coordinates,
)
from lwa_catalog.analyze.vlssr import select_blue_associated_rows
from lwa_catalog.constants import (
    NEDLVS_DEFAULT_CENTROID_SIGMA_DEG,
    NEDLVS_DEFAULT_DIAM_SCALE,
    NEDLVS_DEFAULT_MAX_REDSHIFT,
    NEDLVS_DEFAULT_PATH,
    band_frequency_hz,
)
from lwa_catalog.create.merge import associate_catalogs
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

@dataclass(frozen=True)
class NedlvsMatchConfig:
    """Configuration for :func:`match_catalog_to_nedlvs`.

    Radii use the same :class:`~lwa_catalog.analyze.crossmatch_radius.CrossmatchRadiusSpec`
    pattern as radio / X-ray / MilliQUAS matchers. ``diam_scale`` is NED-LVS-only:
    after applying ``reference_radius``, reference ``BMAJ`` is floored at
    ``diam_scale * Diam_arcsec`` (degrees).
    """

    catalog_path: Path = NEDLVS_DEFAULT_PATH
    target: NedlvsTarget = "metacatalog"
    lwa_radius: CrossmatchRadiusSpec = NEDLVS_LWA_RADIUS
    reference_radius: CrossmatchRadiusSpec = NEDLVS_REFERENCE_RADIUS_FIXED
    # Floor reference match radius as ``diam_scale * Diam_arcsec``. ``None`` disables.
    diam_scale: float | None = NEDLVS_DEFAULT_DIAM_SCALE
    max_redshift: float | None = NEDLVS_DEFAULT_MAX_REDSHIFT


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
        if "quality_flag" not in metacatalog.columns:
            msg = "metacatalog missing quality_flag for selection='quality_all_clear'"
            raise ValueError(msg)
        clear = metacatalog["quality_flag"].to_numpy(dtype=np.uint32) == 0
        out = metacatalog.loc[clear]
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

    Thin wrapper around
    :func:`~lwa_catalog.analyze.crossmatch_radius.resolve_localization_sigma_deg`
    (``match_sigma_deg`` → ``match_source`` → ``E_RA``/``E_DEC`` → jitter →
    default).
    """
    return resolve_localization_sigma_deg(
        row, default_deg=float(default_centroid_sigma_deg)
    )


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


def apply_diam_scale_floor(
    catalog: pd.DataFrame,
    *,
    diam_scale: float | None,
    diam_col: str = "Diam_arcsec",
) -> pd.DataFrame:
    """Floor ``BMAJ`` (degrees) at ``diam_scale * Diam_arcsec``.

    Used on the NED-LVS reference match frame after
    :func:`~lwa_catalog.analyze.crossmatch_radius.apply_match_radius`. ``None``
    or non-positive *diam_scale* leaves *catalog* unchanged (still copied).
    """
    out = catalog.copy()
    if out.empty or diam_scale is None or float(diam_scale) <= 0.0:
        return out
    if "BMAJ" not in out.columns:
        out["BMAJ"] = 0.0
    if diam_col in out.columns:
        diam = pd.to_numeric(out[diam_col], errors="coerce").to_numpy(dtype=float)
    else:
        diam = np.zeros(len(out), dtype=float)
    floor_deg = float(diam_scale) * np.nan_to_num(diam, nan=0.0, posinf=0.0, neginf=0.0) / 3600.0
    bmaj = pd.to_numeric(out["BMAJ"], errors="coerce").to_numpy(dtype=float)
    bmaj = np.nan_to_num(bmaj, nan=0.0, posinf=0.0, neginf=0.0)
    out["BMAJ"] = np.maximum(bmaj, floor_deg)
    return out


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


def _lwa_match_frame(
    catalog: pd.DataFrame,
    spec: CrossmatchRadiusSpec,
) -> pd.DataFrame:
    """Build ``RA`` / ``DEC`` / ``BMAJ`` using cascaded ``match_*`` coords when present."""
    return catalog_match_frame(catalog, spec, prefer_match_coordinates=True)


def match_catalog_to_nedlvs(
    lwa_catalog: pd.DataFrame,
    nedlvs: pd.DataFrame | None = None,
    *,
    config: NedlvsMatchConfig | None = None,
    lwa_match: pd.DataFrame | None = None,
) -> NedlvsMatchResult:
    """Cross-match an LWA catalog against NED-LVS and compute QA metrics.

    Matching uses ``associate_catalogs`` with configured
    :class:`~lwa_catalog.analyze.crossmatch_radius.CrossmatchRadiusSpec`
    radii (``sep ≤ max(r_lwa, r_ned)``). Defaults are Nσ localization on the
    LWA side and a fixed NED catalog position uncertainty on the reference
    side. When ``config.diam_scale`` is set, the reference ``BMAJ`` is floored
    at ``diam_scale * Diam_arcsec``. Coordinates prefer ``match_RA`` /
    ``match_DEC`` when present (cascaded radio best position), else top-level
    ``RA``/``DEC``. Localization prefers ``match_sigma_deg`` or survey defaults
    for ``match_source``, else LWA ``E_RA``/``E_DEC`` /
    ``cluster_jitter_rms_deg``. NED-LVS rows above ``config.max_redshift`` are
    excluded when that limit is set.
    """
    cfg = config or NedlvsMatchConfig()
    warnings: list[str] = []
    warnings.append(
        "radii: LWA="
        + describe_crossmatch_radius(cfg.lwa_radius)
        + "; NED-LVS="
        + describe_crossmatch_radius(cfg.reference_radius)
        + (
            f"; diam_scale={cfg.diam_scale:g}"
            if cfg.diam_scale is not None
            else "; diam_scale=None"
        )
    )

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

    if lwa_match is None:
        resolved_lwa_match = _lwa_match_frame(target, cfg.lwa_radius)
    else:
        if len(lwa_match) != len(target):
            msg = (
                f"lwa_match length {len(lwa_match)} does not match target "
                f"length {len(target)}"
            )
            raise ValueError(msg)
        resolved_lwa_match = lwa_match[["RA", "DEC", "BMAJ"]].copy()
        resolved_lwa_match.index = target.index

    ref_match = apply_match_radius(nedlvs_footprint, cfg.reference_radius)
    ref_match = apply_diam_scale_floor(ref_match, diam_scale=cfg.diam_scale)
    n_nedlvs_footprint = len(nedlvs_footprint)

    meta_hits: dict[int, list[int]] = {}
    nedlvs_hits: dict[int, list[int]] = {}
    if not resolved_lwa_match.empty and not ref_match.empty:
        meta_hits, _ = associate_catalogs(resolved_lwa_match, ref_match)
        nedlvs_hits, _ = associate_catalogs(ref_match, resolved_lwa_match)

    index_to_match_pos = {
        idx: pos for pos, idx in enumerate(resolved_lwa_match.index.tolist())
    }
    match_pos_to_index = {pos: idx for idx, pos in index_to_match_pos.items()}
    has_meta_id = "meta_id" in target.columns
    default_sigma = (
        float(cfg.lwa_radius.default_arcsec) / 3600.0
        if cfg.lwa_radius.default_arcsec is not None
        else NEDLVS_DEFAULT_CENTROID_SIGMA_DEG
    )

    meta_records: list[dict] = []
    for idx, row in target.iterrows():
        match_pos = index_to_match_pos.get(idx)
        hit_nedlvs = meta_hits.get(match_pos, []) if match_pos is not None else []
        n_nedlvs = len(hit_nedlvs)
        match_ra, match_dec = resolve_match_coordinates(row)
        sigma = resolve_centroid_sigma_deg(row, default_centroid_sigma_deg=default_sigma)
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
