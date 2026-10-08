"""eROSITA eRASS:3 Main cross-match against LWA metacatalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from lwa_catalog.analyze.crossmatch_radius import (
    CrossmatchRadiusSpec,
    ERASS3_REFERENCE_RADIUS_LOCALIZATION,
    LWA_CROSSMATCH_RADIUS_BEAM,
    apply_match_radius,
    catalog_match_frame,
)
from lwa_catalog.constants import (
    ERASS3_DEFAULT_PATH,
    ERASS3_GLON_MAX_DEG,
    ERASS3_GLON_MIN_DEG,
    ERASS3_POSITION_ERROR_DEFAULT_ARCSEC,
)
from lwa_catalog.create.merge import associate_catalogs

Erass3Target = Literal["metacatalog"]

# Columns kept for match / classification (user-selected subset of Main+LS10).
ERASS3_LOAD_COLUMNS: tuple[str, ...] = (
    "RA",
    "DEC",
    "POS_ERR",
    "EXT",
    "EXT_ERR",
    "DET_LIKE_0",
    "ML_FLUX_1",
    "ML_FLUX_ERR_1",
    "LS10_OBJID",
    "NWAY_p_i",
    "main_id_simbad",
    "simbad_known_galactic",
    "is_blazar_in_simbad",
    "class_gal_exgal",
)

# Columns copied onto unique-match metacatalog rows (prefixed ``eRASS3_``).
ERASS3_ATTACH_COLUMNS: tuple[str, ...] = ERASS3_LOAD_COLUMNS


@dataclass(frozen=True)
class Erass3MatchConfig:
    """Configuration for :func:`match_catalog_to_erass3`."""

    catalog_path: Path = ERASS3_DEFAULT_PATH
    target: Erass3Target = "metacatalog"
    glon_min_deg: float = ERASS3_GLON_MIN_DEG
    glon_max_deg: float = ERASS3_GLON_MAX_DEG
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM
    reference_radius: CrossmatchRadiusSpec = ERASS3_REFERENCE_RADIUS_LOCALIZATION


@dataclass
class Erass3MatchResult:
    """eRASS:3 cross-match metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    erass3_flags: pd.DataFrame
    erass3_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def galactic_longitude_deg(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Return Galactic longitude in degrees ``[0, 360)`` for ICRS positions."""
    ra_arr = np.asarray(ra, dtype=float)
    dec_arr = np.asarray(dec, dtype=float)
    out = np.full(ra_arr.shape, np.nan, dtype=float)
    ok = np.isfinite(ra_arr) & np.isfinite(dec_arr)
    if not np.any(ok):
        return out
    coords = SkyCoord(ra=ra_arr[ok] * u.deg, dec=dec_arr[ok] * u.deg, frame="icrs")
    out[ok] = coords.galactic.l.wrap_at(360 * u.deg).deg
    return out


def in_erass3_footprint(
    ra: np.ndarray | pd.Series,
    dec: np.ndarray | pd.Series,
    *,
    glon_min_deg: float = ERASS3_GLON_MIN_DEG,
    glon_max_deg: float = ERASS3_GLON_MAX_DEG,
) -> np.ndarray:
    """True where Galactic longitude is in ``[glon_min_deg, glon_max_deg)``.

    eRASS:3 Main (German share) covers the western Galactic hemisphere
    (``180° ≤ l < 360°``), which is **not** the same as ``180 < RA < 360``.
    """
    glon = galactic_longitude_deg(np.asarray(ra, dtype=float), np.asarray(dec, dtype=float))
    return (
        np.isfinite(glon)
        & (glon >= float(glon_min_deg))
        & (glon < float(glon_max_deg))
    )


def _decode_object_column(series: pd.Series) -> pd.Series:
    """Decode FITS bytes / object columns to pandas strings."""
    if series.dtype == object or str(series.dtype).startswith("|S"):
        return series.map(
            lambda v: (
                v.decode("utf-8", errors="replace").strip()
                if isinstance(v, (bytes, bytearray))
                else ("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
            )
        )
    if hasattr(series.dtype, "kind") and series.dtype.kind == "S":
        return series.astype(str).str.strip()
    return series


def _finalize_erass3_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep match/classification columns and normalize dtypes / coordinates."""
    missing = [c for c in ("RA", "DEC") if c not in df.columns]
    if missing:
        msg = f"eRASS:3 table missing required columns: {missing}"
        raise ValueError(msg)

    keep = [c for c in ERASS3_LOAD_COLUMNS if c in df.columns]
    out = df.loc[:, keep].copy()

    for col in (
        "RA",
        "DEC",
        "POS_ERR",
        "EXT",
        "EXT_ERR",
        "DET_LIKE_0",
        "ML_FLUX_1",
        "ML_FLUX_ERR_1",
        "NWAY_p_i",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "LS10_OBJID" in out.columns:
        out["LS10_OBJID"] = pd.to_numeric(out["LS10_OBJID"], errors="coerce")
    if "class_gal_exgal" in out.columns:
        out["class_gal_exgal"] = pd.to_numeric(out["class_gal_exgal"], errors="coerce")

    for col in ("main_id_simbad",):
        if col in out.columns:
            out[col] = _decode_object_column(out[col])

    for col in ("simbad_known_galactic", "is_blazar_in_simbad"):
        if col in out.columns:
            out[col] = out[col].astype("boolean")

    # Localization matcher uses E_RA/E_DEC (degrees). POS_ERR is 1σ radial (arcsec).
    pos_err_arcsec = (
        pd.to_numeric(out["POS_ERR"], errors="coerce")
        if "POS_ERR" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    pos_err_deg = pos_err_arcsec.to_numpy(dtype=float) / 3600.0
    fallback_deg = ERASS3_POSITION_ERROR_DEFAULT_ARCSEC / 3600.0
    pos_err_deg = np.where(np.isfinite(pos_err_deg) & (pos_err_deg > 0.0), pos_err_deg, fallback_deg)
    # hypot(E_RA, E_DEC) == POS_ERR when both axes share the radial σ / √2.
    axis_sigma = pos_err_deg / np.sqrt(2.0)
    out["E_RA"] = axis_sigma
    out["E_DEC"] = axis_sigma

    ra = out["RA"].to_numpy(dtype=float)
    dec = out["DEC"].to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return out.loc[ok].reset_index(drop=True)


def load_erass3_catalog(path: Path | str | None = None) -> pd.DataFrame:
    """Load eRASS:3 Main (+ LS10) columns needed for cross-match / classification.

    Only :data:`ERASS3_LOAD_COLUMNS` are retained. ``POS_ERR`` (arcsec) is
    converted into ``E_RA``/``E_DEC`` (degrees) for localization matching.
    Rows with non-finite ``RA``/``DEC`` are dropped.

    Parameters
    ----------
    path
        Catalog FITS path. Defaults to
        :data:`~lwa_catalog.constants.ERASS3_DEFAULT_PATH`.
    """
    catalog_path = Path(ERASS3_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = f"eRASS:3 catalog not found: {catalog_path}"
        raise FileNotFoundError(msg)

    from astropy.io import fits
    from astropy.table import Table

    # Read only the requested columns (full Main+LS10 table is ~1.8 GB / 189 cols).
    with fits.open(catalog_path, memmap=True) as hdul:
        data = hdul[1].data
        names = set(data.dtype.names or ())
        missing = [c for c in ERASS3_LOAD_COLUMNS if c not in names]
        if missing:
            msg = f"eRASS:3 table missing columns: {missing}"
            raise ValueError(msg)
        table = Table({c: data[c] for c in ERASS3_LOAD_COLUMNS})
    df = table.to_pandas()
    return _finalize_erass3_frame(df)


def select_unique_erass3_matches(meta_flags: pd.DataFrame) -> pd.DataFrame:
    """Return meta rows with exactly one eRASS:3 match (``n_erass3 == 1``)."""
    if meta_flags.empty or "n_erass3" not in meta_flags.columns:
        return meta_flags.iloc[0:0].copy()
    return meta_flags.loc[meta_flags["n_erass3"] == 1].copy()


def _catalog_match_frame(
    catalog: pd.DataFrame,
    spec: CrossmatchRadiusSpec,
) -> pd.DataFrame:
    return catalog_match_frame(catalog, spec)


def _footprint_mask_lwa(
    lwa: pd.DataFrame,
    *,
    glon_min_deg: float,
    glon_max_deg: float,
) -> np.ndarray:
    if lwa.empty or "RA" not in lwa.columns or "DEC" not in lwa.columns:
        return np.zeros(len(lwa), dtype=bool)
    return in_erass3_footprint(
        lwa["RA"],
        lwa["DEC"],
        glon_min_deg=glon_min_deg,
        glon_max_deg=glon_max_deg,
    )


def _footprint_filter_erass3(
    erass3: pd.DataFrame,
    *,
    glon_min_deg: float,
    glon_max_deg: float,
) -> pd.DataFrame:
    """Keep eRASS:3 rows inside the western Galactic hemisphere footprint."""
    if erass3.empty:
        return erass3.copy()
    keep = in_erass3_footprint(
        erass3["RA"],
        erass3["DEC"],
        glon_min_deg=glon_min_deg,
        glon_max_deg=glon_max_deg,
    )
    return erass3.loc[keep].reset_index(drop=True)


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_lwa_in_footprint": 0,
        "n_erass3_footprint": 0,
        "n_meta_matched": 0,
        "match_completeness": float("nan"),
        "n_meta_unique": 0,
        "unique_match_fraction": float("nan"),
        "n_erass3_matched": 0,
        "erass3_recovery": float("nan"),
        "n_erass3_oversplit": 0,
        "n_meta_multi_erass3": 0,
        "meta_erass3_hits_max": 0,
    }


def match_catalog_to_erass3(
    lwa_catalog: pd.DataFrame,
    erass3: pd.DataFrame | None = None,
    *,
    config: Erass3MatchConfig | None = None,
    lwa_match: pd.DataFrame | None = None,
) -> Erass3MatchResult:
    """Cross-match an LWA catalog against eRASS:3 Main and compute association metrics.

    Only LWA rows inside the western Galactic hemisphere
    (``config.glon_min_deg`` ≤ *l* < ``config.glon_max_deg``) are matched.
    Completeness uses that footprint subset as the denominator. Eastern-hemisphere
    LWA rows keep ``in_erass_footprint=False`` and ``n_erass3=0``.

    Matching uses ``associate_catalogs`` with configured
    :class:`~lwa_catalog.analyze.crossmatch_radius.CrossmatchRadiusSpec` radii.
    """
    cfg = config or Erass3MatchConfig()
    warnings: list[str] = []

    if erass3 is None:
        erass3 = load_erass3_catalog(cfg.catalog_path)

    target = lwa_catalog
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty")
        return Erass3MatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=[
                    "meta_id",
                    "RA",
                    "DEC",
                    "in_erass_footprint",
                    "n_erass3",
                    "erass3_positions",
                    "matched",
                ]
            ),
            erass3_flags=pd.DataFrame(
                columns=[
                    "erass3_pos",
                    "RA",
                    "DEC",
                    "DET_LIKE_0",
                    "ML_FLUX_1",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            erass3_footprint=pd.DataFrame(),
            warnings=warnings,
        )

    footprint_mask = _footprint_mask_lwa(
        target,
        glon_min_deg=cfg.glon_min_deg,
        glon_max_deg=cfg.glon_max_deg,
    )
    n_lwa_in_footprint = int(footprint_mask.sum())
    target_in = target.loc[footprint_mask].copy()

    erass3_footprint = _footprint_filter_erass3(
        erass3,
        glon_min_deg=cfg.glon_min_deg,
        glon_max_deg=cfg.glon_max_deg,
    )
    ref_match = apply_match_radius(erass3_footprint, cfg.reference_radius)

    if lwa_match is None:
        resolved_lwa_match = _catalog_match_frame(target_in, cfg.lwa_radius)
    else:
        if len(lwa_match) != len(target):
            msg = (
                f"lwa_match length {len(lwa_match)} does not match target "
                f"length {len(target)}"
            )
            raise ValueError(msg)
        # Restrict cascaded frame to footprint rows (positional; keep labels).
        resolved_lwa_match = lwa_match.iloc[footprint_mask][["RA", "DEC", "BMAJ"]].copy()
        resolved_lwa_match.index = target_in.index

    n_erass3_footprint = len(erass3_footprint)

    meta_hits: dict[int, list[int]] = {}
    erass3_hits: dict[int, list[int]] = {}
    if not resolved_lwa_match.empty and not ref_match.empty:
        meta_hits, _ = associate_catalogs(resolved_lwa_match, ref_match)
        erass3_hits, _ = associate_catalogs(ref_match, resolved_lwa_match)

    index_to_match_pos = {
        idx: pos for pos, idx in enumerate(resolved_lwa_match.index.tolist())
    }
    match_pos_to_index = {pos: idx for idx, pos in index_to_match_pos.items()}
    has_meta_id = "meta_id" in target.columns
    footprint_by_index = pd.Series(footprint_mask, index=target.index)

    meta_records: list[dict] = []
    for idx, row in target.iterrows():
        in_fp = bool(footprint_by_index.loc[idx])
        match_pos = index_to_match_pos.get(idx) if in_fp else None
        hit_erass3 = meta_hits.get(match_pos, []) if match_pos is not None else []
        n_erass3 = len(hit_erass3)
        record: dict = {
            "RA": row.get("RA", np.nan),
            "DEC": row.get("DEC", np.nan),
            "in_erass_footprint": in_fp,
            "n_erass3": n_erass3,
            "erass3_positions": list(hit_erass3),
            "matched": n_erass3 >= 1,
        }
        if has_meta_id:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    cols = [
        "meta_id",
        "RA",
        "DEC",
        "in_erass_footprint",
        "n_erass3",
        "erass3_positions",
        "matched",
    ]
    if "meta_id" not in meta_flags.columns:
        cols = [c for c in cols if c != "meta_id"]
    if not meta_flags.empty:
        meta_flags = meta_flags[cols]

    erass3_records: list[dict] = []
    for pos in range(len(erass3_footprint)):
        row = erass3_footprint.iloc[pos]
        hit_meta = erass3_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                midx = match_pos_to_index.get(match_pos)
                if midx is not None:
                    meta_ids.append(target.loc[midx, "meta_id"])
        erass3_records.append(
            {
                "erass3_pos": pos,
                "RA": row["RA"],
                "DEC": row["DEC"],
                "DET_LIKE_0": row.get("DET_LIKE_0", np.nan),
                "ML_FLUX_1": row.get("ML_FLUX_1", np.nan),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    erass3_cols = [
        "erass3_pos",
        "RA",
        "DEC",
        "DET_LIKE_0",
        "ML_FLUX_1",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        erass3_cols = [c for c in erass3_cols if c != "meta_ids"]
    if erass3_records:
        erass3_flags = pd.DataFrame(erass3_records)[erass3_cols]
    else:
        erass3_flags = pd.DataFrame(columns=erass3_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_meta_unique = (
        int(((meta_flags["n_erass3"] == 1) & meta_flags["in_erass_footprint"]).sum())
        if not meta_flags.empty
        else 0
    )
    n_erass3_matched = (
        int((erass3_flags["n_meta"] > 0).sum()) if not erass3_flags.empty else 0
    )
    n_erass3_oversplit = (
        int(erass3_flags["oversplit"].sum()) if not erass3_flags.empty else 0
    )
    n_meta_multi_erass3 = (
        int((meta_flags["n_erass3"] > 1).sum()) if not meta_flags.empty else 0
    )
    meta_erass3_hits_max = (
        int(meta_flags["n_erass3"].max()) if not meta_flags.empty else 0
    )

    completeness_den = n_lwa_in_footprint if n_lwa_in_footprint else float("nan")
    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_lwa_in_footprint": n_lwa_in_footprint,
        "n_erass3_footprint": n_erass3_footprint,
        "n_meta_matched": n_meta_matched,
        "match_completeness": (
            n_meta_matched / completeness_den if n_lwa_in_footprint else float("nan")
        ),
        "n_meta_unique": n_meta_unique,
        "unique_match_fraction": (
            n_meta_unique / completeness_den if n_lwa_in_footprint else float("nan")
        ),
        "n_erass3_matched": n_erass3_matched,
        "erass3_recovery": (
            n_erass3_matched / n_erass3_footprint
            if n_erass3_footprint
            else float("nan")
        ),
        "n_erass3_oversplit": n_erass3_oversplit,
        "n_meta_multi_erass3": n_meta_multi_erass3,
        "meta_erass3_hits_max": meta_erass3_hits_max,
    }

    return Erass3MatchResult(
        summary=summary,
        meta_flags=meta_flags,
        erass3_flags=erass3_flags,
        erass3_footprint=erass3_footprint,
        warnings=warnings,
    )


def _pick_erass3_hit(
    erass3_footprint: pd.DataFrame,
    hit_positions: list[int],
    *,
    meta_ra: float,
    meta_dec: float,
) -> int | None:
    """Choose one eRASS hit: highest ``DET_LIKE_0``, then closest on sky."""
    if not hit_positions:
        return None
    if len(hit_positions) == 1:
        return int(hit_positions[0])

    rows = erass3_footprint.iloc[hit_positions]
    det = pd.to_numeric(rows["DET_LIKE_0"], errors="coerce").to_numpy(dtype=float)
    # Prefer finite DET_LIKE; non-finite sorts last.
    det_key = np.where(np.isfinite(det), det, -np.inf)
    best_det = np.max(det_key)
    candidates = [
        hit_positions[i] for i, v in enumerate(det_key) if v == best_det
    ]
    if len(candidates) == 1:
        return int(candidates[0])

    meta = SkyCoord(ra=meta_ra * u.deg, dec=meta_dec * u.deg)
    best_pos = candidates[0]
    best_sep = np.inf
    for pos in candidates:
        row = erass3_footprint.iloc[pos]
        sep = meta.separation(
            SkyCoord(ra=float(row["RA"]) * u.deg, dec=float(row["DEC"]) * u.deg)
        ).deg
        if sep < best_sep:
            best_sep = sep
            best_pos = pos
    return int(best_pos)


def attach_erass3_to_metacatalog(
    meta_df: pd.DataFrame,
    erass3: pd.DataFrame,
    meta_flags: pd.DataFrame,
    *,
    column_prefix: str = "eRASS3_",
) -> pd.DataFrame:
    """Attach eRASS:3 classification columns for matched metacatalog rows.

    Row count is unchanged. Matched rows (``n_erass3 >= 1``) receive the
    chosen counterpart's :data:`ERASS3_ATTACH_COLUMNS` under *column_prefix*,
    plus ``n_erass3``, ``in_erass_footprint``, and ``sep_arcsec_eRASS3``.
    Multi-match rows keep the highest-``DET_LIKE_0`` hit (sky-sep tie-break).
    Unmatched / out-of-footprint rows get NaN / False defaults.
    """
    if len(meta_flags) != len(meta_df):
        msg = (
            f"meta_flags length {len(meta_flags)} does not match meta_df "
            f"length {len(meta_df)}"
        )
        raise ValueError(msg)

    out = meta_df.reset_index(drop=True).copy()
    flags = meta_flags.reset_index(drop=True)
    erass = erass3.reset_index(drop=True)

    out["in_erass_footprint"] = flags["in_erass_footprint"].to_numpy(dtype=bool)
    out["n_erass3"] = pd.to_numeric(flags["n_erass3"], errors="coerce").fillna(0).astype(int)

    for col in ERASS3_ATTACH_COLUMNS:
        out[f"{column_prefix}{col}"] = (
            pd.NA if col in ("main_id_simbad", "simbad_known_galactic", "is_blazar_in_simbad")
            else np.nan
        )
    out["sep_arcsec_eRASS3"] = np.nan

    for i, flag_row in flags.iterrows():
        positions = flag_row.get("erass3_positions", [])
        if not isinstance(positions, (list, tuple)) or not positions:
            continue
        meta_ra = float(pd.to_numeric(flag_row.get("RA"), errors="coerce"))
        meta_dec = float(pd.to_numeric(flag_row.get("DEC"), errors="coerce"))
        if not (np.isfinite(meta_ra) and np.isfinite(meta_dec)):
            continue
        pick = _pick_erass3_hit(
            erass,
            list(positions),
            meta_ra=meta_ra,
            meta_dec=meta_dec,
        )
        if pick is None:
            continue
        src = erass.iloc[pick]
        for col in ERASS3_ATTACH_COLUMNS:
            if col in src.index:
                out.at[i, f"{column_prefix}{col}"] = src[col]
        src_coord = SkyCoord(
            ra=float(src["RA"]) * u.deg, dec=float(src["DEC"]) * u.deg
        )
        meta_coord = SkyCoord(ra=meta_ra * u.deg, dec=meta_dec * u.deg)
        out.at[i, "sep_arcsec_eRASS3"] = float(meta_coord.separation(src_coord).arcsec)

    return out


def summarize_erass3_match(result: Erass3MatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:                  {int(s['n_lwa_target']):6d}",
        f"LWA in eRASS footprint (Gal l):   {int(s['n_lwa_in_footprint']):6d}",
        f"eRASS:3 footprint rows:           {int(s['n_erass3_footprint']):6d}",
        f"Meta matched (>=1 eRASS:3):       {int(s['n_meta_matched']):6d}",
        f"Match completeness (in footprint):{s['match_completeness']:.3f}",
        f"Meta unique match (n=1):          {int(s['n_meta_unique']):6d}",
        f"Unique match fraction:            {s['unique_match_fraction']:.3f}",
        f"eRASS:3 matched (>=1 meta):       {int(s['n_erass3_matched']):6d}",
        f"eRASS:3 recovery:                 {s['erass3_recovery']:.3f}",
        f"eRASS:3 over-split (n_meta>1):    {int(s['n_erass3_oversplit']):6d}",
        f"Meta multi-eRASS:3 (n>1):         {int(s['n_meta_multi_erass3']):6d}",
        f"Max eRASS:3 hits per meta:        {int(s['meta_erass3_hits_max']):6d}",
        "",
        "Footprint: western Galactic hemisphere "
        f"({ERASS3_GLON_MIN_DEG:g}° ≤ l < {ERASS3_GLON_MAX_DEG:g}°).",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)
