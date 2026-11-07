"""Wakely & Horan 2008 TeVCat very-high-energy gamma-ray sources vs LWA."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from lwa_catalog.analyze.crossmatch_radius import (
    LWA_CROSSMATCH_RADIUS_BEAM,
    TEVCAT_REFERENCE_RADIUS_FIXED,
    CrossmatchRadiusSpec,
    apply_match_radius,
    catalog_match_frame,
)
from lwa_catalog.constants import (
    TEVCAT_DEFAULT_EXTENT_ARCMIN,
    TEVCAT_DEFAULT_PATH,
)
from lwa_catalog.create.merge import associate_catalogs

TevcatTarget = Literal["metacatalog"]

TEVCAT_LOAD_COLUMNS: tuple[str, ...] = (
    "id",
    "tevcat_name",
    "canonical_name",
    "other_names",
    "RA",
    "DEC",
    "glon",
    "glat",
    "source_type",
    "subcatalog",
    "discovery_date",
    "discovered_by",
    "distance_kpc",
    "redshift",
    "flux_crab",
    "energy_threshold",
    "spectral_index",
    "extended",
    "size_x",
    "size_y",
    "variability",
    "notes",
)


@dataclass(frozen=True)
class TevcatMatchConfig:
    """Configuration for :func:`match_catalog_to_tevcat`."""

    catalog_path: Path = TEVCAT_DEFAULT_PATH
    target: TevcatTarget = "metacatalog"
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM
    reference_radius: CrossmatchRadiusSpec = TEVCAT_REFERENCE_RADIUS_FIXED
    # Optional subset of ``subcatalog`` labels; None keeps all.
    subcatalogs: tuple[str, ...] | None = None


@dataclass
class TevcatMatchResult:
    """TeVCat cross-match metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    tevcat_flags: pd.DataFrame
    tevcat_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _finalize_tevcat_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names / dtypes and set match ``BMAJ`` from size."""
    out = df.copy()
    rename = {"ra": "RA", "dec": "DEC"}
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    missing = [c for c in ("RA", "DEC") if c not in out.columns]
    if missing:
        msg = f"TeVCat table missing required columns: {missing}"
        raise ValueError(msg)

    keep = [c for c in TEVCAT_LOAD_COLUMNS if c in out.columns]
    out = out.loc[:, keep].copy()

    for col in (
        "id",
        "RA",
        "DEC",
        "glon",
        "glat",
        "distance_kpc",
        "redshift",
        "flux_crab",
        "energy_threshold",
        "spectral_index",
        "size_x",
        "size_y",
        "variability",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in (
        "tevcat_name",
        "canonical_name",
        "other_names",
        "source_type",
        "subcatalog",
        "discovery_date",
        "discovered_by",
        "notes",
    ):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()

    if "extended" in out.columns:
        out["extended"] = out["extended"].astype("boolean")

    # size_x / size_y are degrees in the TeVCat VOTable; use max axis as BMAJ.
    sx = (
        pd.to_numeric(out["size_x"], errors="coerce").to_numpy(dtype=float)
        if "size_x" in out.columns
        else np.full(len(out), np.nan)
    )
    sy = (
        pd.to_numeric(out["size_y"], errors="coerce").to_numpy(dtype=float)
        if "size_y" in out.columns
        else np.full(len(out), np.nan)
    )
    size_deg = np.fmax(sx, sy)
    default_deg = TEVCAT_DEFAULT_EXTENT_ARCMIN / 60.0
    size_deg = np.where(np.isfinite(size_deg) & (size_deg > 0.0), size_deg, default_deg)
    out["extent_deg"] = size_deg
    out["BMAJ"] = size_deg
    out["BMIN"] = size_deg
    # Arcsec alias for POS_ERR-style overlays.
    out["POS_ERR"] = size_deg * 3600.0

    ra = out["RA"].to_numpy(dtype=float)
    dec = out["DEC"].to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return out.loc[ok].reset_index(drop=True)


def load_tevcat_catalog(path: Path | str | None = None) -> pd.DataFrame:
    """Load Wakely & Horan 2008 TeVCat (VOTable).

    Reads ``/fast/claw/tevcat.vot`` by default. Lower-case ``ra``/``dec`` become
    ``RA``/``DEC``. Match ``BMAJ`` is ``max(size_x, size_y)`` in degrees when
    finite, else
    :data:`~lwa_catalog.constants.TEVCAT_DEFAULT_EXTENT_ARCMIN`.

    Parameters
    ----------
    path
        Catalog path. Defaults to
        :data:`~lwa_catalog.constants.TEVCAT_DEFAULT_PATH`.
    """
    catalog_path = Path(TEVCAT_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = f"TeVCat catalog not found: {catalog_path}"
        raise FileNotFoundError(msg)

    from astropy.table import Table

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = Table.read(catalog_path)
    return _finalize_tevcat_frame(table.to_pandas())


def filter_tevcat_by_subcatalog(
    catalog: pd.DataFrame,
    subcatalogs: tuple[str, ...] | None,
) -> pd.DataFrame:
    """Keep rows whose ``subcatalog`` is in *subcatalogs* (``None`` keeps all)."""
    if catalog.empty or not subcatalogs:
        return catalog.copy()
    wanted = {str(s).strip().lower() for s in subcatalogs}
    if "subcatalog" not in catalog.columns:
        return catalog.copy()
    labels = catalog["subcatalog"].astype(str).str.strip().str.lower()
    return catalog.loc[labels.isin(wanted)].reset_index(drop=True)


def select_unique_tevcat_matches(meta_flags: pd.DataFrame) -> pd.DataFrame:
    """Return meta rows with exactly one TeVCat match."""
    if meta_flags.empty or "n_tevcat" not in meta_flags.columns:
        return meta_flags.iloc[0:0].copy()
    return meta_flags.loc[meta_flags["n_tevcat"] == 1].copy()


def _catalog_match_frame(
    catalog: pd.DataFrame,
    spec: CrossmatchRadiusSpec,
) -> pd.DataFrame:
    return catalog_match_frame(catalog, spec)


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_tevcat_footprint": 0,
        "n_meta_matched": 0,
        "match_completeness": float("nan"),
        "n_meta_unique": 0,
        "unique_match_fraction": float("nan"),
        "n_tevcat_matched": 0,
        "tevcat_recovery": float("nan"),
        "n_tevcat_oversplit": 0,
        "n_meta_multi_tevcat": 0,
        "meta_tevcat_hits_max": 0,
    }


def match_catalog_to_tevcat(
    lwa_catalog: pd.DataFrame,
    tevcat: pd.DataFrame | None = None,
    *,
    config: TevcatMatchConfig | None = None,
    lwa_match: pd.DataFrame | None = None,
) -> TevcatMatchResult:
    """Cross-match an LWA catalog against TeVCat VHE gamma-ray sources.

    Matching uses ``associate_catalogs`` with configured
    :class:`~lwa_catalog.analyze.crossmatch_radius.CrossmatchRadiusSpec`
    radii. The reference side defaults to a fixed 3′ gate
    (:data:`~lwa_catalog.analyze.crossmatch_radius.TEVCAT_REFERENCE_RADIUS_FIXED`).
    """
    cfg = config or TevcatMatchConfig()
    warnings: list[str] = []

    if tevcat is None:
        tevcat = load_tevcat_catalog(cfg.catalog_path)

    n_loaded = len(tevcat)
    footprint = filter_tevcat_by_subcatalog(tevcat, cfg.subcatalogs)
    if len(footprint) < n_loaded:
        warnings.append(
            f"subcatalog filter kept {len(footprint):,} / {n_loaded:,} TeVCat rows "
            f"(subcatalogs={cfg.subcatalogs})"
        )

    target = lwa_catalog
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty")
        return TevcatMatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=[
                    "meta_id",
                    "RA",
                    "DEC",
                    "n_tevcat",
                    "tevcat_positions",
                    "matched",
                ]
            ),
            tevcat_flags=pd.DataFrame(
                columns=[
                    "tevcat_pos",
                    "RA",
                    "DEC",
                    "tevcat_name",
                    "canonical_name",
                    "source_type",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            tevcat_footprint=pd.DataFrame(),
            warnings=warnings,
        )

    ref_match = apply_match_radius(footprint, cfg.reference_radius)
    if lwa_match is None:
        resolved_lwa_match = _catalog_match_frame(target, cfg.lwa_radius)
    else:
        if len(lwa_match) != len(target):
            msg = (
                f"lwa_match length {len(lwa_match)} does not match target "
                f"length {len(target)}"
            )
            raise ValueError(msg)
        resolved_lwa_match = lwa_match[["RA", "DEC", "BMAJ"]].copy()
        resolved_lwa_match.index = target.index

    n_tevcat_footprint = len(footprint)

    meta_hits: dict[int, list[int]] = {}
    tev_hits: dict[int, list[int]] = {}
    if not resolved_lwa_match.empty and not ref_match.empty:
        meta_hits, _ = associate_catalogs(resolved_lwa_match, ref_match)
        tev_hits, _ = associate_catalogs(ref_match, resolved_lwa_match)

    index_to_match_pos = {
        idx: pos for pos, idx in enumerate(resolved_lwa_match.index.tolist())
    }
    match_pos_to_index = {pos: idx for idx, pos in index_to_match_pos.items()}
    has_meta_id = "meta_id" in target.columns

    meta_records: list[dict] = []
    for idx, row in target.iterrows():
        match_pos = index_to_match_pos.get(idx)
        hit_ref = meta_hits.get(match_pos, []) if match_pos is not None else []
        n_hit = len(hit_ref)
        record: dict = {
            "RA": row.get("RA", np.nan),
            "DEC": row.get("DEC", np.nan),
            "n_tevcat": n_hit,
            "tevcat_positions": list(hit_ref),
            "matched": n_hit >= 1,
        }
        if has_meta_id:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    cols = ["meta_id", "RA", "DEC", "n_tevcat", "tevcat_positions", "matched"]
    if "meta_id" not in meta_flags.columns:
        cols = [c for c in cols if c != "meta_id"]
    if not meta_flags.empty:
        meta_flags = meta_flags[cols]

    tev_records: list[dict] = []
    for pos in range(len(footprint)):
        row = footprint.iloc[pos]
        hit_meta = tev_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                midx = match_pos_to_index.get(match_pos)
                if midx is not None:
                    meta_ids.append(target.loc[midx, "meta_id"])
        tev_records.append(
            {
                "tevcat_pos": pos,
                "RA": row["RA"],
                "DEC": row["DEC"],
                "tevcat_name": row.get("tevcat_name", ""),
                "canonical_name": row.get("canonical_name", ""),
                "source_type": row.get("source_type", ""),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    tev_cols = [
        "tevcat_pos",
        "RA",
        "DEC",
        "tevcat_name",
        "canonical_name",
        "source_type",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        tev_cols = [c for c in tev_cols if c != "meta_ids"]
    if tev_records:
        tevcat_flags = pd.DataFrame(tev_records)[tev_cols]
    else:
        tevcat_flags = pd.DataFrame(columns=tev_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_meta_unique = (
        int((meta_flags["n_tevcat"] == 1).sum()) if not meta_flags.empty else 0
    )
    n_tev_matched = (
        int((tevcat_flags["n_meta"] > 0).sum()) if not tevcat_flags.empty else 0
    )
    n_tev_oversplit = (
        int(tevcat_flags["oversplit"].sum()) if not tevcat_flags.empty else 0
    )
    n_meta_multi = (
        int((meta_flags["n_tevcat"] > 1).sum()) if not meta_flags.empty else 0
    )
    meta_hits_max = int(meta_flags["n_tevcat"].max()) if not meta_flags.empty else 0

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_tevcat_footprint": n_tevcat_footprint,
        "n_meta_matched": n_meta_matched,
        "match_completeness": n_meta_matched / n_lwa_target,
        "n_meta_unique": n_meta_unique,
        "unique_match_fraction": n_meta_unique / n_lwa_target,
        "n_tevcat_matched": n_tev_matched,
        "tevcat_recovery": (
            n_tev_matched / n_tevcat_footprint if n_tevcat_footprint else float("nan")
        ),
        "n_tevcat_oversplit": n_tev_oversplit,
        "n_meta_multi_tevcat": n_meta_multi,
        "meta_tevcat_hits_max": meta_hits_max,
    }

    return TevcatMatchResult(
        summary=summary,
        meta_flags=meta_flags,
        tevcat_flags=tevcat_flags,
        tevcat_footprint=footprint,
        warnings=warnings,
    )


def summarize_tevcat_match(result: TevcatMatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:                  {int(s['n_lwa_target']):6d}",
        f"TeVCat footprint:                 {int(s['n_tevcat_footprint']):6d}",
        f"Meta matched (>=1 TeVCat):        {int(s['n_meta_matched']):6d}",
        f"Match completeness:               {s['match_completeness']:.3f}",
        f"Meta unique match (n=1):          {int(s['n_meta_unique']):6d}",
        f"Unique match fraction:            {s['unique_match_fraction']:.3f}",
        f"TeVCat matched (>=1 meta):        {int(s['n_tevcat_matched']):6d}",
        f"TeVCat recovery:                  {s['tevcat_recovery']:.3f}",
        f"TeVCat over-split (n_meta>1):     {int(s['n_tevcat_oversplit']):6d}",
        f"Meta multi-TeVCat (n>1):          {int(s['n_meta_multi_tevcat']):6d}",
        f"Max TeVCat hits per meta:         {int(s['meta_tevcat_hits_max']):6d}",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)
