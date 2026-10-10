"""Xu et al. 2022 RXGCC ROSAT galaxy clusters vs LWA metacatalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from lwa_catalog.analyze.crossmatch_radius import (
    LWA_CROSSMATCH_RADIUS_BEAM,
    RXGCC_REFERENCE_RADIUS_BEAM,
    CrossmatchRadiusSpec,
    apply_match_radius,
    catalog_match_frame,
)
from lwa_catalog.constants import (
    RXGCC_DEFAULT_EXTENT_ARCMIN,
    RXGCC_DEFAULT_PATH,
)
from lwa_catalog.create.merge import associate_catalogs

Xu2022Target = Literal["metacatalog"]

# Columns kept for match / display (FITS names normalized on load).
XU2022_LOAD_COLUMNS: tuple[str, ...] = (
    "RXGCC",
    "RA",
    "DEC",
    "Ext",
    "Extml",
    "z",
    "e_z",
    "r_z",
    "Class",
    "GCXSZ",
    "GCOPT",
    "GC",
    "Rsig",
    "R500_arcmin",
    "R500_Mpc",
    "CRsig",
    "e_CRsig",
    "CR500",
    "e_CR500",
    "L500",
    "e_L500",
    "F500",
    "e_F500",
    "M500",
    "e_M500",
    "TX",
    "e_TX",
    "Beta",
    "Rc",
)

# Gold / Silver / Bronze class labels from Xu+2022.
RXGCC_CLASS_LABELS: dict[str, str] = {
    "G": "Gold (new)",
    "S": "Silver (first X-ray)",
    "B": "Bronze (known ICM)",
}


@dataclass(frozen=True)
class Xu2022MatchConfig:
    """Configuration for :func:`match_catalog_to_xu2022`."""

    catalog_path: Path = RXGCC_DEFAULT_PATH
    target: Xu2022Target = "metacatalog"
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM
    reference_radius: CrossmatchRadiusSpec = RXGCC_REFERENCE_RADIUS_BEAM
    # Optional subset of Class codes (``G`` / ``S`` / ``B``); None keeps all.
    classes: tuple[str, ...] | None = None


@dataclass
class Xu2022MatchResult:
    """Xu+2022 RXGCC cross-match metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    xu2022_flags: pd.DataFrame
    xu2022_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _decode_bytes(series: pd.Series) -> pd.Series:
    if series.dtype == object or str(series.dtype).startswith("|S"):
        return series.map(
            lambda v: (
                v.decode("utf-8", errors="replace").strip()
                if isinstance(v, (bytes, bytearray))
                else ("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip())
            )
        )
    if hasattr(series.dtype, "kind") and series.dtype.kind == "S":
        return series.astype(str).str.strip()
    return series.astype(str).str.strip()


def _finalize_xu2022_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names / dtypes and set match ``BMAJ`` from extent."""
    out = df.copy()
    rename = {
        "RAdeg": "RA",
        "DEdeg": "DEC",
        "R500*": "R500_arcmin",
        "R500": "R500_Mpc",
        "GC*": "GC",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    missing = [c for c in ("RA", "DEC") if c not in out.columns]
    if missing:
        msg = f"Xu+2022 / RXGCC table missing required columns: {missing}"
        raise ValueError(msg)

    keep = [c for c in XU2022_LOAD_COLUMNS if c in out.columns]
    out = out.loc[:, keep].copy()

    for col in (
        "RXGCC",
        "RA",
        "DEC",
        "Ext",
        "Extml",
        "z",
        "e_z",
        "Rsig",
        "R500_arcmin",
        "R500_Mpc",
        "CRsig",
        "e_CRsig",
        "CR500",
        "e_CR500",
        "L500",
        "e_L500",
        "F500",
        "e_F500",
        "M500",
        "e_M500",
        "TX",
        "e_TX",
        "Beta",
        "Rc",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ("r_z", "Class", "GCXSZ", "GCOPT", "GC"):
        if col in out.columns:
            out[col] = _decode_bytes(out[col])

    # Match scale: prefer R500* (arcmin), else Ext, else catalog default.
    r500 = (
        pd.to_numeric(out["R500_arcmin"], errors="coerce").to_numpy(dtype=float)
        if "R500_arcmin" in out.columns
        else np.full(len(out), np.nan)
    )
    ext = (
        pd.to_numeric(out["Ext"], errors="coerce").to_numpy(dtype=float)
        if "Ext" in out.columns
        else np.full(len(out), np.nan)
    )
    extent_arcmin = np.where(np.isfinite(r500) & (r500 > 0.0), r500, ext)
    extent_arcmin = np.where(
        np.isfinite(extent_arcmin) & (extent_arcmin > 0.0),
        extent_arcmin,
        RXGCC_DEFAULT_EXTENT_ARCMIN,
    )
    out["extent_arcmin"] = extent_arcmin
    out["BMAJ"] = extent_arcmin / 60.0
    out["BMIN"] = out["BMAJ"]
    # Arcsec alias for POS_ERR-style overlays.
    out["POS_ERR"] = extent_arcmin * 60.0

    ra = out["RA"].to_numpy(dtype=float)
    dec = out["DEC"].to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return out.loc[ok].reset_index(drop=True)


def load_xu2022_catalog(path: Path | str | None = None) -> pd.DataFrame:
    """Load Xu et al. 2022 RXGCC galaxy-cluster catalog.

    The operator file ``xu2022table3.dat`` is a FITS table (despite the
    ``.dat`` suffix). ``RAdeg``/``DEdeg`` become ``RA``/``DEC``;
    ``R500*`` → ``R500_arcmin``. Match ``BMAJ`` is set from ``R500_arcmin``
    (fallback ``Ext``, then :data:`~lwa_catalog.constants.RXGCC_DEFAULT_EXTENT_ARCMIN`).

    Parameters
    ----------
    path
        Catalog path. Defaults to
        :data:`~lwa_catalog.constants.RXGCC_DEFAULT_PATH`.
    """
    catalog_path = Path(RXGCC_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = f"Xu+2022 / RXGCC catalog not found: {catalog_path}"
        raise FileNotFoundError(msg)

    from astropy.table import Table

    # Suppress non-standard unit warnings from the CDS FITS product.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = Table.read(catalog_path)
    return _finalize_xu2022_frame(table.to_pandas())


def filter_xu2022_by_class(
    catalog: pd.DataFrame,
    classes: tuple[str, ...] | None,
) -> pd.DataFrame:
    """Keep rows whose ``Class`` is in *classes* (``None`` keeps all)."""
    if catalog.empty or not classes:
        return catalog.copy()
    wanted = {str(c).strip().upper() for c in classes}
    cls = catalog["Class"].astype(str).str.strip().str.upper() if "Class" in catalog.columns else None
    if cls is None:
        return catalog.copy()
    return catalog.loc[cls.isin(wanted)].reset_index(drop=True)


def select_unique_xu2022_matches(meta_flags: pd.DataFrame) -> pd.DataFrame:
    """Return meta rows with exactly one RXGCC match."""
    if meta_flags.empty or "n_xu2022" not in meta_flags.columns:
        return meta_flags.iloc[0:0].copy()
    return meta_flags.loc[meta_flags["n_xu2022"] == 1].copy()


def _catalog_match_frame(
    catalog: pd.DataFrame,
    spec: CrossmatchRadiusSpec,
) -> pd.DataFrame:
    return catalog_match_frame(catalog, spec)


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_xu2022_footprint": 0,
        "n_meta_matched": 0,
        "match_completeness": float("nan"),
        "n_meta_unique": 0,
        "unique_match_fraction": float("nan"),
        "n_xu2022_matched": 0,
        "xu2022_recovery": float("nan"),
        "n_xu2022_oversplit": 0,
        "n_meta_multi_xu2022": 0,
        "meta_xu2022_hits_max": 0,
    }


def match_catalog_to_xu2022(
    lwa_catalog: pd.DataFrame,
    xu2022: pd.DataFrame | None = None,
    *,
    config: Xu2022MatchConfig | None = None,
    lwa_match: pd.DataFrame | None = None,
) -> Xu2022MatchResult:
    """Cross-match an LWA catalog against Xu+2022 RXGCC clusters.

    Matching uses ``associate_catalogs`` with configured
    :class:`~lwa_catalog.analyze.crossmatch_radius.CrossmatchRadiusSpec`
    radii. The reference side defaults to beam mode so per-cluster
    ``BMAJ`` (= ``R500_arcmin``) sets the match scale.
    """
    cfg = config or Xu2022MatchConfig()
    warnings: list[str] = []

    if xu2022 is None:
        xu2022 = load_xu2022_catalog(cfg.catalog_path)

    n_loaded = len(xu2022)
    footprint = filter_xu2022_by_class(xu2022, cfg.classes)
    if len(footprint) < n_loaded:
        warnings.append(
            f"Class filter kept {len(footprint):,} / {n_loaded:,} RXGCC rows "
            f"(classes={cfg.classes})"
        )

    target = lwa_catalog
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty")
        return Xu2022MatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=[
                    "meta_id",
                    "RA",
                    "DEC",
                    "n_xu2022",
                    "xu2022_positions",
                    "matched",
                ]
            ),
            xu2022_flags=pd.DataFrame(
                columns=[
                    "xu2022_pos",
                    "RA",
                    "DEC",
                    "RXGCC",
                    "Class",
                    "z",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            xu2022_footprint=pd.DataFrame(),
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

    n_xu2022_footprint = len(footprint)

    meta_hits: dict[int, list[int]] = {}
    xu_hits: dict[int, list[int]] = {}
    if not resolved_lwa_match.empty and not ref_match.empty:
        meta_hits, _ = associate_catalogs(resolved_lwa_match, ref_match)
        xu_hits, _ = associate_catalogs(ref_match, resolved_lwa_match)

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
            "n_xu2022": n_hit,
            "xu2022_positions": list(hit_ref),
            "matched": n_hit >= 1,
        }
        if has_meta_id:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    cols = ["meta_id", "RA", "DEC", "n_xu2022", "xu2022_positions", "matched"]
    if "meta_id" not in meta_flags.columns:
        cols = [c for c in cols if c != "meta_id"]
    if not meta_flags.empty:
        meta_flags = meta_flags[cols]

    xu_records: list[dict] = []
    for pos in range(len(footprint)):
        row = footprint.iloc[pos]
        hit_meta = xu_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                midx = match_pos_to_index.get(match_pos)
                if midx is not None:
                    meta_ids.append(target.loc[midx, "meta_id"])
        xu_records.append(
            {
                "xu2022_pos": pos,
                "RA": row["RA"],
                "DEC": row["DEC"],
                "RXGCC": row.get("RXGCC", np.nan),
                "Class": row.get("Class", ""),
                "z": row.get("z", np.nan),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    xu_cols = [
        "xu2022_pos",
        "RA",
        "DEC",
        "RXGCC",
        "Class",
        "z",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        xu_cols = [c for c in xu_cols if c != "meta_ids"]
    if xu_records:
        xu2022_flags = pd.DataFrame(xu_records)[xu_cols]
    else:
        xu2022_flags = pd.DataFrame(columns=xu_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_meta_unique = (
        int((meta_flags["n_xu2022"] == 1).sum()) if not meta_flags.empty else 0
    )
    n_xu_matched = (
        int((xu2022_flags["n_meta"] > 0).sum()) if not xu2022_flags.empty else 0
    )
    n_xu_oversplit = (
        int(xu2022_flags["oversplit"].sum()) if not xu2022_flags.empty else 0
    )
    n_meta_multi = (
        int((meta_flags["n_xu2022"] > 1).sum()) if not meta_flags.empty else 0
    )
    meta_hits_max = int(meta_flags["n_xu2022"].max()) if not meta_flags.empty else 0

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_xu2022_footprint": n_xu2022_footprint,
        "n_meta_matched": n_meta_matched,
        "match_completeness": n_meta_matched / n_lwa_target,
        "n_meta_unique": n_meta_unique,
        "unique_match_fraction": n_meta_unique / n_lwa_target,
        "n_xu2022_matched": n_xu_matched,
        "xu2022_recovery": (
            n_xu_matched / n_xu2022_footprint if n_xu2022_footprint else float("nan")
        ),
        "n_xu2022_oversplit": n_xu_oversplit,
        "n_meta_multi_xu2022": n_meta_multi,
        "meta_xu2022_hits_max": meta_hits_max,
    }

    return Xu2022MatchResult(
        summary=summary,
        meta_flags=meta_flags,
        xu2022_flags=xu2022_flags,
        xu2022_footprint=footprint,
        warnings=warnings,
    )


def summarize_xu2022_match(result: Xu2022MatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:                  {int(s['n_lwa_target']):6d}",
        f"RXGCC (Xu+2022) footprint:        {int(s['n_xu2022_footprint']):6d}",
        f"Meta matched (>=1 RXGCC):         {int(s['n_meta_matched']):6d}",
        f"Match completeness:               {s['match_completeness']:.3f}",
        f"Meta unique match (n=1):          {int(s['n_meta_unique']):6d}",
        f"Unique match fraction:            {s['unique_match_fraction']:.3f}",
        f"RXGCC matched (>=1 meta):         {int(s['n_xu2022_matched']):6d}",
        f"RXGCC recovery:                   {s['xu2022_recovery']:.3f}",
        f"RXGCC over-split (n_meta>1):      {int(s['n_xu2022_oversplit']):6d}",
        f"Meta multi-RXGCC (n>1):           {int(s['n_meta_multi_xu2022']):6d}",
        f"Max RXGCC hits per meta:          {int(s['meta_xu2022_hits_max']):6d}",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)
