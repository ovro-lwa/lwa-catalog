"""Boller et al. 2016 2RXS compact source catalog vs LWA metacatalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from lwa_catalog.analyze.crossmatch_radius import (
    LWA_CROSSMATCH_RADIUS_BEAM,
    TWORXS_REFERENCE_RADIUS_LOCALIZATION,
    CrossmatchRadiusSpec,
    apply_match_radius,
    catalog_match_frame,
)
from lwa_catalog.constants import (
    TWORXS_DEFAULT_EXIML_MIN,
    TWORXS_DEFAULT_PATH,
    TWORXS_IMAGE_PIXEL_ARCSEC,
    TWORXS_POSITION_ERROR_DEFAULT_ARCSEC,
)
from lwa_catalog.create.merge import associate_catalogs

TworxsTarget = Literal["metacatalog"]

# CDS J/A+A/588/A103 cat2rxs.dat subset (1-indexed inclusive → 0 half-open).
_TWORXS_COLSPECS: list[tuple[int, int]] = [
    (0, 21),  # 2RXS
    (22, 28),  # Seq
    (29, 32),  # IndDet
    (33, 42),  # ExiML
    (43, 52),  # Cts
    (53, 63),  # e_Cts
    (64, 72),  # CRate
    (73, 80),  # e_CRate
    (81, 89),  # ExpTime
    (90, 99),  # RAdeg
    (100, 109),  # DEdeg
    (176, 185),  # Ext
    (186, 193),  # e_Ext
    (194, 201),  # ExtML
    (202, 208),  # HR1
    (209, 217),  # e_HR1
    (218, 224),  # HR2
    (225, 234),  # e_HR2
    (241, 244),  # Sflag
    (462, 472),  # Fluxp
    (755, 763),  # e_Xima
    (764, 772),  # e_Yima
]
_TWORXS_NAMES: tuple[str, ...] = (
    "2RXS",
    "Seq",
    "IndDet",
    "ExiML",
    "Cts",
    "e_Cts",
    "CRate",
    "e_CRate",
    "ExpTime",
    "RA",
    "DEC",
    "Ext",
    "e_Ext",
    "ExtML",
    "HR1",
    "e_HR1",
    "HR2",
    "e_HR2",
    "Sflag",
    "Fluxp",
    "e_Xima",
    "e_Yima",
)

TWORXS_LOAD_COLUMNS: tuple[str, ...] = _TWORXS_NAMES


@dataclass(frozen=True)
class TworxsMatchConfig:
    """Configuration for :func:`match_catalog_to_tworxs`."""

    catalog_path: Path = TWORXS_DEFAULT_PATH
    target: TworxsTarget = "metacatalog"
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM
    reference_radius: CrossmatchRadiusSpec = TWORXS_REFERENCE_RADIUS_LOCALIZATION
    # Paper cleaner sample (None disables).
    eximl_min: float | None = TWORXS_DEFAULT_EXIML_MIN


@dataclass
class TworxsMatchResult:
    """2RXS cross-match metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    tworxs_flags: pd.DataFrame
    tworxs_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _resolve_tworxs_path(path: Path | str | None) -> Path:
    """Resolve ``.dat`` / ``.dat.gz`` variants for the CDS ASCII product."""
    catalog_path = Path(TWORXS_DEFAULT_PATH if path is None else path)
    if catalog_path.is_file():
        return catalog_path
    # Operator may point at uncompressed name while only ``.gz`` exists.
    if catalog_path.suffix == ".dat":
        gz = catalog_path.with_suffix(".dat.gz")
        if gz.is_file():
            return gz
    if "".join(catalog_path.suffixes).endswith(".dat.gz"):
        plain = Path(str(catalog_path)[: -len(".gz")])
        if plain.is_file():
            return plain
    msg = f"2RXS catalog not found: {catalog_path}"
    raise FileNotFoundError(msg)


def _finalize_tworxs_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes, derive localization axes, drop bad coordinates."""
    missing = [c for c in ("RA", "DEC") if c not in df.columns]
    if missing:
        msg = f"2RXS table missing required columns: {missing}"
        raise ValueError(msg)

    keep = [c for c in TWORXS_LOAD_COLUMNS if c in df.columns]
    out = df.loc[:, keep].copy()

    if "2RXS" in out.columns:
        out["2RXS"] = out["2RXS"].astype(str).str.strip()

    for col in (
        "Seq",
        "IndDet",
        "ExiML",
        "Cts",
        "e_Cts",
        "CRate",
        "e_CRate",
        "ExpTime",
        "RA",
        "DEC",
        "Ext",
        "e_Ext",
        "ExtML",
        "HR1",
        "e_HR1",
        "HR2",
        "e_HR2",
        "Sflag",
        "Fluxp",
        "e_Xima",
        "e_Yima",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # POS_ERR from image-plane 1σ errors (45″/pix) → radial arcsec.
    ex = (
        pd.to_numeric(out["e_Xima"], errors="coerce").to_numpy(dtype=float)
        if "e_Xima" in out.columns
        else np.full(len(out), np.nan)
    )
    ey = (
        pd.to_numeric(out["e_Yima"], errors="coerce").to_numpy(dtype=float)
        if "e_Yima" in out.columns
        else np.full(len(out), np.nan)
    )
    pos_err_arcsec = TWORXS_IMAGE_PIXEL_ARCSEC * np.hypot(ex, ey)
    pos_err_arcsec = np.where(
        np.isfinite(pos_err_arcsec) & (pos_err_arcsec > 0.0),
        pos_err_arcsec,
        TWORXS_POSITION_ERROR_DEFAULT_ARCSEC,
    )
    out["POS_ERR"] = pos_err_arcsec
    axis_sigma = (pos_err_arcsec / 3600.0) / np.sqrt(2.0)
    out["E_RA"] = axis_sigma
    out["E_DEC"] = axis_sigma

    ra = out["RA"].to_numpy(dtype=float)
    dec = out["DEC"].to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return out.loc[ok].reset_index(drop=True)


def load_tworxs_catalog(path: Path | str | None = None) -> pd.DataFrame:
    """Load Boller et al. 2016 2RXS compact sources (CDS ``cat2rxs.dat``).

    Reads a fixed-width subset of columns from the ASCII (or ``.gz``) product.
    Image-plane ``XERR``/``YERR`` become radial ``POS_ERR`` (arcsec) and
    ``E_RA``/``E_DEC`` (degrees) for localization matching. Rows with
    non-finite ``RA``/``DEC`` are dropped.

    Parameters
    ----------
    path
        Catalog path. Defaults to
        :data:`~lwa_catalog.constants.TWORXS_DEFAULT_PATH`. Accepts plain
        ``.dat`` or ``.dat.gz``.
    """
    catalog_path = _resolve_tworxs_path(path)
    compression = "gzip" if "".join(catalog_path.suffixes).endswith(".gz") else None
    raw = pd.read_fwf(
        catalog_path,
        colspecs=_TWORXS_COLSPECS,
        names=list(_TWORXS_NAMES),
        dtype=str,
        compression=compression,
    )
    return _finalize_tworxs_frame(raw)


def filter_tworxs_by_eximl(
    catalog: pd.DataFrame,
    *,
    eximl_min: float | None = TWORXS_DEFAULT_EXIML_MIN,
) -> pd.DataFrame:
    """Keep rows with ``ExiML >= eximl_min`` (``None`` keeps all)."""
    if catalog.empty or eximl_min is None or "ExiML" not in catalog.columns:
        return catalog.copy()
    eximl = pd.to_numeric(catalog["ExiML"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(eximl) & (eximl >= float(eximl_min))
    return catalog.loc[keep].reset_index(drop=True)


def select_unique_tworxs_matches(meta_flags: pd.DataFrame) -> pd.DataFrame:
    """Return meta rows with exactly one 2RXS match."""
    if meta_flags.empty or "n_2rxs" not in meta_flags.columns:
        return meta_flags.iloc[0:0].copy()
    return meta_flags.loc[meta_flags["n_2rxs"] == 1].copy()


def _catalog_match_frame(
    catalog: pd.DataFrame,
    spec: CrossmatchRadiusSpec,
) -> pd.DataFrame:
    return catalog_match_frame(catalog, spec)


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_2rxs_footprint": 0,
        "n_meta_matched": 0,
        "match_completeness": float("nan"),
        "n_meta_unique": 0,
        "unique_match_fraction": float("nan"),
        "n_2rxs_matched": 0,
        "tworxs_recovery": float("nan"),
        "n_2rxs_oversplit": 0,
        "n_meta_multi_2rxs": 0,
        "meta_2rxs_hits_max": 0,
    }


def match_catalog_to_tworxs(
    lwa_catalog: pd.DataFrame,
    tworxs: pd.DataFrame | None = None,
    *,
    config: TworxsMatchConfig | None = None,
    lwa_match: pd.DataFrame | None = None,
) -> TworxsMatchResult:
    """Cross-match an LWA catalog against 2RXS compact sources.

    Matching uses ``associate_catalogs`` with configured
    :class:`~lwa_catalog.analyze.crossmatch_radius.CrossmatchRadiusSpec`
    radii. Optional ``ExiML`` cut filters the reference catalog first.
    """
    cfg = config or TworxsMatchConfig()
    warnings: list[str] = []

    if tworxs is None:
        tworxs = load_tworxs_catalog(cfg.catalog_path)

    n_loaded = len(tworxs)
    footprint = filter_tworxs_by_eximl(tworxs, eximl_min=cfg.eximl_min)
    if len(footprint) < n_loaded:
        warnings.append(
            f"ExiML cut kept {len(footprint):,} / {n_loaded:,} 2RXS rows "
            f"(ExiML>={cfg.eximl_min})"
        )

    target = lwa_catalog
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty")
        return TworxsMatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=[
                    "meta_id",
                    "RA",
                    "DEC",
                    "n_2rxs",
                    "tworxs_positions",
                    "matched",
                ]
            ),
            tworxs_flags=pd.DataFrame(
                columns=[
                    "tworxs_pos",
                    "RA",
                    "DEC",
                    "2RXS",
                    "ExiML",
                    "CRate",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            tworxs_footprint=pd.DataFrame(),
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

    n_2rxs_footprint = len(footprint)

    meta_hits: dict[int, list[int]] = {}
    ref_hits: dict[int, list[int]] = {}
    if not resolved_lwa_match.empty and not ref_match.empty:
        meta_hits, _ = associate_catalogs(resolved_lwa_match, ref_match)
        ref_hits, _ = associate_catalogs(ref_match, resolved_lwa_match)

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
            "n_2rxs": n_hit,
            "tworxs_positions": list(hit_ref),
            "matched": n_hit >= 1,
        }
        if has_meta_id:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    cols = ["meta_id", "RA", "DEC", "n_2rxs", "tworxs_positions", "matched"]
    if "meta_id" not in meta_flags.columns:
        cols = [c for c in cols if c != "meta_id"]
    if not meta_flags.empty:
        meta_flags = meta_flags[cols]

    ref_records: list[dict] = []
    for pos in range(len(footprint)):
        row = footprint.iloc[pos]
        hit_meta = ref_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                midx = match_pos_to_index.get(match_pos)
                if midx is not None:
                    meta_ids.append(target.loc[midx, "meta_id"])
        ref_records.append(
            {
                "tworxs_pos": pos,
                "RA": row["RA"],
                "DEC": row["DEC"],
                "2RXS": row.get("2RXS", ""),
                "ExiML": row.get("ExiML", np.nan),
                "CRate": row.get("CRate", np.nan),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    ref_cols = [
        "tworxs_pos",
        "RA",
        "DEC",
        "2RXS",
        "ExiML",
        "CRate",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        ref_cols = [c for c in ref_cols if c != "meta_ids"]
    if ref_records:
        tworxs_flags = pd.DataFrame(ref_records)[ref_cols]
    else:
        tworxs_flags = pd.DataFrame(columns=ref_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_meta_unique = int((meta_flags["n_2rxs"] == 1).sum()) if not meta_flags.empty else 0
    n_ref_matched = (
        int((tworxs_flags["n_meta"] > 0).sum()) if not tworxs_flags.empty else 0
    )
    n_ref_oversplit = (
        int(tworxs_flags["oversplit"].sum()) if not tworxs_flags.empty else 0
    )
    n_meta_multi = int((meta_flags["n_2rxs"] > 1).sum()) if not meta_flags.empty else 0
    meta_hits_max = int(meta_flags["n_2rxs"].max()) if not meta_flags.empty else 0

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_2rxs_footprint": n_2rxs_footprint,
        "n_meta_matched": n_meta_matched,
        "match_completeness": n_meta_matched / n_lwa_target,
        "n_meta_unique": n_meta_unique,
        "unique_match_fraction": n_meta_unique / n_lwa_target,
        "n_2rxs_matched": n_ref_matched,
        "tworxs_recovery": (
            n_ref_matched / n_2rxs_footprint if n_2rxs_footprint else float("nan")
        ),
        "n_2rxs_oversplit": n_ref_oversplit,
        "n_meta_multi_2rxs": n_meta_multi,
        "meta_2rxs_hits_max": meta_hits_max,
    }

    return TworxsMatchResult(
        summary=summary,
        meta_flags=meta_flags,
        tworxs_flags=tworxs_flags,
        tworxs_footprint=footprint,
        warnings=warnings,
    )


def summarize_tworxs_match(result: TworxsMatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:                  {int(s['n_lwa_target']):6d}",
        f"2RXS footprint (after ExiML cut): {int(s['n_2rxs_footprint']):6d}",
        f"Meta matched (>=1 2RXS):          {int(s['n_meta_matched']):6d}",
        f"Match completeness:               {s['match_completeness']:.3f}",
        f"Meta unique match (n=1):          {int(s['n_meta_unique']):6d}",
        f"Unique match fraction:            {s['unique_match_fraction']:.3f}",
        f"2RXS matched (>=1 meta):          {int(s['n_2rxs_matched']):6d}",
        f"2RXS recovery:                    {s['tworxs_recovery']:.3f}",
        f"2RXS over-split (n_meta>1):       {int(s['n_2rxs_oversplit']):6d}",
        f"Meta multi-2RXS (n>1):            {int(s['n_meta_multi_2rxs']):6d}",
        f"Max 2RXS hits per meta:           {int(s['meta_2rxs_hits_max']):6d}",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)
