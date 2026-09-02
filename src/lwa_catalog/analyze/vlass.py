"""VLASS cross-match against LWA metacatalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from lwa_catalog.analyze.reliability import resolve_bmaj
from lwa_catalog.analyze.vlssr import select_blue_associated_rows
from lwa_catalog.constants import (
    VLASS_BMAJ_DEG,
    VLASS_DEC_MIN_DEG,
    VLASS_DEFAULT_PATH,
    VLASS_FREQ_HZ,
)
from lwa_catalog.create.merge import associate_catalogs

VlassTarget = Literal["metacatalog", "metacatalog_blue"]

VLASS_LOAD_COLUMNS: tuple[str, ...] = (
    "Component_name",
    "RA",
    "DEC",
    "Peak_flux",
    "Total_flux",
    "Maj",
    "Min",
    "PA",
    "BMAJ",
    "BMIN",
    "BPA",
    "Duplicate_flag",
    "Quality_flag",
    "S_Code",
)


@dataclass(frozen=True)
class VlassMatchConfig:
    """Configuration for :func:`match_catalog_to_vlass`."""

    catalog_path: Path = VLASS_DEFAULT_PATH
    target: VlassTarget = "metacatalog"
    dec_min_deg: float = VLASS_DEC_MIN_DEG
    apply_quality_filter: bool = True
    max_duplicate_flag: int = 1
    quality_flags: tuple[int, ...] = (0, 4)
    exclude_s_code_e: bool = True


@dataclass
class VlassMatchResult:
    """VLASS cross-match metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    vlass_flags: pd.DataFrame
    vlass_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _resolve_vlass_bmaj_deg(row: pd.Series) -> float:
    bmaj_arcsec = pd.to_numeric(row.get("BMAJ"), errors="coerce")
    if np.isfinite(bmaj_arcsec) and float(bmaj_arcsec) > 0.0:
        return float(bmaj_arcsec) / 3600.0
    return VLASS_BMAJ_DEG


def _apply_vlass_quality_filter(
    catalog: pd.DataFrame,
    *,
    max_duplicate_flag: int,
    quality_flags: tuple[int, ...],
    exclude_s_code_e: bool,
) -> pd.DataFrame:
    """Apply CIRADA-recommended component-table quality cuts."""
    out = catalog
    if "Duplicate_flag" in out.columns:
        dup = pd.to_numeric(out["Duplicate_flag"], errors="coerce")
        out = out.loc[dup <= max_duplicate_flag]
    if "Quality_flag" in out.columns and quality_flags:
        qual = pd.to_numeric(out["Quality_flag"], errors="coerce")
        out = out.loc[qual.isin(list(quality_flags))]
    if exclude_s_code_e and "S_Code" in out.columns:
        out = out.loc[out["S_Code"].astype(str) != "E"]
    return out.reset_index(drop=True)


def load_vlass_catalog(
    path: Path | str | None = None,
    *,
    config: VlassMatchConfig | None = None,
) -> pd.DataFrame:
    """Load the CIRADA VLASS QL epoch 1 component table for cross-matching.

    Expects the gzipped component CSV from
    `CIRADA VLASS catalogues <https://cirada.ca/catalogues>`_ (Gordon et al.
    2021, ApJS 255, 30). By default applies recommended quality cuts
    (``Duplicate_flag <= 1``, ``Quality_flag in (0, 4)``, ``S_Code != 'E'``).

    Returns columns ``RA``, ``DEC``, ``Peak_flux``, ``Total_flux``, deconvolved
    ``Maj``/``Min``/``PA`` (arcsec), ``Component_name``, and ``BMAJ``/``BMIN``
    in degrees for :func:`~lwa_catalog.create.merge.associate_catalogs`.
    """
    cfg = config or VlassMatchConfig()
    catalog_path = Path(VLASS_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = (
            f"VLASS catalog not found: {catalog_path}. "
            f"Download CIRADA_VLASS1QLv3.1_table1_components.csv.gz into "
            f"{VLASS_DEFAULT_PATH.parent} from https://cirada.ca/catalogues"
        )
        raise FileNotFoundError(msg)

    usecols = list(VLASS_LOAD_COLUMNS)
    df = pd.read_csv(catalog_path, usecols=lambda c: c in usecols)
    missing = [col for col in VLASS_LOAD_COLUMNS if col not in df.columns]
    if missing:
        msg = f"VLASS table missing expected columns: {missing}"
        raise ValueError(msg)

    if cfg.apply_quality_filter:
        n_before = len(df)
        df = _apply_vlass_quality_filter(
            df,
            max_duplicate_flag=cfg.max_duplicate_flag,
            quality_flags=cfg.quality_flags,
            exclude_s_code_e=cfg.exclude_s_code_e,
        )
        if len(df) < n_before:
            pass  # caller may inspect warnings from match if needed

    df["BMAJ"] = df.apply(_resolve_vlass_bmaj_deg, axis=1)
    bmin_arcsec = pd.to_numeric(df["BMIN"], errors="coerce")
    df["BMIN"] = np.where(
        np.isfinite(bmin_arcsec.to_numpy(dtype=float)) & (bmin_arcsec > 0),
        bmin_arcsec / 3600.0,
        df["BMAJ"],
    )

    ra = pd.to_numeric(df["RA"], errors="coerce")
    dec = pd.to_numeric(df["DEC"], errors="coerce")
    ok = np.isfinite(ra.to_numpy(dtype=float)) & np.isfinite(dec.to_numpy(dtype=float))
    return df.loc[ok].reset_index(drop=True)


def select_unique_vlass_matches(meta_flags: pd.DataFrame) -> pd.DataFrame:
    """Return meta rows with exactly one VLASS match (``n_vlass == 1``)."""
    if meta_flags.empty or "n_vlass" not in meta_flags.columns:
        return meta_flags.iloc[0:0].copy()
    return meta_flags.loc[meta_flags["n_vlass"] == 1].copy()


def _catalog_match_frame(catalog: pd.DataFrame) -> pd.DataFrame:
    """Build ``RA`` / ``DEC`` / ``BMAJ`` columns for beam-radius matching."""
    records: list[dict[str, float]] = []
    indices: list[object] = []
    for idx, row in catalog.iterrows():
        try:
            ra = float(row["RA"])
            dec = float(row["DEC"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(ra) or not np.isfinite(dec):
            continue
        bmaj = resolve_bmaj(row)
        if not np.isfinite(bmaj):
            bmaj = 0.0
        records.append({"RA": ra, "DEC": dec, "BMAJ": bmaj})
        indices.append(idx)

    if not records:
        return pd.DataFrame(columns=["RA", "DEC", "BMAJ"])
    return pd.DataFrame(records, index=indices)


def _footprint_filter_vlass(
    vlass: pd.DataFrame,
    lwa: pd.DataFrame,
    *,
    dec_min_deg: float,
) -> pd.DataFrame:
    """Keep VLASS rows in the LWA Dec box and above the VLASS survey limit."""
    if vlass.empty:
        return vlass.copy()
    if lwa.empty or "DEC" not in lwa.columns:
        return vlass.iloc[0:0].copy()

    lwa_dec = pd.to_numeric(lwa["DEC"], errors="coerce")
    finite_lwa = lwa_dec[np.isfinite(lwa_dec.to_numpy(dtype=float))]
    if finite_lwa.empty:
        return vlass.iloc[0:0].copy()

    dec_min = max(float(finite_lwa.min()), float(dec_min_deg))
    dec_max = float(finite_lwa.max())
    vlass_dec = pd.to_numeric(vlass["DEC"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(vlass_dec) & (vlass_dec >= dec_min) & (vlass_dec <= dec_max)
    return vlass.loc[keep].copy()


def _select_lwa_target(lwa_catalog: pd.DataFrame, target: VlassTarget) -> pd.DataFrame:
    if target == "metacatalog":
        return lwa_catalog
    if target == "metacatalog_blue":
        return select_blue_associated_rows(lwa_catalog)
    msg = f"unsupported target: {target!r}"
    raise ValueError(msg)


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_vlass_footprint": 0,
        "n_meta_matched": 0,
        "match_completeness": float("nan"),
        "n_meta_unique": 0,
        "unique_match_fraction": float("nan"),
        "n_vlass_matched": 0,
        "vlass_recovery": float("nan"),
        "n_vlass_oversplit": 0,
        "n_meta_multi_vlass": 0,
        "meta_vlass_hits_max": 0,
    }


def match_catalog_to_vlass(
    lwa_catalog: pd.DataFrame,
    vlass: pd.DataFrame | None = None,
    *,
    config: VlassMatchConfig | None = None,
) -> VlassMatchResult:
    """Cross-match an LWA catalog against VLASS and compute association metrics.

    Default target is the full input metacatalog. Matching uses primary
    ``RA``/``DEC`` and beam radii via
    :func:`~lwa_catalog.create.merge.associate_catalogs`. VLASS components
    outside the survey (Dec ``< config.dec_min_deg``, default −40°) are excluded
    from the footprint.
    """
    cfg = config or VlassMatchConfig()
    warnings: list[str] = []

    if vlass is None:
        vlass = load_vlass_catalog(cfg.catalog_path, config=cfg)

    target = _select_lwa_target(lwa_catalog, cfg.target)
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty after band selection")
        return VlassMatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=["meta_id", "RA", "DEC", "n_vlass", "vlass_positions", "matched"]
            ),
            vlass_flags=pd.DataFrame(
                columns=[
                    "vlass_pos",
                    "RA",
                    "DEC",
                    "Peak_flux",
                    "Component_name",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            vlass_footprint=pd.DataFrame(),
            warnings=warnings,
        )

    vlass_footprint = _footprint_filter_vlass(vlass, target, dec_min_deg=cfg.dec_min_deg)
    lwa_match = _catalog_match_frame(target)
    n_vlass_footprint = len(vlass_footprint)

    meta_hits: dict[int, list[int]] = {}
    vlass_hits: dict[int, list[int]] = {}
    if not lwa_match.empty and not vlass_footprint.empty:
        meta_hits, _ = associate_catalogs(lwa_match, vlass_footprint)
        vlass_hits, _ = associate_catalogs(vlass_footprint, lwa_match)

    index_to_match_pos = {idx: pos for pos, idx in enumerate(lwa_match.index.tolist())}
    match_pos_to_index = {pos: idx for idx, pos in index_to_match_pos.items()}
    has_meta_id = "meta_id" in target.columns

    meta_records: list[dict] = []
    for idx, row in target.iterrows():
        match_pos = index_to_match_pos.get(idx)
        hit_vlass = meta_hits.get(match_pos, []) if match_pos is not None else []
        n_vlass = len(hit_vlass)
        record: dict = {
            "RA": row.get("RA", np.nan),
            "DEC": row.get("DEC", np.nan),
            "n_vlass": n_vlass,
            "vlass_positions": list(hit_vlass),
            "matched": n_vlass >= 1,
        }
        if "meta_id" in target.columns:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    if "meta_id" in meta_flags.columns:
        cols = ["meta_id", "RA", "DEC", "n_vlass", "vlass_positions", "matched"]
        meta_flags = meta_flags[cols]

    vlass_records: list[dict] = []
    for pos in range(len(vlass_footprint)):
        row = vlass_footprint.iloc[pos]
        hit_meta = vlass_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                idx = match_pos_to_index.get(match_pos)
                if idx is not None:
                    meta_ids.append(target.loc[idx, "meta_id"])
        vlass_records.append(
            {
                "vlass_pos": pos,
                "RA": row["RA"],
                "DEC": row["DEC"],
                "Peak_flux": row.get("Peak_flux", np.nan),
                "Component_name": row.get("Component_name", ""),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    vlass_cols = [
        "vlass_pos",
        "RA",
        "DEC",
        "Peak_flux",
        "Component_name",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        vlass_cols = [c for c in vlass_cols if c != "meta_ids"]
    if vlass_records:
        vlass_flags = pd.DataFrame(vlass_records)[vlass_cols]
    else:
        vlass_flags = pd.DataFrame(columns=vlass_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_meta_unique = int((meta_flags["n_vlass"] == 1).sum()) if not meta_flags.empty else 0
    n_vlass_matched = int((vlass_flags["n_meta"] > 0).sum()) if not vlass_flags.empty else 0
    n_vlass_oversplit = int(vlass_flags["oversplit"].sum()) if not vlass_flags.empty else 0
    n_meta_multi_vlass = int((meta_flags["n_vlass"] > 1).sum()) if not meta_flags.empty else 0
    meta_vlass_hits_max = int(meta_flags["n_vlass"].max()) if not meta_flags.empty else 0

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_vlass_footprint": n_vlass_footprint,
        "n_meta_matched": n_meta_matched,
        "match_completeness": n_meta_matched / n_lwa_target,
        "n_meta_unique": n_meta_unique,
        "unique_match_fraction": n_meta_unique / n_lwa_target,
        "n_vlass_matched": n_vlass_matched,
        "vlass_recovery": (
            n_vlass_matched / n_vlass_footprint if n_vlass_footprint else float("nan")
        ),
        "n_vlass_oversplit": n_vlass_oversplit,
        "n_meta_multi_vlass": n_meta_multi_vlass,
        "meta_vlass_hits_max": meta_vlass_hits_max,
    }

    return VlassMatchResult(
        summary=summary,
        meta_flags=meta_flags,
        vlass_flags=vlass_flags,
        vlass_footprint=vlass_footprint,
        warnings=warnings,
    )


def summarize_vlass_match(result: VlassMatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:              {int(s['n_lwa_target']):6d}",
        f"VLASS footprint (Dec box):    {int(s['n_vlass_footprint']):6d}",
        f"Meta matched (>=1 VLASS):     {int(s['n_meta_matched']):6d}",
        f"Match completeness:           {s['match_completeness']:.3f}",
        f"Meta unique match (n=1):      {int(s['n_meta_unique']):6d}",
        f"Unique match fraction:        {s['unique_match_fraction']:.3f}",
        f"VLASS matched (>=1 meta):     {int(s['n_vlass_matched']):6d}",
        f"VLASS recovery:               {s['vlass_recovery']:.3f}",
        f"VLASS over-split (n_meta>1):  {int(s['n_vlass_oversplit']):6d}",
        f"Meta multi-VLASS (n_vlass>1): {int(s['n_meta_multi_vlass']):6d}",
        f"Max VLASS hits per meta:      {int(s['meta_vlass_hits_max']):6d}",
        "",
        f"VLASS reference frequency:    {VLASS_FREQ_HZ / 1e9:.2f} GHz",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)
