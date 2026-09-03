"""VLSSR cross-match QA against LWA metacatalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from lwa_catalog.analyze.crossmatch_radius import (
    CrossmatchRadiusSpec,
    LWA_CROSSMATCH_RADIUS_BEAM,
    VLSSR_REFERENCE_RADIUS_FIXED,
    apply_match_radius,
    catalog_match_frame,
)
from lwa_catalog.analyze.reliability import parse_bands_present
from lwa_catalog.constants import VLSSR_BMAJ_DEG, VLSSR_DEFAULT_PATH
from lwa_catalog.create.merge import associate_catalogs

VlssrTarget = Literal["metacatalog", "metacatalog_blue", "lst_merged_blue"]


@dataclass(frozen=True)
class VlssrMatchConfig:
    """Configuration for :func:`match_catalog_to_vlssr`."""

    catalog_path: Path = VLSSR_DEFAULT_PATH
    target: VlssrTarget = "metacatalog_blue"
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM
    reference_radius: CrossmatchRadiusSpec = VLSSR_REFERENCE_RADIUS_FIXED


@dataclass
class VlssrMatchResult:
    """VLSSR cross-match QA metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    vlssr_flags: pd.DataFrame
    vlssr_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def load_vlssr_catalog(path: Path | str | None = None) -> pd.DataFrame:
    """Load the VLSSR reference catalog from a whitespace-delimited text file.

    The file header is ``RA(2000) DEC(2000) "PEAK INT"``. Returned columns are
    ``RA``, ``DEC``, ``Peak_flux`` (Jy), ``BMAJ``, and ``BMIN`` (circular 80″
    PSF). Rows with non-finite RA/DEC are dropped.

    Parameters
    ----------
    path
        Catalog file path. Defaults to :data:`~lwa_catalog.constants.VLSSR_DEFAULT_PATH`.

    Returns
    -------
    pd.DataFrame
        VLSSR sources ready for :func:`~lwa_catalog.create.merge.associate_catalogs`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    catalog_path = Path(VLSSR_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = f"VLSSR catalog not found: {catalog_path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(
        catalog_path,
        sep=r"\s+",
        skipinitialspace=True,
        quotechar='"',
    )
    df.columns = ["RA", "DEC", "Peak_flux"]
    df["BMAJ"] = VLSSR_BMAJ_DEG
    df["BMIN"] = VLSSR_BMAJ_DEG

    ra = pd.to_numeric(df["RA"], errors="coerce")
    dec = pd.to_numeric(df["DEC"], errors="coerce")
    ok = np.isfinite(ra.to_numpy(dtype=float)) & np.isfinite(dec.to_numpy(dtype=float))
    return df.loc[ok].reset_index(drop=True)


def select_blue_associated_rows(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return metacatalog rows with ``Blue`` in ``bands_present``.

    Original row indices are preserved for cross-match hit mapping.

    Raises
    ------
    ValueError
        If *catalog* lacks a ``bands_present`` column.
    """
    if "bands_present" not in catalog.columns:
        msg = "catalog missing bands_present column"
        raise ValueError(msg)

    keep: list[object] = []
    for idx, row in catalog.iterrows():
        if "Blue" in parse_bands_present(row):
            keep.append(idx)
    return catalog.loc[keep]


def _catalog_match_frame(
    catalog: pd.DataFrame,
    spec: CrossmatchRadiusSpec,
) -> pd.DataFrame:
    return catalog_match_frame(catalog, spec)


def _footprint_filter_vlssr(vlssr: pd.DataFrame, lwa: pd.DataFrame) -> pd.DataFrame:
    """Keep VLSSR rows whose Dec lies within the finite Dec range of *lwa*."""
    if vlssr.empty:
        return vlssr.copy()
    if lwa.empty or "DEC" not in lwa.columns:
        return vlssr.iloc[0:0].copy()

    lwa_dec = pd.to_numeric(lwa["DEC"], errors="coerce")
    finite_lwa = lwa_dec[np.isfinite(lwa_dec.to_numpy(dtype=float))]
    if finite_lwa.empty:
        return vlssr.iloc[0:0].copy()

    dec_min = float(finite_lwa.min())
    dec_max = float(finite_lwa.max())
    vlssr_dec = pd.to_numeric(vlssr["DEC"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(vlssr_dec) & (vlssr_dec >= dec_min) & (vlssr_dec <= dec_max)
    return vlssr.loc[keep].copy()


def _select_lwa_target(lwa_catalog: pd.DataFrame, target: VlssrTarget) -> pd.DataFrame:
    if target in {"metacatalog", "lst_merged_blue"}:
        return lwa_catalog
    if target == "metacatalog_blue":
        return select_blue_associated_rows(lwa_catalog)
    msg = f"unsupported target: {target!r}"
    raise ValueError(msg)


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_vlssr_footprint": 0,
        "n_meta_matched": 0,
        "blue_completeness": float("nan"),
        "n_vlssr_matched": 0,
        "vlssr_recovery": float("nan"),
        "n_vlssr_oversplit": 0,
        "n_meta_multi_vlssr": 0,
        "meta_vlssr_hits_max": 0,
    }


def match_catalog_to_vlssr(
    lwa_catalog: pd.DataFrame,
    vlssr: pd.DataFrame | None = None,
    *,
    config: VlssrMatchConfig | None = None,
) -> VlssrMatchResult:
    """Cross-match an LWA catalog against VLSSR and compute QA metrics.

    Default target is Blue-associated metacatalog rows
    (``config.target == "metacatalog_blue"``). Use ``target="metacatalog"`` to
    match every input row. Matching uses primary ``RA``/``DEC``
    and beam radii via :func:`~lwa_catalog.create.merge.associate_catalogs`.

    Parameters
    ----------
    lwa_catalog
        Metacatalog or LST-merged Blue table.
    vlssr
        Pre-loaded VLSSR catalog. Loaded from ``config.catalog_path`` when omitted.
    config
        Match configuration.

    Returns
    -------
    VlssrMatchResult
        Summary fractions, per-meta flags, and per-VLSSR flags.
    """
    cfg = config or VlssrMatchConfig()
    warnings: list[str] = []

    if vlssr is None:
        vlssr = load_vlssr_catalog(cfg.catalog_path)

    target = _select_lwa_target(lwa_catalog, cfg.target)
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty after band selection")
        return VlssrMatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=["meta_id", "RA", "DEC", "n_vlssr", "vlssr_positions", "matched"]
            ),
            vlssr_flags=pd.DataFrame(
                columns=[
                    "vlssr_pos",
                    "RA",
                    "DEC",
                    "Peak_flux",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            vlssr_footprint=pd.DataFrame(),
            warnings=warnings,
        )

    vlssr_footprint = _footprint_filter_vlssr(vlssr, target)
    ref_match = apply_match_radius(vlssr_footprint, cfg.reference_radius)
    lwa_match = _catalog_match_frame(target, cfg.lwa_radius)
    n_vlssr_footprint = len(vlssr_footprint)

    meta_hits: dict[int, list[int]] = {}
    vlssr_hits: dict[int, list[int]] = {}
    if not lwa_match.empty and not ref_match.empty:
        meta_hits, _ = associate_catalogs(lwa_match, ref_match)
        vlssr_hits, _ = associate_catalogs(ref_match, lwa_match)

    index_to_match_pos = {idx: pos for pos, idx in enumerate(lwa_match.index.tolist())}
    match_pos_to_index = {pos: idx for idx, pos in index_to_match_pos.items()}
    has_meta_id = "meta_id" in target.columns

    meta_records: list[dict] = []
    for idx, row in target.iterrows():
        match_pos = index_to_match_pos.get(idx)
        hit_vlssr = meta_hits.get(match_pos, []) if match_pos is not None else []
        n_vlssr = len(hit_vlssr)
        record: dict = {
            "RA": row.get("RA", np.nan),
            "DEC": row.get("DEC", np.nan),
            "n_vlssr": n_vlssr,
            "vlssr_positions": list(hit_vlssr),
            "matched": n_vlssr >= 1,
        }
        if "meta_id" in target.columns:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    if "meta_id" in meta_flags.columns:
        cols = ["meta_id", "RA", "DEC", "n_vlssr", "vlssr_positions", "matched"]
        meta_flags = meta_flags[cols]

    vlssr_records: list[dict] = []
    for pos in range(len(vlssr_footprint)):
        row = vlssr_footprint.iloc[pos]
        hit_meta = vlssr_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                idx = match_pos_to_index.get(match_pos)
                if idx is not None:
                    meta_ids.append(target.loc[idx, "meta_id"])
        vlssr_records.append(
            {
                "vlssr_pos": pos,
                "RA": row["RA"],
                "DEC": row["DEC"],
                "Peak_flux": row.get("Peak_flux", np.nan),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    vlssr_cols = [
        "vlssr_pos",
        "RA",
        "DEC",
        "Peak_flux",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        vlssr_cols = [c for c in vlssr_cols if c != "meta_ids"]
    if vlssr_records:
        vlssr_flags = pd.DataFrame(vlssr_records)[vlssr_cols]
    else:
        vlssr_flags = pd.DataFrame(columns=vlssr_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_vlssr_matched = int((vlssr_flags["n_meta"] > 0).sum()) if not vlssr_flags.empty else 0
    n_vlssr_oversplit = int(vlssr_flags["oversplit"].sum()) if not vlssr_flags.empty else 0
    n_meta_multi_vlssr = int((meta_flags["n_vlssr"] > 1).sum()) if not meta_flags.empty else 0
    meta_vlssr_hits_max = int(meta_flags["n_vlssr"].max()) if not meta_flags.empty else 0

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_vlssr_footprint": n_vlssr_footprint,
        "n_meta_matched": n_meta_matched,
        "blue_completeness": n_meta_matched / n_lwa_target,
        "n_vlssr_matched": n_vlssr_matched,
        "vlssr_recovery": (
            n_vlssr_matched / n_vlssr_footprint if n_vlssr_footprint else float("nan")
        ),
        "n_vlssr_oversplit": n_vlssr_oversplit,
        "n_meta_multi_vlssr": n_meta_multi_vlssr,
        "meta_vlssr_hits_max": meta_vlssr_hits_max,
    }

    return VlssrMatchResult(
        summary=summary,
        meta_flags=meta_flags,
        vlssr_flags=vlssr_flags,
        vlssr_footprint=vlssr_footprint,
        warnings=warnings,
    )


def summarize_vlssr_match(result: VlssrMatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:              {int(s['n_lwa_target']):6d}",
        f"VLSSR footprint (Dec box):    {int(s['n_vlssr_footprint']):6d}",
        f"Meta matched (>=1 VLSSR):     {int(s['n_meta_matched']):6d}",
        f"Blue completeness:            {s['blue_completeness']:.3f}",
        f"VLSSR matched (>=1 meta):     {int(s['n_vlssr_matched']):6d}",
        f"VLSSR recovery:               {s['vlssr_recovery']:.3f}",
        f"VLSSR over-split (n_meta>1):  {int(s['n_vlssr_oversplit']):6d}",
        f"Meta multi-VLSSR (n_vlssr>1): {int(s['n_meta_multi_vlssr']):6d}",
        f"Max VLSSR hits per meta:      {int(s['meta_vlssr_hits_max']):6d}",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)
