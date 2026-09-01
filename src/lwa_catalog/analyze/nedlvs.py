"""NED-LVS cross-match QA against LWA metacatalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from astropy.table import Table

from lwa_catalog.analyze.reliability import parse_bands_present, resolve_bmaj
from lwa_catalog.analyze.vlssr import select_blue_associated_rows
from lwa_catalog.constants import (
    NEDLVS_DEFAULT_BMAJ_ARCSEC,
    NEDLVS_DEFAULT_BMAJ_DEG,
    NEDLVS_DEFAULT_PATH,
)
from lwa_catalog.create.merge import associate_catalogs

NedlvsTarget = Literal["metacatalog", "metacatalog_blue", "lst_merged_blue"]

NEDLVS_LOAD_COLUMNS: tuple[str, ...] = (
    "objname",
    "ra",
    "dec",
    "objtype",
    "z",
    "DistMpc",
    "Diam",
    "Mstar",
)


@dataclass(frozen=True)
class NedlvsMatchConfig:
    """Configuration for :func:`match_catalog_to_nedlvs`."""

    catalog_path: Path = NEDLVS_DEFAULT_PATH
    target: NedlvsTarget = "metacatalog_blue"
    default_bmaj_arcsec: float = NEDLVS_DEFAULT_BMAJ_ARCSEC


@dataclass
class NedlvsMatchResult:
    """NED-LVS cross-match QA metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    nedlvs_flags: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _diam_to_bmaj_deg(diam_arcsec: np.ndarray, *, default_bmaj_deg: float) -> np.ndarray:
    """Convert NED-LVS ``Diam`` (major-axis arcsec) to match radius in degrees."""
    bmaj = np.full(diam_arcsec.shape, default_bmaj_deg, dtype=float)
    finite = np.isfinite(diam_arcsec) & (diam_arcsec > 0.0)
    bmaj[finite] = diam_arcsec[finite] / 3600.0
    return bmaj


def load_nedlvs_catalog(
    path: Path | str | None = None,
    *,
    default_bmaj_arcsec: float = NEDLVS_DEFAULT_BMAJ_ARCSEC,
) -> pd.DataFrame:
    """Load the NED-LVS FITS table for beam-radius cross-matching.

    Returns columns ``RA``, ``DEC``, ``BMAJ`` (degrees, from ``Diam`` when
    available), plus ``objname``, ``objtype``, ``z``, ``DistMpc``, ``Diam_arcsec``,
    and ``Mstar``. Rows with non-finite ``RA``/``DEC`` are dropped.

    Parameters
    ----------
    path
        FITS file path. Defaults to
        :data:`~lwa_catalog.constants.NEDLVS_DEFAULT_PATH`.
    default_bmaj_arcsec
        Match radius used when ``Diam`` is missing or non-positive.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
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
    default_bmaj_deg = float(default_bmaj_arcsec) / 3600.0
    df["Diam_arcsec"] = diam
    df["BMAJ"] = _diam_to_bmaj_deg(diam, default_bmaj_deg=default_bmaj_deg)
    df = df.drop(columns=["Diam"])

    ok = np.isfinite(ra.to_numpy(dtype=float)) & np.isfinite(dec.to_numpy(dtype=float))
    return df.loc[ok].reset_index(drop=True)


def _footprint_filter_nedlvs(nedlvs: pd.DataFrame, lwa: pd.DataFrame) -> pd.DataFrame:
    """Keep NED-LVS rows whose Dec lies within the finite Dec range of *lwa*."""
    if nedlvs.empty:
        return nedlvs.copy()
    if lwa.empty or "DEC" not in lwa.columns:
        return nedlvs.iloc[0:0].copy()

    lwa_dec = pd.to_numeric(lwa["DEC"], errors="coerce")
    finite_lwa = lwa_dec[np.isfinite(lwa_dec.to_numpy(dtype=float))]
    if finite_lwa.empty:
        return nedlvs.iloc[0:0].copy()

    dec_min = float(finite_lwa.min())
    dec_max = float(finite_lwa.max())
    nedlvs_dec = pd.to_numeric(nedlvs["DEC"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(nedlvs_dec) & (nedlvs_dec >= dec_min) & (nedlvs_dec <= dec_max)
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
        "n_nedlvs_matched": 0,
        "nedlvs_recovery": float("nan"),
        "n_nedlvs_oversplit": 0,
        "n_meta_multi_nedlvs": 0,
        "meta_nedlvs_hits_max": 0,
    }


def match_catalog_to_nedlvs(
    lwa_catalog: pd.DataFrame,
    nedlvs: pd.DataFrame | None = None,
    *,
    config: NedlvsMatchConfig | None = None,
) -> NedlvsMatchResult:
    """Cross-match an LWA catalog against NED-LVS and compute QA metrics.

    Default target is Blue-associated metacatalog rows
    (``config.target == "metacatalog_blue"``). Matching uses primary ``RA``/``DEC``
    and beam radii via :func:`~lwa_catalog.create.merge.associate_catalogs`.
    NED-LVS rows use ``Diam`` (major-axis arcsec) as ``BMAJ`` when available.

    Parameters
    ----------
    lwa_catalog
        Metacatalog or LST-merged Blue table.
    nedlvs
        Pre-loaded NED-LVS catalog. Loaded from ``config.catalog_path`` when omitted.
    config
        Match configuration.

    Returns
    -------
    NedlvsMatchResult
        Summary fractions, per-meta flags, and per-NED-LVS flags.
    """
    cfg = config or NedlvsMatchConfig()
    warnings: list[str] = []

    if nedlvs is None:
        nedlvs = load_nedlvs_catalog(
            cfg.catalog_path,
            default_bmaj_arcsec=cfg.default_bmaj_arcsec,
        )

    target = _select_lwa_target(lwa_catalog, cfg.target)
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty after band selection")
        return NedlvsMatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=["meta_id", "RA", "DEC", "n_nedlvs", "matched"]
            ),
            nedlvs_flags=pd.DataFrame(
                columns=[
                    "nedlvs_pos",
                    "RA",
                    "DEC",
                    "objname",
                    "DistMpc",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            warnings=warnings,
        )

    nedlvs_footprint = _footprint_filter_nedlvs(nedlvs, target)
    lwa_match = _catalog_match_frame(target)
    n_nedlvs_footprint = len(nedlvs_footprint)

    meta_hits: dict[int, list[int]] = {}
    nedlvs_hits: dict[int, list[int]] = {}
    if not lwa_match.empty and not nedlvs_footprint.empty:
        meta_hits, _ = associate_catalogs(lwa_match, nedlvs_footprint)
        nedlvs_hits, _ = associate_catalogs(nedlvs_footprint, lwa_match)

    index_to_match_pos = {idx: pos for pos, idx in enumerate(lwa_match.index.tolist())}
    match_pos_to_index = {pos: idx for idx, pos in index_to_match_pos.items()}
    has_meta_id = "meta_id" in target.columns

    meta_records: list[dict] = []
    for idx, row in target.iterrows():
        match_pos = index_to_match_pos.get(idx)
        hit_nedlvs = meta_hits.get(match_pos, []) if match_pos is not None else []
        n_nedlvs = len(hit_nedlvs)
        record: dict = {
            "RA": row.get("RA", np.nan),
            "DEC": row.get("DEC", np.nan),
            "n_nedlvs": n_nedlvs,
            "matched": n_nedlvs >= 1,
        }
        if "meta_id" in target.columns:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    if "meta_id" in meta_flags.columns:
        cols = ["meta_id", "RA", "DEC", "n_nedlvs", "matched"]
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
    n_nedlvs_matched = int((nedlvs_flags["n_meta"] > 0).sum()) if not nedlvs_flags.empty else 0
    n_nedlvs_oversplit = int(nedlvs_flags["oversplit"].sum()) if not nedlvs_flags.empty else 0
    n_meta_multi_nedlvs = int((meta_flags["n_nedlvs"] > 1).sum()) if not meta_flags.empty else 0
    meta_nedlvs_hits_max = int(meta_flags["n_nedlvs"].max()) if not meta_flags.empty else 0

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_nedlvs_footprint": n_nedlvs_footprint,
        "n_meta_matched": n_meta_matched,
        "meta_nedlvs_fraction": n_meta_matched / n_lwa_target,
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
        warnings=warnings,
    )


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


def summarize_nedlvs_match(result: NedlvsMatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:                {int(s['n_lwa_target']):6d}",
        f"NED-LVS footprint (Dec box):    {int(s['n_nedlvs_footprint']):6d}",
        f"Meta matched (>=1 NED-LVS):     {int(s['n_meta_matched']):6d}",
        f"Meta with NED-LVS host:         {s['meta_nedlvs_fraction']:.3f}",
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
