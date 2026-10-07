"""LoDeSS (LOFAR Decametre Sky Survey) cross-match against LWA metacatalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from lwa_catalog.analyze.crossmatch_radius import (
    CrossmatchRadiusSpec,
    LODES_REFERENCE_RADIUS_LOCALIZATION,
    LWA_CROSSMATCH_RADIUS_BEAM,
    apply_match_radius,
    catalog_match_frame,
)
from lwa_catalog.analyze.vlssr import select_blue_associated_rows
from lwa_catalog.constants import (
    LODES_BMAJ_DEG,
    LODES_DEC_MIN_DEG,
    LODES_DEFAULT_PATH,
    LODES_FREQ_HZ,
)
from lwa_catalog.create.merge import associate_catalogs

LodesTarget = Literal["metacatalog", "metacatalog_blue"]

# Columns kept from the PyBDSF gaul table for match / photometry / overlays.
_LODES_KEEP_COLUMNS: tuple[str, ...] = (
    "RA",
    "DEC",
    "E_RA",
    "E_DEC",
    "Total_flux",
    "E_Total_flux",
    "Peak_flux",
    "E_Peak_flux",
    "Maj",
    "Min",
    "PA",
    "S_Code",
    "Source_id",
    "Isl_id",
    "Gaus_id",
)


@dataclass(frozen=True)
class LodesMatchConfig:
    """Configuration for :func:`match_catalog_to_lodess`."""

    catalog_path: Path = LODES_DEFAULT_PATH
    target: LodesTarget = "metacatalog"
    dec_min_deg: float = LODES_DEC_MIN_DEG
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM
    reference_radius: CrossmatchRadiusSpec = LODES_REFERENCE_RADIUS_LOCALIZATION


@dataclass
class LodesMatchResult:
    """LoDeSS cross-match metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    lodess_flags: pd.DataFrame
    lodess_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _finalize_lodess_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and drop non-finite coordinates."""
    out = df.copy()
    keep = [c for c in _LODES_KEEP_COLUMNS if c in out.columns]
    if "RA" not in keep or "DEC" not in keep:
        msg = "LoDeSS table missing RA/DEC columns"
        raise ValueError(msg)
    out = out.loc[:, keep].copy()
    if "BMAJ" not in out.columns:
        out["BMAJ"] = LODES_BMAJ_DEG
    if "BMIN" not in out.columns:
        out["BMIN"] = LODES_BMAJ_DEG
    for col in (
        "RA",
        "DEC",
        "E_RA",
        "E_DEC",
        "Total_flux",
        "E_Total_flux",
        "Peak_flux",
        "E_Peak_flux",
        "Maj",
        "Min",
        "PA",
        "BMAJ",
        "BMIN",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    ra = out["RA"].to_numpy(dtype=float)
    dec = out["DEC"].to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return out.loc[ok].reset_index(drop=True)


def load_lodess_catalog(path: Path | str | None = None) -> pd.DataFrame:
    """Load the LoDeSS PyBDSF Gaussian catalog for cross-matching.

    Default path is the DR1 ``*.gaul.fits`` table. Also accepts parquet/CSV with
    ``RA``/``DEC`` columns. Returned columns include positions, fluxes (Jy),
    shape, and circular ``BMAJ``/``BMIN`` set to the survey ~45″ FWHM.
    Rows with non-finite ``RA``/``DEC`` are dropped.

    Parameters
    ----------
    path
        Catalog file path. Defaults to
        :data:`~lwa_catalog.constants.LODES_DEFAULT_PATH`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing.
    """
    catalog_path = Path(LODES_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = f"LoDeSS catalog not found: {catalog_path}"
        raise FileNotFoundError(msg)

    suffixes = "".join(catalog_path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        df = pd.read_parquet(catalog_path)
    elif suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        df = pd.read_csv(catalog_path)
    else:
        from astropy.table import Table

        df = Table.read(catalog_path).to_pandas()

    return _finalize_lodess_frame(df)


def select_unique_lodess_matches(meta_flags: pd.DataFrame) -> pd.DataFrame:
    """Return meta rows with exactly one LoDeSS match (``n_lodess == 1``)."""
    if meta_flags.empty or "n_lodess" not in meta_flags.columns:
        return meta_flags.iloc[0:0].copy()
    return meta_flags.loc[meta_flags["n_lodess"] == 1].copy()


def _catalog_match_frame(
    catalog: pd.DataFrame,
    spec: CrossmatchRadiusSpec,
) -> pd.DataFrame:
    return catalog_match_frame(catalog, spec)


def _footprint_filter_lodess(
    lodess: pd.DataFrame,
    lwa: pd.DataFrame,
    *,
    dec_min_deg: float,
) -> pd.DataFrame:
    """Keep LoDeSS rows in the LWA Dec box and above the survey Dec limit."""
    if lodess.empty:
        return lodess.copy()
    if lwa.empty or "DEC" not in lwa.columns:
        return lodess.iloc[0:0].copy()

    lwa_dec = pd.to_numeric(lwa["DEC"], errors="coerce")
    finite_lwa = lwa_dec[np.isfinite(lwa_dec.to_numpy(dtype=float))]
    if finite_lwa.empty:
        return lodess.iloc[0:0].copy()

    dec_min = max(float(finite_lwa.min()), float(dec_min_deg))
    dec_max = float(finite_lwa.max())
    lodess_dec = pd.to_numeric(lodess["DEC"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(lodess_dec) & (lodess_dec >= dec_min) & (lodess_dec <= dec_max)
    return lodess.loc[keep].copy()


def _select_lwa_target(lwa_catalog: pd.DataFrame, target: LodesTarget) -> pd.DataFrame:
    if target == "metacatalog":
        return lwa_catalog
    if target == "metacatalog_blue":
        return select_blue_associated_rows(lwa_catalog)
    msg = f"unsupported target: {target!r}"
    raise ValueError(msg)


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_lodess_footprint": 0,
        "n_meta_matched": 0,
        "match_completeness": float("nan"),
        "n_meta_unique": 0,
        "unique_match_fraction": float("nan"),
        "n_lodess_matched": 0,
        "lodess_recovery": float("nan"),
        "n_lodess_oversplit": 0,
        "n_meta_multi_lodess": 0,
        "meta_lodess_hits_max": 0,
    }


def match_catalog_to_lodess(
    lwa_catalog: pd.DataFrame,
    lodess: pd.DataFrame | None = None,
    *,
    config: LodesMatchConfig | None = None,
    lwa_match: pd.DataFrame | None = None,
) -> LodesMatchResult:
    """Cross-match an LWA catalog against LoDeSS and compute association metrics.

    Default target is the full input metacatalog. Matching uses primary
    ``RA``/``DEC`` and configured
    :class:`~lwa_catalog.analyze.crossmatch_radius.CrossmatchRadiusSpec`
    radii via :func:`~lwa_catalog.create.merge.associate_catalogs`. LoDeSS
    sources below ``config.dec_min_deg`` (default 20°) are excluded from the
    footprint.

    Parameters
    ----------
    lwa_catalog
        Metacatalog table.
    lodess
        Pre-loaded LoDeSS catalog. Loaded from ``config.catalog_path`` when omitted.
    config
        Match configuration.
    lwa_match
        Optional prebuilt match frame (``RA``/``DEC``/``BMAJ``), iloc-aligned
        with the selected target (cascaded bootstrap).

    Returns
    -------
    LodesMatchResult
        Summary fractions, per-meta flags, and per-LoDeSS flags.
    """
    cfg = config or LodesMatchConfig()
    warnings: list[str] = []

    if lodess is None:
        lodess = load_lodess_catalog(cfg.catalog_path)

    target = _select_lwa_target(lwa_catalog, cfg.target)
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty after band selection")
        return LodesMatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=["meta_id", "RA", "DEC", "n_lodess", "lodess_positions", "matched"]
            ),
            lodess_flags=pd.DataFrame(
                columns=[
                    "lodess_pos",
                    "RA",
                    "DEC",
                    "Peak_flux",
                    "Total_flux",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            lodess_footprint=pd.DataFrame(),
            warnings=warnings,
        )

    lodess_footprint = _footprint_filter_lodess(
        lodess, target, dec_min_deg=cfg.dec_min_deg
    )
    ref_match = apply_match_radius(lodess_footprint, cfg.reference_radius)
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
    n_lodess_footprint = len(lodess_footprint)

    meta_hits: dict[int, list[int]] = {}
    lodess_hits: dict[int, list[int]] = {}
    if not resolved_lwa_match.empty and not ref_match.empty:
        meta_hits, _ = associate_catalogs(resolved_lwa_match, ref_match)
        lodess_hits, _ = associate_catalogs(ref_match, resolved_lwa_match)

    index_to_match_pos = {
        idx: pos for pos, idx in enumerate(resolved_lwa_match.index.tolist())
    }
    match_pos_to_index = {pos: idx for idx, pos in index_to_match_pos.items()}
    has_meta_id = "meta_id" in target.columns

    meta_records: list[dict] = []
    for idx, row in target.iterrows():
        match_pos = index_to_match_pos.get(idx)
        hit_lodess = meta_hits.get(match_pos, []) if match_pos is not None else []
        n_lodess = len(hit_lodess)
        record: dict = {
            "RA": row.get("RA", np.nan),
            "DEC": row.get("DEC", np.nan),
            "n_lodess": n_lodess,
            "lodess_positions": list(hit_lodess),
            "matched": n_lodess >= 1,
        }
        if "meta_id" in target.columns:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    if "meta_id" in meta_flags.columns:
        cols = ["meta_id", "RA", "DEC", "n_lodess", "lodess_positions", "matched"]
        meta_flags = meta_flags[cols]

    lodess_records: list[dict] = []
    for pos in range(len(lodess_footprint)):
        row = lodess_footprint.iloc[pos]
        hit_meta = lodess_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                idx = match_pos_to_index.get(match_pos)
                if idx is not None:
                    meta_ids.append(target.loc[idx, "meta_id"])
        lodess_records.append(
            {
                "lodess_pos": pos,
                "RA": row["RA"],
                "DEC": row["DEC"],
                "Peak_flux": row.get("Peak_flux", np.nan),
                "Total_flux": row.get("Total_flux", np.nan),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    lodess_cols = [
        "lodess_pos",
        "RA",
        "DEC",
        "Peak_flux",
        "Total_flux",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        lodess_cols = [c for c in lodess_cols if c != "meta_ids"]
    if lodess_records:
        lodess_flags = pd.DataFrame(lodess_records)[lodess_cols]
    else:
        lodess_flags = pd.DataFrame(columns=lodess_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_meta_unique = int((meta_flags["n_lodess"] == 1).sum()) if not meta_flags.empty else 0
    n_lodess_matched = (
        int((lodess_flags["n_meta"] > 0).sum()) if not lodess_flags.empty else 0
    )
    n_lodess_oversplit = (
        int(lodess_flags["oversplit"].sum()) if not lodess_flags.empty else 0
    )
    n_meta_multi_lodess = (
        int((meta_flags["n_lodess"] > 1).sum()) if not meta_flags.empty else 0
    )
    meta_lodess_hits_max = (
        int(meta_flags["n_lodess"].max()) if not meta_flags.empty else 0
    )

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_lodess_footprint": n_lodess_footprint,
        "n_meta_matched": n_meta_matched,
        "match_completeness": n_meta_matched / n_lwa_target,
        "n_meta_unique": n_meta_unique,
        "unique_match_fraction": n_meta_unique / n_lwa_target,
        "n_lodess_matched": n_lodess_matched,
        "lodess_recovery": (
            n_lodess_matched / n_lodess_footprint if n_lodess_footprint else float("nan")
        ),
        "n_lodess_oversplit": n_lodess_oversplit,
        "n_meta_multi_lodess": n_meta_multi_lodess,
        "meta_lodess_hits_max": meta_lodess_hits_max,
    }

    return LodesMatchResult(
        summary=summary,
        meta_flags=meta_flags,
        lodess_flags=lodess_flags,
        lodess_footprint=lodess_footprint,
        warnings=warnings,
    )


def summarize_lodess_match(result: LodesMatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:               {int(s['n_lwa_target']):6d}",
        f"LoDeSS footprint (Dec box):    {int(s['n_lodess_footprint']):6d}",
        f"Meta matched (>=1 LoDeSS):     {int(s['n_meta_matched']):6d}",
        f"Match completeness:            {s['match_completeness']:.3f}",
        f"Meta unique match (n=1):       {int(s['n_meta_unique']):6d}",
        f"Unique match fraction:         {s['unique_match_fraction']:.3f}",
        f"LoDeSS matched (>=1 meta):     {int(s['n_lodess_matched']):6d}",
        f"LoDeSS recovery:               {s['lodess_recovery']:.3f}",
        f"LoDeSS over-split (n_meta>1):  {int(s['n_lodess_oversplit']):6d}",
        f"Meta multi-LoDeSS (n>1):       {int(s['n_meta_multi_lodess']):6d}",
        f"Max LoDeSS hits per meta:      {int(s['meta_lodess_hits_max']):6d}",
        "",
        f"LoDeSS reference frequency:    {LODES_FREQ_HZ / 1e6:.0f} MHz",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)
