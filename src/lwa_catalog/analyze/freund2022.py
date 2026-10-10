"""Freund et al. 2022 ROSAT/2RXS stellar counterparts vs LWA metacatalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from lwa_catalog.analyze.crossmatch_radius import (
    FREUND2022_REFERENCE_RADIUS_LOCALIZATION,
    LWA_CROSSMATCH_RADIUS_BEAM,
    CrossmatchRadiusSpec,
    apply_match_radius,
    catalog_match_frame,
)
from lwa_catalog.constants import (
    FREUND2022_DEFAULT_PATH,
    FREUND2022_DEFAULT_PIJ_MIN,
    FREUND2022_DEFAULT_PSTELLAR_MIN,
    FREUND2022_POSITION_ERROR_DEFAULT_ARCSEC,
)
from lwa_catalog.create.merge import associate_catalogs

Freund2022Target = Literal["metacatalog"]

# CDS J/A+A/664/A105 main.dat / suppl.dat byte columns (1-indexed inclusive →
# 0-indexed half-open for pandas.read_fwf).
_FREUND2022_COLSPECS: list[tuple[int, int]] = [
    (0, 21),
    (22, 44),
    (45, 51),
    (52, 58),
    (59, 65),
    (66, 72),
    (73, 83),
    (84, 95),
    (96, 104),
    (105, 110),
    (111, 117),
    (118, 124),
    (125, 132),
    (133, 138),
]
_FREUND2022_NAMES: tuple[str, ...] = (
    "2RXS",
    "Match",
    "ePos",
    "Sep",
    "pstellar",
    "pij",
    "RA",
    "DEC",
    "FX",
    "HR",
    "Gmag",
    "BP_RP",
    "plx",
    "subdwarf",
)

FREUND2022_LOAD_COLUMNS: tuple[str, ...] = _FREUND2022_NAMES


@dataclass(frozen=True)
class Freund2022MatchConfig:
    """Configuration for :func:`match_catalog_to_freund2022`."""

    catalog_path: Path = FREUND2022_DEFAULT_PATH
    target: Freund2022Target = "metacatalog"
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM
    reference_radius: CrossmatchRadiusSpec = FREUND2022_REFERENCE_RADIUS_LOCALIZATION
    # Paper science cut (None disables that gate).
    pstellar_min: float | None = FREUND2022_DEFAULT_PSTELLAR_MIN
    pij_min: float | None = FREUND2022_DEFAULT_PIJ_MIN
    exclude_subdwarf: bool = True


@dataclass
class Freund2022MatchResult:
    """Freund+2022 cross-match metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    freund2022_flags: pd.DataFrame
    freund2022_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _finalize_freund2022_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes, derive localization axes, drop bad coordinates."""
    missing = [c for c in ("RA", "DEC") if c not in df.columns]
    if missing:
        msg = f"Freund+2022 table missing required columns: {missing}"
        raise ValueError(msg)

    keep = [c for c in FREUND2022_LOAD_COLUMNS if c in df.columns]
    out = df.loc[:, keep].copy()

    for col in ("2RXS", "Match"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()

    for col in ("ePos", "Sep", "pstellar", "pij", "RA", "DEC", "FX", "HR", "Gmag", "BP_RP", "plx"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "subdwarf" in out.columns:
        raw = out["subdwarf"].astype(str).str.strip().str.lower()
        out["subdwarf"] = raw.map({"true": True, "false": False}).astype("boolean")

    # Localization matcher uses E_RA/E_DEC (degrees). ePos is 1σ radial (arcsec).
    epos_arcsec = (
        pd.to_numeric(out["ePos"], errors="coerce")
        if "ePos" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    epos_deg = epos_arcsec.to_numpy(dtype=float) / 3600.0
    fallback_deg = FREUND2022_POSITION_ERROR_DEFAULT_ARCSEC / 3600.0
    epos_deg = np.where(np.isfinite(epos_deg) & (epos_deg > 0.0), epos_deg, fallback_deg)
    axis_sigma = epos_deg / np.sqrt(2.0)
    out["E_RA"] = axis_sigma
    out["E_DEC"] = axis_sigma
    # Alias for POS_ERR-style overlays / radius specs.
    epos_vals = epos_arcsec.to_numpy(dtype=float)
    out["POS_ERR"] = np.where(
        np.isfinite(epos_vals) & (epos_vals > 0.0),
        epos_vals,
        FREUND2022_POSITION_ERROR_DEFAULT_ARCSEC,
    )

    ra = out["RA"].to_numpy(dtype=float)
    dec = out["DEC"].to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return out.loc[ok].reset_index(drop=True)


def load_freund2022_catalog(path: Path | str | None = None) -> pd.DataFrame:
    """Load Freund et al. 2022 ROSAT stellar counterpart catalog (CDS main.dat).

    Fixed-width ASCII from VizieR ``J/A+A/664/A105/main``. ``ePos`` (arcsec)
    becomes ``E_RA``/``E_DEC`` (degrees) and ``POS_ERR`` for localization
    matching / overlays. Rows with non-finite ``RA``/``DEC`` are dropped.

    Parameters
    ----------
    path
        Catalog path. Defaults to
        :data:`~lwa_catalog.constants.FREUND2022_DEFAULT_PATH`.
    """
    catalog_path = Path(FREUND2022_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = f"Freund+2022 catalog not found: {catalog_path}"
        raise FileNotFoundError(msg)

    raw = pd.read_fwf(
        catalog_path,
        colspecs=_FREUND2022_COLSPECS,
        names=list(_FREUND2022_NAMES),
        dtype=str,
    )
    return _finalize_freund2022_frame(raw)


def filter_freund2022_science_sample(
    catalog: pd.DataFrame,
    *,
    pstellar_min: float | None = FREUND2022_DEFAULT_PSTELLAR_MIN,
    pij_min: float | None = FREUND2022_DEFAULT_PIJ_MIN,
    exclude_subdwarf: bool = True,
) -> pd.DataFrame:
    """Apply the Freund+2022 paper science cut (or a subset of its gates)."""
    if catalog.empty:
        return catalog.copy()
    keep = np.ones(len(catalog), dtype=bool)
    if pstellar_min is not None and "pstellar" in catalog.columns:
        pstellar = pd.to_numeric(catalog["pstellar"], errors="coerce").to_numpy(dtype=float)
        keep &= np.isfinite(pstellar) & (pstellar > float(pstellar_min))
    if pij_min is not None and "pij" in catalog.columns:
        pij = pd.to_numeric(catalog["pij"], errors="coerce").to_numpy(dtype=float)
        keep &= np.isfinite(pij) & (pij > float(pij_min))
    if exclude_subdwarf and "subdwarf" in catalog.columns:
        sub = catalog["subdwarf"]
        # Keep False / missing; drop True.
        keep &= ~sub.fillna(False).astype(bool).to_numpy()
    return catalog.loc[keep].reset_index(drop=True)


def select_unique_freund2022_matches(meta_flags: pd.DataFrame) -> pd.DataFrame:
    """Return meta rows with exactly one Freund+2022 match."""
    if meta_flags.empty or "n_freund2022" not in meta_flags.columns:
        return meta_flags.iloc[0:0].copy()
    return meta_flags.loc[meta_flags["n_freund2022"] == 1].copy()


def _catalog_match_frame(
    catalog: pd.DataFrame,
    spec: CrossmatchRadiusSpec,
) -> pd.DataFrame:
    return catalog_match_frame(catalog, spec)


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_freund2022_footprint": 0,
        "n_meta_matched": 0,
        "match_completeness": float("nan"),
        "n_meta_unique": 0,
        "unique_match_fraction": float("nan"),
        "n_freund2022_matched": 0,
        "freund2022_recovery": float("nan"),
        "n_freund2022_oversplit": 0,
        "n_meta_multi_freund2022": 0,
        "meta_freund2022_hits_max": 0,
    }


def match_catalog_to_freund2022(
    lwa_catalog: pd.DataFrame,
    freund2022: pd.DataFrame | None = None,
    *,
    config: Freund2022MatchConfig | None = None,
    lwa_match: pd.DataFrame | None = None,
) -> Freund2022MatchResult:
    """Cross-match an LWA catalog against Freund+2022 ROSAT stellar counterparts.

    Matching uses ``associate_catalogs`` with configured
    :class:`~lwa_catalog.analyze.crossmatch_radius.CrossmatchRadiusSpec` radii.
    Optional science cuts (``pstellar`` / ``pij`` / subdwarf) filter the
    reference catalog before association.
    """
    cfg = config or Freund2022MatchConfig()
    warnings: list[str] = []

    if freund2022 is None:
        freund2022 = load_freund2022_catalog(cfg.catalog_path)

    n_loaded = len(freund2022)
    footprint = filter_freund2022_science_sample(
        freund2022,
        pstellar_min=cfg.pstellar_min,
        pij_min=cfg.pij_min,
        exclude_subdwarf=cfg.exclude_subdwarf,
    )
    if len(footprint) < n_loaded:
        warnings.append(
            f"science cut kept {len(footprint):,} / {n_loaded:,} Freund+2022 rows "
            f"(pstellar>{cfg.pstellar_min}, pij>{cfg.pij_min}, "
            f"exclude_subdwarf={cfg.exclude_subdwarf})"
        )

    target = lwa_catalog
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty")
        return Freund2022MatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=[
                    "meta_id",
                    "RA",
                    "DEC",
                    "n_freund2022",
                    "freund2022_positions",
                    "matched",
                ]
            ),
            freund2022_flags=pd.DataFrame(
                columns=[
                    "freund2022_pos",
                    "RA",
                    "DEC",
                    "2RXS",
                    "FX",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            freund2022_footprint=pd.DataFrame(),
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

    n_freund2022_footprint = len(footprint)

    meta_hits: dict[int, list[int]] = {}
    freund_hits: dict[int, list[int]] = {}
    if not resolved_lwa_match.empty and not ref_match.empty:
        meta_hits, _ = associate_catalogs(resolved_lwa_match, ref_match)
        freund_hits, _ = associate_catalogs(ref_match, resolved_lwa_match)

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
            "n_freund2022": n_hit,
            "freund2022_positions": list(hit_ref),
            "matched": n_hit >= 1,
        }
        if has_meta_id:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    cols = ["meta_id", "RA", "DEC", "n_freund2022", "freund2022_positions", "matched"]
    if "meta_id" not in meta_flags.columns:
        cols = [c for c in cols if c != "meta_id"]
    if not meta_flags.empty:
        meta_flags = meta_flags[cols]

    freund_records: list[dict] = []
    for pos in range(len(footprint)):
        row = footprint.iloc[pos]
        hit_meta = freund_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                midx = match_pos_to_index.get(match_pos)
                if midx is not None:
                    meta_ids.append(target.loc[midx, "meta_id"])
        freund_records.append(
            {
                "freund2022_pos": pos,
                "RA": row["RA"],
                "DEC": row["DEC"],
                "2RXS": row.get("2RXS", ""),
                "FX": row.get("FX", np.nan),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    freund_cols = [
        "freund2022_pos",
        "RA",
        "DEC",
        "2RXS",
        "FX",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        freund_cols = [c for c in freund_cols if c != "meta_ids"]
    if freund_records:
        freund2022_flags = pd.DataFrame(freund_records)[freund_cols]
    else:
        freund2022_flags = pd.DataFrame(columns=freund_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_meta_unique = (
        int((meta_flags["n_freund2022"] == 1).sum()) if not meta_flags.empty else 0
    )
    n_freund_matched = (
        int((freund2022_flags["n_meta"] > 0).sum()) if not freund2022_flags.empty else 0
    )
    n_freund_oversplit = (
        int(freund2022_flags["oversplit"].sum()) if not freund2022_flags.empty else 0
    )
    n_meta_multi = (
        int((meta_flags["n_freund2022"] > 1).sum()) if not meta_flags.empty else 0
    )
    meta_hits_max = (
        int(meta_flags["n_freund2022"].max()) if not meta_flags.empty else 0
    )

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_freund2022_footprint": n_freund2022_footprint,
        "n_meta_matched": n_meta_matched,
        "match_completeness": n_meta_matched / n_lwa_target,
        "n_meta_unique": n_meta_unique,
        "unique_match_fraction": n_meta_unique / n_lwa_target,
        "n_freund2022_matched": n_freund_matched,
        "freund2022_recovery": (
            n_freund_matched / n_freund2022_footprint
            if n_freund2022_footprint
            else float("nan")
        ),
        "n_freund2022_oversplit": n_freund_oversplit,
        "n_meta_multi_freund2022": n_meta_multi,
        "meta_freund2022_hits_max": meta_hits_max,
    }

    return Freund2022MatchResult(
        summary=summary,
        meta_flags=meta_flags,
        freund2022_flags=freund2022_flags,
        freund2022_footprint=footprint,
        warnings=warnings,
    )


def summarize_freund2022_match(result: Freund2022MatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:                  {int(s['n_lwa_target']):6d}",
        f"Freund+2022 science sample:       {int(s['n_freund2022_footprint']):6d}",
        f"Meta matched (>=1 Freund+2022):   {int(s['n_meta_matched']):6d}",
        f"Match completeness:               {s['match_completeness']:.3f}",
        f"Meta unique match (n=1):          {int(s['n_meta_unique']):6d}",
        f"Unique match fraction:            {s['unique_match_fraction']:.3f}",
        f"Freund+2022 matched (>=1 meta):   {int(s['n_freund2022_matched']):6d}",
        f"Freund+2022 recovery:             {s['freund2022_recovery']:.3f}",
        f"Freund+2022 over-split (n>1):     {int(s['n_freund2022_oversplit']):6d}",
        f"Meta multi-Freund+2022 (n>1):     {int(s['n_meta_multi_freund2022']):6d}",
        f"Max Freund+2022 hits per meta:    {int(s['meta_freund2022_hits_max']):6d}",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)
