"""Two-tier PyBDSF GAUL fusion: replace over-decomposed ``M`` with tier-2 ``S``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from lwa_catalog.create.merge import associate_catalogs

__all__ = [
    "DEFAULT_TIER2_BDSF_KW",
    "fuse_gaul_m_with_tier2_s",
    "merge_tier2_bdsf_kw",
]

# Overrides merged onto the tier-1 ``bdsf_kw`` for the high-threshold pass.
DEFAULT_TIER2_BDSF_KW: dict[str, Any] = {
    "thresh_isl": 7.0,
    "thresh_pix": 4.0,
}


def merge_tier2_bdsf_kw(
    tier1_bdsf_kw: Mapping[str, Any] | None,
    tier2_bdsf_kw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return tier-2 PyBDSF kwargs: tier-1 base + defaults + *tier2_bdsf_kw*."""
    out = dict(tier1_bdsf_kw or {})
    out.update(DEFAULT_TIER2_BDSF_KW)
    if tier2_bdsf_kw is not None:
        out.update(dict(tier2_bdsf_kw))
    return out


def _ensure_bmaj(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "BMAJ" in out.columns:
        out["BMAJ"] = pd.to_numeric(out["BMAJ"], errors="coerce").fillna(0.0)
    else:
        out["BMAJ"] = 0.0
    return out


def _pick_best_tier2_s(
    m_row: pd.Series,
    s_hits: pd.DataFrame,
) -> int:
    """Return iloc into *s_hits* for the preferred coincident ``S`` Gaussian."""
    if len(s_hits) == 1:
        return 0

    import astropy.units as u
    from astropy.coordinates import SkyCoord

    m_sc = SkyCoord(
        ra=float(m_row["RA"]) * u.deg,
        dec=float(m_row["DEC"]) * u.deg,
    )
    s_sc = SkyCoord(
        ra=s_hits["RA"].to_numpy(dtype=float) * u.deg,
        dec=s_hits["DEC"].to_numpy(dtype=float) * u.deg,
    )
    sep = m_sc.separation(s_sc).deg
    peak = pd.to_numeric(s_hits["Peak_flux"], errors="coerce").to_numpy(dtype=float)
    peak = np.where(np.isfinite(peak), peak, -np.inf)
    # Nearest on sky, then brightest Peak_flux
    order = np.lexsort((-peak, sep))
    return int(order[0])


def fuse_gaul_m_with_tier2_s(
    tier1: pd.DataFrame,
    tier2: pd.DataFrame,
) -> pd.DataFrame:
    """Replace tier-1 ``S_Code=M`` clumps with coincident tier-2 ``S`` Gaussians.

    For each multi-Gaussian (``M``) row in *tier1*, search *tier2* for
    ``S_Code=S`` matches within the usual beam radius
    (:func:`~lwa_catalog.create.merge.associate_catalogs`). All tier-1 ``M``
    rows that claim the same tier-2 ``S`` are dropped and that single ``S``
    row is inserted once. Unmatched ``M`` rows and all non-``M`` tier-1 rows
    are kept unchanged.

    Parameters
    ----------
    tier1, tier2
        GAUL-like catalogs (need ``RA``, ``DEC``, ``S_Code``, ``Peak_flux``,
        and ``BMAJ`` when available).

    Returns
    -------
    DataFrame
        Fused catalog (new index). Empty inputs are handled: empty *tier1*
        yields empty; empty *tier2* yields a copy of *tier1*.
    """
    if tier1 is None or tier1.empty:
        return pd.DataFrame() if tier1 is None else tier1.copy()
    if tier2 is None or tier2.empty:
        return tier1.copy()

    t1 = _ensure_bmaj(tier1.reset_index(drop=True))
    t2 = _ensure_bmaj(tier2.reset_index(drop=True))
    if "S_Code" not in t1.columns or "S_Code" not in t2.columns:
        return t1
    for col in ("RA", "DEC", "Peak_flux"):
        if col not in t1.columns or col not in t2.columns:
            return t1

    m_ilocs = np.flatnonzero(t1["S_Code"].astype(str) == "M")
    s_ilocs = np.flatnonzero(t2["S_Code"].astype(str) == "S")
    if m_ilocs.size == 0 or s_ilocs.size == 0:
        return t1

    m_work = t1.iloc[m_ilocs].reset_index(drop=True)
    s_work = t2.iloc[s_ilocs].reset_index(drop=True)
    hits, _matched = associate_catalogs(m_work, s_work)
    if not hits:
        return t1

    replaced_t1: set[int] = set()
    s_t2_to_add: set[int] = set()
    for m_work_i, s_list in hits.items():
        if not s_list:
            continue
        s_hits = s_work.iloc[list(s_list)].reset_index(drop=True)
        # Map reset s_hits row -> original s_work iloc
        s_work_local = _pick_best_tier2_s(m_work.iloc[int(m_work_i)], s_hits)
        s_work_i = int(s_list[s_work_local])
        t1_i = int(m_ilocs[int(m_work_i)])
        t2_i = int(s_ilocs[s_work_i])
        replaced_t1.add(t1_i)
        s_t2_to_add.add(t2_i)

    if not replaced_t1:
        return t1

    keep_mask = ~t1.index.isin(replaced_t1)
    parts = [t1.loc[keep_mask]]
    if s_t2_to_add:
        parts.append(t2.loc[sorted(s_t2_to_add)])
    out = pd.concat(parts, ignore_index=True)
    return out
