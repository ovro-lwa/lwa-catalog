"""Cascaded survey position bootstrap for radio cross-matching.

Build temporary match frames that adopt bijective survey counterparts'
coordinates and uncertainties (VLSSR → NVSS → VLASS) without rewriting
metacatalog astrometry.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from lwa_catalog.analyze.crossmatch_radius import (
    CrossmatchRadiusSpec,
    match_radius_deg,
)


def native_lwa_frame(
    catalog: pd.DataFrame,
    spec: CrossmatchRadiusSpec,
) -> pd.DataFrame:
    """Build a positional match frame aligned to *catalog* row order (iloc).

    Every input row appears once. Non-finite coordinates yield NaN ``RA``/``DEC``
    and zero ``BMAJ`` so :func:`~lwa_catalog.create.merge.associate_catalogs`
    skips them. ``bootstrap_source`` is ``\"LWA\"``.
    """
    records: list[dict[str, Any]] = []
    for _, row in catalog.iterrows():
        try:
            ra = float(row["RA"])
            dec = float(row["DEC"])
        except (KeyError, TypeError, ValueError):
            ra, dec = float("nan"), float("nan")
        if not np.isfinite(ra) or not np.isfinite(dec):
            records.append(
                {
                    "RA": float("nan"),
                    "DEC": float("nan"),
                    "BMAJ": 0.0,
                    "bootstrap_source": "LWA",
                    "bootstrap_ref_pos": -1,
                }
            )
            continue
        radius = match_radius_deg(row, spec)
        if not np.isfinite(radius) or radius < 0.0:
            radius = 0.0
        records.append(
            {
                "RA": ra,
                "DEC": dec,
                "BMAJ": float(radius),
                "bootstrap_source": "LWA",
                "bootstrap_ref_pos": -1,
            }
        )
    if not records:
        return pd.DataFrame(
            columns=["RA", "DEC", "BMAJ", "bootstrap_source", "bootstrap_ref_pos"]
        )
    return pd.DataFrame(records)


def select_bijective_pairs(
    meta_flags: pd.DataFrame,
    ref_flags: pd.DataFrame,
    *,
    n_ref_col: str,
    positions_col: str,
    ref_pos_col: str,
) -> dict[int, int]:
    """Return ``{meta_iloc: ref_footprint_iloc}`` for bijective associations.

    A pair is bijective when the metacatalog row has exactly one reference hit
    and that reference source has exactly one metacatalog hit (``n_meta == 1``).
    """
    if meta_flags.empty or ref_flags.empty:
        return {}

    bijective_refs: set[int] = set()
    for _, row in ref_flags.iterrows():
        try:
            n_meta = int(row["n_meta"])
            ref_pos = int(row[ref_pos_col])
        except (KeyError, TypeError, ValueError):
            continue
        if n_meta == 1:
            bijective_refs.add(ref_pos)

    out: dict[int, int] = {}
    for i in range(len(meta_flags)):
        row = meta_flags.iloc[i]
        try:
            n_ref = int(row[n_ref_col])
        except (TypeError, ValueError):
            continue
        if n_ref != 1:
            continue
        positions = row.get(positions_col, [])
        if positions is None or (isinstance(positions, float) and not np.isfinite(positions)):
            continue
        if not hasattr(positions, "__len__") or len(positions) != 1:
            continue
        try:
            ref_pos = int(positions[0])
        except (TypeError, ValueError):
            continue
        if ref_pos in bijective_refs:
            out[i] = ref_pos
    return out


def advance_bootstrap_frame(
    base_frame: pd.DataFrame,
    ref_footprint: pd.DataFrame,
    bijective_map: Mapping[int, int],
    *,
    radius_spec: CrossmatchRadiusSpec,
    source: str,
) -> pd.DataFrame:
    """Adopt bijective *ref_footprint* coords/radii; keep prior frame otherwise.

    *base_frame* and returned frame are iloc-aligned. Rows in *bijective_map*
    take ``RA``/``DEC`` from the reference footprint and ``BMAJ`` from
    :func:`~lwa_catalog.analyze.crossmatch_radius.match_radius_deg` applied to
    that reference row with *radius_spec*.
    """
    if base_frame.empty:
        return base_frame.copy()

    out = base_frame.copy()
    if "bootstrap_source" not in out.columns:
        out["bootstrap_source"] = "LWA"
    if "bootstrap_ref_pos" not in out.columns:
        out["bootstrap_ref_pos"] = -1

    ra_loc = out.columns.get_loc("RA")
    dec_loc = out.columns.get_loc("DEC")
    bmaj_loc = out.columns.get_loc("BMAJ")
    src_loc = out.columns.get_loc("bootstrap_source")
    pos_loc = out.columns.get_loc("bootstrap_ref_pos")

    for meta_i, ref_pos in bijective_map.items():
        if meta_i < 0 or meta_i >= len(out):
            continue
        if ref_pos < 0 or ref_pos >= len(ref_footprint):
            continue
        ref_row = ref_footprint.iloc[ref_pos]
        try:
            ra = float(ref_row["RA"])
            dec = float(ref_row["DEC"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(ra) or not np.isfinite(dec):
            continue
        radius = match_radius_deg(ref_row, radius_spec)
        if not np.isfinite(radius) or radius < 0.0:
            radius = 0.0
        out.iat[meta_i, ra_loc] = ra
        out.iat[meta_i, dec_loc] = dec
        out.iat[meta_i, bmaj_loc] = float(radius)
        out.iat[meta_i, src_loc] = source
        out.iat[meta_i, pos_loc] = int(ref_pos)
    return out


def _match_result_columns(source: str) -> tuple[str, str, str, str]:
    """Return ``(n_ref_col, positions_col, ref_pos_col, footprint_attr)``."""
    key = source.upper()
    if key == "VLSSR":
        return "n_vlssr", "vlssr_positions", "vlssr_pos", "vlssr_footprint"
    if key == "NVSS":
        return "n_nvss", "nvss_positions", "nvss_pos", "nvss_footprint"
    if key == "VLASS":
        return "n_vlass", "vlass_positions", "vlass_pos", "vlass_footprint"
    msg = f"unsupported bootstrap source: {source!r}"
    raise ValueError(msg)


def advance_from_match(
    base_frame: pd.DataFrame,
    match_result: Any,
    radius_spec: CrossmatchRadiusSpec,
    *,
    source: str,
) -> pd.DataFrame:
    """Advance *base_frame* using bijective hits from a survey QA match result.

    *base_frame* must have one row per metacatalog row in the same order as
    ``match_result.meta_flags``.
    """
    n_ref_col, positions_col, ref_pos_col, footprint_attr = _match_result_columns(source)
    meta_flags = match_result.meta_flags
    ref_flags = getattr(match_result, f"{source.lower()}_flags")
    footprint = getattr(match_result, footprint_attr)

    if len(base_frame) != len(meta_flags):
        msg = (
            f"bootstrap frame length {len(base_frame)} != meta_flags length "
            f"{len(meta_flags)} for source {source}"
        )
        raise ValueError(msg)

    bijective = select_bijective_pairs(
        meta_flags,
        ref_flags,
        n_ref_col=n_ref_col,
        positions_col=positions_col,
        ref_pos_col=ref_pos_col,
    )
    return advance_bootstrap_frame(
        base_frame,
        footprint,
        bijective,
        radius_spec=radius_spec,
        source=source.upper(),
    )


def bootstrap_source_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count rows by ``bootstrap_source`` for notebook logging."""
    if frame.empty or "bootstrap_source" not in frame.columns:
        return {}
    counts = frame["bootstrap_source"].astype(str).value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def bijective_map_from_hits(
    meta_hits: Mapping[int, list[int]],
    ref_hits: Mapping[int, list[int]],
) -> dict[int, int]:
    """Build bijective ``meta_iloc → ref_iloc`` from mutual association hit maps."""
    out: dict[int, int] = {}
    for meta_i, refs in meta_hits.items():
        if len(refs) != 1:
            continue
        ref_i = int(refs[0])
        back = ref_hits.get(ref_i, [])
        if len(back) == 1 and int(back[0]) == int(meta_i):
            out[int(meta_i)] = ref_i
    return out
