"""Cross-match radius configuration for external catalog association."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from lwa_catalog.analyze.reliability import resolve_bmaj
from lwa_catalog.constants import (
    NVSS_BMAJ_ARCSEC,
    VLASS_BMAJ_ARCSEC,
    VLSSR_BMAJ_ARCSEC,
)

CrossmatchRadiusMode = Literal["beam", "localization", "fixed"]


@dataclass(frozen=True, slots=True)
class CrossmatchRadiusSpec:
    """How to set the per-source match radius (stored as ``BMAJ`` in degrees).

    Pairwise matching uses ``sep ≤ max(radius_base, radius_band)`` via
    :func:`~lwa_catalog.create.merge.associate_catalogs`.

    Parameters
    ----------
    mode
        ``beam`` — restoring beam (``BMAJ_match`` / ``BMAJ`` on each row).
        ``localization`` — 1σ sky position uncertainty (``E_RA``/``E_DEC``,
        ``cluster_jitter_rms_deg``, or ``default_arcsec``).
        ``fixed`` — constant ``fixed_arcsec`` for every row.
    fixed_arcsec
        Match radius in arcseconds when ``mode == "fixed"``.
    sigma_scale
        Multiplier applied to the beam or 1σ localization radius (e.g. ``3``
        for a 3σ gate).
    default_arcsec
        Fallback radius in arcseconds when beam or localization columns are
        missing on a row.
    """

    mode: CrossmatchRadiusMode = "beam"
    fixed_arcsec: float | None = None
    sigma_scale: float = 1.0
    default_arcsec: float | None = None

    def __post_init__(self) -> None:
        if self.mode == "fixed" and self.fixed_arcsec is None:
            msg = "fixed_arcsec is required when mode='fixed'"
            raise ValueError(msg)
        if self.sigma_scale <= 0.0:
            msg = f"sigma_scale must be positive, got {self.sigma_scale!r}"
            raise ValueError(msg)


# Survey-specific defaults (reference side) for notebook / config reuse.
VLSSR_REFERENCE_RADIUS_BEAM = CrossmatchRadiusSpec(
    mode="beam",
    default_arcsec=VLSSR_BMAJ_ARCSEC,
)
NVSS_REFERENCE_RADIUS_BEAM = CrossmatchRadiusSpec(
    mode="beam",
    default_arcsec=NVSS_BMAJ_ARCSEC,
)
VLASS_REFERENCE_RADIUS_BEAM = CrossmatchRadiusSpec(
    mode="beam",
    default_arcsec=VLASS_BMAJ_ARCSEC,
)

LWA_CROSSMATCH_RADIUS_BEAM = CrossmatchRadiusSpec(mode="beam")
LWA_CROSSMATCH_RADIUS_LOCALIZATION = CrossmatchRadiusSpec(
    mode="localization",
    default_arcsec=5.0,
)


def _default_deg(spec: CrossmatchRadiusSpec) -> float:
    if spec.default_arcsec is None:
        return 0.0
    return float(spec.default_arcsec) / 3600.0


def _resolve_beam_deg(row: pd.Series) -> float:
    bmaj = resolve_bmaj(row)
    if np.isfinite(bmaj) and bmaj > 0.0:
        return float(bmaj)
    col = pd.to_numeric(row.get("BMAJ"), errors="coerce")
    if np.isfinite(col) and float(col) > 0.0:
        return float(col)
    return float("nan")


def _resolve_localization_sigma_deg(row: pd.Series, *, default_deg: float) -> float:
    e_ra = pd.to_numeric(row.get("E_RA"), errors="coerce")
    e_dec = pd.to_numeric(row.get("E_DEC"), errors="coerce")
    if (
        np.isfinite(e_ra)
        and np.isfinite(e_dec)
        and float(e_ra) > 0.0
        and float(e_dec) > 0.0
    ):
        return float(np.hypot(float(e_ra), float(e_dec)))

    jitter = pd.to_numeric(row.get("cluster_jitter_rms_deg"), errors="coerce")
    if np.isfinite(jitter) and float(jitter) > 0.0:
        return float(jitter)

    return float(default_deg)


def match_radius_deg(row: pd.Series, spec: CrossmatchRadiusSpec) -> float:
    """Return the match radius in degrees for one catalog row."""
    scale = float(spec.sigma_scale)
    fallback_deg = _default_deg(spec)

    if spec.mode == "fixed":
        assert spec.fixed_arcsec is not None
        return (float(spec.fixed_arcsec) / 3600.0) * scale

    if spec.mode == "beam":
        bmaj = _resolve_beam_deg(row)
        if not np.isfinite(bmaj) or bmaj <= 0.0:
            bmaj = fallback_deg
        return bmaj * scale

    sigma = _resolve_localization_sigma_deg(
        row,
        default_deg=fallback_deg if fallback_deg > 0.0 else 1.0 / 3600.0,
    )
    return sigma * scale


def apply_match_radius(
    catalog: pd.DataFrame,
    spec: CrossmatchRadiusSpec,
) -> pd.DataFrame:
    """Return a copy of *catalog* with ``BMAJ`` set to the match radius (degrees)."""
    if catalog.empty:
        return catalog.copy()
    out = catalog.copy()
    out["BMAJ"] = [match_radius_deg(row, spec) for _, row in catalog.iterrows()]
    return out


def catalog_match_frame(
    catalog: pd.DataFrame,
    spec: CrossmatchRadiusSpec,
) -> pd.DataFrame:
    """Build an ``RA`` / ``DEC`` / ``BMAJ`` frame for :func:`~lwa_catalog.create.merge.associate_catalogs`.

    ``BMAJ`` holds the configured match radius. Rows with non-finite coordinates
    are omitted. The index matches *catalog*.
    """
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
        radius = match_radius_deg(row, spec)
        if not np.isfinite(radius) or radius < 0.0:
            radius = 0.0
        records.append({"RA": ra, "DEC": dec, "BMAJ": radius})
        indices.append(idx)

    if not records:
        return pd.DataFrame(columns=["RA", "DEC", "BMAJ"])
    return pd.DataFrame(records, index=indices)


def describe_crossmatch_radius(spec: CrossmatchRadiusSpec) -> str:
    """One-line summary for notebook logging."""
    if spec.mode == "fixed":
        base = f"fixed {spec.fixed_arcsec}″"
    elif spec.mode == "localization":
        base = "localization (E_RA/E_DEC or jitter"
        if spec.default_arcsec is not None:
            base += f", default {spec.default_arcsec}″"
        base += ")"
    else:
        base = "beam (BMAJ"
        if spec.default_arcsec is not None:
            base += f", default {spec.default_arcsec}″"
        base += ")"
    if spec.sigma_scale != 1.0:
        base += f" × {spec.sigma_scale:g}σ"
    return base
