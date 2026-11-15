"""Cross-match radius configuration for external catalog association."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from lwa_catalog.analyze.reliability import resolve_bmaj
from lwa_catalog.constants import (
    ERASS3_POSITION_ERROR_DEFAULT_ARCSEC,
    FREUND2022_POSITION_ERROR_DEFAULT_ARCSEC,
    GLEAM_BMAJ_ARCSEC,
    GLEAM_POSITION_ERROR_DEFAULT_ARCSEC,
    LODES_BMAJ_ARCSEC,
    LODES_POSITION_ERROR_DEFAULT_ARCSEC,
    MILLIQUAS_POSITION_ERROR_DEFAULT_ARCSEC,
    NEDLVS_CATALOG_POSITION_SIGMA_ARCSEC,
    NEDLVS_DEFAULT_CENTROID_SIGMA_ARCSEC,
    NEDLVS_DEFAULT_POSITION_SIGMA_SCALE,
    NVSS_BMAJ_ARCSEC,
    NVSS_POSITION_ERROR_DEFAULT_ARCSEC,
    RFC_POSITION_ERROR_DEFAULT_ARCSEC,
    RXGCC_DEFAULT_EXTENT_ARCMIN,
    TEVCAT_DEFAULT_EXTENT_ARCMIN,
    TWORXS_POSITION_ERROR_DEFAULT_ARCSEC,
    VLASS_BMAJ_ARCSEC,
    VLSSR_BMAJ_ARCSEC,
    VLSSR_POSITION_ERROR_ARCSEC,
)

if TYPE_CHECKING:
    from lwa_catalog.paths import CatalogLayout

# 1σ defaults (degrees) when ``match_source`` is a cascaded radio survey and
# ``match_sigma_deg`` is absent (e.g. older metacatalog_spectral.parquet).
MATCH_SOURCE_SIGMA_DEG: dict[str, float] = {
    "VLSSR": VLSSR_POSITION_ERROR_ARCSEC / 3600.0,
    "NVSS": NVSS_POSITION_ERROR_DEFAULT_ARCSEC / 3600.0,
    "VLASS": VLASS_BMAJ_ARCSEC / 3600.0,
    "LoDeSS": LODES_POSITION_ERROR_DEFAULT_ARCSEC / 3600.0,
}

# Default fraction of high-frequency / astrometry-band BMAJ for LWA external match.
DEFAULT_PRECISE_BEAM_FRAC: float = 0.1

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
    precise_beam_frac
        When ``mode == "localization"`` and this is set, the radius is
        ``max(sigma_scale × σ_loc, precise_beam_frac × BMAJ_precise)`` using
        the row's ``BMAJ_precise`` column (degrees). Ignored for ``beam`` /
        ``fixed``. Attach ``BMAJ_precise`` via
        :func:`attach_bmaj_precise_from_lst`. Missing / non-finite
        ``BMAJ_precise`` falls back to localization only.
    """

    mode: CrossmatchRadiusMode = "beam"
    fixed_arcsec: float | None = None
    sigma_scale: float = 1.0
    default_arcsec: float | None = None
    precise_beam_frac: float | None = None

    def __post_init__(self) -> None:
        if self.mode == "fixed" and self.fixed_arcsec is None:
            msg = "fixed_arcsec is required when mode='fixed'"
            raise ValueError(msg)
        if self.sigma_scale <= 0.0:
            msg = f"sigma_scale must be positive, got {self.sigma_scale!r}"
            raise ValueError(msg)
        if self.precise_beam_frac is not None and self.precise_beam_frac <= 0.0:
            msg = (
                f"precise_beam_frac must be positive when set, "
                f"got {self.precise_beam_frac!r}"
            )
            raise ValueError(msg)


# Survey-specific defaults (reference side) for notebook / config reuse.
VLSSR_REFERENCE_RADIUS_FIXED = CrossmatchRadiusSpec(
    mode="fixed",
    fixed_arcsec=VLSSR_POSITION_ERROR_ARCSEC,
)
VLSSR_REFERENCE_RADIUS_BEAM = CrossmatchRadiusSpec(
    mode="beam",
    default_arcsec=VLSSR_BMAJ_ARCSEC,
)
NVSS_REFERENCE_RADIUS_LOCALIZATION = CrossmatchRadiusSpec(
    mode="localization",
    default_arcsec=NVSS_POSITION_ERROR_DEFAULT_ARCSEC,
)
NVSS_REFERENCE_RADIUS_BEAM = CrossmatchRadiusSpec(
    mode="beam",
    default_arcsec=NVSS_BMAJ_ARCSEC,
)
VLASS_REFERENCE_RADIUS_LOCALIZATION = CrossmatchRadiusSpec(
    mode="localization",
    default_arcsec=VLASS_BMAJ_ARCSEC,
)
VLASS_REFERENCE_RADIUS_BEAM = CrossmatchRadiusSpec(
    mode="beam",
    default_arcsec=VLASS_BMAJ_ARCSEC,
)
LODES_REFERENCE_RADIUS_LOCALIZATION = CrossmatchRadiusSpec(
    mode="localization",
    default_arcsec=LODES_POSITION_ERROR_DEFAULT_ARCSEC,
)
LODES_REFERENCE_RADIUS_BEAM = CrossmatchRadiusSpec(
    mode="beam",
    default_arcsec=LODES_BMAJ_ARCSEC,
)
GLEAM_REFERENCE_RADIUS_LOCALIZATION = CrossmatchRadiusSpec(
    mode="localization",
    default_arcsec=GLEAM_POSITION_ERROR_DEFAULT_ARCSEC,
)
GLEAM_REFERENCE_RADIUS_BEAM = CrossmatchRadiusSpec(
    mode="beam",
    default_arcsec=GLEAM_BMAJ_ARCSEC,
)
RFC_REFERENCE_RADIUS_LOCALIZATION = CrossmatchRadiusSpec(
    mode="localization",
    default_arcsec=RFC_POSITION_ERROR_DEFAULT_ARCSEC,
)
ERASS3_REFERENCE_RADIUS_LOCALIZATION = CrossmatchRadiusSpec(
    mode="localization",
    default_arcsec=ERASS3_POSITION_ERROR_DEFAULT_ARCSEC,
)
FREUND2022_REFERENCE_RADIUS_LOCALIZATION = CrossmatchRadiusSpec(
    mode="localization",
    default_arcsec=FREUND2022_POSITION_ERROR_DEFAULT_ARCSEC,
)
# RXGCC clusters: per-row BMAJ from R500* / Ext (arcmin → deg on load).
RXGCC_REFERENCE_RADIUS_BEAM = CrossmatchRadiusSpec(
    mode="beam",
    default_arcsec=RXGCC_DEFAULT_EXTENT_ARCMIN * 60.0,
)
TWORXS_REFERENCE_RADIUS_LOCALIZATION = CrossmatchRadiusSpec(
    mode="localization",
    default_arcsec=TWORXS_POSITION_ERROR_DEFAULT_ARCSEC,
)
# TeVCat: fixed 3′ match gate (independent of catalog size_x/size_y).
TEVCAT_REFERENCE_RADIUS_FIXED = CrossmatchRadiusSpec(
    mode="fixed",
    fixed_arcsec=TEVCAT_DEFAULT_EXTENT_ARCMIN * 60.0,
)
MILLIQUAS_REFERENCE_RADIUS_FIXED = CrossmatchRadiusSpec(
    mode="fixed",
    fixed_arcsec=MILLIQUAS_POSITION_ERROR_DEFAULT_ARCSEC,
)
# NED-LVS: LWA side uses Nσ localization (cascaded match_sigma / E_RA/E_DEC);
# reference side is a fixed NED catalog position uncertainty. Galaxy size is
# applied as a separate ``diam_scale`` floor on the reference ``BMAJ``.
NEDLVS_LWA_RADIUS = CrossmatchRadiusSpec(
    mode="localization",
    sigma_scale=NEDLVS_DEFAULT_POSITION_SIGMA_SCALE,
    default_arcsec=NEDLVS_DEFAULT_CENTROID_SIGMA_ARCSEC,
)
NEDLVS_REFERENCE_RADIUS_FIXED = CrossmatchRadiusSpec(
    mode="fixed",
    fixed_arcsec=NEDLVS_CATALOG_POSITION_SIGMA_ARCSEC,
)

LWA_CROSSMATCH_RADIUS_BEAM = CrossmatchRadiusSpec(mode="beam")
LWA_CROSSMATCH_RADIUS_LOCALIZATION = CrossmatchRadiusSpec(
    mode="localization",
    default_arcsec=NEDLVS_DEFAULT_CENTROID_SIGMA_ARCSEC,
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


def resolve_localization_sigma_deg(row: pd.Series, *, default_deg: float) -> float:
    """Return a 1σ sky-position radius in degrees for localization matching.

    Preference order:

    1. ``match_sigma_deg`` from cascaded radio attach (bootstrap frame).
    2. Survey default for ``match_source`` ``VLSSR`` / ``NVSS`` / ``VLASS`` /
       ``LoDeSS``.
    3. PyBDSF ``E_RA``/``E_DEC``.
    4. Merge-time ``cluster_jitter_rms_deg``.
    5. *default_deg*.
    """
    match_sigma = pd.to_numeric(row.get("match_sigma_deg"), errors="coerce")
    if np.isfinite(match_sigma) and float(match_sigma) > 0.0:
        return float(match_sigma)

    source = str(row.get("match_source", "") or "").strip().upper()
    if source in MATCH_SOURCE_SIGMA_DEG:
        return float(MATCH_SOURCE_SIGMA_DEG[source])

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

    sigma = resolve_localization_sigma_deg(
        row,
        default_deg=fallback_deg if fallback_deg > 0.0 else 1.0 / 3600.0,
    )
    loc_radius = sigma * scale
    frac = spec.precise_beam_frac
    if frac is None:
        return loc_radius
    precise = pd.to_numeric(row.get("BMAJ_precise"), errors="coerce")
    if not np.isfinite(precise) or float(precise) <= 0.0:
        return loc_radius
    return max(loc_radius, float(frac) * float(precise))


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


def resolve_match_coordinates(row: pd.Series) -> tuple[float, float]:
    """Return ``(RA, DEC)`` degrees, preferring ``match_RA``/``match_DEC``.

    Cascaded radio cross-match writes those columns when a bijective survey
    position is available. Falls back to top-level ``RA``/``DEC``.
    """
    for ra_key, dec_key in (("match_RA", "match_DEC"), ("RA", "DEC")):
        try:
            ra = float(pd.to_numeric(row.get(ra_key), errors="coerce"))
            dec = float(pd.to_numeric(row.get(dec_key), errors="coerce"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(ra) and np.isfinite(dec):
            return ra, dec
    return float("nan"), float("nan")


def catalog_match_frame(
    catalog: pd.DataFrame,
    spec: CrossmatchRadiusSpec,
    *,
    prefer_match_coordinates: bool = False,
) -> pd.DataFrame:
    """Build an ``RA`` / ``DEC`` / ``BMAJ`` frame for :func:`~lwa_catalog.create.merge.associate_catalogs`.

    ``BMAJ`` holds the configured match radius. Rows with non-finite coordinates
    are omitted. The index matches *catalog*.

    When *prefer_match_coordinates* is True, sky position comes from
    :func:`resolve_match_coordinates` (``match_RA``/``match_DEC`` when finite);
    the radius still uses the original row (beam / localization columns).
    """
    records: list[dict[str, float]] = []
    indices: list[object] = []
    for idx, row in catalog.iterrows():
        if prefer_match_coordinates:
            ra, dec = resolve_match_coordinates(row)
        else:
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
    if spec.mode == "localization" and spec.precise_beam_frac is not None:
        base += f"; max with {spec.precise_beam_frac:g}×BMAJ_precise"
    return base


def attach_bmaj_precise_from_lst(
    meta: pd.DataFrame,
    layout: CatalogLayout,
    *,
    band_col: str = "astrometry_band",
) -> pd.DataFrame:
    """Attach ``BMAJ_precise`` from LST-merged catalogs named by *band_col*.

    For each distinct band value in *band_col*, loads
    ``layout.lst_merged(band)`` and associates meta rows to that LST catalog
    with ``associate_catalogs`` (meta search scale = ``BMAJ_match`` when
    present, else the LST catalog median ``BMAJ``). Among hits, copies the
    **nearest** LST row's ``BMAJ`` (degrees). Missing LST files, blank band
    labels, or unmatched rows leave ``BMAJ_precise`` as NaN (localization-only
    fallback when ``precise_beam_frac`` is set).

    Does not write Parquet.
    """
    from lwa_catalog.create.merge import associate_catalogs

    out = meta.copy()
    out["BMAJ_precise"] = np.nan
    if out.empty or "RA" not in out.columns or "DEC" not in out.columns:
        return out
    if band_col not in out.columns:
        return out

    bands = out[band_col].fillna("").astype(str).str.strip()
    for band in sorted({b for b in bands.unique().tolist() if b and b.lower() != "nan"}):
        path = layout.lst_merged(band)
        if not path.is_file():
            continue
        lst = pd.read_parquet(path, columns=["RA", "DEC", "BMAJ"])
        if lst.empty or "BMAJ" not in lst.columns:
            continue
        lst = lst.reset_index(drop=True)
        lst_bmaj = pd.to_numeric(lst["BMAJ"], errors="coerce").to_numpy(dtype=float)
        finite_lst = lst_bmaj[np.isfinite(lst_bmaj) & (lst_bmaj > 0.0)]
        if finite_lst.size == 0:
            continue
        median_lst_bmaj = float(np.median(finite_lst))

        mask = bands == band
        subset = out.loc[mask]
        if subset.empty:
            continue

        if "BMAJ_match" in subset.columns:
            meta_bmaj = pd.to_numeric(subset["BMAJ_match"], errors="coerce").to_numpy(
                dtype=float
            )
        else:
            meta_bmaj = np.full(len(subset), np.nan, dtype=float)
        meta_bmaj = np.where(
            np.isfinite(meta_bmaj) & (meta_bmaj > 0.0), meta_bmaj, median_lst_bmaj
        )

        meta_frame = pd.DataFrame(
            {
                "RA": pd.to_numeric(subset["RA"], errors="coerce").to_numpy(dtype=float),
                "DEC": pd.to_numeric(subset["DEC"], errors="coerce").to_numpy(dtype=float),
                "BMAJ": meta_bmaj,
            }
        )
        lst_frame = pd.DataFrame(
            {
                "RA": pd.to_numeric(lst["RA"], errors="coerce").to_numpy(dtype=float),
                "DEC": pd.to_numeric(lst["DEC"], errors="coerce").to_numpy(dtype=float),
                "BMAJ": lst_bmaj,
            }
        )
        hits, _ = associate_catalogs(meta_frame, lst_frame)
        if not hits:
            continue

        meta_sc = SkyCoord(
            ra=meta_frame["RA"].to_numpy(dtype=float) * u.deg,
            dec=meta_frame["DEC"].to_numpy(dtype=float) * u.deg,
        )
        lst_sc = SkyCoord(
            ra=lst_frame["RA"].to_numpy(dtype=float) * u.deg,
            dec=lst_frame["DEC"].to_numpy(dtype=float) * u.deg,
        )
        subset_index = subset.index.to_numpy()
        for local_i, lst_ilocs in hits.items():
            if not lst_ilocs:
                continue
            seps = meta_sc[local_i].separation(lst_sc[lst_ilocs]).deg
            j_best = int(lst_ilocs[int(np.argmin(seps))])
            bmaj = float(lst_bmaj[j_best])
            if not np.isfinite(bmaj) or bmaj <= 0.0:
                continue
            out.at[subset_index[local_i], "BMAJ_precise"] = bmaj

    return out
