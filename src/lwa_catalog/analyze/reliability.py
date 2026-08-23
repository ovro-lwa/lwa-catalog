"""Metacatalog reliability tiers: soft exclude (cleaned) and hard include (gold)."""

from __future__ import annotations

import warnings as py_warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from lwa_catalog.analyze.trace import (
    _ensure_bmaj_column,
    _load_lst_band,
    _match_bmaj,
    _one_row_base,
    _parse_bands_present,
    _parse_lst_hours,
    _pick_rematch_lst_row,
)
from lwa_catalog.create.merge import associate_catalogs
from lwa_catalog.io import read_sources_catalog
from lwa_catalog.paths import CatalogLayout

_EXCLUDE_TIERS = ("E0", "E1", "E2", "E3", "E4", "E5")
_INCLUDE_TIERS = ("I0", "I1", "I2", "I3", "I4", "I5")


@dataclass(frozen=True)
class ReliabilityConfig:
    """Thresholds and tier depth for reliability filters."""

    resid_rms_thresh_jy: float = 1.0
    resid_mean_thresh_jy: float = 1.0
    flux_unphysical_nsigma: float = 3.0
    jitter_bmaj_frac: float = 0.3
    min_lst_contributions: int = 2
    require_unique_assoc_include: bool = True
    require_unique_assoc_exclude: bool = False
    strict: bool = False
    include_through: str = "I5"
    exclude_through: str = "E4"


@dataclass
class ReliabilityResult:
    """Filtered metacatalog subset plus tier yields and per-row flags."""

    catalog: pd.DataFrame
    meta_ids: np.ndarray
    tier_counts: pd.DataFrame
    flags: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def flux_sigma_total_minus_peak(df: pd.DataFrame) -> pd.Series:
    """Return ``(Total - Peak) / hypot(E_Total, E_Peak)``; NaN when not calculable."""
    need = ("Total_flux", "Peak_flux", "E_Total_flux", "E_Peak_flux")
    if any(c not in df.columns for c in need):
        return pd.Series(np.nan, index=df.index, dtype=float, name="flux_sigma")
    total = pd.to_numeric(df["Total_flux"], errors="coerce")
    peak = pd.to_numeric(df["Peak_flux"], errors="coerce")
    e_tot = pd.to_numeric(df["E_Total_flux"], errors="coerce")
    e_peak = pd.to_numeric(df["E_Peak_flux"], errors="coerce")
    denom = np.hypot(e_tot.to_numpy(dtype=float), e_peak.to_numpy(dtype=float))
    numer = (total - peak).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.where(
            np.isfinite(numer) & np.isfinite(denom) & (denom > 0.0),
            numer / denom,
            np.nan,
        )
    return pd.Series(sigma, index=df.index, dtype=float, name="flux_sigma")


def flag_unphysical_flux(
    df: pd.DataFrame,
    *,
    nsigma: float = 3.0,
) -> pd.Series:
    """True when finite ``σ < -nsigma`` (positive unphysical failure)."""
    sigma = flux_sigma_total_minus_peak(df)
    return (sigma < -float(nsigma)).fillna(False).rename("unphysical_flux")


def flag_residual_absolute(
    df: pd.DataFrame,
    *,
    rms_thresh_jy: float = 1.0,
    mean_thresh_jy: float = 1.0,
) -> pd.DataFrame:
    """Boolean residual flags using absolute Jy/beam thresholds.

    Columns: ``resid_rms_fail``, ``resid_mean_fail``, ``resid_fail`` (either).
    Failures require finite values above threshold (soft-exclude semantics).
    """
    n = len(df)
    if "Resid_Isl_rms" in df.columns:
        rms = pd.to_numeric(df["Resid_Isl_rms"], errors="coerce")
        rms_fail = (rms > float(rms_thresh_jy)).fillna(False)
    else:
        rms_fail = pd.Series(False, index=df.index)
    if "Resid_Isl_mean" in df.columns:
        mean = pd.to_numeric(df["Resid_Isl_mean"], errors="coerce").abs()
        mean_fail = (mean > float(mean_thresh_jy)).fillna(False)
    else:
        mean_fail = pd.Series(False, index=df.index)
    out = pd.DataFrame(
        {
            "resid_rms_fail": rms_fail,
            "resid_mean_fail": mean_fail,
            "resid_fail": rms_fail | mean_fail,
        },
        index=df.index,
    )
    if n == 0:
        return out
    return out


def parse_bands_present(row: pd.Series | Mapping) -> list[str]:
    """Parse comma-separated ``bands_present`` into a band list."""
    if isinstance(row, pd.Series):
        return _parse_bands_present(row)
    return _parse_bands_present(pd.Series(row))


def unique_assoc_band_count(row: pd.Series, *, min_lst: int = 2) -> tuple[int, int]:
    """Return ``(n_lst, n_unique_assoc_bands)``.

    Full in ``bands_present`` counts as uniquely associated (no ``n_assoc_Full``).
    Color bands count only when ``n_assoc_{band} == 1``.
    """
    del min_lst  # reserved for API symmetry with passes_multi_image
    raw = row.get("n_lst_contributions", np.nan)
    try:
        n_lst = int(raw) if pd.notna(raw) else 0
    except (TypeError, ValueError):
        n_lst = 0

    bands = parse_bands_present(row)
    n_unique = 0
    for band in bands:
        if band == "Full":
            n_unique += 1
            continue
        col = f"n_assoc_{band}"
        if col not in row.index:
            continue
        try:
            n_assoc = int(row[col])
        except (TypeError, ValueError):
            continue
        if n_assoc == 1:
            n_unique += 1
    return n_lst, n_unique


def passes_multi_image(
    row: pd.Series,
    *,
    min_lst: int = 2,
) -> bool:
    """True if ``n_lst >= min_lst`` or ≥2 uniquely associated bands."""
    n_lst, n_unique = unique_assoc_band_count(row, min_lst=min_lst)
    return n_lst >= int(min_lst) or n_unique >= 2


def flag_invalid_astrometry_flux(df: pd.DataFrame) -> pd.Series:
    """True when RA, DEC, or Peak_flux is non-finite."""
    ok = pd.Series(True, index=df.index)
    for col in ("RA", "DEC", "Peak_flux"):
        if col not in df.columns:
            ok &= False
            continue
        ok &= pd.to_numeric(df[col], errors="coerce").apply(np.isfinite)
    return (~ok).rename("invalid_astrometry_flux")


def flag_confused_assoc(df: pd.DataFrame) -> pd.Series:
    """True when any band in ``bands_present`` has ``n_assoc_* > 1``."""
    flags = []
    for _idx, row in df.iterrows():
        confused = False
        for band in parse_bands_present(row):
            if band == "Full":
                continue
            col = f"n_assoc_{band}"
            if col not in row.index or pd.isna(row[col]):
                continue
            try:
                if int(row[col]) > 1:
                    confused = True
                    break
            except (TypeError, ValueError):
                continue
        flags.append(confused)
    return pd.Series(flags, index=df.index, dtype=bool, name="confused_assoc")


def all_bands_unique_assoc(row: pd.Series) -> bool:
    """True when every band in ``bands_present`` is uniquely associated (or Full)."""
    bands = parse_bands_present(row)
    if not bands:
        return False
    for band in bands:
        if band == "Full":
            continue
        col = f"n_assoc_{band}"
        if col not in row.index or pd.isna(row[col]):
            return False
        try:
            if int(row[col]) != 1:
                return False
        except (TypeError, ValueError):
            return False
    return True


def cluster_radec_jitter_rms(source_matches: pd.DataFrame) -> float:
    """RMS of member–centroid great-circle offsets in degrees.

    Returns 0.0 for fewer than two finite-coordinate members? Plan: single
    member → RMS 0 (trivial pass). Empty → NaN (not calculable).
    """
    if source_matches is None or source_matches.empty:
        return float("nan")
    if "RA" not in source_matches.columns or "DEC" not in source_matches.columns:
        return float("nan")
    ra = pd.to_numeric(source_matches["RA"], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(source_matches["DEC"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(ra) & np.isfinite(dec)
    if not mask.any():
        return float("nan")
    ra = ra[mask]
    dec = dec[mask]
    if len(ra) == 1:
        return 0.0
    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    # Cartesian mean on the unit sphere for a wrap-safe centroid
    xyz = sc.cartesian.xyz.value  # shape (3, N)
    mean_xyz = np.mean(xyz, axis=1)
    norm = np.linalg.norm(mean_xyz)
    if norm <= 0.0 or not np.isfinite(norm):
        return float("nan")
    mean_xyz = mean_xyz / norm
    centroid = SkyCoord(
        x=mean_xyz[0],
        y=mean_xyz[1],
        z=mean_xyz[2],
        representation_type="cartesian",
        frame="icrs",
    )
    seps = sc.separation(centroid).deg
    return float(np.sqrt(np.mean(seps**2)))


def resolve_bmaj(row: pd.Series) -> float:
    """Finite BMAJ from ``BMAJ_match`` / ``BMAJ_full`` / ``BMAJ``, else NaN."""
    for key in ("BMAJ_match", "BMAJ_full", "BMAJ"):
        if key not in row.index:
            continue
        try:
            val = float(row[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(val) and val > 0.0:
            return val
    return float("nan")


def flag_jitter_exceeds(
    rms_deg: float,
    bmaj_deg: float,
    *,
    frac: float = 0.3,
) -> bool:
    """True when finite RMS exceeds ``frac * BMAJ``."""
    if not np.isfinite(rms_deg) or not np.isfinite(bmaj_deg) or bmaj_deg <= 0.0:
        return False
    return bool(rms_deg > float(frac) * bmaj_deg)


def seed_lst_rows(
    metacatalog: pd.DataFrame,
    layout: CatalogLayout,
    *,
    lst_merged: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Rematch each row's ``origin_band`` to one LST-merged seed row.

    Returns a DataFrame indexed by ``meta_id`` with LST columns plus
    ``seed_band``, ``seed_matched``. Missing matches yield empty/NaN rows with
    ``seed_matched=False``.
    """
    warn: list[str] = []
    if metacatalog.empty:
        return pd.DataFrame(), warn
    work = metacatalog.copy()
    if "meta_id" not in work.columns:
        msg = "metacatalog missing meta_id"
        raise ValueError(msg)

    records: list[dict] = []
    for origin_band, group in work.groupby(work["origin_band"].astype(str), sort=False):
        band = str(origin_band)
        if band.lower() in {"", "nan", "none"}:
            for _, row in group.iterrows():
                records.append(
                    {
                        "meta_id": int(row["meta_id"]),
                        "seed_band": band,
                        "seed_matched": False,
                    }
                )
            warn.append(f"{len(group)} rows have empty origin_band")
            continue

        lst_df = _load_lst_band(layout, band, lst_merged, warn)
        if lst_df is None:
            for _, row in group.iterrows():
                records.append(
                    {
                        "meta_id": int(row["meta_id"]),
                        "seed_band": band,
                        "seed_matched": False,
                    }
                )
            continue

        bases = []
        meta_rows: list[pd.Series] = []
        for _, row in group.iterrows():
            bases.append(
                {
                    "RA": float(row["RA"]) if np.isfinite(float(row["RA"])) else np.nan,
                    "DEC": float(row["DEC"]) if np.isfinite(float(row["DEC"])) else np.nan,
                    "BMAJ": _match_bmaj(row),
                }
            )
            meta_rows.append(row)
        base = pd.DataFrame(bases)
        # Drop non-finite coords for association; mark unmatched later
        finite = np.isfinite(base["RA"]) & np.isfinite(base["DEC"])
        hits_map: dict[int, list[int]] = {}
        if finite.any():
            sub_base = base.loc[finite].reset_index(drop=True)
            hits, _ = associate_catalogs(sub_base, lst_df)
            finite_idxs = np.flatnonzero(finite.to_numpy())
            for local_i, meta_i in enumerate(finite_idxs):
                hits_map[int(meta_i)] = hits.get(local_i, [])

        for i, row in enumerate(meta_rows):
            mid = int(row["meta_id"])
            idxs = hits_map.get(i, [])
            if not idxs:
                records.append({"meta_id": mid, "seed_band": band, "seed_matched": False})
                continue
            best = _pick_rematch_lst_row(lst_df.iloc[idxs], row, band)
            entry = best.to_dict()
            entry["meta_id"] = mid
            entry["seed_band"] = band
            entry["seed_matched"] = True
            records.append(entry)

    seed = pd.DataFrame.from_records(records)
    if seed.empty:
        return seed, warn
    return seed.set_index("meta_id", drop=False), warn


def expand_seed_source_matches(
    seed_lst_row: pd.Series,
    meta_row: pd.Series,
    layout: CatalogLayout,
) -> tuple[pd.DataFrame, list[str]]:
    """Per-hour source rematch for one seed LST row (jitter membership)."""
    warn: list[str] = []
    band = str(
        seed_lst_row.get("seed_band")
        or seed_lst_row.get("band")
        or meta_row.get("origin_band")
    )
    hours = _parse_lst_hours(seed_lst_row.get("lst_hours", ""))
    if not hours:
        hours = _parse_lst_hours(meta_row.get("lst_hours", ""))
    if not hours:
        warn.append(f"No lst_hours for seed band {band}")
        return pd.DataFrame(), warn

    base_bmaj = float(seed_lst_row["BMAJ"]) if "BMAJ" in seed_lst_row.index and np.isfinite(
        float(seed_lst_row.get("BMAJ", np.nan))
    ) else _match_bmaj(meta_row)
    lst_base = _one_row_base(float(seed_lst_row["RA"]), float(seed_lst_row["DEC"]), base_bmaj)

    frames: list[pd.DataFrame] = []
    for hour in hours:
        path = layout.sources(hour, band)
        if not path.is_file():
            warn.append(f"Missing sources catalog: {path.name}")
            continue
        try:
            sources = read_sources_catalog(layout, hour, band, as_pandas=True)
        except FileNotFoundError:
            warn.append(f"Missing sources catalog: {path.name}")
            continue
        assert isinstance(sources, pd.DataFrame)
        if sources.empty:
            continue
        sources = _ensure_bmaj_column(sources)
        hits, _ = associate_catalogs(lst_base, sources)
        idxs = hits.get(0, [])
        if not idxs:
            continue
        hit = sources.iloc[idxs].copy()
        hit["band"] = band
        if "lst_hour" not in hit.columns:
            hit["lst_hour"] = hour
        frames.append(hit)
    if not frames:
        return pd.DataFrame(), warn
    return pd.concat(frames, ignore_index=True), warn


def _tier_cutoff(through: str, order: Sequence[str]) -> list[str]:
    through = str(through).upper()
    if through not in order:
        msg = f"Unknown tier {through!r}; expected one of {list(order)}"
        raise ValueError(msg)
    return list(order[: order.index(through) + 1])


def assert_gold_subset_of_cleaned(
    cleaned: ReliabilityResult | pd.DataFrame,
    gold: ReliabilityResult | pd.DataFrame,
    *,
    strict: bool = False,
) -> None:
    """Enforce ``gold ⊆ cleaned`` by ``meta_id``; warn or raise."""
    def _ids(obj: ReliabilityResult | pd.DataFrame) -> set[int]:
        if isinstance(obj, ReliabilityResult):
            return set(int(x) for x in obj.meta_ids)
        if "meta_id" not in obj.columns:
            return set()
        return set(int(x) for x in obj["meta_id"].tolist())

    extra = _ids(gold) - _ids(cleaned)
    if not extra:
        return
    msg = (
        f"gold ⊆ cleaned violated: {len(extra)} meta_id(s) in gold not in cleaned "
        f"(e.g. {sorted(extra)[:5]})"
    )
    if strict:
        raise ValueError(msg)
    py_warnings.warn(msg, UserWarning, stacklevel=2)


def _empty_result(metacatalog: pd.DataFrame) -> ReliabilityResult:
    return ReliabilityResult(
        catalog=metacatalog.iloc[0:0].copy(),
        meta_ids=np.array([], dtype=int),
        tier_counts=pd.DataFrame(columns=["tier", "n_in", "n_out", "n_removed"]),
        flags=pd.DataFrame(),
    )


def _build_context(
    metacatalog: pd.DataFrame,
    layout: CatalogLayout,
    *,
    config: ReliabilityConfig,
    lst_merged: Mapping[str, pd.DataFrame] | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach seed LST, flux/resid, jitter columns keyed by metacatalog index."""
    warn: list[str] = []
    meta = metacatalog.reset_index(drop=True).copy()
    if "meta_id" not in meta.columns:
        meta["meta_id"] = np.arange(len(meta), dtype=int)

    seed, seed_warn = seed_lst_rows(meta, layout, lst_merged=lst_merged)
    warn.extend(seed_warn)

    n = len(meta)
    seed_matched = np.zeros(n, dtype=bool)
    flux_sigma = np.full(n, np.nan)
    unphys_soft = np.zeros(n, dtype=bool)
    resid_fail_soft = np.zeros(n, dtype=bool)
    resid_rms = np.full(n, np.nan)
    resid_mean = np.full(n, np.nan)
    sigma_finite = np.zeros(n, dtype=bool)
    resid_finite = np.zeros(n, dtype=bool)
    jitter_rms = np.full(n, np.nan)
    n_rematch = np.zeros(n, dtype=int)
    n_lst_seed = np.full(n, np.nan)
    bmaj = np.full(n, np.nan)

    mid_to_i = {int(m): i for i, m in enumerate(meta["meta_id"].tolist())}

    if not seed.empty:
        for mid, srow in seed.iterrows():
            mid_i = int(mid) if not isinstance(mid, (int, np.integer)) else int(mid)
            # seed index may be meta_id
            if "meta_id" in srow.index:
                mid_i = int(srow["meta_id"])
            i = mid_to_i.get(mid_i)
            if i is None:
                continue
            matched = bool(srow.get("seed_matched", False))
            seed_matched[i] = matched
            if not matched:
                continue
            s_df = pd.DataFrame([srow])
            sig = flux_sigma_total_minus_peak(s_df).iloc[0]
            flux_sigma[i] = sig
            sigma_finite[i] = bool(np.isfinite(sig))
            unphys_soft[i] = bool(
                flag_unphysical_flux(s_df, nsigma=config.flux_unphysical_nsigma).iloc[0]
            )
            rf = flag_residual_absolute(
                s_df,
                rms_thresh_jy=config.resid_rms_thresh_jy,
                mean_thresh_jy=config.resid_mean_thresh_jy,
            )
            resid_fail_soft[i] = bool(rf["resid_fail"].iloc[0])
            if "Resid_Isl_rms" in srow.index:
                resid_rms[i] = float(pd.to_numeric(srow["Resid_Isl_rms"], errors="coerce"))
            if "Resid_Isl_mean" in srow.index:
                resid_mean[i] = float(pd.to_numeric(srow["Resid_Isl_mean"], errors="coerce"))
            resid_finite[i] = np.isfinite(resid_rms[i]) and np.isfinite(resid_mean[i])
            try:
                n_lst_seed[i] = float(srow.get("n_lst_contributions", np.nan))
            except (TypeError, ValueError):
                n_lst_seed[i] = np.nan

            # jitter from seed-band sources
            sources, src_warn = expand_seed_source_matches(srow, meta.iloc[i], layout)
            warn.extend(src_warn)
            n_rematch[i] = len(sources)
            jitter_rms[i] = cluster_radec_jitter_rms(sources)

            bmaj[i] = resolve_bmaj(meta.iloc[i])
            if not np.isfinite(bmaj[i]) and "BMAJ" in srow.index:
                try:
                    bmaj[i] = float(srow["BMAJ"])
                except (TypeError, ValueError):
                    pass

    # BMAJ from meta even without seed
    for i, row in meta.iterrows():
        if not np.isfinite(bmaj[i]):
            bmaj[i] = resolve_bmaj(row)

    invalid = flag_invalid_astrometry_flux(meta).to_numpy()
    multi = meta.apply(
        lambda r: passes_multi_image(r, min_lst=config.min_lst_contributions),
        axis=1,
    ).to_numpy()
    confused = flag_confused_assoc(meta).to_numpy()
    unique_ok = meta.apply(all_bands_unique_assoc, axis=1).to_numpy()

    jitter_fail_soft = np.array(
        [
            flag_jitter_exceeds(jitter_rms[i], bmaj[i], frac=config.jitter_bmaj_frac)
            for i in range(n)
        ],
        dtype=bool,
    )

    flags = pd.DataFrame(
        {
            "meta_id": meta["meta_id"].to_numpy(),
            "invalid": invalid,
            "passes_multi_image": multi,
            "seed_matched": seed_matched,
            "flux_sigma": flux_sigma,
            "sigma_finite": sigma_finite,
            "unphysical_soft": unphys_soft,
            "resid_rms": resid_rms,
            "resid_mean": resid_mean,
            "resid_finite": resid_finite,
            "resid_fail_soft": resid_fail_soft,
            "jitter_rms_deg": jitter_rms,
            "bmaj_deg": bmaj,
            "jitter_fail_soft": jitter_fail_soft,
            "n_rematch": n_rematch,
            "n_lst_seed": n_lst_seed,
            "confused_assoc": confused,
            "all_bands_unique": unique_ok,
        }
    )
    return flags, warn


def _exclude_from_flags(
    metacatalog: pd.DataFrame,
    flags: pd.DataFrame,
    cfg: ReliabilityConfig,
    warn: list[str],
) -> ReliabilityResult:
    meta = metacatalog.reset_index(drop=True).copy()
    if "meta_id" not in meta.columns:
        meta["meta_id"] = flags["meta_id"]
    active = _tier_cutoff(cfg.exclude_through, _EXCLUDE_TIERS)
    if cfg.require_unique_assoc_exclude and "E5" not in active:
        active = [*active, "E5"]
    keep = np.ones(len(meta), dtype=bool)
    tier_rows = []
    n_in = int(keep.sum())

    def _apply(name: str, remove: np.ndarray) -> None:
        nonlocal n_in, keep
        if name not in active:
            return
        removed = keep & remove
        keep = keep & ~remove
        tier_rows.append(
            {"tier": name, "n_in": n_in, "n_out": int(keep.sum()), "n_removed": int(removed.sum())}
        )
        n_in = int(keep.sum())

    _apply("E0", flags["invalid"].to_numpy())
    _apply("E1", ~flags["passes_multi_image"].to_numpy())
    _apply("E2", flags["unphysical_soft"].to_numpy())
    _apply("E3", flags["resid_fail_soft"].to_numpy())
    _apply("E4", flags["jitter_fail_soft"].to_numpy())
    _apply("E5", flags["confused_assoc"].to_numpy())
    catalog = meta.loc[keep].copy()
    return ReliabilityResult(
        catalog=catalog,
        meta_ids=catalog["meta_id"].to_numpy(dtype=int),
        tier_counts=pd.DataFrame(tier_rows),
        flags=flags,
        warnings=list(warn),
    )


def _include_from_flags(
    metacatalog: pd.DataFrame,
    flags: pd.DataFrame,
    cfg: ReliabilityConfig,
    warn: list[str],
) -> ReliabilityResult:
    meta = metacatalog.reset_index(drop=True).copy()
    if "meta_id" not in meta.columns:
        meta["meta_id"] = flags["meta_id"]

    active = _tier_cutoff(cfg.include_through, _INCLUDE_TIERS)
    if cfg.require_unique_assoc_include and "I5" not in active:
        active = list(dict.fromkeys([*active, "I5"]))
    if not cfg.require_unique_assoc_include:
        active = [t for t in active if t != "I5"]

    keep = np.ones(len(meta), dtype=bool)
    tier_rows = []
    n_in = int(keep.sum())

    def _require(name: str, ok: np.ndarray) -> None:
        nonlocal n_in, keep
        if name not in active:
            return
        fail = keep & ~ok
        keep = keep & ok
        tier_rows.append(
            {"tier": name, "n_in": n_in, "n_out": int(keep.sum()), "n_removed": int(fail.sum())}
        )
        n_in = int(keep.sum())

    _require("I0", ~flags["invalid"].to_numpy())
    _require("I1", flags["passes_multi_image"].to_numpy())
    i2_ok = (
        flags["seed_matched"].to_numpy()
        & flags["sigma_finite"].to_numpy()
        & (flags["flux_sigma"].to_numpy(dtype=float) >= -float(cfg.flux_unphysical_nsigma))
    )
    _require("I2", i2_ok)
    i3_ok = (
        flags["seed_matched"].to_numpy()
        & flags["resid_finite"].to_numpy()
        & ~flags["resid_fail_soft"].to_numpy()
    )
    _require("I3", i3_ok)
    n_lst_meta = pd.to_numeric(
        meta["n_lst_contributions"] if "n_lst_contributions" in meta.columns else np.nan,
        errors="coerce",
    ).to_numpy(dtype=float)
    need_rematch = n_lst_meta >= float(cfg.min_lst_contributions)
    rematch_ok = (~need_rematch) | (flags["n_rematch"].to_numpy(dtype=int) >= 2)
    jitter_ok = (
        np.isfinite(flags["bmaj_deg"].to_numpy(dtype=float))
        & np.isfinite(flags["jitter_rms_deg"].to_numpy(dtype=float))
        & ~flags["jitter_fail_soft"].to_numpy()
        & rematch_ok
    )
    _require("I4", jitter_ok)
    _require("I5", flags["all_bands_unique"].to_numpy())

    catalog = meta.loc[keep].copy()
    return ReliabilityResult(
        catalog=catalog,
        meta_ids=catalog["meta_id"].to_numpy(dtype=int),
        tier_counts=pd.DataFrame(tier_rows),
        flags=flags,
        warnings=list(warn),
    )


def filter_metacatalog_exclude(
    metacatalog: pd.DataFrame,
    layout: CatalogLayout,
    *,
    config: ReliabilityConfig | None = None,
    lst_merged: Mapping[str, pd.DataFrame] | None = None,
    flags: pd.DataFrame | None = None,
) -> ReliabilityResult:
    """Soft-exclude demonstrably bad sources → cleaned catalog."""
    cfg = config or ReliabilityConfig()
    if metacatalog is None or metacatalog.empty:
        return _empty_result(metacatalog if metacatalog is not None else pd.DataFrame())
    if flags is None or len(flags) != len(metacatalog.reset_index(drop=True)):
        flags, warn = _build_context(metacatalog, layout, config=cfg, lst_merged=lst_merged)
    else:
        warn = []
    return _exclude_from_flags(metacatalog, flags, cfg, warn)


def filter_metacatalog_include(
    metacatalog: pd.DataFrame,
    layout: CatalogLayout,
    *,
    config: ReliabilityConfig | None = None,
    lst_merged: Mapping[str, pd.DataFrame] | None = None,
    flags: pd.DataFrame | None = None,
) -> ReliabilityResult:
    """Hard-include only demonstrably good sources → gold catalog."""
    cfg = config or ReliabilityConfig()
    if metacatalog is None or metacatalog.empty:
        return _empty_result(metacatalog if metacatalog is not None else pd.DataFrame())
    if flags is None or len(flags) != len(metacatalog.reset_index(drop=True)):
        flags, warn = _build_context(metacatalog, layout, config=cfg, lst_merged=lst_merged)
    else:
        warn = []
    return _include_from_flags(metacatalog, flags, cfg, warn)


def filter_metacatalog_reliability(
    metacatalog: pd.DataFrame,
    layout: CatalogLayout,
    *,
    config: ReliabilityConfig | None = None,
    lst_merged: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[ReliabilityResult, ReliabilityResult]:
    """Run cleaned (exclude) and gold (include) once; check nesting contract."""
    cfg = config or ReliabilityConfig()
    if metacatalog is None or metacatalog.empty:
        empty = _empty_result(metacatalog if metacatalog is not None else pd.DataFrame())
        return empty, empty
    flags, warn = _build_context(metacatalog, layout, config=cfg, lst_merged=lst_merged)
    cleaned = _exclude_from_flags(metacatalog, flags, cfg, warn)
    gold = _include_from_flags(metacatalog, flags, cfg, warn)
    assert_gold_subset_of_cleaned(cleaned, gold, strict=cfg.strict)
    return cleaned, gold

