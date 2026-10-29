"""Cross-band forced photometry fill/replace on LST-merged band catalogs.

Post–tile-merge, pre-fusion step: use neighbor-band seed positions to
(1) **insert** forced Gaussian rows where the target band has no near hit
(426 / reverse_confused gap path), and (2) optionally **replace** near-hit
Peak/Total when they sit below a Peak-dip ratio vs neighbor seeds
(861 / unconfused_step path).

Does **not** modify 2-tier PyBDSF fuse. Forced fits use HEALPix TAN cutouts
via :mod:`lwa_catalog.analyze.healpix_cutout` and
:func:`~lwa_catalog.analyze.forced_photometry.fit_gaussian2d_cutout`.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits

from lwa_catalog.analyze.forced_photometry import (
    ForcedPhotometryConfig,
    ForcedPhotometryResult,
    fit_gaussian2d_cutout,
)
from lwa_catalog.analyze.healpix_cutout import HealpixMapCache, healpix_cutout
from lwa_catalog.create.merge import associate_catalogs

__all__ = [
    "CrossbandForcedFillConfig",
    "CrossbandForcedFillResult",
    "classify_crossband_actions",
    "fill_band_crossband_forced",
    "fill_interior_subbands_crossband_forced",
    "forced_fit_on_healpix_band",
    "summarize_crossband_forced_fill",
]

ForcedFitFn = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class CrossbandForcedFillConfig:
    """Configuration for :func:`fill_band_crossband_forced`."""

    sep_frac_bmaj: float = 0.25
    peak_dip_ratio: float = 0.70
    enable_phot_dip: bool = True
    enable_gap_insert: bool = True
    min_seed_peak_jy: float = 0.0
    resid_abs_jy: float = 1.0
    resid_frac: float = 0.5
    fixed_position: bool = True
    cutout_beam_factor: float = 3.0
    min_cutout_deg: float = 0.5
    cutout_size_pix: int = 96
    nside: int = 2048
    nested: bool = True


@dataclass
class CrossbandForcedFillResult:
    """Result of a cross-band forced fill on one target LST catalog."""

    summary: dict[str, Any]
    target: pd.DataFrame
    actions: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def summarize_crossband_forced_fill(result: CrossbandForcedFillResult) -> str:
    """One-line human summary of a fill result."""
    s = result.summary
    return (
        f"crossband forced fill: n_insert={s.get('n_insert', 0)}, "
        f"n_replace={s.get('n_replace', 0)}, "
        f"n_skip_resid={s.get('n_skip_resid', 0)}, "
        f"n_out={s.get('n_out', len(result.target))}"
    )


def _ensure_bmaj(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "BMAJ" in out.columns:
        out["BMAJ"] = pd.to_numeric(out["BMAJ"], errors="coerce").fillna(0.0)
    else:
        out["BMAJ"] = 0.0
    return out


def _sep_deg(ra0: float, dec0: float, ra1: float, dec1: float) -> float:
    a = SkyCoord(ra=ra0 * u.deg, dec=dec0 * u.deg)
    b = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    return float(a.separation(b).deg)


def _seps_to_hits_deg(ra0: float, dec0: float, hits: pd.DataFrame) -> np.ndarray:
    """Vectorized great-circle separations from one seed to *hits* rows (deg)."""
    if hits.empty:
        return np.zeros(0, dtype=float)
    seed = SkyCoord(ra=ra0 * u.deg, dec=dec0 * u.deg)
    hit_sc = SkyCoord(
        ra=hits["RA"].to_numpy(dtype=float) * u.deg,
        dec=hits["DEC"].to_numpy(dtype=float) * u.deg,
    )
    return seed.separation(hit_sc).deg


def _near_hit_mask(
    seed_row: pd.Series,
    hits: pd.DataFrame,
    *,
    sep_frac_bmaj: float,
) -> np.ndarray:
    """Boolean mask of *hits* rows within ``sep_frac × max(BMAJ)`` of the seed."""
    if hits.empty:
        return np.zeros(0, dtype=bool)
    ra0 = float(seed_row["RA"])
    dec0 = float(seed_row["DEC"])
    bmaj0 = float(seed_row.get("BMAJ", 0.0) or 0.0)
    seps = _seps_to_hits_deg(ra0, dec0, hits)
    bmaj1 = pd.to_numeric(hits["BMAJ"], errors="coerce").to_numpy(dtype=float)
    bmaj1 = np.where(np.isfinite(bmaj1), bmaj1, 0.0)
    limits = float(sep_frac_bmaj) * np.maximum(np.maximum(bmaj0, bmaj1), 0.0)
    return np.isfinite(seps) & (seps <= limits)


def _pick_nearest_hit(seed_row: pd.Series, hits: pd.DataFrame) -> int:
    """Return iloc into *hits* for nearest (then brightest Peak) row."""
    if len(hits) == 1:
        return 0
    seps = _seps_to_hits_deg(float(seed_row["RA"]), float(seed_row["DEC"]), hits)
    peak = pd.to_numeric(hits["Peak_flux"], errors="coerce").to_numpy(dtype=float)
    peak = np.where(np.isfinite(peak), peak, -np.inf)
    order = np.lexsort((-peak, seps))
    return int(order[0])


def classify_crossband_actions(
    target: pd.DataFrame,
    seed_frames: Mapping[str, pd.DataFrame],
    *,
    config: CrossbandForcedFillConfig | None = None,
) -> pd.DataFrame:
    """Return action rows (``insert`` / ``replace`` / ``skip``) without I/O.

    *seed_frames* maps neighbor band label → LST-merged catalog. When multiple
    seed bands mark the same target row as a near hit, the phot-dip test uses
    ``Peak_target / median(seed Peaks)``.
    """
    cfg = config or CrossbandForcedFillConfig()
    t = _ensure_bmaj(target.reset_index(drop=True))
    rows: list[dict[str, Any]] = []

    # target_iloc -> list of (seed_band, seed_iloc, seed_peak)
    near_claims: dict[int, list[tuple[str, int, float]]] = {}
    insert_cands: list[dict[str, Any]] = []

    for seed_band, seed_df in seed_frames.items():
        if seed_df is None or seed_df.empty:
            continue
        s = _ensure_bmaj(seed_df.reset_index(drop=True))
        seed_peaks_all = pd.to_numeric(s["Peak_flux"], errors="coerce").to_numpy(dtype=float)
        if float(cfg.min_seed_peak_jy) > 0:
            keep = np.isfinite(seed_peaks_all) & (
                seed_peaks_all >= float(cfg.min_seed_peak_jy)
            )
            if not keep.any():
                continue
            # Preserve original ilocs for seed_iloc provenance.
            orig_ilocs = np.flatnonzero(keep)
            s = s.iloc[keep].reset_index(drop=True)
            seed_peaks_all = seed_peaks_all[keep]
        else:
            orig_ilocs = np.arange(len(s))
        hits_by_seed, _matched = associate_catalogs(s, t)
        for seed_i in range(len(s)):
            seed_peak = float(seed_peaks_all[seed_i])
            seed_row = s.iloc[seed_i]
            seed_iloc_orig = int(orig_ilocs[seed_i])
            hit_ilocs = hits_by_seed.get(seed_i, [])
            if not hit_ilocs:
                if cfg.enable_gap_insert:
                    insert_cands.append(
                        {
                            "action": "insert",
                            "seed_band": str(seed_band),
                            "seed_iloc": seed_iloc_orig,
                            "target_iloc": -1,
                            "seed_ra": float(seed_row["RA"]),
                            "seed_dec": float(seed_row["DEC"]),
                            "seed_peak": seed_peak,
                            "target_peak": float("nan"),
                            "peak_ratio": float("nan"),
                            "sep_deg": float("nan"),
                            "reason": "no_beam_hit",
                        }
                    )
                continue
            hit_df = t.iloc[list(hit_ilocs)].reset_index(drop=True)
            near = _near_hit_mask(seed_row, hit_df, sep_frac_bmaj=cfg.sep_frac_bmaj)
            if not near.any():
                if not cfg.enable_gap_insert:
                    continue
                # Beam hit(s) exist but all are offset → gap / 426 path
                far_i = _pick_nearest_hit(seed_row, hit_df)
                far = hit_df.iloc[far_i]
                seps = _seps_to_hits_deg(
                    float(seed_row["RA"]), float(seed_row["DEC"]), hit_df
                )
                insert_cands.append(
                    {
                        "action": "insert",
                        "seed_band": str(seed_band),
                        "seed_iloc": seed_iloc_orig,
                        "target_iloc": -1,
                        "seed_ra": float(seed_row["RA"]),
                        "seed_dec": float(seed_row["DEC"]),
                        "seed_peak": seed_peak,
                        "target_peak": float(
                            pd.to_numeric(far.get("Peak_flux"), errors="coerce")
                        ),
                        "peak_ratio": float("nan"),
                        "sep_deg": float(seps[far_i]),
                        "reason": "offset_only",
                    }
                )
                continue
            near_hit_df = hit_df.loc[near].reset_index(drop=True)
            # Map back to original target ilocs
            near_orig = [int(hit_ilocs[j]) for j, keep in enumerate(near) if keep]
            local = _pick_nearest_hit(seed_row, near_hit_df)
            t_iloc = near_orig[local]
            near_claims.setdefault(t_iloc, []).append(
                (str(seed_band), seed_iloc_orig, seed_peak)
            )

    # Phot-dip replaces (one action per target row)
    if cfg.enable_phot_dip:
        for t_iloc, claims in near_claims.items():
            t_peak = float(pd.to_numeric(t.iloc[t_iloc].get("Peak_flux"), errors="coerce"))
            seed_peaks = [p for _b, _i, p in claims if np.isfinite(p) and p > 0]
            if not seed_peaks or not (np.isfinite(t_peak) and t_peak > 0):
                rows.append(
                    {
                        "action": "skip",
                        "seed_band": ",".join(sorted({b for b, _, _ in claims})),
                        "seed_iloc": int(claims[0][1]),
                        "target_iloc": int(t_iloc),
                        "seed_ra": float(t.iloc[t_iloc]["RA"]),
                        "seed_dec": float(t.iloc[t_iloc]["DEC"]),
                        "seed_peak": float(np.median(seed_peaks)) if seed_peaks else float("nan"),
                        "target_peak": t_peak,
                        "peak_ratio": float("nan"),
                        "sep_deg": 0.0,
                        "reason": "invalid_peak",
                    }
                )
                continue
            neigh = float(np.median(seed_peaks))
            ratio = t_peak / neigh
            bands_sorted = sorted({b for b, _, _ in claims})
            seed_band_label = ",".join(bands_sorted)
            primary_band = bands_sorted[0]
            primary_iloc = next(i for b, i, _p in claims if b == primary_band)
            if ratio < float(cfg.peak_dip_ratio):
                rows.append(
                    {
                        "action": "replace",
                        "seed_band": seed_band_label,
                        "seed_iloc": int(primary_iloc),
                        "target_iloc": int(t_iloc),
                        "seed_ra": float(t.iloc[t_iloc]["RA"]),
                        "seed_dec": float(t.iloc[t_iloc]["DEC"]),
                        "seed_peak": neigh,
                        "target_peak": t_peak,
                        "peak_ratio": ratio,
                        "sep_deg": 0.0,
                        "reason": "phot_dip",
                    }
                )
            else:
                rows.append(
                    {
                        "action": "skip",
                        "seed_band": seed_band_label,
                        "seed_iloc": int(primary_iloc),
                        "target_iloc": int(t_iloc),
                        "seed_ra": float(t.iloc[t_iloc]["RA"]),
                        "seed_dec": float(t.iloc[t_iloc]["DEC"]),
                        "seed_peak": neigh,
                        "target_peak": t_peak,
                        "peak_ratio": ratio,
                        "sep_deg": 0.0,
                        "reason": "peak_ok",
                    }
                )

    # Dedupe inserts: drop candidates near an existing near-claimed target or
    # a brighter insert; keep brightest seed_peak per sky cluster.
    claimed_sky = []
    for t_iloc in near_claims:
        claimed_sky.append((float(t.iloc[t_iloc]["RA"]), float(t.iloc[t_iloc]["DEC"]),
                            float(t.iloc[t_iloc].get("BMAJ", 0.0) or 0.0)))

    insert_cands.sort(
        key=lambda r: -r["seed_peak"] if np.isfinite(r["seed_peak"]) else float("-inf")
    )
    kept_inserts: list[dict[str, Any]] = []
    for cand in insert_cands:
        ra, dec = cand["seed_ra"], cand["seed_dec"]
        bmaj_s = 0.0
        # Approximate BMAJ from first seed frame row if available
        sb = cand["seed_band"]
        if sb in seed_frames and not seed_frames[sb].empty:
            bmaj_s = float(
                pd.to_numeric(
                    seed_frames[sb].reset_index(drop=True).iloc[cand["seed_iloc"]].get("BMAJ"),
                    errors="coerce",
                )
                or 0.0
            )
        skip = False
        for cra, cdec, cbmaj in claimed_sky:
            lim = float(cfg.sep_frac_bmaj) * max(bmaj_s, cbmaj, 0.0)
            if _sep_deg(ra, dec, cra, cdec) <= lim:
                skip = True
                break
        if skip:
            continue
        for kept in kept_inserts:
            lim = float(cfg.sep_frac_bmaj) * max(bmaj_s, 0.0)
            # Use kept seed BMAJ approx via same limit
            if _sep_deg(ra, dec, kept["seed_ra"], kept["seed_dec"]) <= max(lim, 1e-12):
                skip = True
                break
        if skip:
            continue
        kept_inserts.append(cand)
        claimed_sky.append((ra, dec, bmaj_s))

    rows.extend(kept_inserts)
    if not rows:
        return pd.DataFrame(
            columns=[
                "action",
                "seed_band",
                "seed_iloc",
                "target_iloc",
                "seed_ra",
                "seed_dec",
                "seed_peak",
                "target_peak",
                "peak_ratio",
                "sep_deg",
                "reason",
            ]
        )
    return pd.DataFrame(rows)


def forced_fit_on_healpix_band(
    *,
    cache: HealpixMapCache,
    band: str,
    ra_deg: float,
    dec_deg: float,
    maj_deg: float = float("nan"),
    min_deg: float = float("nan"),
    pa_deg: float = 0.0,
    peak_guess: float = float("nan"),
    bmaj_deg: float = float("nan"),
    bmin_deg: float = float("nan"),
    config: CrossbandForcedFillConfig | None = None,
) -> dict[str, Any]:
    """Forced Gaussian2D on a TAN HEALPix cutout locked to (*ra_deg*, *dec_deg*)."""
    cfg = config or CrossbandForcedFillConfig()
    path = cache.path_for(band)
    if not path.is_file():
        return {"ok": False, "reason": f"missing {path}"}
    try:
        hp_map, weight, meta = cache.get(band)
    except Exception as exc:  # noqa: BLE001 — surface as fit failure
        return {"ok": False, "reason": f"cache.get: {exc!r}"}

    bmaj_use = bmaj_deg if np.isfinite(bmaj_deg) and bmaj_deg > 0 else (
        maj_deg if np.isfinite(maj_deg) and maj_deg > 0 else 0.1
    )
    size_deg = max(float(cfg.min_cutout_deg), float(cfg.cutout_beam_factor) * float(bmaj_use))
    hdu = healpix_cutout(
        hp_map,
        weight,
        float(ra_deg),
        float(dec_deg),
        size_deg=size_deg,
        nside=int(meta.get("nside", cfg.nside)),
        nested=bool(meta.get("nested", cfg.nested)),
    )
    bmin_use = bmin_deg if np.isfinite(bmin_deg) and bmin_deg > 0 else bmaj_use

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        fits.PrimaryHDU(data=np.asarray(hdu.data, dtype=np.float32), header=hdu.header).writeto(
            tmp_path, overwrite=True
        )
        with fits.open(tmp_path, mode="update") as hdul:
            hdr = hdul[0].header
            hdr["BMAJ"] = float(bmaj_use)
            hdr["BMIN"] = float(bmin_use)
            hdr["BPA"] = float(pa_deg) if np.isfinite(pa_deg) else 0.0
            hdul.flush()
        result: ForcedPhotometryResult = fit_gaussian2d_cutout(
            tmp_path,
            ra_deg=float(ra_deg),
            dec_deg=float(dec_deg),
            maj_deg=float(maj_deg) if np.isfinite(maj_deg) else float(bmaj_use),
            min_deg=float(min_deg) if np.isfinite(min_deg) else float(bmin_use),
            pa_deg=float(pa_deg) if np.isfinite(pa_deg) else 0.0,
            peak_guess=float(peak_guess),
            bmaj_deg=float(bmaj_use),
            bmin_deg=float(bmin_use),
            fixed_position=bool(cfg.fixed_position),
            config=ForcedPhotometryConfig(
                cutout_size_pix=min(int(cfg.cutout_size_pix), int(hdu.data.shape[0]))
            ),
        )
        return {
            "ok": True,
            "forced_Peak": result.forced_Peak_flux,
            "forced_Total": result.forced_Total_flux,
            "forced_RA": result.forced_RA,
            "forced_DEC": result.forced_DEC,
            "forced_resid_rms": result.forced_resid_rms,
            "summary": dict(result.summary),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": repr(exc)}
    finally:
        tmp_path.unlink(missing_ok=True)


def _resid_ok(forced: Mapping[str, Any], cfg: CrossbandForcedFillConfig) -> bool:
    resid = float(forced.get("forced_resid_rms", float("nan")))
    peak = float(forced.get("forced_Peak", float("nan")))
    if not np.isfinite(resid):
        return False
    peak_term = float(cfg.resid_frac) * abs(peak) if np.isfinite(peak) else 0.0
    thresh = max(float(cfg.resid_abs_jy), peak_term)
    return resid <= thresh


def _seed_template_row(
    seed_frames: Mapping[str, pd.DataFrame],
    seed_band_field: str,
    seed_iloc: int,
) -> pd.Series:
    """Pick a seed row for cloning insert metadata (first band if comma-list)."""
    bands = [b.strip() for b in str(seed_band_field).split(",") if b.strip()]
    band = bands[0]
    df = seed_frames[band].reset_index(drop=True)
    return df.iloc[int(seed_iloc)]


def _apply_forced_fluxes(
    row: pd.Series,
    forced: Mapping[str, Any],
    *,
    mode: str,
    seed_band: str,
) -> pd.Series:
    out = row.copy()
    out["Peak_flux"] = float(forced["forced_Peak"])
    out["Total_flux"] = float(forced["forced_Total"])
    if "E_Peak_flux" in out.index:
        out["E_Peak_flux"] = float("nan")
    if "E_Total_flux" in out.index:
        out["E_Total_flux"] = float("nan")
    if "forced_RA" in forced and np.isfinite(float(forced["forced_RA"])):
        out["RA"] = float(forced["forced_RA"])
    if "forced_DEC" in forced and np.isfinite(float(forced["forced_DEC"])):
        out["DEC"] = float(forced["forced_DEC"])
    out["forced"] = True
    out["forced_mode"] = str(mode)
    out["forced_from_band"] = str(seed_band)
    return out


def fill_band_crossband_forced(
    target: pd.DataFrame,
    seed_frames: Mapping[str, pd.DataFrame],
    *,
    target_band: str,
    healpix_root: Path | str | None = None,
    config: CrossbandForcedFillConfig | None = None,
    fit_fn: ForcedFitFn | None = None,
    map_cache: HealpixMapCache | None = None,
) -> CrossbandForcedFillResult:
    """Classify gap/phot-dip actions and apply forced Peak+Total on the target band.

    Parameters
    ----------
    target
        LST-merged (or tile-merged) catalog for *target_band*.
    seed_frames
        Neighbor-band catalogs ``{band: DataFrame}``.
    target_band
        Band label for HEALPix map lookup (``healpix_{band}_nside*.fits``).
    healpix_root
        Catalog / OUTPUT_DIR root containing HEALPix coadds. Required unless
        *fit_fn* is supplied (tests).
    fit_fn
        Optional injectable fitter ``(**kwargs) -> dict`` with keys
        ``ok``, ``forced_Peak``, ``forced_Total``, ``forced_resid_rms``, …
        for unit tests without healpy.
    map_cache
        Optional shared :class:`HealpixMapCache`.
    """
    cfg = config or CrossbandForcedFillConfig()
    warnings: list[str] = []
    if target is None or target.empty:
        empty = pd.DataFrame() if target is None else target.copy()
        return CrossbandForcedFillResult(
            summary={"n_insert": 0, "n_replace": 0, "n_skip_resid": 0, "n_out": 0},
            target=empty,
            actions=pd.DataFrame(),
            warnings=["empty target"],
        )

    actions = classify_crossband_actions(target, seed_frames, config=cfg)
    out = target.reset_index(drop=True).copy()
    if "forced" not in out.columns:
        out["forced"] = False
    if "forced_mode" not in out.columns:
        out["forced_mode"] = pd.Series([pd.NA] * len(out), dtype=object)
    if "forced_from_band" not in out.columns:
        out["forced_from_band"] = pd.Series([pd.NA] * len(out), dtype=object)

    cache = map_cache
    if fit_fn is None:
        if healpix_root is None and cache is None:
            raise ValueError("healpix_root or map_cache required when fit_fn is None")
        if cache is None:
            cache = HealpixMapCache(
                Path(healpix_root),  # type: ignore[arg-type]
                nside=cfg.nside,
                nested=cfg.nested,
            )

    def _fit(action: pd.Series) -> dict[str, Any]:
        seed_row = _seed_template_row(seed_frames, action["seed_band"], int(action["seed_iloc"]))
        kwargs = dict(
            band=str(target_band),
            ra_deg=float(action["seed_ra"]),
            dec_deg=float(action["seed_dec"]),
            maj_deg=float(pd.to_numeric(seed_row.get("Maj"), errors="coerce")),
            min_deg=float(pd.to_numeric(seed_row.get("Min"), errors="coerce")),
            pa_deg=float(pd.to_numeric(seed_row.get("PA"), errors="coerce") or 0.0),
            peak_guess=(
                float(action["seed_peak"])
                if np.isfinite(action["seed_peak"])
                else float("nan")
            ),
            bmaj_deg=float(pd.to_numeric(seed_row.get("BMAJ"), errors="coerce")),
            bmin_deg=float(pd.to_numeric(seed_row.get("BMIN"), errors="coerce")),
            config=cfg,
        )
        if fit_fn is not None:
            return fit_fn(**kwargs)
        assert cache is not None
        return forced_fit_on_healpix_band(cache=cache, **kwargs)

    n_insert = 0
    n_replace = 0
    n_skip_resid = 0
    insert_rows: list[pd.Series] = []
    next_neg_id = -1

    for _, action in actions.iterrows():
        act = str(action["action"])
        if act == "skip":
            continue
        forced = _fit(action)
        if not forced.get("ok"):
            warnings.append(
                f"{act} seed={action['seed_band']} iloc={action['seed_iloc']}: "
                f"{forced.get('reason', 'fit_failed')}"
            )
            continue
        if not _resid_ok(forced, cfg):
            n_skip_resid += 1
            warnings.append(
                f"resid reject {act} seed={action['seed_band']} "
                f"resid={forced.get('forced_resid_rms')}"
            )
            continue

        if act == "replace":
            t_iloc = int(action["target_iloc"])
            updated = _apply_forced_fluxes(
                out.iloc[t_iloc],
                forced,
                mode="replace",
                seed_band=str(action["seed_band"]),
            )
            for col in out.columns:
                out.at[t_iloc, col] = updated[col]
            n_replace += 1
        elif act == "insert":
            seed_row = _seed_template_row(
                seed_frames, action["seed_band"], int(action["seed_iloc"])
            )
            new_data = {
                col: seed_row[col] if col in seed_row.index else pd.NA
                for col in out.columns
            }
            new_row = pd.Series(new_data)
            if "band" in new_row.index:
                new_row["band"] = target_band
            if "Source_id" in new_row.index:
                new_row["Source_id"] = next_neg_id
            if "Isl_id" in new_row.index:
                new_row["Isl_id"] = next_neg_id
            next_neg_id -= 1
            new_row = _apply_forced_fluxes(
                new_row,
                forced,
                mode="insert",
                seed_band=str(action["seed_band"]),
            )
            # Keep seed sky position when fixed_position fits still report slight shifts
            if cfg.fixed_position:
                new_row["RA"] = float(action["seed_ra"])
                new_row["DEC"] = float(action["seed_dec"])
            insert_rows.append(new_row)
            n_insert += 1

    if insert_rows:
        out = pd.concat([out, pd.DataFrame(insert_rows)], ignore_index=True)

    summary = {
        "target_band": str(target_band),
        "n_insert": n_insert,
        "n_replace": n_replace,
        "n_skip_resid": n_skip_resid,
        "n_actions": int(len(actions)),
        "n_out": int(len(out)),
        "enable_phot_dip": bool(cfg.enable_phot_dip),
    }
    return CrossbandForcedFillResult(
        summary=summary,
        target=out,
        actions=actions,
        warnings=warnings,
    )


def fill_interior_subbands_crossband_forced(
    lst_merged: Mapping[str, pd.DataFrame],
    bands: Sequence[str],
    *,
    healpix_root: Path | str,
    config: CrossbandForcedFillConfig | None = None,
    fit_fn: ForcedFitFn | None = None,
) -> dict[str, CrossbandForcedFillResult]:
    """Run :func:`fill_band_crossband_forced` for each interior band in *bands*.

    For band index ``i`` (``0 < i < len(bands)-1``), seeds are ``bands[i-1]`` and
    ``bands[i+1]`` when present in *lst_merged*.
    """
    cfg = config or CrossbandForcedFillConfig()
    cache = HealpixMapCache(Path(healpix_root), nside=cfg.nside, nested=cfg.nested)
    results: dict[str, CrossbandForcedFillResult] = {}
    band_list = list(bands)
    for i in range(1, len(band_list) - 1):
        band = band_list[i]
        if band not in lst_merged:
            continue
        seeds: dict[str, pd.DataFrame] = {}
        for nb in (band_list[i - 1], band_list[i + 1]):
            if nb in lst_merged and lst_merged[nb] is not None and not lst_merged[nb].empty:
                seeds[nb] = lst_merged[nb]
        if not seeds:
            continue
        results[band] = fill_band_crossband_forced(
            lst_merged[band],
            seeds,
            target_band=band,
            healpix_root=healpix_root,
            config=cfg,
            fit_fn=fit_fn,
            map_cache=cache,
        )
    return results
