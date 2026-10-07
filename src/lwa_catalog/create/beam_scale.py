"""Per-band restoring-beam scale (α) from Dec-stratified probe residuals.

Estimates one scalar multiplier on the band-median ``BMAJ``/``BMIN`` by
minimizing mean map−GAUL residual RMS on fixed nested probe tiles under the
production tier2fuse detect path. Does **not** use Maj-median size estimators
(rejected experimentally). Fallback is :data:`DEFAULT_BEAM_SCALE` (1.0).
"""

from __future__ import annotations

import warnings as warnings_mod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from astropy.table import Table

from lwa_catalog.constants import GAUL_DETECTION_COLUMNS
from lwa_catalog.create.detect import run_pybdsf_on_hdu
from lwa_catalog.create.healpix_detect import (
    DEFAULT_BEAM_SCALE,
    attach_beam_and_freq,
    scale_beam,
)
from lwa_catalog.create.tiered_detect import (
    fuse_gaul_m_with_tier2_s,
    merge_tier2_bdsf_kw,
)
from lwa_catalog.gaul import cast_gaul_string_columns

# Dec-stratified nside_tile=4 nested probes (high / equator / low Dec).
DEFAULT_BEAM_SCALE_PROBE_TILES: tuple[int, ...] = (28, 102, 189)
DEFAULT_BEAM_SCALE_ALPHA_GRID: tuple[float, ...] = (
    1.0,
    1.1,
    1.2,
    1.3,
    1.5,
    1.7,
    2.0,
)

__all__ = [
    "DEFAULT_BEAM_SCALE_ALPHA_GRID",
    "DEFAULT_BEAM_SCALE_PROBE_TILES",
    "BeamScaleEstimateConfig",
    "BeamScaleEstimateResult",
    "estimate_beam_scale",
    "gaul_model_residual",
    "pick_best_alpha",
    "summarize_beam_scale_estimate",
    "tile_pixel_mask",
]


def _import_lwa_healpix():
    try:
        import lwa_healpix
    except ImportError as exc:  # pragma: no cover
        msg = (
            "lwa_healpix is required for beam-scale estimation; "
            "pip install 'lwa-catalog[analyze]'"
        )
        raise ImportError(msg) from exc
    return lwa_healpix


@dataclass(frozen=True)
class BeamScaleEstimateConfig:
    """Configuration for :func:`estimate_beam_scale`."""

    probe_tiles: tuple[int, ...] = DEFAULT_BEAM_SCALE_PROBE_TILES
    alpha_grid: tuple[float, ...] = DEFAULT_BEAM_SCALE_ALPHA_GRID
    nside_tile: int = 4
    overlap: float = 0.2
    margin: float = 0.05
    align: str = "diamond"
    ctype: str = "TAN"
    coord_frame: str = "icrs"
    nested: bool = True
    min_probe_tiles: int = 1
    min_resid_pix: int = 1
    bunit: str = "JY/BEAM"
    fallback_scale: float = DEFAULT_BEAM_SCALE

    def __post_init__(self) -> None:
        if not self.probe_tiles:
            msg = "probe_tiles must be non-empty"
            raise ValueError(msg)
        if not self.alpha_grid:
            msg = "alpha_grid must be non-empty"
            raise ValueError(msg)
        for a in self.alpha_grid:
            if not np.isfinite(a) or float(a) <= 0.0:
                msg = f"alpha_grid values must be finite and > 0, got {a!r}"
                raise ValueError(msg)
        if int(self.min_probe_tiles) < 1:
            msg = "min_probe_tiles must be >= 1"
            raise ValueError(msg)
        if not np.isfinite(self.fallback_scale) or float(self.fallback_scale) <= 0.0:
            msg = f"fallback_scale must be finite and > 0, got {self.fallback_scale!r}"
            raise ValueError(msg)


@dataclass
class BeamScaleEstimateResult:
    """Outcome of :func:`estimate_beam_scale`."""

    beam_scale: float
    summary: dict[str, Any]
    per_alpha: pd.DataFrame
    per_tile: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def tile_pixel_mask(
    nside_map: int,
    nside_tile: int,
    tile_ipix: int,
    *,
    nested: bool = True,
) -> np.ndarray:
    """Return NESTED pixel indices belonging to tile parent ``tile_ipix``."""
    if not nested:
        msg = "tile_pixel_mask currently requires nested=True"
        raise ValueError(msg)
    nside_map_i = int(nside_map)
    nside_tile_i = int(nside_tile)
    if nside_map_i < nside_tile_i or (nside_map_i % nside_tile_i) != 0:
        msg = f"nside_map={nside_map_i} must be divisible by nside_tile={nside_tile_i}"
        raise ValueError(msg)
    factor = (nside_map_i // nside_tile_i) ** 2
    start = int(tile_ipix) * factor
    return np.arange(start, start + factor, dtype=np.int64)


def gaul_model_residual(
    obs_map: np.ndarray,
    gaul_df: pd.DataFrame | None,
    *,
    nside: int,
    nest: bool,
    tile_ipix: int,
    nside_tile: int,
) -> dict[str, float]:
    """Observed − GAUL Gaussian paint on nested children of ``tile_ipix``."""
    from lwa_catalog.analyze.healpix_map import metacatalog_to_healpix

    pix = tile_pixel_mask(nside, nside_tile, tile_ipix, nested=nest)
    obs = np.asarray(obs_map, dtype=float)
    if gaul_df is None or gaul_df.empty:
        finite = obs[pix]
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return {
                "resid_rms": float("nan"),
                "resid_mean": float("nan"),
                "resid_abs_p95": float("nan"),
                "resid_peak_abs": float("nan"),
                "n_resid_pix": 0,
                "n_model_sources": 0,
            }
        return {
            "resid_rms": float(np.sqrt(np.mean(finite**2))),
            "resid_mean": float(np.mean(finite)),
            "resid_abs_p95": float(np.percentile(np.abs(finite), 95)),
            "resid_peak_abs": float(np.max(np.abs(finite))),
            "n_resid_pix": int(finite.size),
            "n_model_sources": 0,
        }

    model = metacatalog_to_healpix(
        gaul_df,
        nside=int(nside),
        nest=bool(nest),
        profile="gaussian",
        weight_col="Peak_flux",
    )
    resid = obs[pix] - model[pix]
    finite = resid[np.isfinite(obs[pix])]
    if finite.size == 0:
        return {
            "resid_rms": float("nan"),
            "resid_mean": float("nan"),
            "resid_abs_p95": float("nan"),
            "resid_peak_abs": float("nan"),
            "n_resid_pix": 0,
            "n_model_sources": int(len(gaul_df)),
        }
    return {
        "resid_rms": float(np.sqrt(np.mean(finite**2))),
        "resid_mean": float(np.mean(finite)),
        "resid_abs_p95": float(np.percentile(np.abs(finite), 95)),
        "resid_peak_abs": float(np.max(np.abs(finite))),
        "n_resid_pix": int(finite.size),
        "n_model_sources": int(len(gaul_df)),
    }


def pick_best_alpha(
    per_tile: pd.DataFrame,
    *,
    fallback_scale: float = DEFAULT_BEAM_SCALE,
    min_probe_tiles: int = 1,
) -> tuple[float, pd.DataFrame, list[str]]:
    """Pick α minimizing mean finite ``resid_rms``; ties → smallest α.

    Returns ``(beam_scale, per_alpha_df, warnings)``.
    """
    warn: list[str] = []
    empty_alpha = pd.DataFrame(
        columns=["alpha", "mean_resid_rms", "n_probe_tiles", "n_finite"]
    )
    if per_tile is None or per_tile.empty:
        warn.append("no probe residual scores; using fallback beam_scale")
        return float(fallback_scale), empty_alpha, warn

    rows: list[dict[str, Any]] = []
    for alpha, g in per_tile.groupby("alpha", sort=True):
        rms = pd.to_numeric(g["resid_rms"], errors="coerce")
        finite = rms[np.isfinite(rms)]
        rows.append(
            {
                "alpha": float(alpha),
                "mean_resid_rms": float(finite.mean()) if len(finite) else float("nan"),
                "n_probe_tiles": int(len(g)),
                "n_finite": int(len(finite)),
            }
        )
    per_alpha = pd.DataFrame(rows)
    if per_alpha.empty:
        warn.append("empty per_alpha table; using fallback beam_scale")
        return float(fallback_scale), empty_alpha, warn

    usable = per_alpha[np.isfinite(per_alpha["mean_resid_rms"])].copy()
    if usable.empty:
        warn.append("all alpha mean residuals are NaN; using fallback beam_scale")
        return float(fallback_scale), per_alpha, warn

    # Prefer alphas that met min_probe_tiles finite scores when any do.
    enough = usable[usable["n_finite"] >= int(min_probe_tiles)]
    pool = enough if not enough.empty else usable
    if enough.empty:
        warn.append(
            f"no alpha reached min_probe_tiles={min_probe_tiles}; "
            "picking among alphas with any finite residual"
        )

    # Tie-break: smallest alpha among minimum mean residual.
    best_mean = float(pool["mean_resid_rms"].min())
    tied = pool[np.isclose(pool["mean_resid_rms"], best_mean)]
    best_alpha = float(tied["alpha"].min())
    return best_alpha, per_alpha, warn


def summarize_beam_scale_estimate(result: BeamScaleEstimateResult) -> str:
    """Short human-readable summary for notebook logging."""
    s = result.summary
    lines = [
        f"beam_scale={result.beam_scale:g}  "
        f"band={s.get('band', '?')}  "
        f"n_probe={s.get('n_probe_used', '?')}/{s.get('n_probe_requested', '?')}  "
        f"best_mean_resid={s.get('best_mean_resid_rms', float('nan')):.4g}"
    ]
    for w in result.warnings:
        lines.append(f"  warning: {w}")
    if result.per_alpha is not None and not result.per_alpha.empty:
        top = result.per_alpha.sort_values(
            ["mean_resid_rms", "alpha"], na_position="last"
        ).head(3)
        for _, row in top.iterrows():
            lines.append(
                f"  α={row['alpha']:g}: mean_resid={row['mean_resid_rms']:.4g} "
                f"(n_finite={int(row['n_finite'])})"
            )
    return "\n".join(lines)


def _tile_weight_sum(
    weight: np.ndarray,
    *,
    nside_map: int,
    nside_tile: int,
    tile_ipix: int,
    nested: bool,
) -> float:
    pix = tile_pixel_mask(nside_map, nside_tile, tile_ipix, nested=nested)
    return float(np.nansum(np.asarray(weight, dtype=float)[pix]))


def _table_to_tile_df(
    table: Table,
    *,
    gaul_columns: Sequence[str],
    tile_ipix: int,
    nside_tile: int,
    band: str,
    bmaj: float,
    bmin: float,
    bpa: float,
) -> pd.DataFrame:
    df = table.to_pandas()
    keep = [c for c in gaul_columns if c in df.columns]
    df = cast_gaul_string_columns(df[keep].copy())
    df["tile_ipix"] = int(tile_ipix)
    df["nside_tile"] = int(nside_tile)
    df["band"] = band
    df["BMAJ"] = float(bmaj)
    df["BMIN"] = float(bmin)
    df["BPA"] = float(bpa)
    return df


def _detect_probe_tile(
    healpix_map: np.ndarray,
    weight: np.ndarray,
    *,
    tile_ipix: int,
    nside_map: int,
    config: BeamScaleEstimateConfig,
    bmaj_s: float,
    bmin_s: float,
    bpa_s: float,
    restfreq_hz: float | None,
    band: str,
    bdsf_kw: Mapping[str, Any] | None,
    tier2_kw: Mapping[str, Any],
    gaul_columns: Sequence[str],
) -> pd.DataFrame:
    lwa_healpix = _import_lwa_healpix()
    header = lwa_healpix.nested_tile_header(
        int(config.nside_tile),
        int(tile_ipix),
        nside_map=int(nside_map),
        overlap=float(config.overlap),
        margin=float(config.margin),
        align=str(config.align),
        ctype=str(config.ctype),
        coord_frame=str(config.coord_frame),
    )
    hdu = lwa_healpix.healpix_to_hdu(
        np.asarray(healpix_map),
        header,
        weight=np.asarray(weight),
        coord_frame=str(config.coord_frame),
        nested=bool(config.nested),
        header_updates={"BUNIT": config.bunit} if config.bunit else None,
    )
    hdu = attach_beam_and_freq(
        hdu,
        bmaj=bmaj_s,
        bmin=bmin_s,
        bpa=bpa_s,
        restfreq_hz=restfreq_hz,
        bunit=config.bunit,
    )
    if not np.isfinite(hdu.data).any():
        return pd.DataFrame()

    table = run_pybdsf_on_hdu(hdu, bdsf_kw=bdsf_kw)
    if table is None or len(table) == 0:
        return pd.DataFrame()
    df = _table_to_tile_df(
        table,
        gaul_columns=gaul_columns,
        tile_ipix=int(tile_ipix),
        nside_tile=int(config.nside_tile),
        band=band,
        bmaj=bmaj_s,
        bmin=bmin_s,
        bpa=bpa_s,
    )
    table2 = run_pybdsf_on_hdu(hdu, bdsf_kw=tier2_kw)
    if table2 is not None and len(table2) > 0:
        df2 = _table_to_tile_df(
            table2,
            gaul_columns=gaul_columns,
            tile_ipix=int(tile_ipix),
            nside_tile=int(config.nside_tile),
            band=band,
            bmaj=bmaj_s,
            bmin=bmin_s,
            bpa=bpa_s,
        )
        df = fuse_gaul_m_with_tier2_s(df, df2)
    return df


def estimate_beam_scale(
    healpix_map: np.ndarray,
    weight: np.ndarray,
    *,
    nside_map: int,
    bmaj: float,
    bmin: float,
    bpa: float = 0.0,
    restfreq_hz: float | None = None,
    band: str = "Full",
    bdsf_kw: Mapping[str, Any] | None = None,
    tier2_bdsf_kw: Mapping[str, Any] | None = None,
    config: BeamScaleEstimateConfig | None = None,
    gaul_columns: Sequence[str] = GAUL_DETECTION_COLUMNS,
) -> BeamScaleEstimateResult:
    """Estimate one beam scale α for a band HEALPix coadd.

    Runs tier2fuse detect on :data:`DEFAULT_BEAM_SCALE_PROBE_TILES` (or
    ``config.probe_tiles``) for each α in the grid, scores map−GAUL
    ``resid_rms``, and picks the α with the lowest **mean** residual across
    probes with finite scores (tie → smallest α). On failure returns
    ``config.fallback_scale`` (default 1.0).

    Parameters
    ----------
    tier2_bdsf_kw
        Required (non-``None``). Pass ``{}`` for library tier-2 defaults.
    """
    cfg = config if config is not None else BeamScaleEstimateConfig()
    if tier2_bdsf_kw is None:
        msg = (
            "estimate_beam_scale requires tier2_bdsf_kw (tier2fuse path); "
            "pass {} for defaults or notebook overrides"
        )
        raise ValueError(msg)
    for name, val in (("bmaj", bmaj), ("bmin", bmin), ("bpa", bpa)):
        if not np.isfinite(float(val)):
            msg = f"{name} must be finite, got {val!r}"
            raise ValueError(msg)
        if name in ("bmaj", "bmin") and float(val) <= 0.0:
            msg = f"{name} must be > 0, got {val!r}"
            raise ValueError(msg)

    warn: list[str] = []
    map_arr = np.asarray(healpix_map)
    weight_arr = np.asarray(weight)
    tier2_kw = merge_tier2_bdsf_kw(bdsf_kw, tier2_bdsf_kw)

    usable_probes: list[int] = []
    for ipix in cfg.probe_tiles:
        wsum = _tile_weight_sum(
            weight_arr,
            nside_map=int(nside_map),
            nside_tile=int(cfg.nside_tile),
            tile_ipix=int(ipix),
            nested=bool(cfg.nested),
        )
        if wsum > 0.0:
            usable_probes.append(int(ipix))
        else:
            warn.append(f"probe tile {ipix}: weight sum <= 0; skipped")

    per_tile_rows: list[dict[str, Any]] = []
    if len(usable_probes) < int(cfg.min_probe_tiles):
        warn.append(
            f"only {len(usable_probes)} usable probe tiles "
            f"(need {cfg.min_probe_tiles}); using fallback beam_scale"
        )
        result = BeamScaleEstimateResult(
            beam_scale=float(cfg.fallback_scale),
            summary={
                "band": band,
                "beam_scale": float(cfg.fallback_scale),
                "fallback": True,
                "n_probe_requested": len(cfg.probe_tiles),
                "n_probe_used": len(usable_probes),
                "best_mean_resid_rms": float("nan"),
            },
            per_alpha=pd.DataFrame(
                columns=["alpha", "mean_resid_rms", "n_probe_tiles", "n_finite"]
            ),
            per_tile=pd.DataFrame(),
            warnings=warn,
        )
        warnings_mod.warn(
            f"estimate_beam_scale({band}): fallback α={cfg.fallback_scale:g}",
            UserWarning,
            stacklevel=2,
        )
        return result

    for alpha in cfg.alpha_grid:
        bmaj_s, bmin_s, bpa_s = scale_beam(bmaj, bmin, bpa, scale=float(alpha))
        for ipix in usable_probes:
            try:
                cat = _detect_probe_tile(
                    map_arr,
                    weight_arr,
                    tile_ipix=ipix,
                    nside_map=int(nside_map),
                    config=cfg,
                    bmaj_s=bmaj_s,
                    bmin_s=bmin_s,
                    bpa_s=bpa_s,
                    restfreq_hz=restfreq_hz,
                    band=band,
                    bdsf_kw=bdsf_kw,
                    tier2_kw=tier2_kw,
                    gaul_columns=gaul_columns,
                )
            except Exception as exc:  # pragma: no cover - PyBDSF / healpix edge
                warn.append(f"α={alpha:g} tile={ipix}: detect failed ({exc})")
                per_tile_rows.append(
                    {
                        "alpha": float(alpha),
                        "tile_ipix": int(ipix),
                        "resid_rms": float("nan"),
                        "resid_mean": float("nan"),
                        "n_resid_pix": 0,
                        "n_model_sources": 0,
                        "n_catalog": 0,
                    }
                )
                continue

            resid = gaul_model_residual(
                map_arr,
                cat,
                nside=int(nside_map),
                nest=bool(cfg.nested),
                tile_ipix=int(ipix),
                nside_tile=int(cfg.nside_tile),
            )
            if int(resid["n_resid_pix"]) < int(cfg.min_resid_pix):
                resid["resid_rms"] = float("nan")
            per_tile_rows.append(
                {
                    "alpha": float(alpha),
                    "tile_ipix": int(ipix),
                    "resid_rms": resid["resid_rms"],
                    "resid_mean": resid["resid_mean"],
                    "n_resid_pix": resid["n_resid_pix"],
                    "n_model_sources": resid["n_model_sources"],
                    "n_catalog": int(len(cat)),
                }
            )

    per_tile = pd.DataFrame(per_tile_rows)
    best, per_alpha, pick_warn = pick_best_alpha(
        per_tile,
        fallback_scale=float(cfg.fallback_scale),
        min_probe_tiles=int(cfg.min_probe_tiles),
    )
    warn.extend(pick_warn)
    best_row = per_alpha[per_alpha["alpha"] == best] if not per_alpha.empty else None
    best_mean = (
        float(best_row.iloc[0]["mean_resid_rms"])
        if best_row is not None and len(best_row)
        else float("nan")
    )
    fallback = any("fallback beam_scale" in w for w in pick_warn)

    if fallback:
        warnings_mod.warn(
            f"estimate_beam_scale({band}): fallback α={cfg.fallback_scale:g}",
            UserWarning,
            stacklevel=2,
        )

    return BeamScaleEstimateResult(
        beam_scale=float(best),
        summary={
            "band": band,
            "beam_scale": float(best),
            "fallback": fallback,
            "n_probe_requested": len(cfg.probe_tiles),
            "n_probe_used": len(usable_probes),
            "best_mean_resid_rms": best_mean,
            "alpha_grid": list(cfg.alpha_grid),
            "probe_tiles": list(usable_probes),
        },
        per_alpha=per_alpha,
        per_tile=per_tile,
        warnings=warn,
    )
