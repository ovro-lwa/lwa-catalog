"""Joint size + zenith-south κ beam-scale estimate (optional VLSSR hybrid).

Fits per-band isotropic α near zenith and one shared Maj-elongation κ from
compact-source sizes, then optionally rescales all α so near-zenith 73 MHz
Total/VLSSR matches the bootstrap product. See
``.agents/plan-anisotropic-header-beam-alpha.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lwa_catalog.analyze.beam_alpha_bootstrap import (
    write_beam_scale_json,
)
from lwa_catalog.constants import OVRO_LATITUDE_DEG
from lwa_catalog.create.beam_scale_model import (
    MODEL_ZENITH_SOUTH,
    BeamScaleModel,
    zenith_south_u,
)
from lwa_catalog.io import write_table
from lwa_catalog.paths import CatalogLayout

__all__ = [
    "BEAM_ALPHA_KAPPA_BY_BAND_COLUMNS",
    "BeamAlphaKappaEstimateConfig",
    "BeamAlphaKappaEstimateResult",
    "estimate_beam_alpha_kappa",
    "summarize_beam_alpha_kappa_estimate",
    "write_beam_alpha_kappa_result",
]

BEAM_ALPHA_KAPPA_BY_BAND_COLUMNS: tuple[str, ...] = (
    "band",
    "alpha",
    "alpha_size",
    "kappa",
    "n_zenith",
    "n_south",
    "med_maj_over_bmaj_zenith",
    "med_min_over_bmin_zenith",
)

DEFAULT_ZENITH_HALF_WIDTH_DEG: float = 20.0


@dataclass(frozen=True, slots=True)
class BeamAlphaKappaEstimateConfig:
    """Configuration for :func:`estimate_beam_alpha_kappa`."""

    beam_scale_used: float
    zenith_half_width_deg: float = DEFAULT_ZENITH_HALF_WIDTH_DEG
    peak_min_jy: float = 1.0
    s_code: str = "S"
    phi_deg: float = OVRO_LATITUDE_DEG
    alpha_floor: float = 1.0
    bands: tuple[str, ...] | None = None
    anchor_band: str = "73MHz"
    apply_vlssr_hybrid: bool = True
    dec_col: str = "DEC"


@dataclass(frozen=True, slots=True)
class BeamAlphaKappaEstimateResult:
    """Outcome of :func:`estimate_beam_alpha_kappa`."""

    by_band: pd.DataFrame
    per_dec: pd.DataFrame
    kappa: float
    alpha_scale_factor: float
    alpha_vlssr_anchor: float
    median_total_over_vlssr_zenith: float
    median_total_over_vlssr_allsky: float
    n_vlssr_zenith: int
    n_vlssr_allsky: int
    beam_scale_used: float
    phi_deg: float
    zenith_half_width_deg: float
    warnings: tuple[str, ...] = ()

    def as_mapping(self) -> dict[str, BeamScaleModel]:
        """Return ``{band: BeamScaleModel}`` with shared κ."""
        out: dict[str, BeamScaleModel] = {}
        for _, row in self.by_band.iterrows():
            a = float(row["alpha"])
            if not np.isfinite(a) or a <= 0.0:
                continue
            out[str(row["band"])] = BeamScaleModel(
                alpha=a,
                kappa=float(self.kappa),
                model=MODEL_ZENITH_SOUTH,
                phi_deg=float(self.phi_deg),
            )
        return out


def _require_beam_scale_used(beam_scale_used: float) -> float:
    scale = float(beam_scale_used)
    if not np.isfinite(scale) or scale <= 0.0:
        msg = (
            "beam_scale_used must be finite and > 0 "
            f"(detect-time α on catalog BMAJ/BMIN); got {beam_scale_used!r}"
        )
        raise ValueError(msg)
    return scale


def _select_compact(sources: pd.DataFrame, cfg: BeamAlphaKappaEstimateConfig) -> pd.DataFrame:
    need = {"Maj", "Min", "BMAJ", "BMIN", "Peak_flux", "band", cfg.dec_col}
    missing = sorted(need - set(sources.columns))
    if missing:
        msg = f"sources missing required columns: {missing}"
        raise ValueError(msg)
    df = sources.copy()
    if "S_Code" in df.columns and cfg.s_code:
        df = df.loc[df["S_Code"].astype(str) == str(cfg.s_code)]
    peak = pd.to_numeric(df["Peak_flux"], errors="coerce")
    maj = pd.to_numeric(df["Maj"], errors="coerce")
    mn = pd.to_numeric(df["Min"], errors="coerce")
    bmaj = pd.to_numeric(df["BMAJ"], errors="coerce")
    bmin = pd.to_numeric(df["BMIN"], errors="coerce")
    dec = pd.to_numeric(df[cfg.dec_col], errors="coerce")
    ok = (
        (peak >= float(cfg.peak_min_jy))
        & maj.notna()
        & mn.notna()
        & (bmaj > 0)
        & (bmin > 0)
        & dec.notna()
    )
    out = df.loc[ok].copy()
    used = _require_beam_scale_used(cfg.beam_scale_used)
    # Catalog BMAJ already includes used; recover ratios vs unscaled header.
    out["maj_over_bmaj"] = used * (maj.loc[ok] / bmaj.loc[ok])
    out["min_over_bmin"] = used * (mn.loc[ok] / bmin.loc[ok])
    out["_dec"] = dec.loc[ok].to_numpy(dtype=float)
    out["_u"] = [zenith_south_u(d, phi_deg=cfg.phi_deg) for d in out["_dec"]]
    return out


def _tile_or_dec_groups(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (band, sky-bin) with median ratios and n."""
    if "tile_ipix" in df.columns:
        keys = ["band", "tile_ipix"]
        g = df.groupby(keys, sort=False)
        rows = []
        for (band, _ipix), sub in g:
            rows.append(
                {
                    "band": str(band),
                    "dec": float(np.median(sub["_dec"])),
                    "u": float(np.median(sub["_u"])),
                    "med_maj_over_bmaj": float(np.median(sub["maj_over_bmaj"])),
                    "med_min_over_bmin": float(np.median(sub["min_over_bmin"])),
                    "n": int(len(sub)),
                }
            )
        return pd.DataFrame(rows)

    # Fixed 5° Dec bins when tile_ipix is absent.
    dec = df["_dec"].to_numpy(dtype=float)
    edges = np.arange(-40.0, 70.0 + 5.0, 5.0)
    bin_id = np.digitize(dec, edges) - 1
    df = df.assign(_bin=bin_id)
    rows = []
    for (band, bid), sub in df.groupby(["band", "_bin"], sort=False):
        if int(bid) < 0 or int(bid) >= len(edges) - 1:
            continue
        rows.append(
            {
                "band": str(band),
                "dec": float(np.median(sub["_dec"])),
                "u": float(np.median(sub["_u"])),
                "med_maj_over_bmaj": float(np.median(sub["maj_over_bmaj"])),
                "med_min_over_bmin": float(np.median(sub["min_over_bmin"])),
                "n": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def _fit_alpha_per_band(
    bins: pd.DataFrame,
    *,
    phi_deg: float,
    half_width: float,
    alpha_floor: float,
    bands: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Return per-band α and zenith diagnostics."""
    out: dict[str, dict[str, float]] = {}
    for band in bands:
        sub = bins.loc[bins["band"] == band]
        if sub.empty:
            continue
        zen = sub.loc[np.abs(sub["dec"] - phi_deg) <= half_width]
        if zen.empty:
            continue
        med_maj = float(np.median(zen["med_maj_over_bmaj"]))
        med_min = float(np.median(zen["med_min_over_bmin"]))
        alpha = max(float(alpha_floor), med_maj, med_min)
        south = sub.loc[sub["u"] > 0]
        out[band] = {
            "alpha_size": alpha,
            "n_zenith": float(zen["n"].sum()),
            "n_south": float(south["n"].sum()) if len(south) else 0.0,
            "med_maj_over_bmaj_zenith": med_maj,
            "med_min_over_bmin_zenith": med_min,
        }
    return out


def _fit_shared_kappa(
    bins: pd.DataFrame,
    alpha_by_band: Mapping[str, float],
) -> float:
    """n-weighted slope of (Maj/BMAJ)/α − 1 vs u through the origin (κ ≥ 0)."""
    num = 0.0
    den = 0.0
    for _, row in bins.iterrows():
        band = str(row["band"])
        if band not in alpha_by_band:
            continue
        u = float(row["u"])
        if u <= 0.0:
            continue
        alpha = float(alpha_by_band[band])
        if not np.isfinite(alpha) or alpha <= 0.0:
            continue
        resid = float(row["med_maj_over_bmaj"]) / alpha - 1.0
        n = float(row["n"])
        num += n * u * resid
        den += n * u * u
    if den <= 0.0:
        return 0.0
    return float(max(0.0, num / den))


def _vlssr_median_ratio(
    meta: pd.DataFrame,
    *,
    anchor_band: str,
    zenith_mask: pd.Series | None,
) -> tuple[float, int]:
    total_col = f"Total_flux_{anchor_band}"
    if total_col not in meta.columns or "Peak_flux_VLSSR" not in meta.columns:
        return float("nan"), 0
    t = pd.to_numeric(meta[total_col], errors="coerce")
    v = pd.to_numeric(meta["Peak_flux_VLSSR"], errors="coerce")
    if "n_assoc_VLSSR" in meta.columns:
        n_assoc = pd.to_numeric(meta["n_assoc_VLSSR"], errors="coerce")
        ok = (n_assoc == 1) & (t > 0) & (v > 0)
    else:
        ok = (t > 0) & (v > 0)
    if zenith_mask is not None:
        ok = ok & zenith_mask.reindex(meta.index, fill_value=False)
    n = int(ok.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.median((t / v).loc[ok])), n


def estimate_beam_alpha_kappa(
    sources: pd.DataFrame,
    *,
    config: BeamAlphaKappaEstimateConfig,
    metacatalog_vlssr: pd.DataFrame | None = None,
) -> BeamAlphaKappaEstimateResult:
    """Estimate per-band α and shared κ from compact-source sizes.

    Parameters
    ----------
    sources
        Tile GAUL or LST-merged rows with ``Maj``, ``Min``, ``BMAJ``, ``BMIN``,
        ``Peak_flux``, ``band``, and Dec (``config.dec_col``). Prefer
        ``S_Code`` + ``tile_ipix`` when available.
    config
        Selection, zenith window, and hybrid options.
    metacatalog_vlssr
        Optional fused/spectral rows with ``Total_flux_{anchor}``,
        ``Peak_flux_VLSSR``, and Dec for the hybrid α rescale. Ignored when
        ``config.apply_vlssr_hybrid`` is False.
    """
    cfg = config
    used = _require_beam_scale_used(cfg.beam_scale_used)
    warnings: list[str] = []

    selected = _select_compact(sources, cfg)
    if cfg.bands is not None:
        selected = selected.loc[selected["band"].isin(cfg.bands)]
    if selected.empty:
        msg = "no sources passed S / Peak / finite-size selection"
        raise ValueError(msg)

    bins = _tile_or_dec_groups(selected)
    bands = sorted(bins["band"].unique(), key=str)
    alpha_info = _fit_alpha_per_band(
        bins,
        phi_deg=float(cfg.phi_deg),
        half_width=float(cfg.zenith_half_width_deg),
        alpha_floor=float(cfg.alpha_floor),
        bands=bands,
    )
    if not alpha_info:
        msg = (
            f"no near-zenith bins within |δ−φ|≤{cfg.zenith_half_width_deg}°; "
            "cannot estimate α"
        )
        raise ValueError(msg)

    alpha_size = {b: float(v["alpha_size"]) for b, v in alpha_info.items()}
    kappa = _fit_shared_kappa(bins, alpha_size)

    alpha_final = dict(alpha_size)
    alpha_scale = 1.0
    alpha_vlssr = float("nan")
    med_zen = float("nan")
    med_all = float("nan")
    n_zen = 0
    n_all = 0

    if cfg.apply_vlssr_hybrid:
        if metacatalog_vlssr is None:
            warnings.append(
                "apply_vlssr_hybrid=True but metacatalog_vlssr is None; "
                "skipping VLSSR rescale"
            )
        elif cfg.anchor_band not in alpha_final:
            warnings.append(
                f"anchor_band {cfg.anchor_band!r} missing from size α; "
                "skipping VLSSR rescale"
            )
        else:
            meta = metacatalog_vlssr
            dec_col = cfg.dec_col if cfg.dec_col in meta.columns else "DEC"
            if dec_col not in meta.columns:
                warnings.append(
                    f"metacatalog missing {dec_col!r}; skipping VLSSR rescale"
                )
            else:
                dec = pd.to_numeric(meta[dec_col], errors="coerce")
                zen_mask = (dec - float(cfg.phi_deg)).abs() <= float(
                    cfg.zenith_half_width_deg
                )
                med_zen, n_zen = _vlssr_median_ratio(
                    meta, anchor_band=cfg.anchor_band, zenith_mask=zen_mask
                )
                med_all, n_all = _vlssr_median_ratio(
                    meta, anchor_band=cfg.anchor_band, zenith_mask=None
                )
                if n_zen == 0 or not np.isfinite(med_zen) or med_zen <= 0.0:
                    warnings.append(
                        "no near-zenith unique VLSSR matches; skipping rescale"
                    )
                else:
                    alpha_vlssr = float(used * np.sqrt(med_zen))
                    alpha_size_anchor = float(alpha_final[cfg.anchor_band])
                    alpha_scale = float(alpha_vlssr / alpha_size_anchor)
                    alpha_final = {
                        b: float(a * alpha_scale) for b, a in alpha_final.items()
                    }

    rows = []
    for band, info in alpha_info.items():
        rows.append(
            {
                "band": band,
                "alpha": float(alpha_final[band]),
                "alpha_size": float(info["alpha_size"]),
                "kappa": float(kappa),
                "n_zenith": int(info["n_zenith"]),
                "n_south": int(info["n_south"]),
                "med_maj_over_bmaj_zenith": float(info["med_maj_over_bmaj_zenith"]),
                "med_min_over_bmin_zenith": float(info["med_min_over_bmin_zenith"]),
            }
        )
    by_band = pd.DataFrame(rows)
    if len(by_band):
        by_band = by_band.loc[:, list(BEAM_ALPHA_KAPPA_BY_BAND_COLUMNS)]

    return BeamAlphaKappaEstimateResult(
        by_band=by_band,
        per_dec=bins,
        kappa=float(kappa),
        alpha_scale_factor=float(alpha_scale),
        alpha_vlssr_anchor=float(alpha_vlssr),
        median_total_over_vlssr_zenith=float(med_zen),
        median_total_over_vlssr_allsky=float(med_all),
        n_vlssr_zenith=int(n_zen),
        n_vlssr_allsky=int(n_all),
        beam_scale_used=float(used),
        phi_deg=float(cfg.phi_deg),
        zenith_half_width_deg=float(cfg.zenith_half_width_deg),
        warnings=tuple(warnings),
    )


def summarize_beam_alpha_kappa_estimate(result: BeamAlphaKappaEstimateResult) -> str:
    """Human-readable summary of an ``(α, κ)`` estimate."""
    lines = [
        f"shared κ={result.kappa:.6g} deg⁻¹  "
        f"(φ={result.phi_deg:.3f}°, W=±{result.zenith_half_width_deg:g}°)",
        f"beam_scale_used={result.beam_scale_used:g}  "
        f"VLSSR α scale factor s={result.alpha_scale_factor:.6g}",
    ]
    if np.isfinite(result.alpha_vlssr_anchor):
        lines.append(
            f"α_vlssr({result.n_vlssr_zenith} zenith)={result.alpha_vlssr_anchor:.4g}  "
            f"med T/VLSSR zenith={result.median_total_over_vlssr_zenith:.4g}  "
            f"all-sky={result.median_total_over_vlssr_allsky:.4g} "
            f"(n={result.n_vlssr_allsky})"
        )
    for _, row in result.by_band.iterrows():
        lines.append(
            f"  {row['band']}: α={row['alpha']:.4g} "
            f"(size {row['alpha_size']:.4g})  "
            f"n_zenith={int(row['n_zenith'])} n_south={int(row['n_south'])}"
        )
    if result.warnings:
        lines.append(f"warnings: {len(result.warnings)}")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)


def write_beam_alpha_kappa_result(
    result: BeamAlphaKappaEstimateResult,
    path: Path | str | CatalogLayout,
    *,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write ``beam_scale.json`` and ``beam_alpha_kappa_by_band.parquet``."""
    if isinstance(path, CatalogLayout):
        layout = path
        json_path: Path | CatalogLayout = layout
        table_path = layout.root / "beam_alpha_kappa_by_band.parquet"
    else:
        p = Path(path)
        if p.suffix.lower() == ".json" or p.name == "beam_scale.json":
            json_path = p
            table_path = p.with_name("beam_alpha_kappa_by_band.parquet")
        else:
            json_path = p / "beam_scale.json" if p.suffix == "" else p.parent / "beam_scale.json"
            table_path = (
                p / "beam_alpha_kappa_by_band.parquet"
                if p.suffix == ""
                else p.with_name("beam_alpha_kappa_by_band.parquet")
            )

    meta = {
        "kappa": float(result.kappa),
        "phi_deg": float(result.phi_deg),
        "zenith_half_width_deg": float(result.zenith_half_width_deg),
        "beam_scale_used": float(result.beam_scale_used),
        "alpha_scale_factor": float(result.alpha_scale_factor),
        "model": MODEL_ZENITH_SOUTH,
        "source": "alpha_kappa_vlssr_hybrid"
        if np.isfinite(result.alpha_vlssr_anchor)
        else "alpha_kappa_zenith_south",
    }
    if extra:
        meta.update({str(k): v for k, v in extra.items() if k != "BEAM_SCALE"})

    write_beam_scale_json(result.as_mapping(), json_path, extra=meta)
    write_table(result.by_band.loc[:, list(BEAM_ALPHA_KAPPA_BY_BAND_COLUMNS)], table_path)
    if isinstance(json_path, CatalogLayout):
        return json_path.beam_scale_json()
    return Path(json_path)
