"""Constrained elliptical-Gaussian forced photometry on FITS cutouts (option C).

Fits Astropy ``Gaussian2D`` (+ constant background) at a known sky position.
Used by ``notebooks/metacatalog_query.ipynb`` for rematch and missed-hour
remeasurement; no PyBDSF re-run.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.modeling import fitting, models
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from lwa_catalog.create.detect import prepare_hdu
from lwa_catalog.create.discover import FitsMetadata, discover_fits_files, resolve_fits_slot
from lwa_catalog.create.merge import source_elevation_deg

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass(frozen=True)
class ForcedPhotometryConfig:
    """Configuration for :func:`fit_gaussian2d_cutout` and FITS selection."""

    cutout_size_pix: int = 64
    centroid_bound_pix: float = 5.0
    size_bound_factor: float = 2.0
    min_elevation_deg: float = 10.0
    maxiter: int = 200


@dataclass
class ForcedPhotometryResult:
    """Single cutout Gaussian fit (option C)."""

    summary: dict[str, float | int | str | bool]
    cutout_data: np.ndarray
    cutout_model: np.ndarray
    warnings: list[str] = field(default_factory=list)

    @property
    def forced_Peak_flux(self) -> float:
        return float(self.summary["forced_Peak_flux"])

    @property
    def forced_Total_flux(self) -> float:
        return float(self.summary["forced_Total_flux"])

    @property
    def forced_RA(self) -> float:
        return float(self.summary["forced_RA"])

    @property
    def forced_DEC(self) -> float:
        return float(self.summary["forced_DEC"])

    @property
    def forced_resid_rms(self) -> float:
        return float(self.summary["forced_resid_rms"])


def summarize_forced_photometry(result: ForcedPhotometryResult) -> str:
    """One-line human summary of a forced-photometry fit."""
    s = result.summary
    pos = "fixed position" if s.get("fixed_position") else "free centroid"
    return (
        f"forced Peak={float(s['forced_Peak_flux']):.4g}, "
        f"Total={float(s['forced_Total_flux']):.4g}, "
        f"resid RMS={float(s['forced_resid_rms']):.4g} ({pos})"
    )


def index_discovered_fits(
    discovered: Sequence[FitsMetadata],
) -> tuple[dict[str, Path], dict[str, FitsMetadata]]:
    """Build basename → path and basename → metadata indexes."""
    by_name: dict[str, Path] = {}
    meta_by_name: dict[str, FitsMetadata] = {}
    for meta in discovered:
        name = meta.path.name
        by_name[name] = meta.path.resolve()
        meta_by_name[name] = meta
    return by_name, meta_by_name


def discover_and_index_fits(
    root: Path,
    patterns: str | Sequence[str],
) -> tuple[list[FitsMetadata], dict[str, Path], dict[str, FitsMetadata]]:
    """Discover FITS under *root* and return ``(discovered, by_name, meta_by_name)``."""
    pats = (patterns,) if isinstance(patterns, str) else tuple(patterns)
    discovered = list(discover_fits_files(Path(root), patterns=pats))
    by_name, meta_by_name = index_discovered_fits(discovered)
    return discovered, by_name, meta_by_name


def resolve_forced_fits_path(
    source_file: str,
    *,
    by_name: dict[str, Path],
    fits_root: Path | None = None,
    lst_hour: str | None = None,
    band: str | None = None,
) -> Path:
    """Resolve a basename ``source_file`` via discovery index or ``(lst, band)`` slot."""
    name = Path(str(source_file)).name
    if name in by_name:
        return by_name[name]
    if fits_root is not None and lst_hour and band:
        return resolve_fits_slot(Path(fits_root), str(lst_hour), str(band))
    raise FileNotFoundError(
        f"No FITS named {name!r} in discovery index"
        + (f" under {fits_root}" if fits_root is not None else "")
    )


def filter_fits_by_elevation(
    discovered: Iterable[FitsMetadata],
    ra_deg: float,
    dec_deg: float,
    *,
    min_elevation_deg: float = 10.0,
) -> list[FitsMetadata]:
    """Return discovered FITS whose LST has *ra_deg*/*dec_deg* above *min_elevation_deg*."""
    out: list[FitsMetadata] = []
    for meta in discovered:
        try:
            elev = source_elevation_deg(ra_deg, dec_deg, meta.lst_hour)
        except Exception:
            continue
        if elev > float(min_elevation_deg):
            out.append(meta)
    return out


def seed_row_from_meta(
    meta_row: pd.Series,
    *,
    source_file: str,
    lst_hour: str | None,
    band: str | None,
) -> pd.Series:
    """Build fit seeds from a metacatalog row when PyBDSF missed this image."""
    seed: dict[str, Any] = {
        "RA": float(meta_row["RA"]),
        "DEC": float(meta_row["DEC"]),
        "source_file": source_file,
        "lst_hour": lst_hour,
        "band": band,
    }
    for col in ("Maj", "Min", "PA", "Peak_flux", "Total_flux", "BMAJ", "BMIN", "BPA"):
        val = float("nan")
        if col in meta_row.index and pd.notna(meta_row.get(col)):
            try:
                val = float(meta_row[col])
            except (TypeError, ValueError):
                val = float("nan")
        if (not np.isfinite(val)) and band:
            keyed = f"{col}_{band}"
            if keyed in meta_row.index and pd.notna(meta_row.get(keyed)):
                try:
                    val = float(meta_row[keyed])
                except (TypeError, ValueError):
                    val = float("nan")
        seed[col] = val
    return pd.Series(seed)


def select_forced_seed_row(
    *,
    source_file: str,
    rematch: pd.DataFrame | None,
    meta_row: pd.Series,
    lst_hour: str | None,
    band: str | None,
) -> pd.Series:
    """Prefer a rematch ``source_matches`` row for *source_file*; else metacatalog seeds."""
    name = Path(str(source_file)).name
    if rematch is not None and not rematch.empty and "source_file" in rematch.columns:
        hits = rematch.loc[rematch["source_file"].map(lambda x: Path(str(x)).name) == name]
        if len(hits):
            return hits.iloc[0]
    return seed_row_from_meta(
        meta_row, source_file=name, lst_hour=lst_hour, band=band
    )


def fit_gaussian2d_cutout(
    fits_path: Path,
    *,
    ra_deg: float,
    dec_deg: float,
    maj_deg: float = float("nan"),
    min_deg: float = float("nan"),
    pa_deg: float = float("nan"),
    peak_guess: float = float("nan"),
    bmaj_deg: float = float("nan"),
    bmin_deg: float = float("nan"),
    fixed_position: bool = False,
    meta_id: int | None = None,
    config: ForcedPhotometryConfig | None = None,
) -> ForcedPhotometryResult:
    """Option C: constrained Astropy ``Gaussian2D`` (+ const background) on a cutout.

    When *fixed_position* is true, ``x_mean`` / ``y_mean`` stay locked to the
    sky position (*ra_deg*, *dec_deg*) — typically metacatalog coordinates.
    """
    cfg = config or ForcedPhotometryConfig()
    hdu = prepare_hdu(fits_path)
    wcs = WCS(hdu.header).celestial
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    cutout = Cutout2D(
        hdu.data,
        position=coord,
        size=int(cfg.cutout_size_pix),
        wcs=wcs,
        mode="partial",
        fill_value=np.nan,
    )
    data = np.asarray(cutout.data, dtype=float)
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    x0, y0 = cutout.wcs.world_to_pixel(coord)
    x0 = float(x0)
    y0 = float(y0)

    scales = proj_plane_pixel_scales(cutout.wcs)
    pixscale = float(np.mean(np.abs(scales)))
    if not np.isfinite(pixscale) or pixscale <= 0:
        raise ValueError("could not determine pixel scale from cutout WCS")

    maj = maj_deg if np.isfinite(maj_deg) and maj_deg > 0 else (
        bmaj_deg if np.isfinite(bmaj_deg) and bmaj_deg > 0 else 3.0 * pixscale
    )
    minor = min_deg if np.isfinite(min_deg) and min_deg > 0 else (
        bmin_deg if np.isfinite(bmin_deg) and bmin_deg > 0 else maj
    )
    if minor > maj:
        maj, minor = minor, maj
    sig_maj = max(maj * _FWHM_TO_SIGMA / pixscale, 0.5)
    sig_min = max(minor * _FWHM_TO_SIGMA / pixscale, 0.5)
    pa = pa_deg if np.isfinite(pa_deg) else 0.0
    # PA N→E; typical radio cutout +x≈W, +y≈N → Gaussian2D theta CCW from +x.
    theta0 = np.deg2rad(pa) + 0.5 * np.pi

    amp0 = peak_guess if np.isfinite(peak_guess) and peak_guess > 0 else float(
        np.nanmax(data) if np.isfinite(data).any() else 1.0
    )
    bkg0 = float(np.nanmedian(data)) if np.isfinite(data).any() else 0.0

    g0 = models.Gaussian2D(
        amplitude=amp0,
        x_mean=x0,
        y_mean=y0,
        x_stddev=sig_maj,
        y_stddev=sig_min,
        theta=theta0,
    )
    g0.amplitude.min = 0.0
    finite_abs = np.abs(data[np.isfinite(data)])
    amp_cap = float(np.nanmax(finite_abs)) if finite_abs.size else abs(amp0)
    g0.amplitude.max = max(3.0 * abs(amp0), 3.0 * amp_cap)
    if fixed_position:
        g0.x_mean.fixed = True
        g0.y_mean.fixed = True
    else:
        bound = float(cfg.centroid_bound_pix)
        g0.x_mean.min = x0 - bound
        g0.x_mean.max = x0 + bound
        g0.y_mean.min = y0 - bound
        g0.y_mean.max = y0 + bound
    factor = float(cfg.size_bound_factor)
    g0.x_stddev.min = sig_maj / factor
    g0.x_stddev.max = sig_maj * factor
    g0.y_stddev.min = sig_min / factor
    g0.y_stddev.max = sig_min * factor

    bkg = models.Const2D(bkg0)
    model_init = g0 + bkg

    mask = np.isfinite(data)
    if int(mask.sum()) < 10:
        raise ValueError("too few finite cutout pixels to fit")

    fitter = fitting.LevMarLSQFitter(calc_uncertainties=True)
    with np.errstate(all="ignore"):
        fitted = fitter(
            model_init,
            xx[mask],
            yy[mask],
            data[mask],
            maxiter=int(cfg.maxiter),
        )

    model_img = np.full_like(data, np.nan, dtype=float)
    model_img[mask] = fitted(xx[mask], yy[mask])
    residual = data - model_img

    g_fit = fitted[0]
    b_fit = fitted[1]
    amp = float(g_fit.amplitude.value)
    xf = float(g_fit.x_mean.value)
    yf = float(g_fit.y_mean.value)
    sx = float(g_fit.x_stddev.value)
    sy = float(g_fit.y_stddev.value)
    th = float(g_fit.theta.value)
    bkg_fit = float(b_fit.amplitude.value)

    sky = cutout.wcs.pixel_to_world(xf, yf)
    ra_fit = float(sky.ra.deg)
    dec_fit = float(sky.dec.deg)

    forced_peak = amp
    if np.isfinite(bmaj_deg) and np.isfinite(bmin_deg) and bmaj_deg > 0 and bmin_deg > 0:
        sig_bmaj = max(bmaj_deg * _FWHM_TO_SIGMA / pixscale, 1e-6)
        sig_bmin = max(bmin_deg * _FWHM_TO_SIGMA / pixscale, 1e-6)
        forced_total = amp * (sx * sy) / (sig_bmaj * sig_bmin)
    else:
        forced_total = float("nan")

    resid_rms = float(np.nanstd(residual[mask]))
    fit_info = str(getattr(fitter, "fit_info", {}).get("message", "ok"))
    summary: dict[str, float | int | str | bool] = {
        "forced_Peak_flux": forced_peak,
        "forced_Total_flux": forced_total,
        "forced_RA": ra_fit,
        "forced_DEC": dec_fit,
        "forced_x_pix": xf,
        "forced_y_pix": yf,
        "forced_x_stddev_pix": sx,
        "forced_y_stddev_pix": sy,
        "forced_theta_rad": th,
        "forced_bkg": bkg_fit,
        "forced_resid_rms": resid_rms,
        "fixed_position": bool(fixed_position),
        "fit_info": fit_info,
        "fits_name": Path(fits_path).name,
    }
    if meta_id is not None:
        summary["meta_id"] = int(meta_id)

    return ForcedPhotometryResult(
        summary=summary,
        cutout_data=data,
        cutout_model=model_img,
    )


def plot_forced_cutouts(
    result: ForcedPhotometryResult,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (12.0, 3.8),
):
    """Return a Matplotlib figure with input / model / residual cutout panels."""
    import matplotlib.pyplot as plt

    data = result.cutout_data
    model = result.cutout_model
    residual = data - model
    finite = np.isfinite(data) & np.isfinite(model)
    if finite.any():
        vmin = float(np.nanpercentile(data[finite], 1))
        vmax = float(np.nanpercentile(data[finite], 99))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin = float(np.nanmin(data[finite]))
            vmax = float(np.nanmax(data[finite]))
    else:
        vmin, vmax = 0.0, 1.0
    rmax = float(np.nanpercentile(np.abs(residual[finite]), 99)) if finite.any() else 1.0
    if not np.isfinite(rmax) or rmax <= 0:
        rmax = 1.0

    xf = float(result.summary["forced_x_pix"])
    yf = float(result.summary["forced_y_pix"])
    mid = result.summary.get("meta_id")
    mid_label = f"meta_id={int(mid)}" if mid is not None else "forced fit"

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    panels = (
        (data, "Input", "gray", vmin, vmax),
        (model, "Model", "gray", vmin, vmax),
        (residual, "Residual (input − model)", "coolwarm", -rmax, rmax),
    )
    for ax, (img, label, cmap, lo, hi) in zip(axes, panels, strict=True):
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        ax.plot([xf], [yf], "k+", ms=10, mew=1.2, label=mid_label)
        ax.set_title(f"{label} ({mid_label})", fontsize=10)
        ax.set_xlabel("x (pix)")
        ax.set_ylabel("y (pix)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title or summarize_forced_photometry(result), fontsize=10)
    return fig
