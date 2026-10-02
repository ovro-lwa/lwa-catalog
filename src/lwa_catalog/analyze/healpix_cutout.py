"""HEALPix coadd cutouts for band-merge reliability QA.

Reprojects a small TAN patch from an imaging ``healpix_{band}_nside*.fits``
coadd (MAP+WEIGHT via ``lwa_healpix``). Display-only — does not write catalogs
or touch tile membership.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

__all__ = [
    "HealpixMapCache",
    "cutout_grid_shape",
    "healpix_coadd_path",
    "healpix_cutout",
    "plot_band_cutouts",
    "tan_cutout_header",
]


def _import_lwa_healpix():
    try:
        import lwa_healpix
    except ImportError as exc:  # pragma: no cover
        msg = (
            "lwa_healpix is required for HEALPix cutouts; "
            "pip install 'lwa-catalog[analyze]'"
        )
        raise ImportError(msg) from exc
    return lwa_healpix


def healpix_coadd_path(root: Path | str, band: str, *, nside: int) -> Path:
    """Path convention matching ``ovro_lwa_healpix_tile_detect.ipynb``."""
    return Path(root) / f"healpix_{band}_nside{int(nside)}.fits"


def tan_cutout_header(
    ra_deg: float,
    dec_deg: float,
    *,
    size_deg: float,
    pixscale_deg: float,
) -> fits.Header:
    """Build a north-aligned TAN header centered on ``(ra_deg, dec_deg)``.

    Square FOV of *size_deg* on a side; pixel scale *pixscale_deg* (typically
    ``lwa_healpix.pixel_scale_deg_for_nside(nside_map)``).
    """
    if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
        msg = "ra_deg and dec_deg must be finite"
        raise ValueError(msg)
    if not (np.isfinite(size_deg) and size_deg > 0):
        msg = "size_deg must be positive and finite"
        raise ValueError(msg)
    if not (np.isfinite(pixscale_deg) and pixscale_deg > 0):
        msg = "pixscale_deg must be positive and finite"
        raise ValueError(msg)

    naxis = max(8, int(np.ceil(float(size_deg) / float(pixscale_deg))))
    # Prefer odd so CRPIX lands on a pixel center.
    if naxis % 2 == 0:
        naxis += 1
    crpix = (naxis + 1) / 2.0
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = naxis
    hdr["NAXIS2"] = naxis
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = float(ra_deg)
    hdr["CRVAL2"] = float(dec_deg)
    hdr["CRPIX1"] = crpix
    hdr["CRPIX2"] = crpix
    hdr["CDELT1"] = -float(pixscale_deg)
    hdr["CDELT2"] = float(pixscale_deg)
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["RADESYS"] = "ICRS"
    return hdr


def healpix_cutout(
    healpix_map: np.ndarray,
    weight: np.ndarray | None,
    ra_deg: float,
    dec_deg: float,
    *,
    size_deg: float,
    nside: int | None = None,
    nested: bool = True,
    coord_frame: str = "icrs",
    pixscale_deg: float | None = None,
) -> fits.PrimaryHDU:
    """Reproject a TAN stamp centered on ``(ra_deg, dec_deg)`` from a HEALPix map."""
    lwa_healpix = _import_lwa_healpix()
    if pixscale_deg is None:
        if nside is None:
            msg = "pass nside= or pixscale_deg="
            raise ValueError(msg)
        pixscale_deg = float(lwa_healpix.pixel_scale_deg_for_nside(int(nside)))
    header = tan_cutout_header(
        ra_deg, dec_deg, size_deg=size_deg, pixscale_deg=float(pixscale_deg)
    )
    return lwa_healpix.healpix_to_hdu(
        np.asarray(healpix_map),
        header,
        weight=None if weight is None else np.asarray(weight),
        coord_frame=coord_frame,
        nested=nested,
    )


class HealpixMapCache:
    """Lazy loader for ``healpix_{band}_nside*.fits`` under a catalog root."""

    def __init__(
        self,
        root: Path | str,
        *,
        nside: int = 2048,
        nested: bool = True,
        coord_frame: str = "icrs",
    ) -> None:
        self.root = Path(root)
        self.nside = int(nside)
        self.nested = bool(nested)
        self.coord_frame = str(coord_frame)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}

    def path_for(self, band: str) -> Path:
        return healpix_coadd_path(self.root, band, nside=self.nside)

    def get(self, band: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        key = str(band)
        if key in self._cache:
            return self._cache[key]
        path = self.path_for(key)
        if not path.is_file():
            raise FileNotFoundError(path)
        lwa_healpix = _import_lwa_healpix()
        healpix_map, weight, meta = lwa_healpix.read_healpix_fits(path)
        # Prefer file metadata when present.
        nested = bool(meta.get("nested", self.nested))
        frame = str(meta.get("coord_frame", self.coord_frame))
        nside = int(meta.get("nside", self.nside))
        packed = (np.asarray(healpix_map), np.asarray(weight), {
            "nside": nside,
            "nested": nested,
            "coord_frame": frame,
            "path": path,
        })
        self._cache[key] = packed
        return packed

    def clear(self) -> None:
        self._cache.clear()


def cutout_grid_shape(n: int) -> tuple[int, int]:
    """Return ``(nrows, ncols)`` with ``ncols ≈ 2 × nrows`` covering *n* panels."""
    n = max(int(n), 1)
    best: tuple[int, int] | None = None
    best_key: tuple[float, int, int] | None = None
    for nrows in range(1, n + 1):
        ncols = int(np.ceil(n / nrows))
        empty = nrows * ncols - n
        key = (abs(ncols / nrows - 2.0), empty, nrows)
        if best_key is None or key < best_key:
            best_key = key
            best = (nrows, ncols)
    assert best is not None
    return best


def _cutout_size_deg(
    lst_matches: pd.DataFrame,
    *,
    beam_factor: float,
    default_bmaj_deg: float,
) -> float:
    bmaj = default_bmaj_deg
    if lst_matches is not None and not lst_matches.empty and "BMAJ" in lst_matches.columns:
        vals = pd.to_numeric(lst_matches["BMAJ"], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals) & (vals > 0)]
        if finite.size:
            bmaj = float(np.nanmax(finite))
    return float(beam_factor) * bmaj


def _draw_ellipse_on_ax(
    ax,
    wcs: WCS,
    *,
    ra_deg: float,
    dec_deg: float,
    maj_deg: float,
    min_deg: float,
    pa_deg: float,
    color: str,
    lw: float = 1.2,
) -> None:
    """Draw a Maj/Min FWHM ellipse (PA N→E) in pixel coordinates."""
    from matplotlib.patches import Ellipse

    if not all(np.isfinite(v) for v in (ra_deg, dec_deg, maj_deg, min_deg)):
        return
    if maj_deg <= 0 or min_deg <= 0:
        return
    x, y = wcs.world_to_pixel_values(ra_deg, dec_deg)
    scales = np.abs(wcs.wcs.cdelt) if wcs.wcs.cdelt is not None else None
    if scales is None or len(scales) < 2:
        return
    pixscale = float(np.mean(np.abs(scales[:2])))
    if not (np.isfinite(pixscale) and pixscale > 0):
        return
    width = float(maj_deg) / pixscale
    height = float(min_deg) / pixscale
    # Matplotlib Ellipse angle is CCW from +x; radio PA is N→E.
    # TAN +x ≈ −RA (west), +y ≈ +Dec (north) → angle = 90° − PA.
    angle = 90.0 - (float(pa_deg) if np.isfinite(pa_deg) else 0.0)
    ax.add_patch(
        Ellipse(
            (float(x), float(y)),
            width=width,
            height=height,
            angle=angle,
            fill=False,
            edgecolor=color,
            linewidth=lw,
        )
    )


def plot_band_cutouts(
    lst_matches: pd.DataFrame,
    meta_row: pd.Series,
    map_cache: HealpixMapCache,
    *,
    bands: Sequence[str] | None = None,
    beam_factor: float = 6.0,
    default_bmaj_deg: float = 0.5,
    figsize_per: tuple[float, float] = (3.2, 3.2),
    cmap: str = "gray_r",
    percentile: tuple[float, float] = (2.0, 98.0),
    fig=None,
):
    """Grid of HEALPix coadd stamps centered on the fused position.

    Panel layout is roughly 2∶1 (columns∶rows). Overlays the fused RA/DEC
    cross and each band's ``Maj``/``Min``/``PA`` ellipse from *lst_matches*
    when present. Missing coadd files become titled empty panels (no
    exception). Bands with ``n_confused_{band} > 1`` on *meta_row* get a
    dashed yellow box around the subplot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from lwa_catalog.analyze.trace import _band_palette, confused_bands

    ref_ra = float(meta_row["RA"])
    ref_dec = float(meta_row["DEC"])
    if not (np.isfinite(ref_ra) and np.isfinite(ref_dec)):
        msg = "meta_row needs finite RA and DEC for cutouts"
        raise ValueError(msg)

    if bands is None:
        if lst_matches is not None and not lst_matches.empty and "band" in lst_matches.columns:
            band_list = list(dict.fromkeys(lst_matches["band"].astype(str).tolist()))
        else:
            raw = str(meta_row.get("bands_present", "") or "")
            band_list = [
                p.strip() for p in raw.split(",") if p.strip() and p.strip().lower() != "nan"
            ]
    else:
        band_list = [str(b) for b in bands]

    if not band_list:
        if fig is None:
            fig, ax = plt.subplots(figsize=(6.0, 2.5))
        else:
            fig.clf()
            ax = fig.add_subplot(111)
        ax.set_title("Band cutouts (no bands)")
        ax.axis("off")
        return fig

    size_deg = _cutout_size_deg(
        lst_matches if lst_matches is not None else pd.DataFrame(),
        beam_factor=beam_factor,
        default_bmaj_deg=default_bmaj_deg,
    )
    palette = _band_palette(band_list)
    confused = confused_bands(meta_row)
    n = len(band_list)
    nrows, ncols = cutout_grid_shape(n)
    fig_w = figsize_per[0] * ncols
    fig_h = figsize_per[1] * nrows
    if fig is None:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_w, fig_h),
            squeeze=False,
        )
    else:
        fig.clf()
        fig.set_size_inches(fig_w, fig_h, forward=True)
        axes = fig.subplots(nrows, ncols, squeeze=False)

    match_by_band: dict[str, pd.Series] = {}
    if lst_matches is not None and not lst_matches.empty and "band" in lst_matches.columns:
        for _, row in lst_matches.iterrows():
            match_by_band[str(row["band"])] = row

    for i, band in enumerate(band_list):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        color = palette.get(band, "#ffcc00")
        try:
            healpix_map, weight, meta = map_cache.get(band)
            hdu = healpix_cutout(
                healpix_map,
                weight,
                ref_ra,
                ref_dec,
                size_deg=size_deg,
                nside=int(meta.get("nside", map_cache.nside)),
                nested=bool(meta.get("nested", map_cache.nested)),
                coord_frame=str(meta.get("coord_frame", map_cache.coord_frame)),
            )
            data = np.asarray(hdu.data, dtype=float)
            finite = data[np.isfinite(data)]
            if finite.size:
                vmin, vmax = np.percentile(finite, list(percentile))
            else:
                vmin, vmax = 0.0, 1.0
            ax.imshow(
                data,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            wcs = WCS(hdu.header)
            band_row = match_by_band.get(band)
            if band_row is not None:
                try:
                    bra = float(band_row["RA"])
                    bdec = float(band_row["DEC"])
                except (TypeError, ValueError, KeyError):
                    bra, bdec = float("nan"), float("nan")
                try:
                    maj = float(band_row.get("Maj", np.nan))
                    minor = float(band_row.get("Min", np.nan))
                    pa = float(band_row.get("PA", 0.0))
                except (TypeError, ValueError):
                    maj = minor = pa = float("nan")
                _draw_ellipse_on_ax(
                    ax,
                    wcs,
                    ra_deg=bra if np.isfinite(bra) else ref_ra,
                    dec_deg=bdec if np.isfinite(bdec) else ref_dec,
                    maj_deg=maj,
                    min_deg=minor,
                    pa_deg=pa,
                    color=color,
                )
            title = f"{band} (confused)" if band in confused else band
            ax.set_title(title, fontsize=8, color=color, pad=2)
        except FileNotFoundError:
            title = f"{band} (confused)" if band in confused else band
            ax.set_title(title, fontsize=8, color=color, pad=2)
            ax.text(
                0.5,
                0.5,
                f"missing\n{map_cache.path_for(band).name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=6,
                wrap=True,
            )
        except Exception as exc:  # noqa: BLE001 — show failure in panel
            title = f"{band} (confused)" if band in confused else band
            ax.set_title(title, fontsize=8, color=color, pad=2)
            ax.text(
                0.5,
                0.5,
                f"error\n{exc}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=6,
                wrap=True,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        if band in confused:
            ax.add_patch(
                Rectangle(
                    (0.0, 0.0),
                    1.0,
                    1.0,
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor="yellow",
                    linewidth=2.0,
                    linestyle="--",
                    clip_on=False,
                    zorder=20,
                )
            )

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"HEALPix cutouts @ RA={ref_ra:.4f}, Dec={ref_dec:.4f} "
        f"(FOV≈{size_deg:.2f}°)",
        fontsize=9,
    )
    fig.tight_layout()
    return fig
