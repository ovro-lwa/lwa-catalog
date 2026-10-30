"""Detect sources on nested HEALPix tiles (Option 2 packaging).

Requires ``lwa-catalog[analyze]`` (``lwa-healpix``, ``healpy``) and
``lwa-catalog[detect]`` (PyBDSF) for the full tile loop. Does **not** call
``blank_below_elevation`` (tile CRVAL is not zenith).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from lwa_catalog.constants import GAUL_DETECTION_COLUMNS
from lwa_catalog.coords import normalize_ra_columns
from lwa_catalog.create.beam_scale_model import (
    HEALPIX_ATTACH_BPA_DEG,
    BeamScaleModel,
    beam_scale_model_from_value,
    resolve_beam_scale_model,
)
from lwa_catalog.create.detect import run_pybdsf_on_hdu
from lwa_catalog.create.tiered_detect import (
    fuse_gaul_m_with_tier2_s,
    merge_tier2_bdsf_kw,
)
from lwa_catalog.gaul import cast_gaul_string_columns

# Global scale on band-median BMAJ/BMIN for HEALPix-tile PyBDSF (both tiers).
# Default 1.0 (unscaled median); use estimate_beam_scale for per-band residual-optimal α.
# Prefer BeamScaleModel (α, κ) for Dec-dependent Maj elongation; see beam_scale_model.
DEFAULT_BEAM_SCALE: float = 1.0

__all__ = [
    "DEFAULT_BEAM_SCALE",
    "HEALPIX_ATTACH_BPA_DEG",
    "BeamScaleModel",
    "attach_beam_and_freq",
    "beam_scale_model_from_value",
    "detect_sources_on_healpix_tiles",
    "fuse_gaul_m_with_tier2_s",
    "median_beam_from_paths",
    "merge_tier2_bdsf_kw",
    "resolve_beam_scale",
    "resolve_beam_scale_model",
    "restfreq_hz_from_header",
    "sample_tile_onto_healpix",
    "scale_beam",
]


def _import_lwa_healpix():
    try:
        import lwa_healpix
    except ImportError as exc:  # pragma: no cover
        msg = (
            "lwa_healpix is required for HEALPix tile detection; "
            "pip install 'lwa-catalog[analyze]' "
            "(and editable lwa-healpix if developing locally)"
        )
        raise ImportError(msg) from exc
    return lwa_healpix


def median_beam_from_paths(paths: Sequence[str | Path]) -> tuple[float, float, float]:
    """Return median ``(BMAJ, BMIN, BPA)`` in degrees from FITS headers."""
    if not paths:
        msg = "paths must be non-empty"
        raise ValueError(msg)
    bmaj: list[float] = []
    bmin: list[float] = []
    bpa: list[float] = []
    for path in paths:
        hdr = fits.getheader(path)
        if "BMAJ" not in hdr or "BMIN" not in hdr:
            msg = f"FITS header missing BMAJ/BMIN: {path}"
            raise ValueError(msg)
        bmaj.append(float(hdr["BMAJ"]))
        bmin.append(float(hdr["BMIN"]))
        bpa.append(float(hdr.get("BPA", 0.0)))
    return float(np.median(bmaj)), float(np.median(bmin)), float(np.median(bpa))


def scale_beam(
    bmaj: float,
    bmin: float,
    bpa: float = 0.0,
    *,
    scale: float = DEFAULT_BEAM_SCALE,
) -> tuple[float, float, float]:
    """Return ``(scale * BMAJ, scale * BMIN, BPA)`` in degrees.

    Used to scale the band-median restoring beam before PyBDSF on HEALPix
    tiles (default ``scale=1.0``, identity). ``BPA`` is unchanged. Raises
    ``ValueError`` if *scale* is non-finite or ``<= 0``.
    """
    s = float(scale)
    if not np.isfinite(s) or s <= 0.0:
        msg = f"beam scale must be finite and > 0, got {scale!r}"
        raise ValueError(msg)
    return float(s * float(bmaj)), float(s * float(bmin)), float(bpa)


def resolve_beam_scale(
    band: str,
    beam_scale: float | Mapping[str, float],
    *,
    default: float | None = None,
) -> float:
    """Return a finite ``> 0`` beam α for *band*.

    *beam_scale* may be a scalar (same α for every band) or a ``{band: α}``
    mapping from :func:`~lwa_catalog.analyze.bootstrap_beam_alpha`. When the
    mapping omits *band*, *default* is used (or :data:`DEFAULT_BEAM_SCALE`
    when *default* is ``None``).
    """
    if isinstance(beam_scale, Mapping):
        if band in beam_scale:
            raw = beam_scale[band]
        else:
            raw = DEFAULT_BEAM_SCALE if default is None else default
    else:
        raw = beam_scale
    s = float(raw)
    if not np.isfinite(s) or s <= 0.0:
        msg = f"beam scale for {band!r} must be finite and > 0, got {raw!r}"
        raise ValueError(msg)
    return s


def restfreq_hz_from_header(header: fits.Header) -> float:
    """Return rest frequency in Hz from common FITS keywords."""
    for key in ("RESTFREQ", "RESTFRQ", "CRVAL3", "FREQ"):
        if key in header:
            return float(header[key])
    msg = "No RESTFREQ/RESTFRQ/CRVAL3/FREQ in header"
    raise KeyError(msg)


def attach_beam_and_freq(
    hdu: fits.PrimaryHDU,
    *,
    bmaj: float,
    bmin: float,
    bpa: float = 0.0,
    restfreq_hz: float | None = None,
    bunit: str = "JY/BEAM",
) -> fits.PrimaryHDU:
    """Copy *hdu* and set beam / frequency keywords for PyBDSF."""
    header = hdu.header.copy()
    data = np.asarray(hdu.data, dtype=np.float32)
    header["BMAJ"] = float(bmaj)
    header["BMIN"] = float(bmin)
    header["BPA"] = float(bpa)
    if restfreq_hz is not None:
        header["RESTFREQ"] = float(restfreq_hz)
        header["RESTFRQ"] = float(restfreq_hz)
    if bunit is not None:
        header["BUNIT"] = bunit
    return fits.PrimaryHDU(data=data, header=header)


def sample_tile_onto_healpix(
    out: np.ndarray,
    tile_data: np.ndarray,
    tile_header: fits.Header,
    *,
    tile_ipix: int,
    nside_map: int,
    nside_tile: int,
) -> None:
    """Write *tile_data* into the NESTED HEALPix children of *tile_ipix*.

    Nearest-pixel sampling of the tile at each child pixel centre (lon/lat in
    the tile header's celestial frame). Children outside the tile or on NaN
    tile pixels are left unchanged. Intended for smooth products such as the
    PyBDSF background mean map, not for point-like model images.
    """
    import healpy as hp
    from astropy.wcs import WCS

    from lwa_catalog.create.beam_scale import tile_pixel_mask

    pix = tile_pixel_mask(nside_map, nside_tile, tile_ipix, nested=True)
    lon, lat = hp.pix2ang(int(nside_map), pix, nest=True, lonlat=True)
    x, y = WCS(tile_header).celestial.world_to_pixel_values(lon, lat)
    xi = np.rint(np.asarray(x)).astype(np.int64)
    yi = np.rint(np.asarray(y)).astype(np.int64)
    ny, nx = tile_data.shape
    inside = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    vals = np.full(pix.size, np.nan, dtype=float)
    vals[inside] = tile_data[yi[inside], xi[inside]]
    ok = np.isfinite(vals)
    out[pix[ok]] = vals[ok]


def detect_sources_on_healpix_tiles(
    healpix_map: np.ndarray,
    weight: np.ndarray,
    *,
    nside_map: int,
    nside_tile: int = 4,
    overlap: float = 0.2,
    margin: float = 0.05,
    align: str = "diamond",
    coord_frame: str = "icrs",
    nested: bool = True,
    bmaj: float,
    bmin: float,
    bpa: float = 0.0,
    beam_scale: float | BeamScaleModel = DEFAULT_BEAM_SCALE,
    restfreq_hz: float | None = None,
    band: str = "Full",
    bdsf_kw: Mapping[str, Any] | None = None,
    tier2_bdsf_kw: Mapping[str, Any] | None = None,
    gaul_columns: Sequence[str] = GAUL_DETECTION_COLUMNS,
    skip_empty: bool = True,
    ctype: str = "TAN",
    bunit: str = "JY/BEAM",
    background_out: np.ndarray | None = None,
) -> pd.DataFrame:
    """Project nested tiles from a HEALPix map and run PyBDSF on each.

    Parameters
    ----------
    healpix_map, weight
        1-D HEALPix arrays (same ordering; typically NESTED).
    nside_map, nside_tile
        Map and tiling NSIDE (defaults match the Option 2 plan: 2048 / 4).
    overlap, margin, align, ctype, coord_frame, nested
        Forwarded to ``lwa_healpix.iter_nested_tile_headers`` /
        ``healpix_to_hdu``. Default ``align="diamond"`` rotates each TAN
        tile onto the HEALPix cell edges (``margin`` pad); use
        ``align="celestial"`` with ``overlap`` for the legacy north-aligned
        squares. Use reproject healpix frame names (``\"icrs\"``,
        ``\"galactic\"``, ``\"c\"``, ``\"g\"``).
    bmaj, bmin, bpa, restfreq_hz, bunit
        Band-median restoring beam and frequency. Axes are scaled by
        *beam_scale* before attach (not stored in lwa-healpix). Catalog /
        attach ``BPA`` is always :data:`HEALPIX_ATTACH_BPA_DEG` (90°);
        the *bpa* argument is ignored on this path.
    beam_scale
        Isotropic α (float) or :class:`BeamScaleModel` ``(α, κ)`` evaluated
        at each tile center Dec (default isotropic
        :data:`DEFAULT_BEAM_SCALE` ``1.0``). Catalog ``BMAJ``/``BMIN`` use
        the tile-effective beam. Residual-optimal scalar α:
        :func:`lwa_catalog.create.estimate_beam_scale`.
    band
        Catalog ``band`` column value.
    bdsf_kw
        Tier-1 PyBDSF keywords (deep / low-threshold catalog).
    tier2_bdsf_kw
        If not ``None``, run a second PyBDSF pass per tile with
        ``merge_tier2_bdsf_kw(bdsf_kw, tier2_bdsf_kw)`` (defaults
        ``thresh_isl=4``, ``thresh_pix=7``) and fuse with
        :func:`fuse_gaul_m_with_tier2_s` so over-decomposed ``M`` sources
        are replaced by coincident tier-2 ``S`` Gaussians when present.
    skip_empty
        If ``True``, skip tiles whose nested child weight sum is ``<= 0``.
    background_out
        Optional 1-D NESTED array (``12 * nside_map**2``, usually NaN-filled)
        filled in place with each tile's tier-1 PyBDSF background mean
        (:func:`~lwa_catalog.create.detect.pybdsf_background_image`), sampled
        onto that tile's own nested children. Gaussians are fit above this
        mean, so residuals should be ``obs − background − model``. Requires
        ``nested=True``.

    Returns
    -------
    pandas.DataFrame
        Concatenated GAUL rows with ``tile_ipix``, ``nside_tile``, ``band``,
        and beam columns. Empty if no detections.
    """
    lwa_healpix = _import_lwa_healpix()
    weight_arr = np.asarray(weight)
    map_arr = np.asarray(healpix_map)
    model = beam_scale_model_from_value(beam_scale)
    _ = bpa  # ignored: HEALPix-tile attach always uses HEALPIX_ATTACH_BPA_DEG
    run_tier2 = tier2_bdsf_kw is not None
    tier2_kw = merge_tier2_bdsf_kw(bdsf_kw, tier2_bdsf_kw) if run_tier2 else None
    if background_out is not None:
        if not nested:
            msg = "background_out requires nested=True"
            raise ValueError(msg)
        if background_out.shape != (12 * int(nside_map) ** 2,):
            msg = (
                f"background_out shape {background_out.shape} != "
                f"(12*nside_map**2,) for nside_map={nside_map}"
            )
            raise ValueError(msg)

    tiles = lwa_healpix.iter_nested_tile_headers(
        nside_tile,
        nside_map,
        overlap=overlap,
        margin=margin,
        align=align,
        ctype=ctype,
        coord_frame=coord_frame,
        weight=weight_arr if skip_empty else None,
        min_weight_sum=0.0,
    )

    frames: list[pd.DataFrame] = []
    for ipix, target_header in tiles:
        dec_deg = float(target_header["CRVAL2"])
        bmaj_s, bmin_s, bpa_s = model.effective_beam(
            bmaj, bmin, dec_deg=dec_deg
        )
        hdu = lwa_healpix.healpix_to_hdu(
            map_arr,
            target_header,
            weight=weight_arr,
            coord_frame=coord_frame,
            nested=nested,
            header_updates={"BUNIT": bunit} if bunit else None,
        )
        hdu = attach_beam_and_freq(
            hdu,
            bmaj=bmaj_s,
            bmin=bmin_s,
            bpa=bpa_s,
            restfreq_hz=restfreq_hz,
            bunit=bunit,
        )
        if not np.isfinite(hdu.data).any():
            continue

        try:
            if background_out is None:
                table = run_pybdsf_on_hdu(hdu, bdsf_kw=bdsf_kw)
            else:
                table, background = run_pybdsf_on_hdu(
                    hdu, bdsf_kw=bdsf_kw, return_background=True
                )
                if background is not None and background.shape == hdu.data.shape:
                    sample_tile_onto_healpix(
                        background_out,
                        background,
                        hdu.header,
                        tile_ipix=ipix,
                        nside_map=nside_map,
                        nside_tile=nside_tile,
                    )
        except Exception as exc:  # noqa: BLE001 — skip bad tiles; continue band
            warnings.warn(
                f"PyBDSF failed on tile_ipix={ipix} band={band!r} ({exc}); skipping tile",
                UserWarning,
                stacklevel=2,
            )
            continue
        if table is None or len(table) == 0:
            continue
        df = _table_to_tile_dataframe(
            table,
            gaul_columns=gaul_columns,
            tile_ipix=ipix,
            nside_tile=nside_tile,
            band=band,
            bmaj=bmaj_s,
            bmin=bmin_s,
            bpa=bpa_s,
        )
        if run_tier2:
            try:
                table2 = run_pybdsf_on_hdu(hdu, bdsf_kw=tier2_kw)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"PyBDSF tier-2 failed on tile_ipix={ipix} band={band!r} ({exc}); "
                    "keeping tier-1 only",
                    UserWarning,
                    stacklevel=2,
                )
                table2 = None
            if table2 is not None and len(table2) > 0:
                df2 = _table_to_tile_dataframe(
                    table2,
                    gaul_columns=gaul_columns,
                    tile_ipix=ipix,
                    nside_tile=nside_tile,
                    band=band,
                    bmaj=bmaj_s,
                    bmin=bmin_s,
                    bpa=bpa_s,
                )
                df = fuse_gaul_m_with_tier2_s(df, df2)
        frames.append(df)

    if not frames:
        cols = list(gaul_columns) + [
            "tile_ipix",
            "nside_tile",
            "band",
            "BMAJ",
            "BMIN",
            "BPA",
        ]
        return pd.DataFrame(columns=cols)
    return normalize_ra_columns(pd.concat(frames, ignore_index=True))


def _table_to_tile_dataframe(
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
