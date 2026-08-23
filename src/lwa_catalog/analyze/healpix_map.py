"""Catalog → HEALPix map helpers, with HiPS export via ``lwa-healpix``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _require_healpy():
    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - exercised when healpy absent
        msg = (
            "healpy is required for HEALPix maps; "
            "install with: pip install 'lwa-catalog[analyze]'"
        )
        raise ImportError(msg) from exc
    return hp


def _require_lwa_healpix():
    try:
        from lwa_healpix import healpix_to_hips
    except ImportError as exc:  # pragma: no cover
        msg = (
            "lwa-healpix is required for HiPS export; "
            "install with: pip install 'lwa-catalog[analyze]' "
            "(or pip install /path/to/lwa-healpix)"
        )
        raise ImportError(msg) from exc
    return healpix_to_hips


def metacatalog_to_healpix(
    catalog: pd.DataFrame,
    *,
    nside: int = 64,
    weight_col: str | None = "Peak_flux",
    nest: bool = False,
) -> np.ndarray:
    """Bin metacatalog rows into a Peak_flux-weighted (or count) HEALPix map.

    The map is in **equatorial** (RA/Dec) RING ordering by default (``nest=False``),
    suitable for :func:`write_healpix_hips` with ``coord_frame="equatorial"``.

    Parameters
    ----------
    catalog
        Table with ``RA``, ``DEC`` in degrees; optional weight column.
    nside
        HEALPix NSIDE.
    weight_col
        Column to sum into pixels; ``None`` uses counts (weight 1).
    nest
        If True, use NESTED numbering; else RING.

    Returns
    -------
    ndarray
        Length ``12 * nside**2`` map (zeros where empty).
    """
    hp = _require_healpy()
    npix = hp.nside2npix(int(nside))
    out = np.zeros(npix, dtype=float)
    if catalog is None or catalog.empty:
        return out
    if "RA" not in catalog.columns or "DEC" not in catalog.columns:
        msg = "catalog needs RA and DEC columns"
        raise ValueError(msg)

    ra = pd.to_numeric(catalog["RA"], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(catalog["DEC"], errors="coerce").to_numpy(dtype=float)
    if weight_col is None:
        w = np.ones(len(catalog), dtype=float)
    else:
        if weight_col not in catalog.columns:
            msg = f"weight column {weight_col!r} not in catalog"
            raise ValueError(msg)
        w = pd.to_numeric(catalog[weight_col], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(w)
    if not ok.any():
        return out
    ra = ra[ok]
    dec = dec[ok]
    w = w[ok]
    theta = np.deg2rad(90.0 - dec)
    phi = np.deg2rad(ra)
    pix = hp.ang2pix(int(nside), theta, phi, nest=bool(nest))
    np.add.at(out, pix, w)
    return out


def write_healpix_hips(
    m: np.ndarray,
    output_directory: str | Path,
    *,
    nest: bool = False,
    coord_frame: str = "equatorial",
    threads: bool = True,
    cut_percentiles: tuple[float, float] | None = (1.0, 99.0),
    properties: dict[str, str] | None = None,
) -> Path:
    """Write a HEALPix map as a HiPS tile set via :func:`lwa_healpix.healpix_to_hips`.

    Catalog maps from :func:`metacatalog_to_healpix` use equatorial RA/Dec, so the
    default ``coord_frame`` is ``\"equatorial\"``.

    Parameters
    ----------
    m
        1-D HEALPix map.
    output_directory
        Directory for HiPS tiles, ``properties``, and Aladin Lite ``index.html``.
    nest
        If True, *m* uses NESTED ordering.
    coord_frame
        Coordinate frame of *m* (``\"equatorial\"``, ``\"galactic\"``, …).
    threads
        Multi-threaded reprojection (set ``False`` in tests).
    cut_percentiles, properties
        Forwarded to :func:`lwa_healpix.healpix_to_hips`.

    Returns
    -------
    Path
        Resolved HiPS output directory.
    """
    healpix_to_hips = _require_lwa_healpix()
    out = Path(output_directory)
    # reproject_to_hips creates the directory itself (exist_ok=False)
    if out.exists() and any(out.iterdir()):
        msg = f"HiPS output directory is not empty: {out}"
        raise FileExistsError(msg)
    healpix_to_hips(
        np.asarray(m, dtype=float),
        coord_frame=coord_frame,
        output_directory=out,
        nested=bool(nest),
        threads=bool(threads),
        cut_percentiles=cut_percentiles,
        properties=properties,
    )
    return out.resolve()


def metacatalog_to_hips(
    catalog: pd.DataFrame,
    output_directory: str | Path,
    *,
    nside: int = 64,
    weight_col: str | None = "Peak_flux",
    nest: bool = False,
    coord_frame: str = "equatorial",
    threads: bool = True,
    properties: dict[str, str] | None = None,
) -> Path:
    """Bin a metacatalog to HEALPix and write a HiPS tile set in one step."""
    m = metacatalog_to_healpix(
        catalog, nside=nside, weight_col=weight_col, nest=nest
    )
    return write_healpix_hips(
        m,
        output_directory,
        nest=nest,
        coord_frame=coord_frame,
        threads=threads,
        properties=properties,
    )
