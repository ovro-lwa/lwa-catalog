"""Catalog → HEALPix map helpers (optional ``healpy`` dependency)."""

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


def metacatalog_to_healpix(
    catalog: pd.DataFrame,
    *,
    nside: int = 64,
    weight_col: str | None = "Peak_flux",
    nest: bool = False,
) -> np.ndarray:
    """Bin metacatalog rows into a Peak_flux-weighted (or count) HEALPix map.

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


def write_healpix_fits(
    m: np.ndarray,
    path: str | Path,
    *,
    nside: int,
    nest: bool = False,
    coord: str = "C",
) -> Path:
    """Write a HEALPix map to FITS via healpy."""
    hp = _require_healpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hp.write_map(
        str(path),
        m,
        nest=bool(nest),
        coord=coord,
        column_names=["SIGNAL"],
        dtype=np.float64,
        overwrite=True,
        extra_header=[("NSIDE", int(nside), "HEALPix NSIDE")],
    )
    return path.resolve()
