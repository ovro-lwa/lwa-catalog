"""Sky coordinate parsing and nearest-neighbor helpers for Aladin / Panel viewers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord


def parse_coordinate(text: str) -> SkyCoord:
    """Parse a single sky position from free-form text."""
    text = text.strip()
    if not text:
        msg = "Coordinate string is empty"
        raise ValueError(msg)
    parts = text.replace(",", " ").split()
    if len(parts) == 2:
        try:
            ra = float(parts[0])
            dec = float(parts[1])
            return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        except ValueError:
            pass
    try:
        return SkyCoord(text, unit=(u.hourangle, u.deg), frame="icrs")
    except Exception:
        return SkyCoord(text, frame="icrs")


def format_coordinate_deg(ra: float, dec: float) -> str:
    """Format RA/Dec as decimal degrees for text inputs."""
    return f"{ra:.6f} {dec:.6f}"


def nearest_sources(
    df: pd.DataFrame,
    coord: SkyCoord,
    *,
    n: int = 5,
    ra_col: str = "RA",
    dec_col: str = "DEC",
) -> pd.DataFrame:
    """Return the ``n`` catalog rows closest to *coord* (great-circle).

    Adds ``sep_arcmin`` and ``sep_deg`` as the leading columns. The returned
    frame is a defragmented copy so Tabulator / Panel can mutate it safely.
    """
    if ra_col not in df.columns or dec_col not in df.columns:
        raise KeyError(f"Need {ra_col!r} and {dec_col!r} columns for sky matching")
    if df.empty:
        raise ValueError("Catalog is empty")

    catalog = SkyCoord(
        ra=np.asarray(df[ra_col], dtype=float) * u.deg,
        dec=np.asarray(df[dec_col], dtype=float) * u.deg,
        frame="icrs",
    )
    seps = coord.separation(catalog)
    order = np.argsort(seps.deg)
    n = max(1, min(int(n), len(df)))
    out = df.iloc[order[:n]].copy()
    out.insert(0, "sep_arcmin", np.round(seps.deg[order[:n]] * 60.0, 4))
    out.insert(1, "sep_deg", np.round(seps.deg[order[:n]], 6))
    return out.reset_index(drop=True).copy()
