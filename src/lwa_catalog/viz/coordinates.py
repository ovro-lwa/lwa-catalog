"""Sky coordinate parsing for Aladin / Panel viewers."""

from __future__ import annotations

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
