"""Discover OVRO-LWA FITS images and parse LST hour / band metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

HOUR_DIR_RE = re.compile(r"^(\d{2})h$", re.IGNORECASE)

# Deep wideband color products: I_01h_deep_Taper_R0_Full.fits
DEEP_COLOR_RE = re.compile(
    r"^I_(\d{2})h_.*_(Full|Blue|Green|Red)\.fits$",
    re.IGNORECASE,
)

# Portal lst-color products: Blue_I_..._20250508_LST22h_t0001.fits
LST_COLOR_RE = re.compile(
    r"^(Full|Blue|Green|Red)_I_.*_(\d{8})_LST(\d{1,2})h_(t\d+)\.fits$",
    re.IGNORECASE,
)

# Band prefix fallback: Blue_I_....fits (LST from directory or LSTnnh elsewhere in name)
BAND_PREFIX_RE = re.compile(
    r"^(Full|Blue|Green|Red)_I_.*\.fits$",
    re.IGNORECASE,
)

# Frequency subband coadd: 18MHz_I_..._LST01h_....fits
SUBBAND_COADD_RE = re.compile(
    r"^(\d+MHz)_I_.*\.fits$",
    re.IGNORECASE,
)
LST_IN_NAME_RE = re.compile(r"LST(\d{1,2})h", re.IGNORECASE)
# Parent dir like 01h_18MHz or 01h_Blue
SUBBAND_DIR_RE = re.compile(r"^(\d{2})h_(.+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class FitsMetadata:
    """FITS path plus parsed LST hour, band, and optional time key."""

    path: Path
    lst_hour: str  # e.g. "01h"
    band: str  # Full | Blue | Green | Red | 18MHz | ...
    time_key: str | None = None  # optional lst-color time bin key


def format_lst_hour(hour: int | str) -> str:
    """Normalize an hour value to ``NNh`` (zero-padded)."""
    return f"{int(hour):02d}h"


def _lst_from_parents(path: Path) -> str | None:
    for parent in path.parents:
        m = HOUR_DIR_RE.match(parent.name)
        if m:
            return format_lst_hour(m.group(1))
        m = SUBBAND_DIR_RE.match(parent.name)
        if m:
            return format_lst_hour(m.group(1))
    return None


def _lst_from_name_or_parents(path: Path) -> str | None:
    m_lst = LST_IN_NAME_RE.search(path.name)
    if m_lst:
        return format_lst_hour(m_lst.group(1))
    return _lst_from_parents(path)


def parse_fits_metadata(path: Path) -> FitsMetadata | None:
    """Return LST hour and band for a FITS path, or None if unrecognized."""
    path = Path(path)
    name = path.name

    m = DEEP_COLOR_RE.match(name)
    if m:
        return FitsMetadata(
            path=path,
            lst_hour=format_lst_hour(m.group(1)),
            band=m.group(2).title(),
        )

    m = LST_COLOR_RE.match(name)
    if m:
        band, ymd, lst_h, t_bin = m.group(1), m.group(2), m.group(3), m.group(4)
        return FitsMetadata(
            path=path,
            lst_hour=format_lst_hour(lst_h),
            band=band.title(),
            time_key=f"{ymd}_LST{format_lst_hour(lst_h)}_{t_bin}",
        )

    m = BAND_PREFIX_RE.match(name)
    if m:
        band = m.group(1).title()
        lst_h = _lst_from_name_or_parents(path)
        if lst_h is None:
            return None
        return FitsMetadata(path=path, lst_hour=lst_h, band=band)

    m = SUBBAND_COADD_RE.match(name)
    if m:
        band = m.group(1)  # keep original casing, e.g. 18MHz
        lst_h = _lst_from_name_or_parents(path)
        if lst_h is None:
            return None
        return FitsMetadata(path=path, lst_hour=lst_h, band=band)

    return None


def discover_fits_files(
    root: Path,
    *,
    patterns: Iterable[str] = ("*.fits",),
) -> list[FitsMetadata]:
    """Recursively find FITS files and parse metadata."""
    root = Path(root)
    found: list[FitsMetadata] = []
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            meta = parse_fits_metadata(path)
            if meta is not None:
                found.append(meta)
    return found


def lst_hours_from_discovery(found: Iterable[FitsMetadata]) -> list[str]:
    """Return sorted unique LST hours (00h–23h) present in discovered FITS."""
    hours: set[str] = set()
    for meta in found:
        m = HOUR_DIR_RE.match(meta.lst_hour)
        if m is None:
            continue
        hour = int(m.group(1))
        if 0 <= hour <= 23:
            hours.add(format_lst_hour(hour))
    return sorted(hours)


def discovered_slots(
    found: Iterable[FitsMetadata],
) -> dict[tuple[str, str], FitsMetadata]:
    """Map ``(lst_hour, band)`` → metadata for every discovered FITS."""
    return {(m.lst_hour, m.band): m for m in found}


def slot_glob_patterns(lst_hour: str, band: str) -> list[str]:
    """Glob patterns used to resolve a single ``(lst_hour, band)`` FITS slot."""
    patterns = [
        f"**/*_{lst_hour}_*_{band}.fits",
        f"**/I_{lst_hour}_*_{band}.fits",
        f"I_{lst_hour}_*_{band}.fits",
    ]
    if band != "Full":
        patterns.insert(0, f"**/{band}_I_*_LST{lst_hour[0:2]}h_*.fits")
        patterns.insert(1, f"**/{band}_I_*_LST{int(lst_hour[:-1])}h_*.fits")
    else:
        patterns.insert(0, f"**/Full_I_*_LST{lst_hour[0:2]}h_*.fits")
    return patterns


def resolve_fits_slot(root: Path, lst_hour: str, band: str) -> Path:
    """Resolve exactly one FITS path for an ``(lst_hour, band)`` slot."""
    root = Path(root)
    for pattern in slot_glob_patterns(lst_hour, band):
        matches = sorted({p.resolve() for p in root.glob(pattern)})
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Ambiguous glob for ({lst_hour}, {band}): pattern {pattern!r} matched "
                f"{len(matches)} files: {[m.name for m in matches]}"
            )
    raise FileNotFoundError(f"No FITS found for ({lst_hour}, {band}) under {root}")
