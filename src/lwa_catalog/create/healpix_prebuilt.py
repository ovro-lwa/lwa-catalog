"""Discover and load prebuilt per-subband HEALPix mosaic FITS.

Mosaic trees look like::

    HEALPIX_ROOT/
      82MHz/I/healpix/82MHz_I_deep_..._jybeam_feather_allsky*.fits
      78MHz/...

Files are typically healpy BinTable ``SIGNAL`` maps in RING order (not
``lwa_healpix`` MAP+WEIGHT). Load converts to NESTED and synthesizes a
finite-pixel weight so tile detect / HiPS can use the usual contract.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_PREBUILT_HEALPIX_GLOB",
    "DEFAULT_PREBUILT_PREFER_SUBSTRING",
    "beam_and_restfreq_from_healpix_fits",
    "discover_prebuilt_healpix_maps",
    "list_mhz_subband_dirs",
    "load_prebuilt_healpix_map",
    "resolve_prebuilt_healpix_for_band",
]

# Relative to HEALPIX_ROOT; ``{band}`` is substituted with e.g. ``82MHz``.
DEFAULT_PREBUILT_HEALPIX_GLOB = (
    "{band}/I/healpix/{band}_I_deep_Taper_Robust-0.75_coadd"
    "*mosaic_jybeam_feather_allsky*.fits"
)
DEFAULT_PREBUILT_PREFER_SUBSTRING = "ssub-rst8"

_MHZ_DIR_RE = re.compile(r"^(\d+)MHz$", re.IGNORECASE)


def _mhz_sort_key(name: str) -> int:
    m = _MHZ_DIR_RE.match(name)
    if m is None:
        return -1
    return int(m.group(1))


def list_mhz_subband_dirs(root: Path | str) -> list[str]:
    """Return ``*MHz`` subdirectory names under *root*, high → low MHz.

    Ignores non-directories and names that do not match ``\\d+MHz``
    (e.g. ``Blue``, ``Full``, ``tables``).
    """
    root = Path(root)
    if not root.is_dir():
        msg = f"HEALPIX_ROOT is not a directory: {root}"
        raise FileNotFoundError(msg)
    names = [
        path.name
        for path in root.iterdir()
        if path.is_dir() and _MHZ_DIR_RE.match(path.name)
    ]
    # Preserve canonical casing from disk but sort by numeric MHz high→low.
    return sorted(names, key=_mhz_sort_key, reverse=True)


def _format_band_glob(relative_glob: str, band: str) -> str:
    try:
        return relative_glob.format(band=band)
    except (KeyError, IndexError, ValueError):
        # Glob may use only ``??MHz`` wildcards with no ``{band}`` placeholder.
        return relative_glob


def resolve_prebuilt_healpix_for_band(
    root: Path | str,
    band: str,
    *,
    relative_glob: str = DEFAULT_PREBUILT_HEALPIX_GLOB,
    prefer_substring: str | None = DEFAULT_PREBUILT_PREFER_SUBSTRING,
) -> Path:
    """Resolve exactly one prebuilt HEALPix FITS for *band* under *root*.

    *relative_glob* is matched with :meth:`Path.glob` from *root* after
    ``str.format(band=band)`` (optional). When *prefer_substring* is set and
    at least one match contains that substring, only those matches are kept.
    Raises ``FileNotFoundError`` if zero or ambiguous matches remain.
    """
    root = Path(root)
    pattern = _format_band_glob(relative_glob, band)
    # Wide globs like ``??MHz/...`` match every subband; keep only paths whose
    # components include *band* (e.g. ``.../82MHz/I/healpix/...``).
    matches = sorted(
        {
            path.resolve()
            for path in root.glob(pattern)
            if path.is_file() and band in path.resolve().parts
        }
    )
    if not matches:
        msg = f"No prebuilt HEALPix FITS for band {band!r} under {root} matching {pattern!r}"
        raise FileNotFoundError(msg)

    if prefer_substring:
        preferred = [path for path in matches if prefer_substring in path.name]
        if preferred:
            matches = preferred

    if len(matches) == 1:
        return matches[0]

    names = [path.name for path in matches]
    msg = (
        f"Ambiguous prebuilt HEALPix FITS for band {band!r} under {root} "
        f"(pattern {pattern!r}"
        + (f", prefer={prefer_substring!r}" if prefer_substring else "")
        + f"): {names}"
    )
    raise FileNotFoundError(msg)


def discover_prebuilt_healpix_maps(
    root: Path | str,
    *,
    relative_glob: str = DEFAULT_PREBUILT_HEALPIX_GLOB,
    prefer_substring: str | None = DEFAULT_PREBUILT_PREFER_SUBSTRING,
    bands: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Map band → resolved prebuilt HEALPix path.

    When *bands* is ``None``, uses :func:`list_mhz_subband_dirs`. Bands with
    no matching file are omitted (not an error). Ambiguous matches still raise.
    """
    root = Path(root)
    band_list: Iterable[str] = bands if bands is not None else list_mhz_subband_dirs(root)
    out: dict[str, Path] = {}
    for band in band_list:
        pattern = _format_band_glob(relative_glob, band)
        matches = sorted(
            {
                path.resolve()
                for path in root.glob(pattern)
                if path.is_file() and band in path.resolve().parts
            }
        )
        if not matches:
            continue
        out[band] = resolve_prebuilt_healpix_for_band(
            root,
            band,
            relative_glob=relative_glob,
            prefer_substring=prefer_substring,
        )
    return out


def beam_and_restfreq_from_healpix_fits(
    path: Path | str,
) -> tuple[float, float, float, float]:
    """Return ``(BMAJ, BMIN, BPA, restfreq_hz)`` from a prebuilt HEALPix Primary.

    Mosaic products under ``…/I/healpix/`` store restoring-beam and frequency
    keywords on the Primary HDU (``BMAJ``/``BMIN`` required; ``BPA`` defaults
    to 0). Rest frequency uses the same key order as
    :func:`~lwa_catalog.create.healpix_detect.restfreq_hz_from_header`
    (``RESTFREQ``, ``RESTFRQ``, ``CRVAL3``, ``FREQ``).
    """
    from astropy.io import fits

    from lwa_catalog.create.healpix_detect import restfreq_hz_from_header

    path = Path(path)
    header = fits.getheader(path, 0)
    if "BMAJ" not in header or "BMIN" not in header:
        msg = f"Prebuilt HEALPix Primary missing BMAJ/BMIN: {path}"
        raise ValueError(msg)
    bmaj = float(header["BMAJ"])
    bmin = float(header["BMIN"])
    bpa = float(header.get("BPA", 0.0))
    restfreq = float(restfreq_hz_from_header(header))
    return bmaj, bmin, bpa, restfreq


def _coordsys_to_frame(code: str | None) -> str:
    key = str(code or "C").strip().upper()
    if key == "G":
        return "galactic"
    if key in {"C", "E"}:
        # Ecliptic uncommon for LWA; treat C as icrs (healpix detect default).
        return "icrs" if key == "C" else "ecliptic"
    return "icrs"


def load_prebuilt_healpix_map(
    path: Path | str,
    *,
    to_nested: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a mosaic/healpy HEALPix FITS as ``(map, weight, meta)``.

    Reads via ``healpy.read_map`` (BinTable ``SIGNAL`` or compatible). Converts
    RING → NESTED when *to_nested* is True and the file is RING. Weight is
    ``1`` where the map is finite, else ``0`` (``float32``).

    *meta* keys: ``nside``, ``nested``, ``coord_frame``, ``source_ordering``,
    ``path``.

    Requires ``healpy`` (``lwa-catalog[analyze]``).
    """
    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - optional dep
        msg = (
            "healpy is required to load prebuilt HEALPix maps; "
            "pip install 'lwa-catalog[analyze]'"
        )
        raise ImportError(msg) from exc

    path = Path(path)
    coord_frame = "icrs"
    source_ordering = "RING"
    try:
        from astropy.io import fits

        with fits.open(path, memmap=False) as hdul:
            primary = hdul[0].header
            ordering = None
            coordsys = primary.get("COORDSYS")
            for hdu in hdul:
                if ordering is None and "ORDERING" in hdu.header:
                    ordering = str(hdu.header["ORDERING"]).upper()
                if coordsys is None and "COORDSYS" in hdu.header:
                    coordsys = hdu.header["COORDSYS"]
            if ordering is not None:
                source_ordering = "NESTED" if ordering.startswith("NEST") else "RING"
            coord_frame = _coordsys_to_frame(None if coordsys is None else str(coordsys))
    except OSError:
        pass

    # Always materialize RING, then optionally convert to NESTED.
    healpix_map = np.asarray(hp.read_map(str(path), nest=False, dtype=np.float64), dtype=np.float64)
    nside = int(hp.get_nside(healpix_map))
    nested = False
    if to_nested:
        healpix_map = np.asarray(hp.reorder(healpix_map, r2n=True), dtype=np.float64)
        nested = True

    map_f32 = np.asarray(healpix_map, dtype=np.float32)
    weight = np.isfinite(map_f32).astype(np.float32)

    meta: dict[str, Any] = {
        "nside": nside,
        "nested": bool(nested),
        "coord_frame": coord_frame,
        "source_ordering": source_ordering,
        "path": path,
    }
    return map_f32, weight, meta
