"""Detect sources on nested HEALPix tiles (Option 2 packaging).

Requires ``lwa-catalog[analyze]`` (``lwa-healpix``, ``healpy``) and
``lwa-catalog[detect]`` (PyBDSF) for the full tile loop. Does **not** call
``blank_below_elevation`` (tile CRVAL is not zenith).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from lwa_catalog.constants import GAUL_DETECTION_COLUMNS
from lwa_catalog.coords import normalize_ra_columns
from lwa_catalog.create.detect import run_pybdsf_on_hdu
from lwa_catalog.gaul import cast_gaul_string_columns

__all__ = [
    "attach_beam_and_freq",
    "detect_sources_on_healpix_tiles",
    "median_beam_from_paths",
    "restfreq_hz_from_header",
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


def detect_sources_on_healpix_tiles(
    healpix_map: np.ndarray,
    weight: np.ndarray,
    *,
    nside_map: int,
    nside_tile: int = 4,
    overlap: float = 0.2,
    coord_frame: str = "equatorial",
    nested: bool = True,
    bmaj: float,
    bmin: float,
    bpa: float = 0.0,
    restfreq_hz: float | None = None,
    band: str = "Full",
    bdsf_kw: Mapping[str, Any] | None = None,
    gaul_columns: Sequence[str] = GAUL_DETECTION_COLUMNS,
    skip_empty: bool = True,
    ctype: str = "TAN",
    bunit: str = "JY/BEAM",
) -> pd.DataFrame:
    """Project nested tiles from a HEALPix map and run PyBDSF on each.

    Parameters
    ----------
    healpix_map, weight
        1-D HEALPix arrays (same ordering; typically NESTED).
    nside_map, nside_tile
        Map and tiling NSIDE (defaults match the Option 2 plan: 2048 / 4).
    overlap, ctype, coord_frame, nested
        Forwarded to ``lwa_healpix.iter_nested_tile_headers`` / ``healpix_to_hdu``.
    bmaj, bmin, bpa, restfreq_hz, bunit
        Attached to each tile HDU before PyBDSF (not stored in lwa-healpix).
    band
        Catalog ``band`` column value.
    bdsf_kw
        Extra PyBDSF keywords.
    skip_empty
        If ``True``, skip tiles whose nested child weight sum is ``<= 0``.

    Returns
    -------
    pandas.DataFrame
        Concatenated GAUL rows with ``tile_ipix``, ``nside_tile``, ``band``,
        and beam columns. Empty if no detections.
    """
    lwa_healpix = _import_lwa_healpix()
    weight_arr = np.asarray(weight)
    map_arr = np.asarray(healpix_map)

    tiles = lwa_healpix.iter_nested_tile_headers(
        nside_tile,
        nside_map,
        overlap=overlap,
        ctype=ctype,
        coord_frame=coord_frame,
        weight=weight_arr if skip_empty else None,
        min_weight_sum=0.0,
    )

    frames: list[pd.DataFrame] = []
    for ipix, target_header in tiles:
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
            bmaj=bmaj,
            bmin=bmin,
            bpa=bpa,
            restfreq_hz=restfreq_hz,
            bunit=bunit,
        )
        if not np.isfinite(hdu.data).any():
            continue

        table = run_pybdsf_on_hdu(hdu, bdsf_kw=bdsf_kw)
        if table is None or len(table) == 0:
            continue
        df = _table_to_tile_dataframe(
            table,
            gaul_columns=gaul_columns,
            tile_ipix=ipix,
            nside_tile=nside_tile,
            band=band,
            bmaj=bmaj,
            bmin=bmin,
            bpa=bpa,
        )
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
