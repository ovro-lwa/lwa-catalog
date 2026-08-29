"""Per-image source detection with PyBDSF."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from lwa_catalog.constants import GAUL_COLUMNS
from lwa_catalog.coords import normalize_ra_columns
from lwa_catalog.create.discover import FitsMetadata

DEFAULT_BDSF_KW: dict[str, Any] = {
    "thresh": "hard",
    "thresh_isl": 7.0,
    "thresh_pix": 4.0,
    "atrous_do": False,
    "psf_vary_do": False,
    "quiet": True,
    "ncores": 16,
}


def _restfreq_hz(header: fits.Header) -> float | None:
    for key in ("RESTFREQ", "RESTFRQ", "CRVAL3", "FREQ"):
        if key in header:
            try:
                return float(header[key])
            except (TypeError, ValueError):
                continue
    return None


def prepare_hdu(path: Path) -> fits.PrimaryHDU:
    """Read FITS, squeeze to 2D, and fix header for PyBDSF.

    Non-finite pixels (±inf) are converted to NaN. NaNs are preserved so
    PyBDSF can blank them; zero-filling blanked primary-beam regions creates
    constant patches that trigger ``unphysical rms`` errors.
    """
    path = Path(path)
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[0]
        data = np.squeeze(np.asarray(hdu.data, dtype=np.float32))
        if data.ndim != 2:
            msg = f"Expected 2D image in {path.name}, got shape {data.shape}"
            raise ValueError(msg)
        # Copy so we do not mutate memmap-backed arrays in place.
        data = np.array(data, dtype=np.float32, copy=True)
        data[~np.isfinite(data)] = np.nan
        header = hdu.header.copy()
    rf = _restfreq_hz(header)
    if rf is not None:
        header["RESTFREQ"] = rf
        header["RESTFRQ"] = rf
    return fits.PrimaryHDU(data=data, header=header)


def upsample_hdu(hdu: fits.PrimaryHDU, *, factor: int = 2) -> fits.PrimaryHDU:
    """Upsample a 2D image HDU by an integer factor, updating WCS pixel keywords.

    Non-finite pixels are blanked in the upsampled array. ``BMAJ`` / ``BMIN`` /
    ``BPA`` are unchanged (physical beam size is independent of pixel grid).
    """
    if factor == 1:
        return hdu
    if factor < 1 or factor != int(factor):
        msg = f"upsample factor must be a positive integer, got {factor!r}"
        raise ValueError(msg)
    factor = int(factor)

    try:
        from scipy.ndimage import zoom
    except ImportError as exc:  # pragma: no cover
        msg = "scipy is required for upsample_hdu; pip install 'lwa-catalog[detect]'"
        raise ImportError(msg) from exc

    data = np.asarray(hdu.data, dtype=np.float32)
    if data.ndim != 2:
        msg = f"Expected 2D image, got shape {data.shape}"
        raise ValueError(msg)

    mask = ~np.isfinite(data)
    filled = np.where(mask, 0.0, data)
    upsampled = zoom(filled, factor, order=1).astype(np.float32, copy=False)
    mask_up = zoom(mask.astype(np.float32), factor, order=0) > 0.5
    upsampled[mask_up] = np.nan

    header = hdu.header.copy()
    for axis in (1, 2):
        naxis_key = f"NAXIS{axis}"
        cdelt_key = f"CDELT{axis}"
        crpix_key = f"CRPIX{axis}"
        if naxis_key in header:
            header[naxis_key] = int(header[naxis_key]) * factor
        if cdelt_key in header:
            header[cdelt_key] = float(header[cdelt_key]) / factor
        if crpix_key in header:
            header[crpix_key] = (float(header[crpix_key]) - 1.0) * factor + 1.0

    return fits.PrimaryHDU(data=upsampled, header=header)


def beam_from_header(header: fits.Header) -> tuple[float, float, float]:
    """Return ``(BMAJ, BMIN, BPA)`` from a FITS header."""
    if "BMAJ" not in header or "BMIN" not in header:
        raise ValueError("FITS header missing BMAJ/BMIN beam keywords")
    return (
        float(header["BMAJ"]),
        float(header["BMIN"]),
        float(header.get("BPA", 0.0)),
    )


def empty_sources_dataframe(
    meta: FitsMetadata,
    *,
    bmaj: float,
    bmin: float,
    bpa: float,
    gaul_columns: Sequence[str] = GAUL_COLUMNS,
) -> pd.DataFrame:
    """Return an empty per-image catalog with expected columns."""
    columns = list(gaul_columns) + ["lst_hour", "band", "source_file", "BMAJ", "BMIN", "BPA"]
    if meta.time_key is not None:
        columns.append("time_key")
    return pd.DataFrame(columns=columns)


def run_pybdsf_on_hdu(
    hdu: fits.PrimaryHDU,
    *,
    bdsf_kw: Mapping[str, Any] | None = None,
    **process_kw: Any,
) -> Table | None:
    """Run PyBDSF on an in-memory HDU and return the Gaussian catalog table."""
    try:
        import bdsf
    except ImportError as exc:  # pragma: no cover
        msg = "PyBDSF (bdsf) is required for detect_sources; pip install 'lwa-catalog[detect]'"
        raise ImportError(msg) from exc

    beam = beam_from_header(hdu.header)
    kw = dict(DEFAULT_BDSF_KW)
    if bdsf_kw is not None:
        kw.update(dict(bdsf_kw))
    kw.update(process_kw)
    kw["beam"] = beam

    img = bdsf.process_image(hdu, **kw)

    with tempfile.NamedTemporaryFile(suffix=".gaul.fits", delete=False) as tmp:
        cat_path = tmp.name
    try:
        img.write_catalog(
            outfile=cat_path,
            format="fits",
            catalog_type="gaul",
            clobber=True,
        )
        if not os.path.isfile(cat_path) or os.path.getsize(cat_path) == 0:
            return None
        return Table.read(cat_path)
    except Exception:
        return None
    finally:
        try:
            os.unlink(cat_path)
        except OSError:
            pass


def detect_sources(
    meta: FitsMetadata,
    *,
    bdsf_kw: Mapping[str, Any] | None = None,
    gaul_columns: Sequence[str] = GAUL_COLUMNS,
    upsample_factor: int = 1,
    **process_kw: Any,
) -> pd.DataFrame:
    """Detect sources in one FITS image; return a catalog DataFrame.

    Parameters
    ----------
    meta
        Parsed FITS path / LST / band metadata.
    bdsf_kw
        Base PyBDSF ``process_image`` keywords (merged with library defaults).
    gaul_columns
        Gaussian catalog columns to keep when present.
    upsample_factor
        Integer upsampling factor applied to the image before PyBDSF (``1`` =
        native pixels). WCS ``CDELT*`` / ``CRPIX*`` are updated; ``BMAJ`` /
        ``BMIN`` / ``BPA`` are taken from the native image header.
    **process_kw
        Extra keywords forwarded to ``bdsf.process_image``.
    """
    hdu = prepare_hdu(meta.path)
    bmaj, bmin, bpa = beam_from_header(hdu.header)
    if upsample_factor != 1:
        hdu = upsample_hdu(hdu, factor=upsample_factor)
    table = run_pybdsf_on_hdu(hdu, bdsf_kw=bdsf_kw, **process_kw)
    if table is None or len(table) == 0:
        return empty_sources_dataframe(
            meta,
            bmaj=bmaj,
            bmin=bmin,
            bpa=bpa,
            gaul_columns=gaul_columns,
        )

    df = table.to_pandas()
    keep = [c for c in gaul_columns if c in df.columns]
    df = df[keep].copy()
    df["lst_hour"] = meta.lst_hour
    df["band"] = meta.band
    df["source_file"] = meta.path.name
    if meta.time_key is not None:
        df["time_key"] = meta.time_key
    df["BMAJ"] = bmaj
    df["BMIN"] = bmin
    df["BPA"] = bpa
    return normalize_ra_columns(df)
