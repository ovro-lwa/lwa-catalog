"""Per-image source detection with PyBDSF."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    "ncores": 1,
}


def _available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        import multiprocessing

        return multiprocessing.cpu_count()


def _clear_pybdsf_modules() -> None:
    for mod in list(sys.modules):
        if mod == "bdsf" or mod.startswith("bdsf."):
            del sys.modules[mod]


def _import_pybdsf_safely() -> Any:
    """Import PyBDSF, tolerating a pre-set multiprocessing start method.

    PyBDSF unconditionally calls ``multiprocessing.set_start_method('fork')``
    at import time, which raises in Jupyter/IPython where the context is
    already ``spawn``.
    """
    if "bdsf" in sys.modules:
        return sys.modules["bdsf"]

    try:
        return importlib.import_module("bdsf")
    except RuntimeError as exc:
        if "context has already been set" not in str(exc):
            raise
        import multiprocessing

        orig = multiprocessing.set_start_method

        def _ignore_start_method(method: str, force: bool = False) -> None:
            try:
                orig(method, force=force)
            except RuntimeError:
                return None

        multiprocessing.set_start_method = _ignore_start_method  # type: ignore[method-assign]
        _clear_pybdsf_modules()
        try:
            return importlib.import_module("bdsf")
        finally:
            multiprocessing.set_start_method = orig


def _fork_process_pool(max_workers: int) -> ProcessPoolExecutor:
    """Return a process pool that forks workers (required for PyBDSF)."""
    import multiprocessing as mp

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)


def _detect_sources_task(
    meta: FitsMetadata,
    bdsf_kw: Mapping[str, Any] | None,
    gaul_columns: Sequence[str],
    upsample_factor: int,
    process_kw: Mapping[str, Any],
) -> pd.DataFrame:
    return detect_sources(
        meta,
        bdsf_kw=bdsf_kw,
        gaul_columns=gaul_columns,
        upsample_factor=upsample_factor,
        **process_kw,
    )


def _detect_sources_packed(task: tuple[Any, ...]) -> pd.DataFrame:
    return _detect_sources_task(*task)


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
        bdsf = _import_pybdsf_safely()
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


def iter_detect_sources(
    metas: Sequence[FitsMetadata],
    *,
    n_jobs: int | None = None,
    bdsf_kw: Mapping[str, Any] | None = None,
    gaul_columns: Sequence[str] = GAUL_COLUMNS,
    upsample_factor: int = 1,
    **process_kw: Any,
) -> Iterator[tuple[FitsMetadata, pd.DataFrame]]:
    """Yield ``(meta, catalog)`` as each image finishes.

    Workers only return DataFrames. The caller (parent process) can write
    Parquet as results arrive — completion order is not ``metas`` order when
    ``n_jobs > 1``.

    Parameters
    ----------
    metas
        One ``FitsMetadata`` per image to process.
    n_jobs
        Number of worker processes. ``None`` uses all CPUs available to this
        process. ``1`` runs serially in the current process.
    bdsf_kw, gaul_columns, upsample_factor, **process_kw
        Forwarded to :func:`detect_sources` for every image.
    """
    if not metas:
        return

    if n_jobs is None:
        n_jobs = _available_cpu_count()
    n_jobs = max(1, min(n_jobs, len(metas)))

    gaul_cols = tuple(gaul_columns)
    proc_kw = dict(process_kw)
    tasks = [
        (meta, bdsf_kw, gaul_cols, upsample_factor, proc_kw) for meta in metas
    ]

    if n_jobs == 1:
        for meta, task in zip(metas, tasks, strict=True):
            yield meta, _detect_sources_packed(task)
        return

    # Load PyBDSF in the parent before forking so workers inherit the module
    # without re-running PyBDSF's ``set_start_method('fork')`` import hook.
    _import_pybdsf_safely()

    with _fork_process_pool(n_jobs) as pool:
        future_to_index = {
            pool.submit(_detect_sources_packed, task): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            yield metas[i], future.result()


def detect_sources_many(
    metas: Sequence[FitsMetadata],
    *,
    n_jobs: int | None = None,
    bdsf_kw: Mapping[str, Any] | None = None,
    gaul_columns: Sequence[str] = GAUL_COLUMNS,
    upsample_factor: int = 1,
    **process_kw: Any,
) -> list[pd.DataFrame]:
    """Detect sources in many FITS images, parallelizing across images.

    Each image runs PyBDSF in its own worker process. Use ``ncores=1`` in
    ``bdsf_kw`` (the library default) so workers do not compete for CPUs.

    Collects :func:`iter_detect_sources` into a list in the same order as
    ``metas``. Prefer the iterator when the caller should persist each catalog
    as soon as it finishes.

    Parameters
    ----------
    metas
        One ``FitsMetadata`` per image to process.
    n_jobs
        Number of worker processes. ``None`` uses all CPUs available to this
        process. ``1`` runs serially in the current process.
    bdsf_kw, gaul_columns, upsample_factor, **process_kw
        Forwarded to :func:`detect_sources` for every image.

    Returns
    -------
    list[pd.DataFrame]
        Catalogs in the same order as ``metas``.
    """
    by_path = {
        meta.path: catalog
        for meta, catalog in iter_detect_sources(
            metas,
            n_jobs=n_jobs,
            bdsf_kw=bdsf_kw,
            gaul_columns=gaul_columns,
            upsample_factor=upsample_factor,
            **process_kw,
        )
    }
    return [by_path[meta.path] for meta in metas]
