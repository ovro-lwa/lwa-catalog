"""Per-image source detection with PyBDSF."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import warnings
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from lwa_catalog.constants import GAUL_DETECTION_COLUMNS
from lwa_catalog.coords import normalize_ra_columns
from lwa_catalog.gaul import cast_gaul_string_columns
from lwa_catalog.create.discover import FitsMetadata
from lwa_catalog.io import write_table
from lwa_catalog.schemas import sources_schema

DEFAULT_BDSF_KW: dict[str, Any] = {
    "thresh": "hard",
    "thresh_isl": 7.0,
    "thresh_pix": 4.0,
    "atrous_do": False,
    "psf_vary_do": False,
    "quiet": True,
    "ncores": 1,
}

# Force a constant background so PyBDSF skips sliding-box ``bstat`` maps.
_CONSTANT_RMS_FALLBACK_KW: dict[str, Any] = {
    "adaptive_rms_box": False,
    "rms_map": False,
    "mean_map": "const",
    "rms_box": None,
    "rms_box_bright": None,
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


def _detect_and_write_sources(
    meta: FitsMetadata,
    catalog_path: Path,
    bdsf_kw: Mapping[str, Any] | None,
    gaul_columns: Sequence[str],
    upsample_factor: int,
    process_kw: Mapping[str, Any],
) -> tuple[str, int]:
    """Run detection and write Parquet in the worker; return path and row count."""
    catalog = detect_sources(
        meta,
        bdsf_kw=bdsf_kw,
        gaul_columns=gaul_columns,
        upsample_factor=upsample_factor,
        **process_kw,
    )
    path = Path(catalog_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_table(catalog, path, schema=sources_schema())
    return str(path.resolve()), len(catalog)


def _detect_and_write_packed(task: tuple[Any, ...]) -> tuple[str, int]:
    return _detect_and_write_sources(*task)


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
    gaul_columns: Sequence[str] = GAUL_DETECTION_COLUMNS,
) -> pd.DataFrame:
    """Return an empty per-image catalog with expected columns."""
    columns = list(gaul_columns) + ["lst_hour", "band", "source_file", "BMAJ", "BMIN", "BPA"]
    if meta.time_key is not None:
        columns.append("time_key")
    return pd.DataFrame(columns=columns)


def _is_recoverable_rms_error(exc: BaseException) -> bool:
    """True for PyBDSF sliding-box RMS failures on constant/blank patches."""
    msg = str(exc).lower()
    if isinstance(exc, FloatingPointError):
        return "divide" in msg and "zero" in msg
    if isinstance(exc, RuntimeError):
        return "unphysical rms" in msg or "clipped rms appears to be zero" in msg
    return False


def _with_constant_rms_fallback(kw: Mapping[str, Any]) -> dict[str, Any]:
    """Return process_image kwargs that skip 2-D mean/rms map calculation."""
    out = dict(kw)
    out.update(_CONSTANT_RMS_FALLBACK_KW)
    return out


def _isolate_in_memory_outdir(
    kw: Mapping[str, Any],
    *,
    lst_hour: str,
    band: str,
) -> dict[str, Any]:
    """Nest ``outdir`` under ``{outdir}/{lst}_{band}`` for in-memory PyBDSF runs.

    ``bdsf.process_image`` on an HDU always uses the placeholder filename
    ``in_memory.fits``, so parallel detections that share ``outdir`` would
    clobber logs and ``savefits_rmsim`` products. Isolate each image under a
    unique subdirectory when an output directory or RMS-map save is requested.
    """
    out = dict(kw)
    base = out.get("outdir")
    if base is None and not out.get("savefits_rmsim"):
        return out
    if base is None:
        msg = (
            "savefits_rmsim=True requires outdir to be set "
            "(otherwise RMS maps are written under the process cwd)"
        )
        raise ValueError(msg)
    out["outdir"] = str(Path(base) / f"{lst_hour}_{band}")
    return out


def run_pybdsf_on_hdu(
    hdu: fits.PrimaryHDU,
    *,
    bdsf_kw: Mapping[str, Any] | None = None,
    **process_kw: Any,
) -> Table | None:
    """Run PyBDSF on an in-memory HDU and return the Gaussian catalog table.

    If sliding-box RMS estimation fails on a constant/blank patch
    (``FloatingPointError: divide by zero`` or ``unphysical rms``), retries once
    with a constant background (``rms_map=False``, ``mean_map='const'``).
    """
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

    try:
        img = bdsf.process_image(hdu, **kw)
    except (FloatingPointError, RuntimeError) as exc:
        if not _is_recoverable_rms_error(exc):
            raise
        fallback_kw = _with_constant_rms_fallback(kw)
        warnings.warn(
            f"PyBDSF RMS map failed ({exc}); retrying with constant rms/mean maps",
            UserWarning,
            stacklevel=2,
        )
        img = bdsf.process_image(hdu, **fallback_kw)

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
    gaul_columns: Sequence[str] = GAUL_DETECTION_COLUMNS,
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

    Notes
    -----
    When ``outdir`` and/or ``savefits_rmsim`` are set, PyBDSF outputs (RMS maps,
    logs) are written under ``{outdir}/{lst_hour}_{band}/`` so parallel
    in-memory runs do not overwrite each other. RMS maps land at
    ``{outdir}/{lst_hour}_{band}/in_memory_pybdsf/background/in_memory.pybdsf.rmsd_I.fits``.
    """
    hdu = prepare_hdu(meta.path)
    bmaj, bmin, bpa = beam_from_header(hdu.header)
    if upsample_factor != 1:
        hdu = upsample_hdu(hdu, factor=upsample_factor)
    merged_kw = dict(bdsf_kw) if bdsf_kw is not None else {}
    merged_kw.update(process_kw)
    merged_kw = _isolate_in_memory_outdir(
        merged_kw, lst_hour=meta.lst_hour, band=meta.band
    )
    table = run_pybdsf_on_hdu(hdu, bdsf_kw=merged_kw)
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
    df = cast_gaul_string_columns(df[keep].copy())
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
    catalog_paths: Sequence[Path],
    *,
    n_jobs: int | None = None,
    bdsf_kw: Mapping[str, Any] | None = None,
    gaul_columns: Sequence[str] = GAUL_DETECTION_COLUMNS,
    upsample_factor: int = 1,
    **process_kw: Any,
) -> Iterator[tuple[FitsMetadata, Path, int]]:
    """Yield ``(meta, catalog_path, n_sources)`` as each image finishes.

    Each worker runs PyBDSF and writes its Parquet file before returning only
    the output path and source count (avoids shipping large DataFrames through
    a forked process pool from Jupyter). Completion order is not ``metas``
    order when ``n_jobs > 1``.

    Parameters
    ----------
    metas
        One ``FitsMetadata`` per image to process.
    catalog_paths
        Parquet output path per image (same length and order as ``metas``).
    n_jobs
        Number of worker processes. ``None`` uses all CPUs available to this
        process. ``1`` runs serially in the current process.
    bdsf_kw, gaul_columns, upsample_factor, **process_kw
        Forwarded to :func:`detect_sources` for every image.
    """
    if not metas:
        return
    if len(catalog_paths) != len(metas):
        msg = (
            f"catalog_paths length ({len(catalog_paths)}) must match "
            f"metas length ({len(metas)})"
        )
        raise ValueError(msg)

    if n_jobs is None:
        n_jobs = _available_cpu_count()
    n_jobs = max(1, min(n_jobs, len(metas)))

    gaul_cols = tuple(gaul_columns)
    proc_kw = dict(process_kw)
    tasks = [
        (meta, Path(path), bdsf_kw, gaul_cols, upsample_factor, proc_kw)
        for meta, path in zip(metas, catalog_paths, strict=True)
    ]

    if n_jobs == 1:
        for meta, task in zip(metas, tasks, strict=True):
            out_path, n_sources = _detect_and_write_packed(task)
            yield meta, Path(out_path), n_sources
        return

    # Load PyBDSF in the parent before forking so workers inherit the module
    # without re-running PyBDSF's ``set_start_method('fork')`` import hook.
    _import_pybdsf_safely()

    with _fork_process_pool(n_jobs) as pool:
        future_to_index = {
            pool.submit(_detect_and_write_packed, task): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            out_path, n_sources = future.result()
            yield metas[i], Path(out_path), n_sources


def detect_sources_many(
    metas: Sequence[FitsMetadata],
    catalog_paths: Sequence[Path],
    *,
    n_jobs: int | None = None,
    bdsf_kw: Mapping[str, Any] | None = None,
    gaul_columns: Sequence[str] = GAUL_DETECTION_COLUMNS,
    upsample_factor: int = 1,
    **process_kw: Any,
) -> list[pd.DataFrame]:
    """Detect sources in many FITS images, parallelizing across images.

    Each image runs PyBDSF in its own worker process, writes Parquet under
    *catalog_paths*, and the caller receives catalogs read back in ``metas``
    order. Use ``ncores=1`` in ``bdsf_kw`` (the library default) so workers
    do not compete for CPUs.

    Parameters
    ----------
    metas
        One ``FitsMetadata`` per image to process.
    catalog_paths
        Parquet output path per image (same length and order as ``metas``).
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
    from lwa_catalog.io import read_table

    by_path: dict[Path, pd.DataFrame] = {}
    for meta, out_path, _n_sources in iter_detect_sources(
        metas,
        catalog_paths,
        n_jobs=n_jobs,
        bdsf_kw=bdsf_kw,
        gaul_columns=gaul_columns,
        upsample_factor=upsample_factor,
        **process_kw,
    ):
        df = read_table(out_path, as_pandas=True)
        assert isinstance(df, pd.DataFrame)
        by_path[meta.path] = df
    return [by_path[meta.path] for meta in metas]
