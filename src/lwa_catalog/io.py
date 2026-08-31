"""Read and write catalog tables as Apache Parquet (PyArrow).

Catalog persistence uses ``.parquet`` files. Image FITS remains separate for
beam keywords and overlays. Legacy CSV/FITS catalogs can be imported once via
``import_legacy_*`` / ``migrate_output_dir``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits

from lwa_catalog.constants import (
    COLOR_BANDS,
    DROPPED_GAUL_COLUMNS,
    METACATALOG_REQUIRED_COLUMNS,
    SOURCES_REQUIRED_COLUMNS,
)
from lwa_catalog.paths import CatalogLayout
from lwa_catalog.schemas import (
    lst_merged_schema,
    metacatalog_schema,
    sources_schema,
    table_from_dataframe,
)

_PARQUET_SUFFIXES = {".parquet", ".pq"}


def _ensure_parquet_path(path: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix not in _PARQUET_SUFFIXES:
        msg = (
            f"Catalog I/O requires a Parquet path (got suffix {path.suffix!r} "
            f"for {path}). Use .parquet"
        )
        raise ValueError(msg)
    return path


def _as_table(
    data: pd.DataFrame | pa.Table,
    *,
    schema: pa.Schema | None = None,
    include_extras: bool = True,
) -> pa.Table:
    if isinstance(data, pa.Table):
        if schema is None:
            return data
        # Re-align via pandas so missing schema fields become nulls.
        return table_from_dataframe(
            data.to_pandas(),
            schema,
            include_extras=include_extras,
        )
    if schema is None:
        return pa.Table.from_pandas(data, preserve_index=False)
    return table_from_dataframe(data, schema, include_extras=include_extras)


@overload
def read_table(
    path: str | Path,
    *,
    as_pandas: Literal[True] = True,
    **kwargs: Any,
) -> pd.DataFrame: ...


@overload
def read_table(
    path: str | Path,
    *,
    as_pandas: Literal[False],
    **kwargs: Any,
) -> pa.Table: ...


def read_table(
    path: str | Path,
    *,
    as_pandas: bool = True,
    **kwargs: Any,
) -> pd.DataFrame | pa.Table:
    """Load a catalog Parquet file."""
    path = _ensure_parquet_path(Path(path))
    table = pq.read_table(path, **kwargs)
    if as_pandas:
        return table.to_pandas()
    return table


def write_table(
    data: pd.DataFrame | pa.Table,
    path: str | Path,
    *,
    schema: pa.Schema | None = None,
    include_extras: bool = True,
    **kwargs: Any,
) -> Path:
    """Write a catalog table to Parquet."""
    path = _ensure_parquet_path(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _as_table(data, schema=schema, include_extras=include_extras)
    pq.write_table(table, path, **kwargs)
    return path.resolve()


def read_beam_from_fits(path: str | Path) -> tuple[float, float, float]:
    """Return ``(BMAJ, BMIN, BPA)`` degrees from a FITS image primary header."""
    path = Path(path)
    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header
        if "BMAJ" not in header or "BMIN" not in header:
            msg = f"FITS header missing BMAJ/BMIN beam keywords: {path}"
            raise ValueError(msg)
        return (
            float(header["BMAJ"]),
            float(header["BMIN"]),
            float(header.get("BPA", 0.0)),
        )


def ensure_beam_columns(
    catalog: pd.DataFrame | pa.Table,
    *,
    fits_path: str | Path | None = None,
) -> pd.DataFrame | pa.Table:
    """Ensure ``BMAJ``/``BMIN``/``BPA`` columns exist, optionally from a FITS image.

    If beam columns are already present with any non-null ``BMAJ``, *catalog* is
    returned unchanged. If they are missing/empty and *fits_path* is set, values
    are copied from the FITS header. If *fits_path* is ``None``, *catalog* is
    returned unchanged.
    """
    as_table = isinstance(catalog, pa.Table)
    df = catalog.to_pandas() if as_table else catalog
    if "BMAJ" in df.columns and df["BMAJ"].notna().any():
        return catalog
    if fits_path is None:
        return catalog
    bmaj, bmin, bpa = read_beam_from_fits(fits_path)
    out = df.copy()
    out["BMAJ"] = bmaj
    out["BMIN"] = bmin
    out["BPA"] = bpa
    if as_table:
        return pa.Table.from_pandas(out, preserve_index=False)
    return out


def validate_metacatalog(
    catalog: pd.DataFrame | pa.Table,
    *,
    required: set[str] | frozenset[str] | None = None,
) -> None:
    """Check that a metacatalog has expected columns."""
    cols = set(required) if required is not None else set(METACATALOG_REQUIRED_COLUMNS)
    if isinstance(catalog, pa.Table):
        names = set(catalog.column_names)
    else:
        names = set(catalog.columns)
    missing = sorted(cols - names)
    if missing:
        msg = f"metacatalog missing columns: {missing}"
        raise ValueError(msg)


def write_sources_catalog(
    data: pd.DataFrame | pa.Table,
    layout: CatalogLayout,
    lst_hour: str,
    band: str,
    **kwargs: Any,
) -> Path:
    """Write a per-image sources catalog under *layout*."""
    return write_table(
        data,
        layout.sources(lst_hour, band),
        schema=sources_schema(),
        **kwargs,
    )


def read_sources_catalog(
    layout: CatalogLayout,
    lst_hour: str,
    band: str,
    *,
    fits_path: str | Path | None = None,
    as_pandas: bool = True,
) -> pd.DataFrame | pa.Table:
    """Read a per-image sources catalog, optionally backfilling beam columns."""
    path = layout.sources(lst_hour, band)
    if not path.is_file():
        raise FileNotFoundError(f"Missing sources catalog: {path}")
    df = read_table(path, as_pandas=True)
    df = ensure_beam_columns(df, fits_path=fits_path)
    assert isinstance(df, pd.DataFrame)
    if as_pandas:
        return df
    return pa.Table.from_pandas(df, preserve_index=False)


def write_lst_merged(
    data: pd.DataFrame | pa.Table,
    layout: CatalogLayout,
    band: str,
    **kwargs: Any,
) -> Path:
    """Write an LST-merged per-band catalog under *layout*."""
    return write_table(
        data,
        layout.lst_merged(band),
        schema=lst_merged_schema(),
        **kwargs,
    )


def read_lst_merged(
    layout: CatalogLayout,
    band: str,
    *,
    as_pandas: bool = True,
) -> pd.DataFrame | pa.Table:
    """Read one LST-merged per-band catalog."""
    path = layout.lst_merged(band)
    if not path.is_file():
        raise FileNotFoundError(f"Missing LST-merged catalog: {path}")
    return read_table(path, as_pandas=as_pandas)


def read_all_lst_merged(
    layout: CatalogLayout,
    bands: Iterable[str] = COLOR_BANDS,
    *,
    as_pandas: bool = True,
) -> dict[str, pd.DataFrame] | dict[str, pa.Table]:
    """Read LST-merged catalogs for each band in *bands*."""
    out: dict[str, pd.DataFrame] | dict[str, pa.Table] = {}
    for band in bands:
        out[band] = read_lst_merged(layout, band, as_pandas=as_pandas)  # type: ignore[assignment]
    return out


def write_metacatalog(
    data: pd.DataFrame | pa.Table,
    layout: CatalogLayout,
    **kwargs: Any,
) -> Path:
    """Write the global metacatalog Parquet file (validates required columns)."""
    validate_metacatalog(data)
    return write_table(
        data,
        layout.metacatalog(),
        schema=metacatalog_schema(),
        **kwargs,
    )


def read_metacatalog(
    layout_or_path: CatalogLayout | str | Path,
    *,
    as_pandas: bool = True,
    validate: bool = True,
) -> pd.DataFrame | pa.Table:
    """Read the global metacatalog from a layout root or explicit Parquet path."""
    if isinstance(layout_or_path, CatalogLayout):
        path = layout_or_path.metacatalog()
    else:
        path = Path(layout_or_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing metacatalog: {path}")
    catalog = read_table(path, as_pandas=as_pandas)
    if validate:
        validate_metacatalog(catalog)
    return catalog


def sources_cache_complete(
    layout: CatalogLayout,
    slots: Iterable[tuple[str, str]],
) -> bool:
    """True when every ``(lst_hour, band)`` slot has a sources Parquet file."""
    return all(layout.sources(lst, band).is_file() for lst, band in slots)


def lst_merged_cache_complete(
    layout: CatalogLayout,
    bands: Iterable[str] = COLOR_BANDS,
) -> bool:
    """True when every band has an LST-merged Parquet file."""
    return all(layout.lst_merged(band).is_file() for band in bands)


def _layout_catalog_parquet_files(layout: CatalogLayout) -> list[tuple[Path, pa.Schema]]:
    """Return ``(path, schema)`` for sources / LST-merged / metacatalog Parquet files."""
    root = layout.root
    items: list[tuple[Path, pa.Schema]] = []
    if not root.is_dir():
        return items
    for path in sorted(root.glob("sources_*.parquet")):
        items.append((path, sources_schema()))
    for path in sorted(root.glob("metacatalog_lst_*.parquet")):
        items.append((path, lst_merged_schema()))
    meta = layout.metacatalog()
    if meta.is_file():
        items.append((meta, metacatalog_schema()))
    return items


def rewrite_output_dir_gaul_columns(
    layout: CatalogLayout,
    *,
    drop: Iterable[str] = DROPPED_GAUL_COLUMNS,
    dry_run: bool = False,
) -> list[Path]:
    """Drop retired GAUL columns from catalog Parquet files under *layout.root*.

    Rewrites ``sources_*.parquet``, ``metacatalog_lst_*.parquet``, and
    ``metacatalog.parquet`` so on-disk tables match the current detection
    default (:data:`lwa_catalog.constants.GAUL_COLUMNS`). Files that already
    lack *drop* columns are left untouched.

    Returns
    -------
    list of Path
        Files that were rewritten (or would be, when *dry_run* is true).
    """
    drop_set = frozenset(drop)
    rewritten: list[Path] = []
    for path, schema in _layout_catalog_parquet_files(layout):
        table = pq.read_table(path)
        present = [name for name in table.column_names if name in drop_set]
        if not present:
            continue
        rewritten.append(path)
        if dry_run:
            continue
        table = table.drop_columns(present)
        write_table(table, path, schema=schema, include_extras=True)
    return rewritten


def empty_sources_table(
    lst_hour: str,
    band: str,
    *,
    source_file: str = "",
    bmaj: float | None = None,
    bmin: float | None = None,
    bpa: float | None = None,
    time_key: str | None = None,
) -> pa.Table:
    """Return a 0-row sources table with the full :func:`sources_schema`.

    Provenance arguments are stored in Arrow schema metadata (and are available
    for callers that later append rows). Beam values are recorded in metadata
    when provided; columns remain null for the empty table.
    """
    schema = sources_schema()
    meta: dict[bytes, bytes] = {
        b"lst_hour": str(lst_hour).encode(),
        b"band": str(band).encode(),
        b"source_file": str(source_file).encode(),
    }
    if time_key is not None:
        meta[b"time_key"] = str(time_key).encode()
    if bmaj is not None:
        meta[b"BMAJ"] = repr(float(bmaj)).encode()
    if bmin is not None:
        meta[b"BMIN"] = repr(float(bmin)).encode()
    if bpa is not None:
        meta[b"BPA"] = repr(float(bpa)).encode()
    schema = schema.with_metadata(meta)
    arrays = [pa.array([], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def validate_sources_catalog(
    catalog: pd.DataFrame | pa.Table,
    *,
    required: set[str] | frozenset[str] | None = None,
) -> None:
    """Check that a sources catalog has required columns (empty frames allowed)."""
    cols = set(required) if required is not None else set(SOURCES_REQUIRED_COLUMNS)
    if isinstance(catalog, pa.Table):
        names = set(catalog.column_names)
    else:
        names = set(catalog.columns)
    missing = sorted(cols - names)
    if missing:
        msg = f"sources catalog missing columns: {missing}"
        raise ValueError(msg)


def import_legacy_sources_csv(path: str | Path) -> pa.Table:
    """Load a notebook-era ``sources_*.csv`` into an Arrow table."""
    path = Path(path)
    df = pd.read_csv(path)
    table = table_from_dataframe(df, sources_schema(), include_extras=True)
    validate_sources_catalog(table)
    return table


def import_legacy_metacatalog(
    csv_path: str | Path | None = None,
    fits_path: str | Path | None = None,
) -> pa.Table:
    """Load a notebook-era global metacatalog from CSV and/or FITS.

    When both paths exist, **CSV is preferred**. At least one readable path is
    required.
    """
    csv_p = Path(csv_path) if csv_path is not None else None
    fits_p = Path(fits_path) if fits_path is not None else None

    df: pd.DataFrame | None = None
    if csv_p is not None and csv_p.is_file():
        df = pd.read_csv(csv_p)
    elif fits_p is not None and fits_p.is_file():
        from astropy.table import Table

        df = Table.read(fits_p).to_pandas()
    elif csv_p is not None and fits_p is not None:
        msg = f"Neither metacatalog file exists: {csv_p}, {fits_p}"
        raise FileNotFoundError(msg)
    elif csv_p is not None:
        raise FileNotFoundError(f"Missing metacatalog CSV: {csv_p}")
    elif fits_p is not None:
        raise FileNotFoundError(f"Missing metacatalog FITS: {fits_p}")
    else:
        raise ValueError("Provide csv_path and/or fits_path")

    assert df is not None
    validate_metacatalog(df)
    return table_from_dataframe(df, metacatalog_schema(), include_extras=True)


_SOURCES_CSV_RE = re.compile(
    r"^sources_(?P<lst>\d{2}h)_(?P<band>Full|Blue|Green|Red)\.csv$",
    re.IGNORECASE,
)
_LST_MERGED_CSV_RE = re.compile(
    r"^metacatalog_lst_(?P<band>Full|Blue|Green|Red)\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MigrationItem:
    """One legacy file → Parquet destination planned by :func:`migrate_output_dir`."""

    kind: str
    source: Path
    destination: Path
    lst_hour: str | None = None
    band: str | None = None


def _discover_legacy_items(layout: CatalogLayout) -> list[MigrationItem]:
    root = layout.root
    items: list[MigrationItem] = []
    if not root.is_dir():
        return items

    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        m = _SOURCES_CSV_RE.match(name)
        if m:
            lst = m.group("lst")
            # Normalize hour casing: 01h
            lst = f"{int(lst[:-1]):02d}h"
            band = m.group("band").title()
            items.append(
                MigrationItem(
                    kind="sources",
                    source=path,
                    destination=layout.sources(lst, band),
                    lst_hour=lst,
                    band=band,
                )
            )
            continue
        m = _LST_MERGED_CSV_RE.match(name)
        if m:
            band = m.group("band").title()
            items.append(
                MigrationItem(
                    kind="lst_merged",
                    source=path,
                    destination=layout.lst_merged(band),
                    band=band,
                )
            )
            continue

    meta_csv = root / "metacatalog.csv"
    meta_fits = root / "metacatalog.fits"
    if meta_csv.is_file() or meta_fits.is_file():
        # Prefer CSV as source when both exist (matches import_legacy_metacatalog).
        source = meta_csv if meta_csv.is_file() else meta_fits
        items.append(
            MigrationItem(
                kind="metacatalog",
                source=source,
                destination=layout.metacatalog(),
            )
        )
    return items


def migrate_output_dir(
    layout: CatalogLayout,
    *,
    fits_paths: Mapping[tuple[str, str], str | Path] | None = None,
    dry_run: bool = False,
    delete_legacy: bool = False,
    overwrite: bool = True,
) -> list[MigrationItem]:
    """Convert notebook-era CSV/FITS catalogs under *layout.root* to Parquet.

    Parameters
    ----------
    layout
        Target catalog directory (also the scan root).
    fits_paths
        Optional ``(lst_hour, band) → image FITS`` map used to backfill beam
        columns when migrating sources CSVs.
    dry_run
        If ``True``, only discover and return planned items (no writes).
    delete_legacy
        If ``True``, remove the legacy source file after a successful write
        (default keeps legacy files).
    overwrite
        If ``False``, skip destinations that already exist.

    Returns
    -------
    list of MigrationItem
        Planned or completed migrations.
    """
    items = _discover_legacy_items(layout)
    if dry_run:
        return items

    fits_map = {key: Path(val) for key, val in (fits_paths or {}).items()}

    for item in items:
        if item.destination.is_file() and not overwrite:
            continue
        if item.kind == "sources":
            table = import_legacy_sources_csv(item.source)
            assert item.lst_hour is not None and item.band is not None
            fits_path = fits_map.get((item.lst_hour, item.band))
            table = ensure_beam_columns(table, fits_path=fits_path)
            assert isinstance(table, pa.Table)
            write_sources_catalog(table, layout, item.lst_hour, item.band)
        elif item.kind == "lst_merged":
            df = pd.read_csv(item.source)
            assert item.band is not None
            write_lst_merged(df, layout, item.band)
        elif item.kind == "metacatalog":
            csv_path = item.source if item.source.suffix.lower() == ".csv" else None
            fits_path = item.source if item.source.suffix.lower() in {".fits", ".fit"} else None
            # If CSV was chosen but FITS also exists, import_legacy still prefers CSV.
            if csv_path is None:
                # source is fits-only
                table = import_legacy_metacatalog(fits_path=fits_path)
            else:
                sibling_fits = layout.root / "metacatalog.fits"
                table = import_legacy_metacatalog(
                    csv_path=csv_path,
                    fits_path=sibling_fits if sibling_fits.is_file() else None,
                )
            write_metacatalog(table, layout)
        else:  # pragma: no cover
            msg = f"Unknown migration kind: {item.kind}"
            raise ValueError(msg)

        if delete_legacy:
            item.source.unlink(missing_ok=True)
            # When metacatalog CSV was migrated, optionally remove FITS twin.
            if item.kind == "metacatalog":
                twin = layout.root / "metacatalog.fits"
                if twin.is_file() and twin != item.source:
                    twin.unlink(missing_ok=True)

    return items
