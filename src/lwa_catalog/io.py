"""Read and write catalog tables (CSV / FITS).

Stub module. Intended destination for loaders/writers currently embedded in
``notebooks/ovro_lwa_metacatalog.ipynb`` and ``notebooks/metacatalog_sky_view.ipynb``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_catalog(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a catalog table from CSV or FITS.

    Parameters
    ----------
    path
        Path to ``.csv`` or ``.fits`` catalog.
    **kwargs
        Forwarded to :func:`pandas.read_csv` or :func:`astropy.table.Table.read`.

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, **kwargs)
    if path.suffix.lower() in {".fits", ".fit"}:
        from astropy.table import Table

        return Table.read(path, **kwargs).to_pandas()
    msg = f"Unsupported catalog format: {path.suffix!r} ({path})"
    raise ValueError(msg)


def write_catalog(catalog: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Write a catalog table to CSV or FITS.

    Parameters
    ----------
    catalog
        Catalog rows.
    path
        Destination ``.csv`` or ``.fits`` path.
    **kwargs
        Forwarded to :meth:`pandas.DataFrame.to_csv` or
        :meth:`astropy.table.Table.write`.

    Returns
    -------
    pathlib.Path
        Resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        catalog.to_csv(path, index=False, **kwargs)
        return path.resolve()
    if path.suffix.lower() in {".fits", ".fit"}:
        from astropy.table import Table

        Table.from_pandas(catalog).write(path, overwrite=True, **kwargs)
        return path.resolve()
    msg = f"Unsupported catalog format: {path.suffix!r} ({path})"
    raise ValueError(msg)


def validate_metacatalog(catalog: pd.DataFrame, *, required: set[str] | None = None) -> None:
    """Check that a metacatalog DataFrame has expected columns.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    cols = required or {
        "RA",
        "DEC",
        "Peak_flux",
        "origin_band",
        "bands_present",
    }
    missing = sorted(cols - set(catalog.columns))
    if missing:
        msg = f"metacatalog missing columns: {missing}"
        raise ValueError(msg)
