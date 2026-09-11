"""Discover and classify on-disk Parquet catalog trees.

Helpers shared by query / QA notebooks: inventory tables, metacatalog path
classification, catalog-directory discovery, row filters, and display-column
prefs persistence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lwa_catalog.analyze.reliability import filter_by_quality_mask
from lwa_catalog.io import read_metacatalog
from lwa_catalog.paths import CatalogLayout

_SOURCES_RE = re.compile(
    r"^sources_(?P<lst>\d+h)_(?P<band>[^.]+)\.parquet$",
    re.IGNORECASE,
)
_LST_RE = re.compile(
    r"^metacatalog_lst_(?P<band>[^.]+)\.parquet$",
    re.IGNORECASE,
)
_METACATALOG_RE = re.compile(
    r"^metacatalog(?:_[^/]+)?\.parquet$",
    re.IGNORECASE,
)

DEFAULT_DISPLAY_COLUMNS: tuple[str, ...] = (
    "meta_id",
    "Source_id",
    "RA",
    "DEC",
    "Peak_flux",
    "Total_flux",
    "band",
    "lst_hour",
    "bands_present",
    "origin_band",
    "n_lst_contributions",
    "source_file",
)

DEFAULT_DISPLAY_COLUMN_PREFS_PATH: Path = (
    Path.home() / ".config" / "lwa-catalog" / "metacatalog_query_columns.json"
)


def discover_catalog_dirs(root: Path, *, pattern: str = "metacatalog*") -> list[Path]:
    """Discover catalog trees under *root* that contain at least one Parquet file."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p.resolve()
        for p in root.glob(pattern)
        if p.is_dir() and any(p.glob("*.parquet"))
    )


def is_metacatalog_parquet(name: str) -> bool:
    """True for ``metacatalog.parquet`` and derived ``metacatalog_*.parquet`` tables.

    Excludes LST-merged ``metacatalog_lst_*.parquet`` products.
    """
    base = Path(str(name)).name
    return bool(_METACATALOG_RE.match(base)) and not base.lower().startswith(
        "metacatalog_lst_"
    )


def classify_catalog(path: Path | str) -> dict[str, str | None]:
    """Return kind / LST / band metadata for a catalog Parquet path."""
    name = Path(path).name
    m = _LST_RE.match(name)
    if m:
        return {"kind": "lst_merged", "lst_hour": None, "band": m.group("band")}
    if is_metacatalog_parquet(name):
        return {"kind": "metacatalog", "lst_hour": None, "band": None}
    m = _SOURCES_RE.match(name)
    if m:
        return {
            "kind": "sources",
            "lst_hour": m.group("lst"),
            "band": m.group("band"),
        }
    return {"kind": "other", "lst_hour": None, "band": None}


def inventory_catalogs(catalog_dir: Path | str) -> pd.DataFrame:
    """List Parquet catalogs under *catalog_dir* with size and row counts."""
    catalog_dir = Path(catalog_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(catalog_dir.glob("*.parquet")):
        meta = classify_catalog(path)
        try:
            import pyarrow.parquet as pq

            n_rows = pq.ParquetFile(path).metadata.num_rows
        except Exception:
            n_rows = None
        rows.append(
            {
                "file": path.name,
                "kind": meta["kind"],
                "lst_hour": meta["lst_hour"],
                "band": meta["band"],
                "n_rows": n_rows,
                "size_mb": round(path.stat().st_size / 1e6, 2),
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def apply_quality_mask(
    df: pd.DataFrame,
    quality_mask: int | None,
) -> pd.DataFrame:
    """Apply *quality_mask* when not ``None``; otherwise return *df* unchanged."""
    if quality_mask is None:
        return df
    return filter_by_quality_mask(df, quality_mask)


def apply_radio_qa_filter(
    df: pd.DataFrame,
    row_filter: pd.Series | pd.Index | np.ndarray | list | str | None,
) -> pd.DataFrame:
    """Subset *df* with a boolean Series, Index, label array, or pandas query string.

    ``None`` returns a copy of *df*. Used by notebooks' ``RADIO_QA_FILTER`` config.
    """
    if row_filter is None:
        return df.copy()
    if isinstance(row_filter, str):
        return df.query(row_filter).copy()
    if isinstance(row_filter, pd.Series):
        if not pd.api.types.is_bool_dtype(row_filter):
            raise TypeError("row filter Series must be boolean")
        aligned = row_filter.reindex(df.index)
        if aligned.isna().any():
            missing = int(aligned.isna().sum())
            raise ValueError(
                f"row filter boolean Series has {missing} index labels "
                "not present in the catalog"
            )
        return df.loc[aligned.astype(bool)].copy()
    if isinstance(row_filter, pd.Index):
        missing = row_filter.difference(df.index)
        if len(missing):
            raise ValueError(
                f"row filter Index has {len(missing)} labels not in the catalog"
            )
        return df.loc[row_filter].copy()
    labels = pd.Index(row_filter)
    missing = labels.difference(df.index)
    if len(missing):
        raise ValueError(
            f"row filter labels have {len(missing)} entries not in the catalog"
        )
    return df.loc[labels].copy()


def load_metacatalog_frame(
    layout: CatalogLayout,
    *,
    prefer_spectral: bool = True,
    quality_mask: int | None = None,
) -> pd.DataFrame:
    """Read the preferred metacatalog Parquet, optionally filtering by *quality_mask*."""
    return read_metacatalog(
        layout,
        prefer_spectral=prefer_spectral,
        quality_mask=quality_mask,
    )


def load_display_column_prefs(
    path: Path | None = None,
    *,
    default_columns: tuple[str, ...] | list[str] = DEFAULT_DISPLAY_COLUMNS,
) -> list[str]:
    """Load saved display-column preference, else fall back to *default_columns*."""
    prefs_path = Path(path) if path is not None else DEFAULT_DISPLAY_COLUMN_PREFS_PATH
    try:
        raw = json.loads(prefs_path.read_text())
        cols = [str(c) for c in raw.get("columns", []) if str(c).strip()]
        if cols:
            return cols
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        pass
    return list(default_columns)


def save_display_column_prefs(
    columns: list[str],
    path: Path | None = None,
) -> Path:
    """Persist display-column preference for later notebook sessions."""
    prefs_path = Path(path) if path is not None else DEFAULT_DISPLAY_COLUMN_PREFS_PATH
    prefs_path.parent.mkdir(parents=True, exist_ok=True)
    prefs_path.write_text(json.dumps({"columns": list(columns)}, indent=2) + "\n")
    return prefs_path
