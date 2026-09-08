"""OVRO-LWA catalog creation and analysis."""

from __future__ import annotations

from lwa_catalog.io import (
    empty_sources_table,
    migrate_output_dir,
    read_metacatalog,
    read_sources_catalog,
    resolve_metacatalog_path,
    rewrite_output_dir_gaul_columns,
    write_metacatalog,
    write_sources_catalog,
)
from lwa_catalog.catalog_index import (
    DEFAULT_DISPLAY_COLUMNS,
    apply_quality_mask,
    apply_radio_qa_filter,
    classify_catalog,
    discover_catalog_dirs,
    inventory_catalogs,
    is_metacatalog_parquet,
    load_display_column_prefs,
    load_metacatalog_frame,
    save_display_column_prefs,
)
from lwa_catalog.paths import CatalogLayout

try:
    from lwa_catalog._version import __version__
except ImportError:  # pragma: no cover - editable / missing hatch-vcs build
    __version__ = "0.1.0.dev0"

__all__ = [
    "CatalogLayout",
    "DEFAULT_DISPLAY_COLUMNS",
    "__version__",
    "apply_quality_mask",
    "apply_radio_qa_filter",
    "classify_catalog",
    "discover_catalog_dirs",
    "empty_sources_table",
    "inventory_catalogs",
    "is_metacatalog_parquet",
    "load_display_column_prefs",
    "load_metacatalog_frame",
    "migrate_output_dir",
    "read_metacatalog",
    "read_sources_catalog",
    "resolve_metacatalog_path",
    "rewrite_output_dir_gaul_columns",
    "save_display_column_prefs",
    "write_metacatalog",
    "write_sources_catalog",
]
