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
from lwa_catalog.paths import CatalogLayout

try:
    from lwa_catalog._version import __version__
except ImportError:  # pragma: no cover - editable / missing hatch-vcs build
    __version__ = "0.1.0.dev0"

__all__ = [
    "CatalogLayout",
    "__version__",
    "empty_sources_table",
    "migrate_output_dir",
    "read_metacatalog",
    "read_sources_catalog",
    "resolve_metacatalog_path",
    "rewrite_output_dir_gaul_columns",
    "write_metacatalog",
    "write_sources_catalog",
]
