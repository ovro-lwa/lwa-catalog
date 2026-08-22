"""Build per-image catalogs and fuse them into a metacatalog."""

from __future__ import annotations

from lwa_catalog.create.detect import detect_sources
from lwa_catalog.create.discover import (
    FitsMetadata,
    discover_fits_files,
    discovered_slots,
    lst_hours_from_discovery,
    parse_fits_metadata,
    resolve_fits_slot,
)
from lwa_catalog.create.merge import (
    add_spectral_indices,
    associate_band_into_metacatalog,
    associate_catalogs,
    build_global_metacatalog,
    merge_lst_metacatalog,
    pick_highest_elevation_row,
)

__all__ = [
    "FitsMetadata",
    "add_spectral_indices",
    "associate_band_into_metacatalog",
    "associate_catalogs",
    "build_global_metacatalog",
    "detect_sources",
    "discover_fits_files",
    "discovered_slots",
    "lst_hours_from_discovery",
    "merge_lst_metacatalog",
    "parse_fits_metadata",
    "pick_highest_elevation_row",
    "resolve_fits_slot",
]
