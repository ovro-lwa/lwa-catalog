"""Build per-image catalogs and fuse them into a metacatalog."""

from __future__ import annotations

from lwa_catalog.create.detect import (
    blank_below_elevation,
    detect_sources,
    detect_sources_many,
    iter_detect_sources,
)
from lwa_catalog.create.discover import (
    FitsMetadata,
    discover_fits_files,
    discovered_slots,
    lst_hours_from_discovery,
    parse_fits_metadata,
    resolve_fits_slot,
)
from lwa_catalog.create.healpix_detect import (
    attach_beam_and_freq,
    detect_sources_on_healpix_tiles,
    median_beam_from_paths,
    restfreq_hz_from_header,
)
from lwa_catalog.create.merge import (
    add_spectral_indices,
    associate_band_into_metacatalog,
    associate_catalogs,
    build_global_metacatalog,
    build_subband_metacatalog,
    catalog_elevation_deg,
    merge_lst_metacatalog,
    pick_highest_elevation_row,
    source_elevation_deg,
)

__all__ = [
    "FitsMetadata",
    "add_spectral_indices",
    "associate_band_into_metacatalog",
    "associate_catalogs",
    "attach_beam_and_freq",
    "blank_below_elevation",
    "build_global_metacatalog",
    "build_subband_metacatalog",
    "catalog_elevation_deg",
    "detect_sources",
    "detect_sources_many",
    "detect_sources_on_healpix_tiles",
    "iter_detect_sources",
    "discover_fits_files",
    "discovered_slots",
    "lst_hours_from_discovery",
    "median_beam_from_paths",
    "merge_lst_metacatalog",
    "parse_fits_metadata",
    "pick_highest_elevation_row",
    "resolve_fits_slot",
    "restfreq_hz_from_header",
    "source_elevation_deg",
]
