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
from lwa_catalog.create.beam_scale import (
    DEFAULT_BEAM_SCALE_ALPHA_GRID,
    DEFAULT_BEAM_SCALE_PROBE_TILES,
    BeamScaleEstimateConfig,
    BeamScaleEstimateResult,
    estimate_beam_scale,
    gaul_model_residual,
    pick_best_alpha,
    summarize_beam_scale_estimate,
    tile_pixel_mask,
)
from lwa_catalog.create.healpix_detect import (
    DEFAULT_BEAM_SCALE,
    attach_beam_and_freq,
    detect_sources_on_healpix_tiles,
    median_beam_from_paths,
    restfreq_hz_from_header,
    scale_beam,
)
from lwa_catalog.create.tiered_detect import (
    DEFAULT_TIER2_BDSF_KW,
    fuse_gaul_m_with_tier2_s,
    merge_tier2_bdsf_kw,
)
from lwa_catalog.create.merge import (
    add_spectral_indices,
    associate_band_into_metacatalog,
    associate_catalogs,
    build_global_metacatalog,
    build_subband_metacatalog,
    catalog_elevation_deg,
    filter_detections_near_transit,
    merge_lst_metacatalog,
    merge_tile_metacatalog,
    pick_highest_elevation_row,
    source_elevation_deg,
)

__all__ = [
    "FitsMetadata",
    "add_spectral_indices",
    "associate_band_into_metacatalog",
    "associate_catalogs",
    "DEFAULT_BEAM_SCALE",
    "DEFAULT_BEAM_SCALE_ALPHA_GRID",
    "DEFAULT_BEAM_SCALE_PROBE_TILES",
    "BeamScaleEstimateConfig",
    "BeamScaleEstimateResult",
    "attach_beam_and_freq",
    "blank_below_elevation",
    "build_global_metacatalog",
    "build_subband_metacatalog",
    "catalog_elevation_deg",
    "detect_sources",
    "estimate_beam_scale",
    "filter_detections_near_transit",
    "detect_sources_many",
    "detect_sources_on_healpix_tiles",
    "DEFAULT_TIER2_BDSF_KW",
    "fuse_gaul_m_with_tier2_s",
    "gaul_model_residual",
    "iter_detect_sources",
    "discover_fits_files",
    "discovered_slots",
    "lst_hours_from_discovery",
    "median_beam_from_paths",
    "merge_lst_metacatalog",
    "merge_tier2_bdsf_kw",
    "merge_tile_metacatalog",
    "parse_fits_metadata",
    "pick_best_alpha",
    "pick_highest_elevation_row",
    "resolve_fits_slot",
    "restfreq_hz_from_header",
    "scale_beam",
    "source_elevation_deg",
    "summarize_beam_scale_estimate",
    "tile_pixel_mask",
]
