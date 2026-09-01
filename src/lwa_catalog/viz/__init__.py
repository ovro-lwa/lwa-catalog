"""Visualization helpers for HiPS / Aladin catalog overlays.

Requires the ``lwa-catalog[viz]`` extra (``ipyaladin``, ``panel``) for
``overlay_catalog_by_band``; band resolution and FOV filtering work with core deps.
"""

from __future__ import annotations

from lwa_catalog.viz.aladin import (
    OverlayResult,
    catalog_name_from_file,
    catalog_to_astropy_table,
    clear_catalog_overlays,
    clear_trace_overlays,
    filter_catalog_fov,
    overlay_catalog_by_band,
    overlay_trace_members,
    shape_complete_mask,
)
from lwa_catalog.viz.aladin_view import DebouncedAladinViewRefresh, aladin_view_center_fov
from lwa_catalog.viz.bands import band_overlay_color, resolve_band_labels
from lwa_catalog.viz.coordinates import format_coordinate_deg, parse_coordinate
from lwa_catalog.viz.hips import (
    default_hips_survey,
    discover_local_hips_surveys,
    fetch_hips_surveys,
    hips_survey_url,
)

__all__ = [
    "DebouncedAladinViewRefresh",
    "OverlayResult",
    "aladin_view_center_fov",
    "band_overlay_color",
    "catalog_name_from_file",
    "catalog_to_astropy_table",
    "clear_catalog_overlays",
    "clear_trace_overlays",
    "default_hips_survey",
    "discover_local_hips_surveys",
    "fetch_hips_surveys",
    "filter_catalog_fov",
    "format_coordinate_deg",
    "hips_survey_url",
    "overlay_catalog_by_band",
    "overlay_trace_members",
    "parse_coordinate",
    "resolve_band_labels",
    "shape_complete_mask",
]
