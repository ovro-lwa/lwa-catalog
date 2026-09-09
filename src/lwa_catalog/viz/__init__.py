"""Visualization helpers for HiPS / Aladin catalog overlays.

Requires the ``lwa-catalog[viz]`` extra (``ipyaladin``, ``panel``) for
``overlay_catalog_by_band``; band resolution and FOV filtering work with core deps.
"""

from __future__ import annotations

from lwa_catalog.viz.aladin import (
    OverlayResult,
    catalog_name_from_file,
    catalog_to_astropy_table,
    catalog_with_survey_beam,
    clear_catalog_overlays,
    clear_trace_overlays,
    filter_catalog_fov,
    overlay_catalog_by_band,
    overlay_trace_members,
    shape_complete_mask,
)
from lwa_catalog.viz.aladin_view import (
    DebouncedAladinViewRefresh,
    aladin_view_center_fov,
    apply_aladin_view,
    cancel_aladin_view_timers,
    restore_aladin_view,
)
from lwa_catalog.viz.bands import band_overlay_color, resolve_band_labels
from lwa_catalog.viz.browser import CatalogBrowser, CatalogBrowserConfig
from lwa_catalog.viz.coordinates import format_coordinate_deg, nearest_sources, parse_coordinate
from lwa_catalog.viz.hips import (
    SURVEY_HIPS_URLS,
    VLASS_MEDIAN_HIPS_ID,
    default_hips_survey,
    discover_local_hips_surveys,
    fetch_catalog_hips_surveys,
    fetch_hips_surveys,
    hips_survey_url,
    preferred_hips_survey,
    survey_hips_url,
)

__all__ = [
    "CatalogBrowser",
    "CatalogBrowserConfig",
    "DebouncedAladinViewRefresh",
    "OverlayResult",
    "SURVEY_HIPS_URLS",
    "VLASS_MEDIAN_HIPS_ID",
    "aladin_view_center_fov",
    "apply_aladin_view",
    "band_overlay_color",
    "cancel_aladin_view_timers",
    "catalog_name_from_file",
    "catalog_to_astropy_table",
    "catalog_with_survey_beam",
    "clear_catalog_overlays",
    "clear_trace_overlays",
    "default_hips_survey",
    "discover_local_hips_surveys",
    "fetch_catalog_hips_surveys",
    "fetch_hips_surveys",
    "filter_catalog_fov",
    "format_coordinate_deg",
    "hips_survey_url",
    "nearest_sources",
    "overlay_catalog_by_band",
    "overlay_trace_members",
    "parse_coordinate",
    "preferred_hips_survey",
    "resolve_band_labels",
    "restore_aladin_view",
    "shape_complete_mask",
    "survey_hips_url",
]
