"""Visualization helpers for HiPS / Aladin catalog overlays.

Requires the ``lwa-catalog[viz]`` extra (``ipyaladin``, ``panel``) for
``overlay_catalog_by_band``; band resolution and FOV filtering work with core deps.
"""

from __future__ import annotations

from lwa_catalog.viz.aladin import (
    OverlayResult,
    catalog_to_astropy_table,
    filter_catalog_fov,
    overlay_catalog_by_band,
    shape_complete_mask,
)
from lwa_catalog.viz.bands import band_overlay_color, resolve_band_labels

__all__ = [
    "OverlayResult",
    "band_overlay_color",
    "catalog_to_astropy_table",
    "filter_catalog_fov",
    "overlay_catalog_by_band",
    "resolve_band_labels",
    "shape_complete_mask",
]
