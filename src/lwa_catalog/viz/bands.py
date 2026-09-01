"""Band label resolution and overlay colors for catalog HiPS overlays."""

from __future__ import annotations

import re

import pandas as pd

from lwa_catalog.constants import (
    BAND_OVERLAY_COLOR_UNKNOWN,
    BAND_OVERLAY_COLORS,
    normalize_band_label,
    subband_frequency_color,
)

_LST_MERGED_RE = re.compile(
    r"(?:^|_)lst[_-]?(?P<band>Full|Blue|Green|Red|\d+MHz)(?:\.|$|_)",
    re.IGNORECASE,
)
_METACATALOG_LST_RE = re.compile(
    r"metacatalog_lst_(?P<band>Full|Blue|Green|Red|\d+MHz)",
    re.IGNORECASE,
)


def infer_band_from_catalog_name(catalog_name: str) -> str | None:
    """Infer a band from catalog keys like ``metacatalog_lst_Blue`` or ``lst_Blue``."""
    for pattern in (_METACATALOG_LST_RE, _LST_MERGED_RE):
        match = pattern.search(catalog_name)
        if match is not None:
            return normalize_band_label(match.group("band"))
    return None


def band_overlay_color(band: str) -> str:
    """Return the hex overlay color for a band label."""
    canonical = normalize_band_label(band)
    if canonical is None:
        return BAND_OVERLAY_COLOR_UNKNOWN
    subband_color = subband_frequency_color(canonical)
    if subband_color is not None:
        return subband_color
    return BAND_OVERLAY_COLORS.get(canonical, BAND_OVERLAY_COLOR_UNKNOWN)


def resolve_band_labels(df: pd.DataFrame, catalog_name: str) -> pd.Series:
    """Resolve per-row band labels for overlay coloring.

    Precedence: ``origin_band`` → ``band`` → band inferred from ``catalog_name``.
    Unrecognized values become ``"unknown"``.
    """
    inferred = infer_band_from_catalog_name(catalog_name)
    n = len(df)
    if n == 0:
        return pd.Series(dtype=object)

    if "origin_band" in df.columns:
        labels = df["origin_band"].astype(object)
    elif "band" in df.columns:
        labels = df["band"].astype(object)
    else:
        labels = pd.Series([None] * n, index=df.index, dtype=object)

    normalized = labels.map(
        lambda value: normalize_band_label(value) if pd.notna(value) else None
    )
    if inferred is not None:
        normalized = normalized.fillna(inferred)
    return normalized.fillna("unknown").astype(str)
