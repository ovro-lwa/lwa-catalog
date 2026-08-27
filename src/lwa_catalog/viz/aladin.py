"""Catalog → Aladin Lite overlay helpers (ipyaladin)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from lwa_catalog.constants import COLOR_BANDS
from lwa_catalog.viz.bands import band_overlay_color, resolve_band_labels

if TYPE_CHECKING:
    from ipyaladin import Aladin

_SHAPE_COLUMNS = ("Maj", "Min", "PA")
_OVERLAY_TABLE_COLUMNS = (
    "RA",
    "DEC",
    "Maj",
    "Min",
    "PA",
    "Peak_flux",
    "meta_id",
    "origin_band",
    "band",
)


@dataclass(frozen=True)
class OverlayResult:
    """Summary returned by :func:`overlay_catalog_by_band`."""

    drawn: int
    in_fov: int
    truncated: bool
    per_band: dict[str, int] = field(default_factory=dict)


def catalog_to_astropy_table(
    df: pd.DataFrame,
    *,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    columns: Sequence[str] | None = None,
    attach_beam_units: bool = False,
) -> Table:
    """Build an Astropy ``Table`` from a catalog DataFrame for ipyaladin."""
    if ra_col not in df.columns or dec_col not in df.columns:
        msg = f"catalog missing position columns {ra_col!r} and/or {dec_col!r}"
        raise ValueError(msg)
    if columns is None:
        use_cols = [c for c in _OVERLAY_TABLE_COLUMNS if c in df.columns]
    else:
        use_cols = list(columns)
        for required in (ra_col, dec_col):
            if required not in use_cols:
                use_cols.insert(0, required)
    table = Table.from_pandas(df.loc[:, use_cols].copy())
    if attach_beam_units:
        for col in _SHAPE_COLUMNS:
            if col in table.colnames and table[col].unit is None:
                table[col].unit = u.deg
    return table


def shape_complete_mask(df: pd.DataFrame) -> pd.Series:
    """True where ``Maj``/``Min``/``PA`` are finite and positive (ellipse drawable)."""
    if not all(col in df.columns for col in _SHAPE_COLUMNS):
        return pd.Series(False, index=df.index)
    maj = pd.to_numeric(df["Maj"], errors="coerce")
    min_ = pd.to_numeric(df["Min"], errors="coerce")
    pa = pd.to_numeric(df["PA"], errors="coerce")
    return maj.notna() & min_.notna() & pa.notna() & (maj > 0) & (min_ > 0)


def filter_catalog_fov(
    df: pd.DataFrame,
    center: SkyCoord,
    fov_deg: float,
    *,
    margin_deg: float = 0.1,
    ra_col: str = "RA",
    dec_col: str = "DEC",
) -> pd.DataFrame:
    """Return rows whose sky position lies inside the circular FOV."""
    if df.empty:
        return df.copy()
    if ra_col not in df.columns or dec_col not in df.columns:
        msg = f"catalog missing position columns {ra_col!r} and/or {dec_col!r}"
        raise ValueError(msg)

    ra = pd.to_numeric(df[ra_col], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(df[dec_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(ra) & np.isfinite(dec)
    if not valid.any():
        return df.iloc[0:0].copy()

    coords = SkyCoord(ra=ra[valid] * u.deg, dec=dec[valid] * u.deg, frame="icrs")
    sep = center.separation(coords)
    limit = (fov_deg / 2.0 + margin_deg) * u.deg
    in_view = np.zeros(len(df), dtype=bool)
    in_view[np.flatnonzero(valid)] = sep <= limit
    return df.loc[in_view].copy()


def _cap_rows_by_center(
    df: pd.DataFrame,
    center: SkyCoord,
    max_rows: int,
    *,
    ra_col: str = "RA",
    dec_col: str = "DEC",
) -> tuple[pd.DataFrame, bool]:
    if len(df) <= max_rows:
        return df, False
    ra = pd.to_numeric(df[ra_col], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(df[dec_col], errors="coerce").to_numpy(dtype=float)
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    sep_arcsec = center.separation(coords).arcsec
    order = np.argsort(sep_arcsec)
    capped = df.iloc[order[:max_rows]].copy()
    return capped, True


def _overlay_names(name_prefix: str) -> list[str]:
    names = [f"{name_prefix}_selection"]
    for band in (*COLOR_BANDS, "unknown"):
        names.append(f"{name_prefix}_{band}")
        names.append(f"{name_prefix}_{band}_cross")
    return names


def _remove_overlays(aladin: Aladin, name_prefix: str) -> None:
    remove = getattr(aladin, "remove_overlay", None)
    if remove is None:
        msg = (
            "ipyaladin>=0.8 is required for catalog overlay refresh; "
            "install with: pip install 'lwa-catalog[viz]'"
        )
        raise ImportError(msg)
    for name in _overlay_names(name_prefix):
        try:
            remove(name)
        except ValueError:
            # Overlay layer was not present from a prior draw.
            continue


def _catalog_pa_to_regions_angle(pa_deg: float) -> u.Quantity:
    """Map catalog PA (major axis N→E) to ``regions.EllipseSkyRegion.angle``.

    Aladin Lite ``rotationDeg`` uses the same N→E convention as our HiPS
    painting. ipyaladin subtracts 90° when converting regions ellipses, so we
    add it here to cancel that offset.
    """
    return (pa_deg + 90.0) * u.deg


def _dataframe_to_ellipse_regions(df: pd.DataFrame) -> list[Any]:
    """Build ``regions.EllipseSkyRegion`` footprints from beam columns.

    ``Maj`` / ``Min`` are FWHM in degrees; ``PA`` is major-axis position angle
    from North toward East (PyBDSF convention).
    """
    from regions import EllipseSkyRegion

    regions: list[Any] = []
    ra = pd.to_numeric(df["RA"], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(df["DEC"], errors="coerce").to_numpy(dtype=float)
    maj = pd.to_numeric(df["Maj"], errors="coerce").to_numpy(dtype=float)
    min_ = pd.to_numeric(df["Min"], errors="coerce").to_numpy(dtype=float)
    pa = pd.to_numeric(df["PA"], errors="coerce").to_numpy(dtype=float)

    for ra_i, dec_i, maj_i, min_i, pa_i in zip(ra, dec, maj, min_, pa, strict=True):
        if not (
            np.isfinite(ra_i)
            and np.isfinite(dec_i)
            and np.isfinite(maj_i)
            and np.isfinite(min_i)
            and np.isfinite(pa_i)
            and maj_i > 0
            and min_i > 0
        ):
            continue
        if min_i > maj_i:
            maj_i, min_i = min_i, maj_i
        center = SkyCoord(ra=ra_i * u.deg, dec=dec_i * u.deg, frame="icrs")
        regions.append(
            EllipseSkyRegion(
                center=center,
                width=maj_i * u.deg,
                height=min_i * u.deg,
                angle=_catalog_pa_to_regions_angle(pa_i),
            )
        )
    return regions


def _add_cross_table_overlay(
    aladin: Aladin,
    df: pd.DataFrame,
    *,
    overlay_name: str,
    color: str,
    source_size: int,
    line_width: int | None = None,
) -> int:
    if df.empty:
        return 0

    preferred = ["RA", "DEC", "Peak_flux", "meta_id", "origin_band", "band"]
    use_cols = [col for col in preferred if col in df.columns]
    table = catalog_to_astropy_table(df, columns=use_cols)
    options: dict[str, Any] = {
        "shape": "cross",
        "color": color,
        "source_size": source_size,
        "name": overlay_name,
    }
    if line_width is not None:
        options["line_width"] = line_width
    aladin.add_table(table, **options)
    return len(df)


def _add_ellipse_overlay(
    aladin: Aladin,
    df: pd.DataFrame,
    *,
    overlay_name: str,
    color: str,
    line_width: int | None = None,
) -> int:
    if df.empty:
        return 0

    regions = _dataframe_to_ellipse_regions(df)
    if not regions:
        return 0

    options: dict[str, Any] = {"color": color, "name": overlay_name}
    if line_width is not None:
        options["line_width"] = line_width
    aladin.add_graphic_overlay_from_region(regions, **options)
    return len(regions)


def _add_band_overlay(
    aladin: Aladin,
    df: pd.DataFrame,
    *,
    overlay_name: str,
    color: str,
    source_size: int,
    line_width: int | None = None,
    cross_suffix: bool = True,
) -> int:
    if df.empty:
        return 0

    complete = shape_complete_mask(df)
    ellipse_df = df.loc[complete]
    cross_df = df.loc[~complete]
    drawn = 0
    drawn += _add_ellipse_overlay(
        aladin,
        ellipse_df,
        overlay_name=overlay_name,
        color=color,
        line_width=line_width,
    )
    if not cross_df.empty:
        cross_name = f"{overlay_name}_cross" if cross_suffix else overlay_name
        drawn += _add_cross_table_overlay(
            aladin,
            cross_df,
            overlay_name=cross_name,
            color=color,
            source_size=source_size,
            line_width=line_width,
        )
    return drawn


def _draw_band_colored_overlays(
    aladin: Aladin,
    df: pd.DataFrame,
    band_labels: pd.Series,
    *,
    name_prefix: str,
    source_size: int,
    line_width: int | None = None,
    color_override: str | None = None,
) -> dict[str, int]:
    per_band: dict[str, int] = {}
    if df.empty:
        return per_band

    bands_present = sorted({str(b) for b in band_labels.loc[df.index]})
    for band in bands_present:
        band_mask = band_labels.loc[df.index].astype(str) == band
        band_df = df.loc[band_mask]
        color = color_override or band_overlay_color(band)
        count = _add_band_overlay(
            aladin,
            band_df,
            overlay_name=f"{name_prefix}_{band}",
            color=color,
            source_size=source_size,
            line_width=line_width,
        )
        if count:
            per_band[band] = count

    return per_band


def overlay_catalog_by_band(
    aladin: Aladin,
    df: pd.DataFrame,
    catalog_name: str,
    center: SkyCoord,
    fov_deg: float,
    *,
    name_prefix: str = "catalog",
    max_rows: int = 500,
    margin_deg: float = 0.1,
    selection_idx: int | None = None,
    replace: bool = True,
    source_size: int = 8,
    selection_source_size: int = 12,
) -> OverlayResult:
    """FOV-filter, cap, and draw band-colored beam ellipses (cross fallback).

    Rows with complete ``Maj``/``Min``/``PA`` are drawn as ellipses via
    ``add_graphic_overlay_from_region``; incomplete rows use cross markers.

    Parameters
    ----------
    selection_idx
        ``.iloc`` index into ``df`` for the emphasized selection overlay.
    """
    if replace:
        _remove_overlays(aladin, name_prefix)

    if df.empty:
        return OverlayResult(drawn=0, in_fov=0, truncated=False)

    band_labels = resolve_band_labels(df, catalog_name)
    in_fov = filter_catalog_fov(df, center, fov_deg, margin_deg=margin_deg)
    capped, truncated = _cap_rows_by_center(in_fov, center, max_rows)
    per_band = _draw_band_colored_overlays(
        aladin,
        capped,
        band_labels,
        name_prefix=name_prefix,
        source_size=source_size,
    )
    drawn = int(sum(per_band.values()))

    if selection_idx is not None and 0 <= selection_idx < len(df):
        row = df.iloc[selection_idx : selection_idx + 1]
        row_labels = resolve_band_labels(row, catalog_name)
        sel_color = band_overlay_color(str(row_labels.iloc[0]))
        _add_band_overlay(
            aladin,
            row,
            overlay_name=f"{name_prefix}_selection",
            color=sel_color,
            source_size=selection_source_size,
            line_width=3,
            cross_suffix=False,
        )

    return OverlayResult(
        drawn=drawn,
        in_fov=len(in_fov),
        truncated=truncated,
        per_band=per_band,
    )
