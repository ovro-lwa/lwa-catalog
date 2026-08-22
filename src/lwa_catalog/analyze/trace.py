"""Rematch a metacatalog source back to LST-merged and per-hour catalogs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lwa_catalog.create.merge import associate_catalogs, pick_highest_elevation_row
from lwa_catalog.io import read_lst_merged, read_sources_catalog
from lwa_catalog.paths import CatalogLayout

_KEY_COLUMNS = ("meta_id", "band", "lst_hour", "Source_id", "source_file")


@dataclass
class SourceTrace:
    """Positional rematch result for one metacatalog source."""

    meta_id: int
    meta_row: pd.Series
    lst_matches: pd.DataFrame
    source_matches: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _parse_bands_present(row: pd.Series) -> list[str]:
    raw = str(row.get("bands_present", "") or "")
    bands = [part.strip() for part in raw.split(",") if part.strip() and part.strip().lower() != "nan"]
    return bands


def _parse_lst_hours(text: object) -> list[str]:
    raw = str(text or "")
    return [part.strip() for part in raw.split(",") if part.strip() and part.strip().lower() != "nan"]


def _resolve_meta_row(
    meta: pd.Series | pd.DataFrame,
    *,
    meta_id: int | None,
) -> tuple[int, pd.Series]:
    if isinstance(meta, pd.Series):
        row = meta
        if meta_id is not None and "meta_id" in row.index and int(row["meta_id"]) != int(meta_id):
            msg = f"Series meta_id={row['meta_id']} does not match requested meta_id={meta_id}"
            raise ValueError(msg)
        if "meta_id" in row.index and pd.notna(row["meta_id"]):
            return int(row["meta_id"]), row
        if meta_id is not None:
            return int(meta_id), row
        msg = "meta Series needs a meta_id value or pass meta_id="
        raise ValueError(msg)

    if meta_id is None:
        msg = "meta_id is required when meta is a DataFrame"
        raise ValueError(msg)
    if "meta_id" not in meta.columns:
        msg = "metacatalog DataFrame missing meta_id column"
        raise ValueError(msg)
    hits = meta.loc[meta["meta_id"] == meta_id]
    if hits.empty:
        msg = f"meta_id={meta_id} not found in metacatalog ({len(meta)} rows)"
        raise ValueError(msg)
    return int(meta_id), hits.iloc[0]


def _match_bmaj(row: pd.Series) -> float:
    for key in ("BMAJ_match", "BMAJ_full", "BMAJ"):
        if key in row.index:
            try:
                val = float(row[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(val):
                return val
    return 0.0


def _one_row_base(ra: float, dec: float, bmaj: float) -> pd.DataFrame:
    return pd.DataFrame({"RA": [ra], "DEC": [dec], "BMAJ": [bmaj]})


def _ensure_bmaj_column(df: pd.DataFrame) -> pd.DataFrame:
    if "BMAJ" in df.columns:
        out = df.copy()
        out["BMAJ"] = pd.to_numeric(out["BMAJ"], errors="coerce").fillna(0.0)
        return out
    out = df.copy()
    out["BMAJ"] = 0.0
    return out


def _load_lst_band(
    layout: CatalogLayout,
    band: str,
    lst_merged: Mapping[str, pd.DataFrame] | None,
    warnings: list[str],
) -> pd.DataFrame | None:
    if lst_merged is not None and band in lst_merged:
        frame = lst_merged[band]
        if frame is None or (isinstance(frame, pd.DataFrame) and frame.empty):
            warnings.append(f"LST-merged cache for {band} is empty")
            return None
        return _ensure_bmaj_column(frame)
    try:
        loaded = read_lst_merged(layout, band, as_pandas=True)
    except FileNotFoundError:
        warnings.append(f"Missing LST-merged catalog for band {band}: {layout.lst_merged(band)}")
        return None
    assert isinstance(loaded, pd.DataFrame)
    if loaded.empty:
        warnings.append(f"LST-merged catalog for {band} is empty")
        return None
    return _ensure_bmaj_column(loaded)


def rematch_meta_source(
    meta: pd.Series | pd.DataFrame,
    layout: CatalogLayout,
    *,
    meta_id: int | None = None,
    lst_merged: Mapping[str, pd.DataFrame] | None = None,
) -> SourceTrace:
    """Positionally rematch one metacatalog source to LST and per-hour catalogs.

    Step A associates the meta row against each band in ``bands_present`` using
    the same beam-radius matcher as merge, keeping the highest-elevation hit.
    Step B expands each LST match across its ``lst_hours`` and keeps all
    per-hour positional hits (durable key: ``band``, ``lst_hour``, ``Source_id``).

    Parameters
    ----------
    meta
        Metacatalog DataFrame (requires ``meta_id=``) or a single row Series.
    layout
        Catalog directory layout for Parquet reads.
    meta_id
        Required when *meta* is a DataFrame.
    lst_merged
        Optional in-memory band → LST-merged DataFrame cache (skips disk reads).

    Returns
    -------
    SourceTrace
        ``lst_matches``, ``source_matches``, and any skip warnings.
    """
    mid, row = _resolve_meta_row(meta, meta_id=meta_id)
    for col in ("RA", "DEC"):
        if col not in row.index or not np.isfinite(float(row[col])):
            msg = f"metacatalog row meta_id={mid} missing finite {col}"
            raise ValueError(msg)

    warnings: list[str] = []
    bands = _parse_bands_present(row)
    if not bands:
        warnings.append(f"meta_id={mid} has empty bands_present")

    base_bmaj = _match_bmaj(row)
    base = _one_row_base(float(row["RA"]), float(row["DEC"]), base_bmaj)

    lst_rows: list[pd.Series] = []
    for band in bands:
        lst_df = _load_lst_band(layout, band, lst_merged, warnings)
        if lst_df is None:
            continue
        hits, _ = associate_catalogs(base, lst_df)
        idxs = hits.get(0, [])
        if not idxs:
            warnings.append(f"No LST-merged positional match for band {band}")
            continue
        sub = lst_df.iloc[idxs]
        best = pick_highest_elevation_row(sub)
        entry = best.copy()
        entry["meta_id"] = mid
        entry["band"] = band
        lst_rows.append(entry)

    lst_matches = pd.DataFrame(lst_rows) if lst_rows else pd.DataFrame()

    source_frames: list[pd.DataFrame] = []
    for _, lst_row in lst_matches.iterrows():
        band = str(lst_row["band"])
        hours = _parse_lst_hours(lst_row.get("lst_hours", ""))
        if not hours:
            hours = _parse_lst_hours(row.get("lst_hours", ""))
        if not hours:
            warnings.append(f"No lst_hours for LST match in band {band}")
            continue

        lst_bmaj = float(lst_row["BMAJ"]) if np.isfinite(float(lst_row.get("BMAJ", np.nan))) else base_bmaj
        lst_base = _one_row_base(float(lst_row["RA"]), float(lst_row["DEC"]), lst_bmaj)

        for hour in hours:
            path = layout.sources(hour, band)
            if not path.is_file():
                warnings.append(f"Missing sources catalog: {path.name}")
                continue
            try:
                sources = read_sources_catalog(layout, hour, band, as_pandas=True)
            except FileNotFoundError:
                warnings.append(f"Missing sources catalog: {path.name}")
                continue
            assert isinstance(sources, pd.DataFrame)
            if sources.empty:
                continue
            sources = _ensure_bmaj_column(sources)
            hits, _ = associate_catalogs(lst_base, sources)
            idxs = hits.get(0, [])
            if not idxs:
                continue
            hit = sources.iloc[idxs].copy()
            hit["meta_id"] = mid
            hit["band"] = band
            if "lst_hour" not in hit.columns:
                hit["lst_hour"] = hour
            source_frames.append(hit)

    if source_frames:
        source_matches = pd.concat(source_frames, ignore_index=True)
    else:
        source_matches = pd.DataFrame()

    return SourceTrace(
        meta_id=mid,
        meta_row=row,
        lst_matches=lst_matches,
        source_matches=source_matches,
        warnings=warnings,
    )


def _band_colors(bands: list[str]) -> list[str]:
    palette = {
        "Full": "#4c4c4c",
        "Blue": "#1f77b4",
        "Green": "#2ca02c",
        "Red": "#d62728",
    }
    return [palette.get(b, "#7f7f7f") for b in bands]


def plot_peak_flux_vs_lst(
    source_matches: pd.DataFrame,
    *,
    ax=None,
):
    """Scatter ``Peak_flux`` vs LST hour, colored by band.

    Parameters
    ----------
    source_matches
        Rematched per-hour detections (from :func:`rematch_meta_source`).
    ax
        Optional Matplotlib axes; created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    if source_matches is None or source_matches.empty:
        ax.set_title("Peak_flux vs LST (no members)")
        ax.set_xlabel("lst_hour")
        ax.set_ylabel("Peak_flux")
        return ax

    work = source_matches.copy()
    if "lst_hour" not in work.columns or "Peak_flux" not in work.columns:
        ax.set_title("Peak_flux vs LST (missing columns)")
        return ax

    # Order hours numerically when labels look like NNh.
    hours = work["lst_hour"].astype(str)
    hour_keys = []
    for h in hours:
        try:
            hour_keys.append(float(h.strip().lower().rstrip("h")))
        except ValueError:
            hour_keys.append(np.nan)
    work = work.assign(_hour_key=hour_keys)
    work = work.sort_values(["_hour_key", "band"], kind="mergesort")

    bands = work["band"].astype(str).tolist() if "band" in work.columns else ["?"] * len(work)
    colors = _band_colors(bands)
    x_labels = work["lst_hour"].astype(str).tolist()
    # Categorical positions preserving sorted unique order
    unique_hours = list(dict.fromkeys(x_labels))
    x_pos = [unique_hours.index(h) for h in x_labels]
    ax.scatter(x_pos, work["Peak_flux"].to_numpy(dtype=float), c=colors, s=36, alpha=0.85)
    ax.set_xticks(range(len(unique_hours)))
    ax.set_xticklabels(unique_hours, rotation=45, ha="right")
    ax.set_xlabel("lst_hour")
    ax.set_ylabel("Peak_flux")
    ax.set_title("Peak_flux vs LST (by band)")

    # Simple legend for bands present
    if "band" in work.columns:
        seen: dict[str, str] = {}
        for band, color in zip(bands, colors, strict=True):
            seen.setdefault(band, color)
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markersize=8, label=b)
            for b, c in seen.items()
        ]
        ax.legend(handles=handles, title="band", fontsize=8)

    return ax


def plot_member_property_scatter(
    source_matches: pd.DataFrame,
    *,
    x: str = "Peak_flux",
    y: str = "Total_flux",
    ax=None,
):
    """Scatter two member properties, colored by band.

    Parameters
    ----------
    source_matches
        Rematched per-hour detections.
    x, y
        Column names to plot.
    ax
        Optional Matplotlib axes; created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    if source_matches is None or source_matches.empty:
        ax.set_title(f"{y} vs {x} (no members)")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return ax

    if x not in source_matches.columns or y not in source_matches.columns:
        ax.set_title(f"{y} vs {x} (missing columns)")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return ax

    bands = (
        source_matches["band"].astype(str).tolist()
        if "band" in source_matches.columns
        else ["?"] * len(source_matches)
    )
    colors = _band_colors(bands)
    ax.scatter(
        source_matches[x].to_numpy(dtype=float),
        source_matches[y].to_numpy(dtype=float),
        c=colors,
        s=36,
        alpha=0.85,
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x} (by band)")
    if "band" in source_matches.columns:
        seen: dict[str, str] = {}
        for band, color in zip(bands, colors, strict=True):
            seen.setdefault(band, color)
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markersize=8, label=b)
            for b, c in seen.items()
        ]
        ax.legend(handles=handles, title="band", fontsize=8)
    return ax


def preferred_trace_columns(df: pd.DataFrame) -> list[str]:
    """Column order favoring durable keys then common science fields."""
    preferred = [
        *_KEY_COLUMNS,
        "RA",
        "DEC",
        "Peak_flux",
        "Total_flux",
        "lst_hours",
        "n_lst_contributions",
        "representative_lst",
        "BMAJ",
    ]
    front = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return front + rest
