"""Rematch a metacatalog source back to LST-merged and per-hour catalogs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lwa_catalog.constants import BAND_OVERLAY_COLORS, subband_frequency_color
from lwa_catalog.create.merge import associate_catalogs
from lwa_catalog.io import read_lst_merged, read_sources_catalog
from lwa_catalog.paths import CatalogLayout

_KEY_COLUMNS = ("meta_id", "band", "lst_hour", "Source_id", "source_file")

# Default absolute-error column names for GAUL flux / shape / position fields.
_DEFAULT_ERR: dict[str, str] = {
    "Peak_flux": "E_Peak_flux",
    "Total_flux": "E_Total_flux",
    "RA": "E_RA",
    "DEC": "E_DEC",
    "Maj": "E_Maj",
    "Min": "E_Min",
}


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


def _target_peak_flux(meta_row: pd.Series, band: str) -> float | None:
    """Seeded Peak_flux for *band* from the metacatalog row, if available."""
    if band != "Full":
        col = f"Peak_flux_{band}"
        if col in meta_row.index:
            try:
                val = float(meta_row[col])
            except (TypeError, ValueError):
                val = float("nan")
            if np.isfinite(val):
                return val
    try:
        val = float(meta_row["Peak_flux"])
    except (TypeError, ValueError, KeyError):
        return None
    return val if np.isfinite(val) else None


def _pick_rematch_lst_row(
    hits: pd.DataFrame,
    meta_row: pd.Series,
    band: str,
) -> pd.Series:
    """Choose the LST-merged hit that recovers the metacatalog seed.

    Beam search can return confused neighbors. Prefer the hit whose
    ``Peak_flux`` matches the seeded metacatalog value for *band*, then the
    nearest on-sky neighbor. (Merge uses elevation when *building* the catalog;
    rematch must recover that seeded row, not re-rank neighbors by elevation.)
    """
    if len(hits) == 1:
        return hits.iloc[0]

    import astropy.units as u
    from astropy.coordinates import SkyCoord

    meta_sc = SkyCoord(
        ra=float(meta_row["RA"]) * u.deg,
        dec=float(meta_row["DEC"]) * u.deg,
    )
    hit_sc = SkyCoord(
        ra=hits["RA"].to_numpy(dtype=float) * u.deg,
        dec=hits["DEC"].to_numpy(dtype=float) * u.deg,
    )
    sep = meta_sc.separation(hit_sc).deg
    peak = hits["Peak_flux"].to_numpy(dtype=float)
    target = _target_peak_flux(meta_row, band)
    if target is None:
        dpeak = np.zeros(len(hits), dtype=float)
    else:
        dpeak = np.abs(peak - target)
        dpeak = np.where(np.isfinite(dpeak), dpeak, np.inf)

    # lexsort: last key is primary → seeded flux match, then sky separation
    order = np.lexsort((sep, dpeak))
    return hits.iloc[int(order[0])]


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
    the same beam-radius matcher as merge. Among multiple beam hits, the match
    closest in seeded ``Peak_flux`` (then on-sky separation) is kept so confused
    neighbors are not preferred over the catalog seed. Step B expands each LST
    match across its ``lst_hours`` and keeps all per-hour positional hits
    (durable key: ``band``, ``lst_hour``, ``Source_id``).

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
        best = _pick_rematch_lst_row(sub, row, band)
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


def _band_palette(bands: list[str] | None = None) -> dict[str, str]:
    palette = {
        "Full": "#4c4c4c",
        "Blue": "#1f77b4",
        "Green": "#2ca02c",
        "Red": "#d62728",
    }
    if not bands:
        return palette

    subbands = [band for band in bands if subband_frequency_color(band) is not None]
    if not subbands:
        return palette

    freqs_mhz = [float(band.removesuffix("MHz")) for band in subbands]
    freq_min = min(freqs_mhz)
    freq_max = max(freqs_mhz)
    for band in subbands:
        color = subband_frequency_color(
            band,
            freq_mhz_min=freq_min,
            freq_mhz_max=freq_max,
            color_low=BAND_OVERLAY_COLORS["Red"],
            color_high=BAND_OVERLAY_COLORS["Blue"],
        )
        if color is not None:
            palette[band] = color
    return palette


def _err_array(df: pd.DataFrame, err_col: str | None, n: int) -> np.ndarray | None:
    """Return non-negative finite errors, or ``None`` if unavailable."""
    if not err_col or err_col not in df.columns:
        return None
    err = pd.to_numeric(df[err_col], errors="coerce").to_numpy(dtype=float)
    err = np.where(np.isfinite(err) & (err >= 0.0), err, np.nan)
    if not np.isfinite(err).any():
        return None
    return err


def _errorbar_by_band(
    ax,
    df: pd.DataFrame,
    *,
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray | None = None,
    yerr: np.ndarray | None = None,
) -> None:
    """Draw per-band errorbars (colors) so multi-band members stay distinguishable."""
    if "band" in df.columns:
        bands = df["band"].astype(str).to_numpy()
        unique_bands = list(dict.fromkeys(bands.tolist()))
        palette = _band_palette(unique_bands)
        for band in unique_bands:
            mask = bands == band
            color = palette.get(band, "#7f7f7f")
            xe = None if xerr is None else xerr[mask]
            ye = None if yerr is None else yerr[mask]
            ax.errorbar(
                x[mask],
                y[mask],
                xerr=xe,
                yerr=ye,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2,
                markersize=5,
                alpha=0.85,
                label=band,
            )
        ax.legend(title="band", fontsize=8)
    else:
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color="#7f7f7f",
            ecolor="#7f7f7f",
            elinewidth=1.0,
            capsize=2,
            markersize=5,
            alpha=0.85,
        )


def plot_peak_flux_vs_lst(
    source_matches: pd.DataFrame,
    *,
    ax=None,
):
    """Scatter ``Peak_flux`` vs LST hour, colored by band, with ``E_Peak_flux`` bars.

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

    hours = work["lst_hour"].astype(str)
    hour_keys = []
    for h in hours:
        try:
            hour_keys.append(float(h.strip().lower().rstrip("h")))
        except ValueError:
            hour_keys.append(np.nan)
    work = work.assign(_hour_key=hour_keys)
    work = work.sort_values(["_hour_key", "band"], kind="mergesort")

    x_labels = work["lst_hour"].astype(str).tolist()
    unique_hours = list(dict.fromkeys(x_labels))
    x_pos = np.asarray([unique_hours.index(h) for h in x_labels], dtype=float)
    y = work["Peak_flux"].to_numpy(dtype=float)
    yerr = _err_array(work, "E_Peak_flux", len(work))

    _errorbar_by_band(ax, work, x=x_pos, y=y, yerr=yerr)
    ax.set_xticks(range(len(unique_hours)))
    ax.set_xticklabels(unique_hours, rotation=45, ha="right")
    ax.set_xlabel("lst_hour")
    ax.set_ylabel("Peak_flux")
    ax.set_title("Peak_flux vs LST (by band)")
    return ax


def plot_member_property_scatter(
    source_matches: pd.DataFrame,
    *,
    x: str = "Peak_flux",
    y: str = "Total_flux",
    xerr: str | None = None,
    yerr: str | None = None,
    ax=None,
):
    """Scatter two member properties, colored by band, with optional error bars.

    Default error columns are ``E_{x}`` / ``E_{y}`` when those fields exist
    (e.g. ``E_Peak_flux``, ``E_Total_flux``).

    Parameters
    ----------
    source_matches
        Rematched per-hour detections.
    x, y
        Column names to plot.
    xerr, yerr
        Optional absolute-error column names; ``None`` selects library defaults.
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

    if xerr is None:
        xerr = _DEFAULT_ERR.get(x)
    if yerr is None:
        yerr = _DEFAULT_ERR.get(y)

    xv = source_matches[x].to_numpy(dtype=float)
    yv = source_matches[y].to_numpy(dtype=float)
    xe = _err_array(source_matches, xerr, len(source_matches))
    ye = _err_array(source_matches, yerr, len(source_matches))
    _errorbar_by_band(ax, source_matches, x=xv, y=yv, xerr=xe, yerr=ye)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x} (by band)")
    return ax


def plot_ra_dec_scatter(
    source_matches: pd.DataFrame,
    *,
    ax=None,
):
    """Scatter ``RA`` vs ``DEC`` with ``E_RA`` / ``E_DEC`` error bars."""
    return plot_member_property_scatter(
        source_matches,
        x="RA",
        y="DEC",
        xerr="E_RA",
        yerr="E_DEC",
        ax=ax,
    )


def plot_maj_min_scatter(
    source_matches: pd.DataFrame,
    *,
    ax=None,
):
    """Scatter ``Maj`` vs ``Min`` with ``E_Maj`` / ``E_Min`` error bars."""
    return plot_member_property_scatter(
        source_matches,
        x="Maj",
        y="Min",
        xerr="E_Maj",
        yerr="E_Min",
        ax=ax,
    )


def preferred_trace_columns(df: pd.DataFrame) -> list[str]:
    """Column order favoring durable keys then common science fields."""
    preferred = [
        *_KEY_COLUMNS,
        "RA",
        "DEC",
        "E_RA",
        "E_DEC",
        "Peak_flux",
        "E_Peak_flux",
        "Total_flux",
        "E_Total_flux",
        "Maj",
        "E_Maj",
        "Min",
        "E_Min",
        "lst_hours",
        "n_lst_contributions",
        "representative_lst",
        "BMAJ",
    ]
    front = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return front + rest
