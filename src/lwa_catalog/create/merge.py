"""Fuse per-image catalogs into LST-band and global metacatalogs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from lwa_catalog.constants import ASSOC_BANDS, BAND_FIELDS, COLOR_BANDS


def _pick_median_flux_row(df: pd.DataFrame, flux_col: str = "Peak_flux") -> pd.Series:
    """Return the row whose flux is closest to the median over *df*."""
    flux = df[flux_col].to_numpy(dtype=float)
    finite = np.isfinite(flux)
    if not finite.any():
        return df.iloc[0]
    med = float(np.nanmedian(flux[finite]))
    idx = int(np.nanargmin(np.abs(flux - med)))
    return df.iloc[idx]


def _skycoord_from_columns(df: pd.DataFrame) -> SkyCoord:
    return SkyCoord(
        ra=df["RA"].to_numpy(dtype=float) * u.deg,
        dec=df["DEC"].to_numpy(dtype=float) * u.deg,
    )


def _associate_catalogs(
    base_df: pd.DataFrame,
    band_df: pd.DataFrame,
) -> tuple[dict[int, list[int]], set[int]]:
    """Vectorized base↔band matching within per-pair beam radii.

    Returns ``{base_iloc: [band_iloc, ...]}`` and the set of matched band ilocs.
    """
    if base_df.empty or band_df.empty:
        return {}, set()

    base_sc = _skycoord_from_columns(base_df)
    band_sc = _skycoord_from_columns(band_df)
    bmaj_base = base_df["BMAJ"].to_numpy(dtype=float)
    bmaj_band = band_df["BMAJ"].to_numpy(dtype=float)

    search_radius = float(max(bmaj_base.max(), bmaj_band.max())) * u.deg
    # Astropy returns (idx_searcharound, idx_self) = (band, base)
    idx_band, idx_base, sep2d, _ = base_sc.search_around_sky(band_sc, search_radius)
    if len(idx_base) == 0:
        return {}, set()

    sep_deg = sep2d.to(u.deg).value
    limits = np.maximum(bmaj_base[idx_base], bmaj_band[idx_band])
    keep = sep_deg <= limits
    idx_base = idx_base[keep]
    idx_band = idx_band[keep]

    hits_by_base: dict[int, list[int]] = {}
    matched: set[int] = set()
    for i, j in zip(idx_base.tolist(), idx_band.tolist(), strict=True):
        hits_by_base.setdefault(i, []).append(j)
        matched.add(j)
    return hits_by_base, matched


def _cluster_by_sky_position(
    df: pd.DataFrame,
    *,
    sort_col: str = "Peak_flux",
    bmaj_col: str = "BMAJ",
) -> list[pd.DataFrame]:
    """Greedy beam-sized clustering; return one member DataFrame per cluster."""
    if df.empty:
        return []

    work = df[np.isfinite(df["RA"]) & np.isfinite(df["DEC"])].copy()
    work = work.sort_values(sort_col, ascending=False, na_position="last")

    cluster_ra: list[float] = []
    cluster_dec: list[float] = []
    cluster_bmaj: list[float] = []
    cluster_members: list[list[dict]] = []

    for row in work.itertuples(index=False):
        rd = row._asdict()
        ra_row = float(rd["RA"])
        dec_row = float(rd["DEC"])
        bmaj_row = float(rd.get(bmaj_col, np.nan))
        if not np.isfinite(bmaj_row):
            bmaj_row = 0.0

        best_idx: int | None = None
        if cluster_ra:
            sc = SkyCoord(ra=ra_row * u.deg, dec=dec_row * u.deg)
            cluster_sc = SkyCoord(
                ra=np.asarray(cluster_ra, dtype=float) * u.deg,
                dec=np.asarray(cluster_dec, dtype=float) * u.deg,
            )
            seps = sc.separation(cluster_sc).deg
            radii = np.maximum(bmaj_row, np.asarray(cluster_bmaj, dtype=float))
            within = seps <= radii
            if within.any():
                candidates = np.where(within)[0]
                best_idx = int(candidates[np.argmin(seps[candidates])])

        if best_idx is None:
            cluster_ra.append(ra_row)
            cluster_dec.append(dec_row)
            cluster_bmaj.append(bmaj_row)
            cluster_members.append([rd])
        else:
            cluster_members[best_idx].append(rd)
            rep = _pick_median_flux_row(pd.DataFrame(cluster_members[best_idx]))
            cluster_ra[best_idx] = float(rep["RA"])
            cluster_dec[best_idx] = float(rep["DEC"])
            cluster_bmaj[best_idx] = max(cluster_bmaj[best_idx], bmaj_row)

    return [pd.DataFrame(members) for members in cluster_members]


def merge_lst_metacatalog(catalogs: Iterable[pd.DataFrame], *, band: str) -> pd.DataFrame:
    """Fuse per-LST detections within one band → one row per source."""
    frames = list(catalogs)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for members in _cluster_by_sky_position(combined):
        rep = _pick_median_flux_row(members)
        entry = rep.to_dict()
        entry["band"] = band
        entry["n_lst_contributions"] = len(members)
        entry["lst_hours"] = ",".join(sorted(members["lst_hour"].unique()))
        entry["representative_lst"] = rep["lst_hour"]
        rows.append(entry)

    meta = pd.DataFrame(rows)
    return meta.sort_values("Peak_flux", ascending=False, na_position="last").reset_index(drop=True)


def _empty_band_cols(
    *,
    assoc_bands: Sequence[str] = ASSOC_BANDS,
    band_fields: Sequence[str] = BAND_FIELDS,
) -> dict:
    out: dict = {}
    for band in assoc_bands:
        for field in band_fields:
            out[f"{field}_{band}"] = np.nan
        out[f"n_assoc_{band}"] = 0
    return out


def _lst_meta_from_band_row(row: pd.Series) -> tuple[int, str, str]:
    return (
        int(row.get("n_lst_contributions", 1)),
        str(row.get("lst_hours", row.get("lst_hour", ""))),
        str(row.get("representative_lst", row.get("lst_hour", ""))),
    )


def _primary_fields_from_band(row: pd.Series) -> dict:
    return {
        "RA": row["RA"],
        "DEC": row["DEC"],
        "Peak_flux": row["Peak_flux"],
        "Total_flux": row["Total_flux"],
        "Maj": row["Maj"],
        "Min": row["Min"],
        "PA": row["PA"],
        "DC_Maj": row.get("DC_Maj", np.nan),
        "DC_Min": row.get("DC_Min", np.nan),
        "DC_PA": row.get("DC_PA", np.nan),
    }


def _attach_band_columns(
    entry: dict,
    band_row: pd.Series,
    band: str,
    n_assoc: int,
    *,
    band_fields: Sequence[str] = BAND_FIELDS,
) -> None:
    entry[f"n_assoc_{band}"] = n_assoc
    for field in band_fields:
        entry[f"{field}_{band}"] = band_row[field]
    entry[f"source_file_{band}"] = band_row.get("source_file", "")


def _update_bands_present(
    entry: dict,
    *bands: str,
    color_bands: Sequence[str] = COLOR_BANDS,
) -> None:
    present = {b for b in str(entry.get("bands_present", "")).split(",") if b}
    present.update(bands)
    entry["bands_present"] = ",".join(b for b in color_bands if b in present)


def _seed_row_from_band(
    band_row: pd.Series,
    band: str,
    *,
    assoc_bands: Sequence[str] = ASSOC_BANDS,
    band_fields: Sequence[str] = BAND_FIELDS,
) -> dict:
    """One metacatalog row seeded from a single-band LST-merged detection."""
    n_lst, lst_hours, rep_lst = _lst_meta_from_band_row(band_row)
    entry = {
        "origin_band": band,
        "bands_present": band,
        **_primary_fields_from_band(band_row),
        "BMAJ_match": float(band_row["BMAJ"]),
        "n_lst_contributions": n_lst,
        "lst_hours": lst_hours,
        "representative_lst": rep_lst,
    }
    entry.update(_empty_band_cols(assoc_bands=assoc_bands, band_fields=band_fields))
    if band == "Full":
        entry["BMAJ_full"] = float(band_row["BMAJ"])
        entry["source_file_Full"] = band_row.get("source_file", "")
    else:
        entry["BMAJ_full"] = np.nan
        _attach_band_columns(entry, band_row, band, 1, band_fields=band_fields)
    return entry


def merge_full_and_blue(
    full_df: pd.DataFrame,
    blue_df: pd.DataFrame,
    *,
    assoc_bands: Sequence[str] = ASSOC_BANDS,
    band_fields: Sequence[str] = BAND_FIELDS,
    color_bands: Sequence[str] = COLOR_BANDS,
) -> pd.DataFrame:
    """Cross-match Blue onto Full; one row per deduplicated sky position."""
    full_df = full_df.reset_index(drop=True)
    blue_df = blue_df.reset_index(drop=True)
    hits, matched_blue = _associate_catalogs(full_df, blue_df)

    rows: list[dict] = []
    for i, frow in full_df.iterrows():
        entry = _seed_row_from_band(
            frow, "Full", assoc_bands=assoc_bands, band_fields=band_fields
        )
        blues = hits.get(i, [])
        if blues:
            sub = blue_df.iloc[blues]
            best = _pick_median_flux_row(sub)
            _attach_band_columns(entry, best, "Blue", len(blues), band_fields=band_fields)
            _update_bands_present(entry, "Full", "Blue", color_bands=color_bands)
            entry["BMAJ_match"] = max(float(entry["BMAJ_match"]), float(best["BMAJ"]))
        rows.append(entry)

    for j, brow in blue_df.iterrows():
        if j not in matched_blue:
            rows.append(
                _seed_row_from_band(
                    brow, "Blue", assoc_bands=assoc_bands, band_fields=band_fields
                )
            )
    return pd.DataFrame(rows)


def associate_band_into_metacatalog(
    meta_df: pd.DataFrame,
    band_df: pd.DataFrame,
    band: str,
    *,
    assoc_bands: Sequence[str] = ASSOC_BANDS,
    band_fields: Sequence[str] = BAND_FIELDS,
    color_bands: Sequence[str] = COLOR_BANDS,
) -> pd.DataFrame:
    """Cross-match one color band onto the current metacatalog; append unmatched band rows."""
    band_df = band_df.reset_index(drop=True)
    if meta_df.empty:
        return pd.DataFrame(
            [
                _seed_row_from_band(
                    brow, band, assoc_bands=assoc_bands, band_fields=band_fields
                )
                for _, brow in band_df.iterrows()
            ]
        )

    meta_df = meta_df.reset_index(drop=True)
    match_base = meta_df[["RA", "DEC"]].copy()
    match_base["BMAJ"] = meta_df["BMAJ_match"].to_numpy(dtype=float)
    hits, matched_band = _associate_catalogs(match_base, band_df)

    rows: list[dict] = []
    for i, mrow in meta_df.iterrows():
        entry = mrow.to_dict()
        band_hits = hits.get(i, [])
        if band_hits:
            sub = band_df.iloc[band_hits]
            best = _pick_median_flux_row(sub)
            _attach_band_columns(entry, best, band, len(band_hits), band_fields=band_fields)
            _update_bands_present(entry, band, color_bands=color_bands)
            entry["BMAJ_match"] = max(float(entry["BMAJ_match"]), float(best["BMAJ"]))
        rows.append(entry)

    for j, brow in band_df.iterrows():
        if j not in matched_band:
            rows.append(
                _seed_row_from_band(
                    brow, band, assoc_bands=assoc_bands, band_fields=band_fields
                )
            )
    return pd.DataFrame(rows)


def build_global_metacatalog(
    lst_merged: dict[str, pd.DataFrame],
    *,
    assoc_bands: Sequence[str] = ASSOC_BANDS,
    band_fields: Sequence[str] = BAND_FIELDS,
    color_bands: Sequence[str] = COLOR_BANDS,
) -> pd.DataFrame:
    """Fuse LST-merged per-band catalogs via sequential cross-matching.

    Parameters
    ----------
    lst_merged
        Mapping of band name → LST-merged catalog DataFrame. Must include
        ``Full`` and each name in *assoc_bands*.
    assoc_bands
        Color bands associated onto Full (default Blue, Green, Red).
    band_fields
        Per-band measurement columns copied into ``{field}_{band}``.
    color_bands
        Canonical band order for ``bands_present``.
    """
    seed_assoc = assoc_bands[0] if assoc_bands else "Blue"
    temp = merge_full_and_blue(
        lst_merged["Full"],
        lst_merged[seed_assoc],
        assoc_bands=assoc_bands,
        band_fields=band_fields,
        color_bands=color_bands,
    )
    for band in assoc_bands[1:]:
        temp = associate_band_into_metacatalog(
            temp,
            lst_merged[band],
            band,
            assoc_bands=assoc_bands,
            band_fields=band_fields,
            color_bands=color_bands,
        )
    meta = temp
    meta.insert(0, "meta_id", range(len(meta)))
    return meta.sort_values("Peak_flux", ascending=False, na_position="last").reset_index(
        drop=True
    )
