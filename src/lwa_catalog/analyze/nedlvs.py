"""NED-LVS cross-match QA against LWA metacatalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.table import Table

from lwa_catalog.analyze.reliability import parse_bands_present, resolve_bmaj
from lwa_catalog.analyze.vlssr import select_blue_associated_rows
from lwa_catalog.constants import (
    NEDLVS_DEFAULT_BMAJ_ARCSEC,
    NEDLVS_DEFAULT_BMAJ_DEG,
    NEDLVS_DEFAULT_PATH,
    band_frequency_hz,
)
from lwa_catalog.create.merge import associate_catalogs
from lwa_catalog.io import read_table
from lwa_catalog.paths import CatalogLayout

NedlvsTarget = Literal["metacatalog", "metacatalog_blue", "lst_merged_blue"]
MetacatalogSelection = Literal[
    "full",
    "blue",
    "quality_all_clear",
    "query",
]

NEDLVS_LOAD_COLUMNS: tuple[str, ...] = (
    "objname",
    "ra",
    "dec",
    "objtype",
    "z",
    "DistMpc",
    "Diam",
    "Mstar",
    "SFR_hybrid",
    "SFR_W4",
)


@dataclass(frozen=True)
class NedlvsMatchConfig:
    """Configuration for :func:`match_catalog_to_nedlvs`."""

    catalog_path: Path = NEDLVS_DEFAULT_PATH
    target: NedlvsTarget = "metacatalog"
    default_bmaj_arcsec: float = NEDLVS_DEFAULT_BMAJ_ARCSEC


@dataclass
class NedlvsMatchResult:
    """NED-LVS cross-match QA metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    nedlvs_flags: pd.DataFrame
    nedlvs_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def select_metacatalog(
    metacatalog: pd.DataFrame,
    *,
    selection: MetacatalogSelection = "full",
    layout: CatalogLayout | None = None,
    query: str | None = None,
) -> pd.DataFrame:
    """Return a metacatalog subset before NED-LVS cross-matching.

    Parameters
    ----------
    metacatalog
        Full global metacatalog table.
    selection
        ``full`` — all rows; ``blue`` — rows with ``Blue`` in ``bands_present``;
        ``quality_all_clear`` — rows with ``quality_flag == 0`` from cached
        ``metacatalog_quality.parquet`` when *layout* is provided;
        ``query`` — pandas ``DataFrame.query`` on *metacatalog* (requires *query*).
    layout
        Catalog tree root for loading cached quality flags.
    query
        Pandas query string used when ``selection == "query"``.
    """
    if selection == "full":
        out = metacatalog
    elif selection == "blue":
        out = select_blue_associated_rows(metacatalog)
    elif selection == "quality_all_clear":
        if layout is None:
            msg = "layout is required for selection='quality_all_clear'"
            raise ValueError(msg)
        quality_path = layout.root / "metacatalog_quality.parquet"
        if not quality_path.is_file():
            msg = f"quality catalog not found: {quality_path}"
            raise FileNotFoundError(msg)
        quality = read_table(quality_path)
        if "quality_flag" not in quality.columns:
            msg = f"{quality_path} missing quality_flag column"
            raise ValueError(msg)
        clear_ids = quality.loc[quality["quality_flag"] == 0, "meta_id"]
        if "meta_id" not in metacatalog.columns:
            msg = "metacatalog missing meta_id column for quality join"
            raise ValueError(msg)
        out = metacatalog.loc[metacatalog["meta_id"].isin(clear_ids)]
    elif selection == "query":
        if not query:
            msg = "query is required for selection='query'"
            raise ValueError(msg)
        out = metacatalog.query(query, engine="python")
    else:
        msg = f"unsupported selection: {selection!r}"
        raise ValueError(msg)

    return out.reset_index(drop=True)


def resolve_highest_frequency_peak_flux(row: pd.Series) -> tuple[float, float, str]:
    """Return ``(peak_flux_jy, frequency_hz, band_name)`` for the highest-frequency band.

    Checks ``Peak_flux_{band}`` columns in descending frequency order, then falls
    back to primary ``Peak_flux`` when the highest-frequency band is ``origin_band``.
  """
    bands = parse_bands_present(row)
    if not bands:
        return float("nan"), float("nan"), ""

    ranked = sorted(bands, key=band_frequency_hz, reverse=True)
    for band in ranked:
        freq = band_frequency_hz(band)
        if not np.isfinite(freq) or freq <= 0.0:
            continue
        col = f"Peak_flux_{band}"
        if col in row.index:
            flux = pd.to_numeric(row[col], errors="coerce")
            if np.isfinite(flux) and float(flux) > 0.0:
                return float(flux), float(freq), band
        if band == row.get("origin_band") and "Peak_flux" in row.index:
            flux = pd.to_numeric(row["Peak_flux"], errors="coerce")
            if np.isfinite(flux) and float(flux) > 0.0:
                return float(flux), float(freq), band

    return float("nan"), float("nan"), ""


def radio_luminosity_nu(
    flux_jy: float | np.ndarray,
    z: float | np.ndarray,
    frequency_hz: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute monochromatic and ``nu*L_nu`` radio luminosities.

    Uses :data:`astropy.cosmology.Planck18` luminosity distance and observed
    frequency *frequency_hz*. Returns ``(L_nu, nu_L_nu)`` in erg/s/Hz and erg/s.
    """
    flux = np.asarray(flux_jy, dtype=float)
    redshift = np.asarray(z, dtype=float)
    freq = np.asarray(frequency_hz, dtype=float)

    d_l = Planck18.luminosity_distance(redshift)
    f_nu = flux * u.Jy
    l_nu_q = (4 * np.pi * d_l**2 * f_nu).to(u.erg / u.s / u.Hz)
    nu_l_nu_q = (freq * u.Hz * l_nu_q).to(u.erg / u.s)
    l_nu = np.atleast_1d(l_nu_q.value)
    nu_l_nu = np.atleast_1d(nu_l_nu_q.value)
    if l_nu.size == 1:
        return float(l_nu[0]), float(nu_l_nu[0])
    return l_nu, nu_l_nu


def _diam_to_bmaj_deg(diam_arcsec: np.ndarray, *, default_bmaj_deg: float) -> np.ndarray:
    """Convert NED-LVS ``Diam`` (major-axis arcsec) to match radius in degrees."""
    bmaj = np.full(diam_arcsec.shape, default_bmaj_deg, dtype=float)
    finite = np.isfinite(diam_arcsec) & (diam_arcsec > 0.0)
    bmaj[finite] = diam_arcsec[finite] / 3600.0
    return bmaj


def load_nedlvs_catalog(
    path: Path | str | None = None,
    *,
    default_bmaj_arcsec: float = NEDLVS_DEFAULT_BMAJ_ARCSEC,
) -> pd.DataFrame:
    """Load the NED-LVS FITS table for beam-radius cross-matching.

    Returns columns ``RA``, ``DEC``, ``BMAJ`` (degrees, from ``Diam`` when
    available), plus ``objname``, ``objtype``, ``z``, ``DistMpc``, ``Diam_arcsec``,
    ``Mstar``, ``SFR_hybrid``, and ``SFR_W4``. Rows with non-finite ``RA``/``DEC``
    are dropped.

    Parameters
    ----------
    path
        FITS file path. Defaults to
        :data:`~lwa_catalog.constants.NEDLVS_DEFAULT_PATH`.
    default_bmaj_arcsec
        Match radius used when ``Diam`` is missing or non-positive.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    catalog_path = Path(NEDLVS_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = f"NED-LVS catalog not found: {catalog_path}"
        raise FileNotFoundError(msg)

    table = Table.read(catalog_path)
    missing = [col for col in NEDLVS_LOAD_COLUMNS if col not in table.colnames]
    if missing:
        msg = f"NED-LVS FITS missing expected columns: {missing}"
        raise ValueError(msg)

    sub = table[NEDLVS_LOAD_COLUMNS]
    df = sub.to_pandas()
    df = df.rename(columns={"ra": "RA", "dec": "DEC"})

    ra = pd.to_numeric(df["RA"], errors="coerce")
    dec = pd.to_numeric(df["DEC"], errors="coerce")
    diam = pd.to_numeric(df["Diam"], errors="coerce").to_numpy(dtype=float)
    default_bmaj_deg = float(default_bmaj_arcsec) / 3600.0
    df["Diam_arcsec"] = diam
    df["BMAJ"] = _diam_to_bmaj_deg(diam, default_bmaj_deg=default_bmaj_deg)
    df = df.drop(columns=["Diam"])

    ok = np.isfinite(ra.to_numpy(dtype=float)) & np.isfinite(dec.to_numpy(dtype=float))
    return df.loc[ok].reset_index(drop=True)


def _footprint_filter_nedlvs(nedlvs: pd.DataFrame, lwa: pd.DataFrame) -> pd.DataFrame:
    """Keep NED-LVS rows whose Dec lies within the finite Dec range of *lwa*."""
    if nedlvs.empty:
        return nedlvs.copy()
    if lwa.empty or "DEC" not in lwa.columns:
        return nedlvs.iloc[0:0].copy()

    lwa_dec = pd.to_numeric(lwa["DEC"], errors="coerce")
    finite_lwa = lwa_dec[np.isfinite(lwa_dec.to_numpy(dtype=float))]
    if finite_lwa.empty:
        return nedlvs.iloc[0:0].copy()

    dec_min = float(finite_lwa.min())
    dec_max = float(finite_lwa.max())
    nedlvs_dec = pd.to_numeric(nedlvs["DEC"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(nedlvs_dec) & (nedlvs_dec >= dec_min) & (nedlvs_dec <= dec_max)
    return nedlvs.loc[keep].copy()


def _select_lwa_target(lwa_catalog: pd.DataFrame, target: NedlvsTarget) -> pd.DataFrame:
    if target in {"metacatalog", "lst_merged_blue"}:
        return lwa_catalog
    if target == "metacatalog_blue":
        return select_blue_associated_rows(lwa_catalog)
    msg = f"unsupported target: {target!r}"
    raise ValueError(msg)


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_nedlvs_footprint": 0,
        "n_meta_matched": 0,
        "meta_nedlvs_fraction": float("nan"),
        "n_nedlvs_matched": 0,
        "nedlvs_recovery": float("nan"),
        "n_nedlvs_oversplit": 0,
        "n_meta_multi_nedlvs": 0,
        "meta_nedlvs_hits_max": 0,
    }


def _pick_closest_nedlvs_position(
    meta_row: pd.Series,
    nedlvs_footprint: pd.DataFrame,
    positions: list[int],
) -> int | None:
    if not positions:
        return None
    if len(positions) == 1:
        return positions[0]

    try:
        meta_ra = float(meta_row["RA"])
        meta_dec = float(meta_row["DEC"])
    except (KeyError, TypeError, ValueError):
        return positions[0]
    if not np.isfinite(meta_ra) or not np.isfinite(meta_dec):
        return positions[0]

    meta_sc = SkyCoord(ra=meta_ra * u.deg, dec=meta_dec * u.deg)
    best_pos = positions[0]
    best_sep = float("inf")
    for pos in positions:
        nrow = nedlvs_footprint.iloc[pos]
        ned_ra = float(nrow["RA"])
        ned_dec = float(nrow["DEC"])
        if not np.isfinite(ned_ra) or not np.isfinite(ned_dec):
            continue
        sep = meta_sc.separation(SkyCoord(ra=ned_ra * u.deg, dec=ned_dec * u.deg)).deg
        if sep < best_sep:
            best_sep = sep
            best_pos = pos
    return best_pos


def _resolve_sfr(row: pd.Series) -> tuple[float, str]:
    for col in ("SFR_hybrid", "SFR_W4"):
        if col not in row.index:
            continue
        val = pd.to_numeric(row[col], errors="coerce")
        if np.isfinite(val) and float(val) > 0.0:
            return float(val), col
    return float("nan"), ""


def build_sfr_radio_luminosity_table(
    metacatalog: pd.DataFrame,
    nedlvs_footprint: pd.DataFrame,
    meta_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Pair cross-matched rows and compute radio ``nu*L_nu`` vs NED-LVS SFR.

    Uses the highest-frequency positive ``Peak_flux`` from each metacatalog row,
    NED-LVS redshift for luminosity distance, and ``SFR_hybrid`` with fallback to
    ``SFR_W4``. When multiple NED-LVS galaxies match one meta row, the closest
    angular separation is chosen.
    """
    if meta_flags.empty or "meta_id" not in meta_flags.columns:
        return pd.DataFrame(
            columns=[
                "meta_id",
                "objname",
                "z",
                "SFR",
                "SFR_column",
                "radio_band",
                "radio_freq_hz",
                "Peak_flux_jy",
                "L_nu_erg_s_hz",
                "nuL_nu_erg_s",
            ]
        )

    meta_by_id = metacatalog.set_index("meta_id", drop=False)
    records: list[dict] = []
    for _, flag in meta_flags.iterrows():
        if not bool(flag.get("matched", False)):
            continue
        positions = flag.get("nedlvs_positions", [])
        if not positions:
            continue
        meta_id = flag["meta_id"]
        if meta_id not in meta_by_id.index:
            continue
        meta_row = meta_by_id.loc[meta_id]
        if isinstance(meta_row, pd.DataFrame):
            meta_row = meta_row.iloc[0]

        ned_pos = _pick_closest_nedlvs_position(meta_row, nedlvs_footprint, list(positions))
        if ned_pos is None:
            continue
        ned_row = nedlvs_footprint.iloc[ned_pos]

        flux, freq_hz, band = resolve_highest_frequency_peak_flux(meta_row)
        z = pd.to_numeric(ned_row.get("z"), errors="coerce")
        sfr, sfr_col = _resolve_sfr(ned_row)
        if not np.isfinite(flux) or not np.isfinite(freq_hz) or not np.isfinite(z) or z <= 0:
            continue
        if not np.isfinite(sfr) or sfr <= 0:
            continue

        l_nu, nu_l_nu = radio_luminosity_nu(flux, float(z), freq_hz)
        records.append(
            {
                "meta_id": meta_id,
                "objname": ned_row.get("objname", ""),
                "z": float(z),
                "SFR": sfr,
                "SFR_column": sfr_col,
                "radio_band": band,
                "radio_freq_hz": freq_hz,
                "Peak_flux_jy": flux,
                "L_nu_erg_s_hz": float(l_nu),
                "nuL_nu_erg_s": float(nu_l_nu),
            }
        )

    return pd.DataFrame.from_records(records)


def match_catalog_to_nedlvs(
    lwa_catalog: pd.DataFrame,
    nedlvs: pd.DataFrame | None = None,
    *,
    config: NedlvsMatchConfig | None = None,
) -> NedlvsMatchResult:
    """Cross-match an LWA catalog against NED-LVS and compute QA metrics.

    Matching uses primary ``RA``/``DEC`` and beam radii via
    :func:`~lwa_catalog.create.merge.associate_catalogs`. NED-LVS rows use
    ``Diam`` (major-axis arcsec) as ``BMAJ`` when available.

    Parameters
    ----------
    lwa_catalog
        Metacatalog or LST-merged table (often pre-filtered via
        :func:`select_metacatalog`).
    nedlvs
        Pre-loaded NED-LVS catalog. Loaded from ``config.catalog_path`` when omitted.
    config
        Match configuration.

    Returns
    -------
    NedlvsMatchResult
        Summary fractions, per-meta flags, per-NED-LVS flags, and footprint table.
    """
    cfg = config or NedlvsMatchConfig()
    warnings: list[str] = []

    if nedlvs is None:
        nedlvs = load_nedlvs_catalog(
            cfg.catalog_path,
            default_bmaj_arcsec=cfg.default_bmaj_arcsec,
        )

    target = _select_lwa_target(lwa_catalog, cfg.target)
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty after band selection")
        return NedlvsMatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=["meta_id", "RA", "DEC", "n_nedlvs", "nedlvs_positions", "matched"]
            ),
            nedlvs_flags=pd.DataFrame(
                columns=[
                    "nedlvs_pos",
                    "RA",
                    "DEC",
                    "objname",
                    "DistMpc",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            nedlvs_footprint=pd.DataFrame(),
            warnings=warnings,
        )

    nedlvs_footprint = _footprint_filter_nedlvs(nedlvs, target)
    lwa_match = _catalog_match_frame(target)
    n_nedlvs_footprint = len(nedlvs_footprint)

    meta_hits: dict[int, list[int]] = {}
    nedlvs_hits: dict[int, list[int]] = {}
    if not lwa_match.empty and not nedlvs_footprint.empty:
        meta_hits, _ = associate_catalogs(lwa_match, nedlvs_footprint)
        nedlvs_hits, _ = associate_catalogs(nedlvs_footprint, lwa_match)

    index_to_match_pos = {idx: pos for pos, idx in enumerate(lwa_match.index.tolist())}
    match_pos_to_index = {pos: idx for idx, pos in index_to_match_pos.items()}
    has_meta_id = "meta_id" in target.columns

    meta_records: list[dict] = []
    for idx, row in target.iterrows():
        match_pos = index_to_match_pos.get(idx)
        hit_nedlvs = meta_hits.get(match_pos, []) if match_pos is not None else []
        n_nedlvs = len(hit_nedlvs)
        record: dict = {
            "RA": row.get("RA", np.nan),
            "DEC": row.get("DEC", np.nan),
            "n_nedlvs": n_nedlvs,
            "nedlvs_positions": list(hit_nedlvs),
            "matched": n_nedlvs >= 1,
        }
        if "meta_id" in target.columns:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    if "meta_id" in meta_flags.columns:
        cols = ["meta_id", "RA", "DEC", "n_nedlvs", "nedlvs_positions", "matched"]
        meta_flags = meta_flags[cols]

    nedlvs_records: list[dict] = []
    for pos in range(len(nedlvs_footprint)):
        row = nedlvs_footprint.iloc[pos]
        hit_meta = nedlvs_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                idx = match_pos_to_index.get(match_pos)
                if idx is not None:
                    meta_ids.append(target.loc[idx, "meta_id"])
        nedlvs_records.append(
            {
                "nedlvs_pos": pos,
                "RA": row["RA"],
                "DEC": row["DEC"],
                "objname": row.get("objname", ""),
                "DistMpc": row.get("DistMpc", np.nan),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    nedlvs_cols = [
        "nedlvs_pos",
        "RA",
        "DEC",
        "objname",
        "DistMpc",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        nedlvs_cols = [c for c in nedlvs_cols if c != "meta_ids"]
    if nedlvs_records:
        nedlvs_flags = pd.DataFrame(nedlvs_records)[nedlvs_cols]
    else:
        nedlvs_flags = pd.DataFrame(columns=nedlvs_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_nedlvs_matched = int((nedlvs_flags["n_meta"] > 0).sum()) if not nedlvs_flags.empty else 0
    n_nedlvs_oversplit = int(nedlvs_flags["oversplit"].sum()) if not nedlvs_flags.empty else 0
    n_meta_multi_nedlvs = int((meta_flags["n_nedlvs"] > 1).sum()) if not meta_flags.empty else 0
    meta_nedlvs_hits_max = int(meta_flags["n_nedlvs"].max()) if not meta_flags.empty else 0

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_nedlvs_footprint": n_nedlvs_footprint,
        "n_meta_matched": n_meta_matched,
        "meta_nedlvs_fraction": n_meta_matched / n_lwa_target,
        "n_nedlvs_matched": n_nedlvs_matched,
        "nedlvs_recovery": (
            n_nedlvs_matched / n_nedlvs_footprint if n_nedlvs_footprint else float("nan")
        ),
        "n_nedlvs_oversplit": n_nedlvs_oversplit,
        "n_meta_multi_nedlvs": n_meta_multi_nedlvs,
        "meta_nedlvs_hits_max": meta_nedlvs_hits_max,
    }

    return NedlvsMatchResult(
        summary=summary,
        meta_flags=meta_flags,
        nedlvs_flags=nedlvs_flags,
        nedlvs_footprint=nedlvs_footprint.reset_index(drop=True),
        warnings=warnings,
    )


def _catalog_match_frame(catalog: pd.DataFrame) -> pd.DataFrame:
    """Build ``RA`` / ``DEC`` / ``BMAJ`` columns for beam-radius matching."""
    records: list[dict[str, float]] = []
    indices: list[object] = []
    for idx, row in catalog.iterrows():
        try:
            ra = float(row["RA"])
            dec = float(row["DEC"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(ra) or not np.isfinite(dec):
            continue
        bmaj = resolve_bmaj(row)
        if not np.isfinite(bmaj):
            bmaj = 0.0
        records.append({"RA": ra, "DEC": dec, "BMAJ": bmaj})
        indices.append(idx)

    if not records:
        return pd.DataFrame(columns=["RA", "DEC", "BMAJ"])
    return pd.DataFrame(records, index=indices)


def summarize_nedlvs_match(result: NedlvsMatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:                {int(s['n_lwa_target']):6d}",
        f"NED-LVS footprint (Dec box):    {int(s['n_nedlvs_footprint']):6d}",
        f"Meta matched (>=1 NED-LVS):     {int(s['n_meta_matched']):6d}",
        f"Meta with NED-LVS host:         {s['meta_nedlvs_fraction']:.3f}",
        f"NED-LVS matched (>=1 meta):     {int(s['n_nedlvs_matched']):6d}",
        f"NED-LVS recovery:               {s['nedlvs_recovery']:.3f}",
        f"NED-LVS over-split (n_meta>1):  {int(s['n_nedlvs_oversplit']):6d}",
        f"Meta multi-NED-LVS (n>1):       {int(s['n_meta_multi_nedlvs']):6d}",
        f"Max NED-LVS hits per meta:      {int(s['meta_nedlvs_hits_max']):6d}",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)
