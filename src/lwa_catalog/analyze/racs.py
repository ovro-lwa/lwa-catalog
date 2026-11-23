"""ASKAP RACS Stokes I source-catalogue cross-match against LWA metacatalogs.

Supports three bands:

* ``RACSlow2`` — 887.5 MHz primary sources (``AS110.racs_low2_v01``)
* ``RACSmid`` — 1367.5 MHz **fixed 25″** auxiliary sources
  (``AS110.racs_mid_25arcsec_sources_v01``; Dec ≤ +30°)
* ``RACShigh`` — 1655.5 MHz primary sources (``AS110.racs_high_sources_v01``)

Download once (requires ``astroquery``; not needed for load/match)::

    python -m lwa_catalog.analyze.racs download --out /fast/claw/catalogs/racs

Do **not** use the low2 component table (``racs_low2_v02``) or the primary mid
sources table.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from lwa_catalog.analyze.crossmatch_radius import (
    LWA_CROSSMATCH_RADIUS_BEAM,
    RACS_REFERENCE_RADIUS_LOCALIZATION,
    CrossmatchRadiusSpec,
    apply_match_radius,
    catalog_match_frame,
)
from lwa_catalog.constants import (
    RACS_BANDS,
    RACS_BMAJ_ARCSEC,
    RACS_BMAJ_DEG,
    RACS_CATALOG_DIR,
    RACS_DEC_MAX_DEG,
    RACS_DEFAULT_PATHS,
    RACS_FREQ_HZ,
)
from lwa_catalog.create.merge import associate_catalogs

RacsBand = Literal["RACSlow2", "RACSmid", "RACShigh"]

CASDA_TAP_URL: str = "https://casda.csiro.au/casda_vo_tools/tap"

# Public CASDA TAP Stokes I **source** tables (not components).
RACS_TAP_TABLES: dict[str, str] = {
    "RACSlow2": "AS110.racs_low2_v01",
    "RACSmid": "AS110.racs_mid_25arcsec_sources_v01",
    "RACShigh": "AS110.racs_high_sources_v01",
}

RACS_PARQUET_NAMES: dict[str, str] = {
    "RACSlow2": "racs_low2_sources.parquet",
    "RACSmid": "racs_mid_25arcsec_sources.parquet",
    "RACShigh": "racs_high_sources.parquet",
}

_KEEP_COLUMNS: tuple[str, ...] = (
    "RA",
    "DEC",
    "E_RA",
    "E_DEC",
    "Total_flux",
    "E_Total_flux",
    "Peak_flux",
    "E_Peak_flux",
    "Maj",
    "Min",
    "PA",
    "BMAJ",
    "BMIN",
    "S_Code",
    "Flag",
    "source_file",
    "Source_ID",
    "Name",
)


def _validate_band(band: str) -> RacsBand:
    key = str(band).strip()
    if key not in RACS_BANDS:
        msg = f"unsupported RACS band: {band!r}; expected one of {RACS_BANDS}"
        raise ValueError(msg)
    return key  # type: ignore[return-value]


def default_path_for_band(band: str) -> Path:
    """Return the default local Parquet path for *band*."""
    return Path(RACS_DEFAULT_PATHS[_validate_band(band)])


@dataclass(frozen=True)
class RacsMatchConfig:
    """Configuration for :func:`match_catalog_to_racs`."""

    band: RacsBand = "RACSmid"
    catalog_path: Path | None = None
    dec_max_deg: float | None = None
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM
    reference_radius: CrossmatchRadiusSpec = RACS_REFERENCE_RADIUS_LOCALIZATION

    def resolved_path(self) -> Path:
        if self.catalog_path is not None:
            return Path(self.catalog_path)
        return default_path_for_band(self.band)

    def resolved_dec_max_deg(self) -> float:
        if self.dec_max_deg is not None:
            return float(self.dec_max_deg)
        return float(RACS_DEC_MAX_DEG[self.band])


@dataclass
class RacsMatchResult:
    """RACS cross-match metrics and per-row flags.

    ``meta_flags`` use shared ``n_racs`` / ``racs_positions`` column names for
    all three bands so :func:`~lwa_catalog.analyze.bootstrap.advance_from_match`
    can treat RACS uniformly via ``racs_flags`` / ``racs_footprint``.
    """

    band: str
    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    racs_flags: pd.DataFrame
    racs_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _first_present(columns: Sequence[str], aliases: Sequence[str]) -> str | None:
    colset = {c.lower(): c for c in columns}
    for name in aliases:
        hit = colset.get(name.lower())
        if hit is not None:
            return hit
    return None


def _series_from_aliases(df: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    name = _first_present(df.columns, aliases)
    if name is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def _string_from_aliases(df: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    name = _first_present(df.columns, aliases)
    if name is None:
        return pd.Series("", index=df.index, dtype=object)
    return df[name].astype(str)


def _finalize_racs_frame(df: pd.DataFrame, *, band: RacsBand) -> pd.DataFrame:
    """Normalize TAP / local columns to merge-ready Jy / degrees."""
    if df.empty:
        cols = [c for c in _KEEP_COLUMNS]
        return pd.DataFrame(columns=cols)

    ra = _series_from_aliases(df, ("RA", "ra"))
    # Prefer Dec_corr for cross-matching (RACS papers).
    dec = _series_from_aliases(df, ("Dec_corr", "dec_corr", "DEC", "Dec", "dec"))
    e_ra = _series_from_aliases(df, ("E_RA", "e_ra", "e_RA"))
    e_dec = _series_from_aliases(
        df, ("E_Dec_corr", "e_dec_corr", "E_DEC", "E_Dec", "e_dec", "e_DEC")
    )

    total = _series_from_aliases(df, ("Total_flux", "total_flux"))
    e_total = _series_from_aliases(df, ("E_Total_flux", "e_total_flux"))
    peak = _series_from_aliases(df, ("Peak_flux", "peak_flux"))
    e_peak = _series_from_aliases(df, ("E_Peak_flux", "e_peak_flux"))

    # CASDA TAP Stokes I source tables publish mJy (median Peak ~2–4 mJy).
    # Treat median > 0.5 as mJy; already-Jy frames (median ≪ 1) stay unchanged.
    def _mjy_to_jy(values: pd.Series) -> pd.Series:
        finite = values[np.isfinite(values.to_numpy(dtype=float))]
        if not finite.empty and float(finite.median()) > 0.5:
            return values / 1000.0
        return values

    total = _mjy_to_jy(total)
    e_total = _mjy_to_jy(e_total)
    peak = _mjy_to_jy(peak)
    e_peak = _mjy_to_jy(e_peak)

    maj = _series_from_aliases(df, ("Maj_axis", "maj_axis", "Maj", "maj"))
    min_ax = _series_from_aliases(df, ("Min_axis", "min_axis", "Min", "min"))
    pa = _series_from_aliases(df, ("PA", "pa"))
    # Shape axes from RACS are arcsec.
    for series_name, series in (("maj", maj), ("min", min_ax)):
        finite = series[np.isfinite(series.to_numpy(dtype=float))]
        if not finite.empty and float(finite.median()) > 0.1:
            if series_name == "maj":
                maj = series / 3600.0
            else:
                min_ax = series / 3600.0

    psf_maj = _series_from_aliases(df, ("PSF_Maj", "psf_maj", "BMAJ", "bmaj"))
    finite_psf = psf_maj[np.isfinite(psf_maj.to_numpy(dtype=float))]
    if finite_psf.empty:
        bmaj = pd.Series(RACS_BMAJ_DEG[band], index=df.index, dtype=float)
    elif float(finite_psf.median()) > 0.1:
        # arcsec → degrees
        bmaj = psf_maj / 3600.0
    else:
        bmaj = psf_maj
    bmaj = bmaj.fillna(RACS_BMAJ_DEG[band])

    psf_min = _series_from_aliases(df, ("PSF_Min", "psf_min", "BMIN", "bmin"))
    finite_psf_min = psf_min[np.isfinite(psf_min.to_numpy(dtype=float))]
    if finite_psf_min.empty:
        bmin = bmaj.copy()
    elif float(finite_psf_min.median()) > 0.1:
        bmin = psf_min / 3600.0
    else:
        bmin = psf_min
    bmin = bmin.fillna(bmaj)

    # Position errors: TAP E_RA / E_Dec are degrees.
    flag = _series_from_aliases(df, ("Flag", "flag"))
    s_code = _string_from_aliases(df, ("S_Code", "s_code"))
    source_file = _string_from_aliases(df, ("Name", "name", "source_file"))
    source_id = _string_from_aliases(df, ("Source_ID", "source_id"))
    name = _string_from_aliases(df, ("Name", "name"))

    out = pd.DataFrame(
        {
            "RA": ra,
            "DEC": dec,
            "E_RA": e_ra,
            "E_DEC": e_dec,
            "Total_flux": total,
            "E_Total_flux": e_total,
            "Peak_flux": peak,
            "E_Peak_flux": e_peak,
            "Maj": maj,
            "Min": min_ax,
            "PA": pa,
            "BMAJ": bmaj,
            "BMIN": bmin,
            "S_Code": s_code,
            "Flag": flag,
            "source_file": source_file,
            "Source_ID": source_id,
            "Name": name,
        }
    )

    if "Flag" in out.columns:
        flag_num = pd.to_numeric(out["Flag"], errors="coerce")
        # Drop only explicit Flag == 2 (spurious); keep missing / 0 / 1.
        keep_flag = ~((flag_num == 2) & np.isfinite(flag_num.to_numpy(dtype=float)))
        out = out.loc[keep_flag].copy()

    ra_ok = pd.to_numeric(out["RA"], errors="coerce")
    dec_ok = pd.to_numeric(out["DEC"], errors="coerce")
    ok = np.isfinite(ra_ok.to_numpy(dtype=float)) & np.isfinite(dec_ok.to_numpy(dtype=float))
    return out.loc[ok].reset_index(drop=True)


def load_racs_catalog(
    path: Path | str | None = None,
    *,
    band: str = "RACSmid",
) -> pd.DataFrame:
    """Load a local RACS Stokes I source catalogue for cross-matching.

    Accepts Parquet (preferred), CSV, or FITS. TAP dumps use mJy fluxes and
    lowercase columns; this normalizes to Jy, prefers ``Dec_corr`` for ``DEC``,
    sets ``BMAJ`` from ``PSF_Maj`` (arcsec→deg), and drops ``Flag == 2``.

    Parameters
    ----------
    path
        Catalog file. Defaults to :data:`~lwa_catalog.constants.RACS_DEFAULT_PATHS`
        for *band*.
    band
        ``RACSlow2``, ``RACSmid``, or ``RACShigh``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist (message includes the download command).
    ValueError
        If *band* is unsupported.
    """
    resolved_band = _validate_band(band)
    catalog_path = Path(
        default_path_for_band(resolved_band) if path is None else path
    )
    if not catalog_path.is_file():
        msg = (
            f"RACS catalog not found: {catalog_path}. "
            f"Download with: python -m lwa_catalog.analyze.racs download "
            f"--out {RACS_CATALOG_DIR}"
        )
        raise FileNotFoundError(msg)

    suffixes = "".join(catalog_path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        df = pd.read_parquet(catalog_path)
    elif suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        df = pd.read_csv(catalog_path)
    else:
        from astropy.table import Table

        df = Table.read(catalog_path).to_pandas()

    return _finalize_racs_frame(df, band=resolved_band)


def select_unique_racs_matches(meta_flags: pd.DataFrame) -> pd.DataFrame:
    """Return meta rows with exactly one RACS match (``n_racs == 1``)."""
    if meta_flags.empty or "n_racs" not in meta_flags.columns:
        return meta_flags.iloc[0:0].copy()
    return meta_flags.loc[meta_flags["n_racs"] == 1].copy()


def _footprint_filter_racs(
    racs: pd.DataFrame,
    lwa: pd.DataFrame,
    *,
    dec_max_deg: float,
) -> pd.DataFrame:
    """Keep RACS rows in the LWA Dec box and at/below the survey Dec ceiling."""
    if racs.empty:
        return racs.copy()
    if lwa.empty or "DEC" not in lwa.columns:
        return racs.iloc[0:0].copy()

    lwa_dec = pd.to_numeric(lwa["DEC"], errors="coerce")
    finite_lwa = lwa_dec[np.isfinite(lwa_dec.to_numpy(dtype=float))]
    if finite_lwa.empty:
        return racs.iloc[0:0].copy()

    dec_min = float(finite_lwa.min())
    dec_max = min(float(finite_lwa.max()), float(dec_max_deg))
    racs_dec = pd.to_numeric(racs["DEC"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(racs_dec) & (racs_dec >= dec_min) & (racs_dec <= dec_max)
    return racs.loc[keep].copy()


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_racs_footprint": 0,
        "n_meta_matched": 0,
        "match_completeness": float("nan"),
        "n_meta_unique": 0,
        "unique_match_fraction": float("nan"),
        "n_racs_matched": 0,
        "racs_recovery": float("nan"),
        "n_racs_oversplit": 0,
        "n_meta_multi_racs": 0,
        "meta_racs_hits_max": 0,
    }


def match_catalog_to_racs(
    lwa_catalog: pd.DataFrame,
    racs: pd.DataFrame | None = None,
    *,
    config: RacsMatchConfig | None = None,
    lwa_match: pd.DataFrame | None = None,
) -> RacsMatchResult:
    """Cross-match an LWA catalog against one RACS band and compute metrics.

    Matching uses :func:`~lwa_catalog.create.merge.associate_catalogs` with
    configured :class:`~lwa_catalog.analyze.crossmatch_radius.CrossmatchRadiusSpec`
    radii. The RACS footprint is the LWA Dec box clipped to the survey
    ``dec_max_deg`` (mid: +30°; low2/high: +49°).
    """
    cfg = config or RacsMatchConfig()
    band = _validate_band(cfg.band)
    warnings: list[str] = []

    if racs is None:
        racs = load_racs_catalog(cfg.resolved_path(), band=band)

    target = lwa_catalog
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty")
        return RacsMatchResult(
            band=band,
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=["meta_id", "RA", "DEC", "n_racs", "racs_positions", "matched"]
            ),
            racs_flags=pd.DataFrame(
                columns=[
                    "racs_pos",
                    "RA",
                    "DEC",
                    "Peak_flux",
                    "Total_flux",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            racs_footprint=pd.DataFrame(),
            warnings=warnings,
        )

    dec_max = cfg.resolved_dec_max_deg()
    racs_footprint = _footprint_filter_racs(racs, target, dec_max_deg=dec_max)
    ref_match = apply_match_radius(racs_footprint, cfg.reference_radius)
    if lwa_match is None:
        resolved_lwa_match = catalog_match_frame(target, cfg.lwa_radius)
    else:
        if len(lwa_match) != len(target):
            msg = (
                f"lwa_match length {len(lwa_match)} does not match target "
                f"length {len(target)}"
            )
            raise ValueError(msg)
        resolved_lwa_match = lwa_match[["RA", "DEC", "BMAJ"]].copy()
        resolved_lwa_match.index = target.index
    n_racs_footprint = len(racs_footprint)

    meta_hits: dict[int, list[int]] = {}
    racs_hits: dict[int, list[int]] = {}
    if not resolved_lwa_match.empty and not ref_match.empty:
        meta_hits, _ = associate_catalogs(resolved_lwa_match, ref_match)
        racs_hits, _ = associate_catalogs(ref_match, resolved_lwa_match)

    index_to_match_pos = {
        idx: pos for pos, idx in enumerate(resolved_lwa_match.index.tolist())
    }
    match_pos_to_index = {pos: idx for idx, pos in index_to_match_pos.items()}
    has_meta_id = "meta_id" in target.columns

    meta_records: list[dict] = []
    for idx, row in target.iterrows():
        match_pos = index_to_match_pos.get(idx)
        hit_racs = meta_hits.get(match_pos, []) if match_pos is not None else []
        n_racs = len(hit_racs)
        record: dict = {
            "RA": row.get("RA", np.nan),
            "DEC": row.get("DEC", np.nan),
            "n_racs": n_racs,
            "racs_positions": list(hit_racs),
            "matched": n_racs >= 1,
        }
        if "meta_id" in target.columns:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    if "meta_id" in meta_flags.columns:
        cols = ["meta_id", "RA", "DEC", "n_racs", "racs_positions", "matched"]
        meta_flags = meta_flags[cols]

    racs_records: list[dict] = []
    for pos in range(len(racs_footprint)):
        row = racs_footprint.iloc[pos]
        hit_meta = racs_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                idx = match_pos_to_index.get(match_pos)
                if idx is not None:
                    meta_ids.append(target.loc[idx, "meta_id"])
        racs_records.append(
            {
                "racs_pos": pos,
                "RA": row["RA"],
                "DEC": row["DEC"],
                "Peak_flux": row.get("Peak_flux", np.nan),
                "Total_flux": row.get("Total_flux", np.nan),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    racs_cols = [
        "racs_pos",
        "RA",
        "DEC",
        "Peak_flux",
        "Total_flux",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        racs_cols = [c for c in racs_cols if c != "meta_ids"]
    if racs_records:
        racs_flags = pd.DataFrame(racs_records)[racs_cols]
    else:
        racs_flags = pd.DataFrame(columns=racs_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_meta_unique = int((meta_flags["n_racs"] == 1).sum()) if not meta_flags.empty else 0
    n_racs_matched = int((racs_flags["n_meta"] > 0).sum()) if not racs_flags.empty else 0
    n_racs_oversplit = int(racs_flags["oversplit"].sum()) if not racs_flags.empty else 0
    n_meta_multi_racs = int((meta_flags["n_racs"] > 1).sum()) if not meta_flags.empty else 0
    meta_racs_hits_max = int(meta_flags["n_racs"].max()) if not meta_flags.empty else 0

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_racs_footprint": n_racs_footprint,
        "n_meta_matched": n_meta_matched,
        "match_completeness": n_meta_matched / n_lwa_target,
        "n_meta_unique": n_meta_unique,
        "unique_match_fraction": n_meta_unique / n_lwa_target,
        "n_racs_matched": n_racs_matched,
        "racs_recovery": (
            n_racs_matched / n_racs_footprint if n_racs_footprint else float("nan")
        ),
        "n_racs_oversplit": n_racs_oversplit,
        "n_meta_multi_racs": n_meta_multi_racs,
        "meta_racs_hits_max": meta_racs_hits_max,
    }

    return RacsMatchResult(
        band=band,
        summary=summary,
        meta_flags=meta_flags,
        racs_flags=racs_flags,
        racs_footprint=racs_footprint,
        warnings=warnings,
    )


def summarize_racs_match(result: RacsMatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    band = result.band
    freq_mhz = RACS_FREQ_HZ.get(band, float("nan")) / 1e6
    bmaj = RACS_BMAJ_ARCSEC.get(band, float("nan"))
    dec_max = RACS_DEC_MAX_DEG.get(band, float("nan"))
    lines = [
        f"RACS band:                     {band}",
        f"LWA target rows:               {int(s['n_lwa_target']):6d}",
        f"RACS footprint (Dec box):      {int(s['n_racs_footprint']):6d}",
        f"Meta matched (>=1 RACS):       {int(s['n_meta_matched']):6d}",
        f"Match completeness:            {s['match_completeness']:.3f}",
        f"Meta unique match (n=1):       {int(s['n_meta_unique']):6d}",
        f"Unique match fraction:         {s['unique_match_fraction']:.3f}",
        f"RACS matched (>=1 meta):       {int(s['n_racs_matched']):6d}",
        f"RACS recovery:                 {s['racs_recovery']:.3f}",
        f"RACS over-split (n_meta>1):    {int(s['n_racs_oversplit']):6d}",
        f"Meta multi-RACS (n>1):         {int(s['n_meta_multi_racs']):6d}",
        f"Max RACS hits per meta:        {int(s['meta_racs_hits_max']):6d}",
        "",
        f"RACS reference frequency:      {freq_mhz:.1f} MHz",
        f"RACS default PSF major:        {bmaj:.1f} arcsec",
        f"RACS Dec max:                  {dec_max:.1f} deg",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)


def download_racs_catalogs(
    out_dir: Path | str | None = None,
    *,
    bands: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Download RACS source tables from CASDA TAP to local Parquet files.

    Requires ``astroquery``. Skips existing files unless *force* is True.

    Returns
    -------
    dict
        Mapping of band key → written Parquet path.
    """
    try:
        from astroquery.utils.tap.core import TapPlus
    except ImportError as exc:
        msg = (
            "astroquery is required for RACS download. "
            "Install with: pip install astroquery"
        )
        raise ImportError(msg) from exc

    dest = Path(RACS_CATALOG_DIR if out_dir is None else out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    selected = tuple(RACS_BANDS if bands is None else bands)
    for band in selected:
        _validate_band(band)

    tap = TapPlus(url=CASDA_TAP_URL)
    written: dict[str, Path] = {}
    for band in selected:
        out_path = dest / RACS_PARQUET_NAMES[band]
        if out_path.is_file() and not force:
            written[band] = out_path
            continue
        table = RACS_TAP_TABLES[band]
        # Prefer a single async dump; fall back to Dec-strip chunks on failure.
        try:
            job = tap.launch_job_async(f"SELECT * FROM {table}")
            table_result = job.get_results()
            df = table_result.to_pandas()
        except Exception as exc:
            chunks: list[pd.DataFrame] = []
            # Cover Dec −90 … +50 in 10° strips (surveys stop near +49°).
            for dec_lo in range(-90, 50, 10):
                dec_hi = dec_lo + 10
                adql = (
                    f"SELECT * FROM {table} "
                    f"WHERE dec >= {dec_lo} AND dec < {dec_hi}"
                )
                job = tap.launch_job_async(adql)
                part = job.get_results().to_pandas()
                if len(part):
                    chunks.append(part)
            if not chunks:
                msg = f"CASDA TAP returned no rows for {table}"
                raise RuntimeError(msg) from exc
            df = pd.concat(chunks, ignore_index=True)
        df.to_parquet(out_path, index=False)
        written[band] = out_path
    return written


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lwa_catalog.analyze.racs",
        description="ASKAP RACS catalogue helpers (download / load paths).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser(
        "download",
        help="Download RACS Stokes I source tables from CASDA TAP to Parquet",
    )
    download.add_argument(
        "--out",
        type=Path,
        default=RACS_CATALOG_DIR,
        help=f"Output directory (default: {RACS_CATALOG_DIR})",
    )
    download.add_argument(
        "--bands",
        nargs="+",
        choices=list(RACS_BANDS),
        default=None,
        help="Subset of bands to download (default: all three)",
    )
    download.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Parquet files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``python -m lwa_catalog.analyze.racs``."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "download":
        try:
            written = download_racs_catalogs(
                args.out, bands=args.bands, force=args.force
            )
        except ImportError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        for band, path in written.items():
            n = len(pd.read_parquet(path))
            print(f"{band}: {n:,} rows → {path}")
        return 0
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

