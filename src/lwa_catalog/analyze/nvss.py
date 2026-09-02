"""NVSS cross-match against LWA metacatalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from astropy.table import Table

from lwa_catalog.analyze.reliability import resolve_bmaj
from lwa_catalog.analyze.vlssr import select_blue_associated_rows
from lwa_catalog.constants import (
    NVSS_BMAJ_DEG,
    NVSS_DEC_MIN_DEG,
    NVSS_DEFAULT_PATH,
    NVSS_FREQ_HZ,
)
from lwa_catalog.create.merge import associate_catalogs

NvssTarget = Literal["metacatalog", "metacatalog_blue"]

NVSS_FITS_COLUMNS: tuple[str, ...] = (
    "RA(2000)",
    "DEC(2000)",
    "PEAK INT",
    "MAJOR AX",
    "MINOR AX",
    "POSANGLE",
    "P FLUX",
    "FIELD",
)


@dataclass(frozen=True)
class NvssMatchConfig:
    """Configuration for :func:`match_catalog_to_nvss`."""

    catalog_path: Path = NVSS_DEFAULT_PATH
    target: NvssTarget = "metacatalog"
    dec_min_deg: float = NVSS_DEC_MIN_DEG


@dataclass
class NvssMatchResult:
    """NVSS cross-match metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    nvss_flags: pd.DataFrame
    nvss_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def load_nvss_catalog(path: Path | str | None = None) -> pd.DataFrame:
    """Load the NVSS FITS catalog for cross-matching.

    Expects the NRAO ``CATALOG.FIT`` table (see
    `NVSS at HEASARC <https://heasarc.gsfc.nasa.gov/w3browse/all/nvss.html>`_).
    Returned columns include ``RA``, ``DEC``, ``Peak_intensity`` (Jy/beam),
    deconvolved ``Maj``/``Min``/``PA`` (degrees on sky), ``Pol_flux`` (Jy),
  ``Field``, and circular ``BMAJ``/``BMIN`` set to the survey 45″ FWHM.
    Rows with non-finite ``RA``/``DEC`` are dropped.

    Parameters
    ----------
    path
        Catalog file path. Defaults to
        :data:`~lwa_catalog.constants.NVSS_DEFAULT_PATH`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required FITS columns are missing.
    """
    catalog_path = Path(NVSS_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = (
            f"NVSS catalog not found: {catalog_path}. "
            f"Download CATALOG.FIT into {NVSS_DEFAULT_PATH.parent} from "
            "ftp://ftp.cv.nrao.edu/nvss/CATALOG/CATALOG.FIT"
        )
        raise FileNotFoundError(msg)

    table = Table.read(catalog_path)
    missing = [col for col in NVSS_FITS_COLUMNS if col not in table.colnames]
    if missing:
        msg = f"NVSS FITS missing expected columns: {missing}"
        raise ValueError(msg)

    df = table[list(NVSS_FITS_COLUMNS)].to_pandas()
    df = df.rename(
        columns={
            "RA(2000)": "RA",
            "DEC(2000)": "DEC",
            "PEAK INT": "Peak_intensity",
            "MAJOR AX": "Maj",
            "MINOR AX": "Min",
            "POSANGLE": "PA",
            "P FLUX": "Pol_flux",
            "FIELD": "Field",
        }
    )
    df["BMAJ"] = NVSS_BMAJ_DEG
    df["BMIN"] = NVSS_BMAJ_DEG

    ra = pd.to_numeric(df["RA"], errors="coerce")
    dec = pd.to_numeric(df["DEC"], errors="coerce")
    ok = np.isfinite(ra.to_numpy(dtype=float)) & np.isfinite(dec.to_numpy(dtype=float))
    return df.loc[ok].reset_index(drop=True)


def select_unique_nvss_matches(meta_flags: pd.DataFrame) -> pd.DataFrame:
    """Return meta rows with exactly one NVSS match (``n_nvss == 1``)."""
    if meta_flags.empty or "n_nvss" not in meta_flags.columns:
        return meta_flags.iloc[0:0].copy()
    return meta_flags.loc[meta_flags["n_nvss"] == 1].copy()


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


def _footprint_filter_nvss(
    nvss: pd.DataFrame,
    lwa: pd.DataFrame,
    *,
    dec_min_deg: float,
) -> pd.DataFrame:
    """Keep NVSS rows in the LWA Dec box and above the NVSS survey limit."""
    if nvss.empty:
        return nvss.copy()
    if lwa.empty or "DEC" not in lwa.columns:
        return nvss.iloc[0:0].copy()

    lwa_dec = pd.to_numeric(lwa["DEC"], errors="coerce")
    finite_lwa = lwa_dec[np.isfinite(lwa_dec.to_numpy(dtype=float))]
    if finite_lwa.empty:
        return nvss.iloc[0:0].copy()

    dec_min = max(float(finite_lwa.min()), float(dec_min_deg))
    dec_max = float(finite_lwa.max())
    nvss_dec = pd.to_numeric(nvss["DEC"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(nvss_dec) & (nvss_dec >= dec_min) & (nvss_dec <= dec_max)
    return nvss.loc[keep].copy()


def _select_lwa_target(lwa_catalog: pd.DataFrame, target: NvssTarget) -> pd.DataFrame:
    if target == "metacatalog":
        return lwa_catalog
    if target == "metacatalog_blue":
        return select_blue_associated_rows(lwa_catalog)
    msg = f"unsupported target: {target!r}"
    raise ValueError(msg)


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_nvss_footprint": 0,
        "n_meta_matched": 0,
        "match_completeness": float("nan"),
        "n_meta_unique": 0,
        "unique_match_fraction": float("nan"),
        "n_nvss_matched": 0,
        "nvss_recovery": float("nan"),
        "n_nvss_oversplit": 0,
        "n_meta_multi_nvss": 0,
        "meta_nvss_hits_max": 0,
    }


def match_catalog_to_nvss(
    lwa_catalog: pd.DataFrame,
    nvss: pd.DataFrame | None = None,
    *,
    config: NvssMatchConfig | None = None,
) -> NvssMatchResult:
    """Cross-match an LWA catalog against NVSS and compute association metrics.

    Default target is the full input metacatalog (``config.target ==
    "metacatalog"``). Matching uses primary ``RA``/``DEC`` and beam radii via
    :func:`~lwa_catalog.create.merge.associate_catalogs`. NVSS sources outside
    the survey (Dec ``< config.dec_min_deg``, default −40°) are excluded from
    the footprint.

    Parameters
    ----------
    lwa_catalog
        Metacatalog table.
    nvss
        Pre-loaded NVSS catalog. Loaded from ``config.catalog_path`` when omitted.
    config
        Match configuration.

    Returns
    -------
    NvssMatchResult
        Summary fractions, per-meta flags, and per-NVSS flags.
    """
    cfg = config or NvssMatchConfig()
    warnings: list[str] = []

    if nvss is None:
        nvss = load_nvss_catalog(cfg.catalog_path)

    target = _select_lwa_target(lwa_catalog, cfg.target)
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty after band selection")
        return NvssMatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=["meta_id", "RA", "DEC", "n_nvss", "nvss_positions", "matched"]
            ),
            nvss_flags=pd.DataFrame(
                columns=[
                    "nvss_pos",
                    "RA",
                    "DEC",
                    "Peak_intensity",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            nvss_footprint=pd.DataFrame(),
            warnings=warnings,
        )

    nvss_footprint = _footprint_filter_nvss(nvss, target, dec_min_deg=cfg.dec_min_deg)
    lwa_match = _catalog_match_frame(target)
    n_nvss_footprint = len(nvss_footprint)

    meta_hits: dict[int, list[int]] = {}
    nvss_hits: dict[int, list[int]] = {}
    if not lwa_match.empty and not nvss_footprint.empty:
        meta_hits, _ = associate_catalogs(lwa_match, nvss_footprint)
        nvss_hits, _ = associate_catalogs(nvss_footprint, lwa_match)

    index_to_match_pos = {idx: pos for pos, idx in enumerate(lwa_match.index.tolist())}
    match_pos_to_index = {pos: idx for idx, pos in index_to_match_pos.items()}
    has_meta_id = "meta_id" in target.columns

    meta_records: list[dict] = []
    for idx, row in target.iterrows():
        match_pos = index_to_match_pos.get(idx)
        hit_nvss = meta_hits.get(match_pos, []) if match_pos is not None else []
        n_nvss = len(hit_nvss)
        record: dict = {
            "RA": row.get("RA", np.nan),
            "DEC": row.get("DEC", np.nan),
            "n_nvss": n_nvss,
            "nvss_positions": list(hit_nvss),
            "matched": n_nvss >= 1,
        }
        if "meta_id" in target.columns:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    if "meta_id" in meta_flags.columns:
        cols = ["meta_id", "RA", "DEC", "n_nvss", "nvss_positions", "matched"]
        meta_flags = meta_flags[cols]

    nvss_records: list[dict] = []
    for pos in range(len(nvss_footprint)):
        row = nvss_footprint.iloc[pos]
        hit_meta = nvss_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                idx = match_pos_to_index.get(match_pos)
                if idx is not None:
                    meta_ids.append(target.loc[idx, "meta_id"])
        nvss_records.append(
            {
                "nvss_pos": pos,
                "RA": row["RA"],
                "DEC": row["DEC"],
                "Peak_intensity": row.get("Peak_intensity", np.nan),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    nvss_cols = [
        "nvss_pos",
        "RA",
        "DEC",
        "Peak_intensity",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        nvss_cols = [c for c in nvss_cols if c != "meta_ids"]
    if nvss_records:
        nvss_flags = pd.DataFrame(nvss_records)[nvss_cols]
    else:
        nvss_flags = pd.DataFrame(columns=nvss_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_meta_unique = int((meta_flags["n_nvss"] == 1).sum()) if not meta_flags.empty else 0
    n_nvss_matched = int((nvss_flags["n_meta"] > 0).sum()) if not nvss_flags.empty else 0
    n_nvss_oversplit = int(nvss_flags["oversplit"].sum()) if not nvss_flags.empty else 0
    n_meta_multi_nvss = int((meta_flags["n_nvss"] > 1).sum()) if not meta_flags.empty else 0
    meta_nvss_hits_max = int(meta_flags["n_nvss"].max()) if not meta_flags.empty else 0

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_nvss_footprint": n_nvss_footprint,
        "n_meta_matched": n_meta_matched,
        "match_completeness": n_meta_matched / n_lwa_target,
        "n_meta_unique": n_meta_unique,
        "unique_match_fraction": n_meta_unique / n_lwa_target,
        "n_nvss_matched": n_nvss_matched,
        "nvss_recovery": (
            n_nvss_matched / n_nvss_footprint if n_nvss_footprint else float("nan")
        ),
        "n_nvss_oversplit": n_nvss_oversplit,
        "n_meta_multi_nvss": n_meta_multi_nvss,
        "meta_nvss_hits_max": meta_nvss_hits_max,
    }

    return NvssMatchResult(
        summary=summary,
        meta_flags=meta_flags,
        nvss_flags=nvss_flags,
        nvss_footprint=nvss_footprint,
        warnings=warnings,
    )


def predict_flux_at_frequency_hz(
    row: pd.Series,
    frequency_hz: float,
    *,
    flux_kind: str = "peak",
) -> float:
    """Extrapolate metacatalog flux to *frequency_hz* using Taylor ``spec_*`` coeffs.

    Returns NaN when spectral-fit columns are missing or invalid.
    """
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        return float("nan")

    prefix = "spec_peak" if flux_kind == "peak" else "spec_total"
    n_terms_col = f"{prefix}_n_terms"
    nu0_col = f"{prefix}_nu0_mhz"
    if n_terms_col not in row.index or nu0_col not in row.index:
        return float("nan")

    n_terms = int(pd.to_numeric(row.get(n_terms_col), errors="coerce") or 0)
    if n_terms < 2:
        return float("nan")

    nu0_mhz = pd.to_numeric(row.get(nu0_col), errors="coerce")
    if not np.isfinite(nu0_mhz) or float(nu0_mhz) <= 0.0:
        return float("nan")

    coeffs: list[float] = []
    for idx in range(n_terms):
        col = f"{prefix}_a{idx}"
        if col not in row.index:
            return float("nan")
        coeff = pd.to_numeric(row.get(col), errors="coerce")
        if not np.isfinite(coeff):
            return float("nan")
        coeffs.append(float(coeff))

    x = np.log(frequency_hz / (float(nu0_mhz) * 1e6))
    ln_s = sum(c * x**j for j, c in enumerate(coeffs))
    return float(np.exp(ln_s))


def summarize_nvss_match(result: NvssMatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:              {int(s['n_lwa_target']):6d}",
        f"NVSS footprint (Dec box):     {int(s['n_nvss_footprint']):6d}",
        f"Meta matched (>=1 NVSS):      {int(s['n_meta_matched']):6d}",
        f"Match completeness:           {s['match_completeness']:.3f}",
        f"Meta unique match (n=1):      {int(s['n_meta_unique']):6d}",
        f"Unique match fraction:        {s['unique_match_fraction']:.3f}",
        f"NVSS matched (>=1 meta):      {int(s['n_nvss_matched']):6d}",
        f"NVSS recovery:                {s['nvss_recovery']:.3f}",
        f"NVSS over-split (n_meta>1):   {int(s['n_nvss_oversplit']):6d}",
        f"Meta multi-NVSS (n_nvss>1):   {int(s['n_meta_multi_nvss']):6d}",
        f"Max NVSS hits per meta:       {int(s['meta_nvss_hits_max']):6d}",
        "",
        f"NVSS reference frequency:     {NVSS_FREQ_HZ / 1e9:.2f} GHz",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)
