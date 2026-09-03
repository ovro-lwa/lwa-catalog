"""NVSS cross-match against LWA metacatalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from lwa_catalog.analyze.crossmatch_radius import (
    CrossmatchRadiusSpec,
    LWA_CROSSMATCH_RADIUS_BEAM,
    NVSS_REFERENCE_RADIUS_LOCALIZATION,
    apply_match_radius,
    catalog_match_frame,
)
from lwa_catalog.analyze.vlssr import select_blue_associated_rows
from lwa_catalog.constants import (
    NVSS_BMAJ_DEG,
    NVSS_DEC_MIN_DEG,
    NVSS_DEFAULT_PATH,
    NVSS_FREQ_HZ,
)
from lwa_catalog.create.merge import associate_catalogs

NvssTarget = Literal["metacatalog", "metacatalog_blue"]

# VizieR VIII/65 fixed-width layout (0-based half-open), from ReadMe.vizier.
_NVSS_VIZIER_COLSPECS: list[tuple[int, int]] = [
    (0, 8),
    (9, 16),
    (17, 24),
    (25, 39),
    (40, 42),
    (43, 45),
    (46, 51),
    (52, 53),
    (53, 55),
    (56, 58),
    (59, 63),
    (64, 69),
    (70, 74),
    (75, 83),
    (84, 91),
    (92, 93),
    (93, 98),
    (99, 100),
    (100, 105),
    (106, 111),
    (112, 116),
    (117, 121),
    (122, 126),
    (127, 129),
    (130, 134),
    (135, 141),
    (142, 147),
    (148, 153),
    (154, 158),
]
_NVSS_VIZIER_NAMES: tuple[str, ...] = (
    "Field",
    "Xpos",
    "Ypos",
    "NVSS",
    "RAh",
    "RAm",
    "RAs",
    "DE-",
    "DEd",
    "DEm",
    "DEs",
    "e_RAs",
    "e_DEs",
    "S1.4",
    "e_S1.4",
    "l_MajAxis",
    "MajAxis",
    "l_MinAxis",
    "MinAxis",
    "PA",
    "e_MajAxis",
    "e_MinAxis",
    "e_PA",
    "f_resFlux",
    "resFlux",
    "polFlux",
    "polPA",
    "e_polFlux",
    "e_polPA",
)


@dataclass(frozen=True)
class NvssMatchConfig:
    """Configuration for :func:`match_catalog_to_nvss`."""

    catalog_path: Path = NVSS_DEFAULT_PATH
    target: NvssTarget = "metacatalog"
    dec_min_deg: float = NVSS_DEC_MIN_DEG
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM
    reference_radius: CrossmatchRadiusSpec = NVSS_REFERENCE_RADIUS_LOCALIZATION


@dataclass
class NvssMatchResult:
    """NVSS cross-match metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    nvss_flags: pd.DataFrame
    nvss_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _sexagesimal_to_deg(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rah = pd.to_numeric(raw["RAh"], errors="coerce")
    ram = pd.to_numeric(raw["RAm"], errors="coerce")
    ras = pd.to_numeric(raw["RAs"], errors="coerce")
    ded = pd.to_numeric(raw["DEd"], errors="coerce")
    dem = pd.to_numeric(raw["DEm"], errors="coerce")
    des = pd.to_numeric(raw["DEs"], errors="coerce")
    sign = np.where(raw["DE-"].astype(str).str.strip() == "-", -1.0, 1.0)
    ra = (15.0 * (rah + ram / 60.0 + ras / 3600.0)).to_numpy(dtype=float)
    dec = (sign * (ded + dem / 60.0 + des / 3600.0)).to_numpy(dtype=float)
    return ra, dec


def _finalize_nvss_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and drop non-finite coordinates."""
    out = df.copy()
    if "BMAJ" not in out.columns:
        out["BMAJ"] = NVSS_BMAJ_DEG
    if "BMIN" not in out.columns:
        out["BMIN"] = NVSS_BMAJ_DEG
    ra = pd.to_numeric(out["RA"], errors="coerce")
    dec = pd.to_numeric(out["DEC"], errors="coerce")
    ok = np.isfinite(ra.to_numpy(dtype=float)) & np.isfinite(dec.to_numpy(dtype=float))
    return out.loc[ok].reset_index(drop=True)


def _load_nvss_vizier_fwf(path: Path) -> pd.DataFrame:
    """Parse VizieR VIII/65 ``nvss.dat`` / ``.dat.gz`` fixed-width ASCII."""
    compression = "gzip" if path.suffix == ".gz" else None
    raw = pd.read_fwf(
        path,
        colspecs=_NVSS_VIZIER_COLSPECS,
        names=list(_NVSS_VIZIER_NAMES),
        compression=compression,
    )
    ra, dec = _sexagesimal_to_deg(raw)
    e_ras = pd.to_numeric(raw["e_RAs"], errors="coerce").to_numpy(dtype=float)
    e_des = pd.to_numeric(raw["e_DEs"], errors="coerce").to_numpy(dtype=float)
    # On-sky degrees: e_RAs is seconds of time; e_DEs is arcsec.
    e_ra = (e_ras * 15.0 * np.cos(np.deg2rad(dec))) / 3600.0
    e_dec = e_des / 3600.0

    s14_jy = pd.to_numeric(raw["S1.4"], errors="coerce").to_numpy(dtype=float) / 1000.0
    e_s14_jy = (
        pd.to_numeric(raw["e_S1.4"], errors="coerce").to_numpy(dtype=float) / 1000.0
    )
    maj_deg = pd.to_numeric(raw["MajAxis"], errors="coerce").to_numpy(dtype=float) / 3600.0
    min_deg = pd.to_numeric(raw["MinAxis"], errors="coerce").to_numpy(dtype=float) / 3600.0
    pa = pd.to_numeric(raw["PA"], errors="coerce").to_numpy(dtype=float)
    pol_jy = pd.to_numeric(raw["polFlux"], errors="coerce").to_numpy(dtype=float) / 1000.0

    return pd.DataFrame(
        {
            "RA": ra,
            "DEC": dec,
            "E_RA": e_ra,
            "E_DEC": e_dec,
            "Total_flux": s14_jy,
            # VIII/65 publishes integrated S1.4 only; keep Peak_intensity for
            # existing NVSS QA / attach code paths.
            "Peak_intensity": s14_jy,
            "E_Total_flux": e_s14_jy,
            "Maj": maj_deg,
            "Min": min_deg,
            "PA": pa,
            "Pol_flux": pol_jy,
            "Field": raw["Field"].astype(str).str.strip().to_numpy(),
            "NVSS": raw["NVSS"].astype(str).str.strip().to_numpy(),
            "BMAJ": np.full(len(raw), NVSS_BMAJ_DEG),
            "BMIN": np.full(len(raw), NVSS_BMAJ_DEG),
        }
    )


def load_nvss_catalog(path: Path | str | None = None) -> pd.DataFrame:
    """Load the NVSS catalog for cross-matching.

    Default path is the VizieR VIII/65 packaging converted to parquet
    (``nvss_vizier.parquet``), which includes per-source ``E_RA``/``E_DEC``
    (on-sky degrees). Also accepts the raw ``nvss_vizier.dat.gz`` / ``.dat``
    ASCII and a pre-normalized parquet/FITS/CSV with ``RA``/``DEC`` columns.

    Returned columns include ``RA``, ``DEC``, ``E_RA``, ``E_DEC``,
    ``Total_flux`` / ``Peak_intensity`` (Jy; VizieR integrated ``S1.4``),
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
        If required columns are missing from a pre-normalized table.
    """
    catalog_path = Path(NVSS_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = (
            f"NVSS catalog not found: {catalog_path}. "
            f"Download VizieR VIII/65 into {NVSS_DEFAULT_PATH.parent} "
            "(e.g. nvss_vizier.dat.gz from "
            "https://cdsarc.cds.unistra.fr/ftp/cats/VIII/65/) "
            "and convert/point NVSS_DEFAULT_PATH at the parquet or ASCII file."
        )
        raise FileNotFoundError(msg)

    suffixes = "".join(catalog_path.suffixes).lower()
    if suffixes.endswith(".dat") or suffixes.endswith(".dat.gz"):
        df = _load_nvss_vizier_fwf(catalog_path)
        return _finalize_nvss_frame(df)

    if suffixes.endswith(".parquet"):
        df = pd.read_parquet(catalog_path)
    elif suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        df = pd.read_csv(catalog_path)
    else:
        from astropy.table import Table

        df = Table.read(catalog_path).to_pandas()

    if "RA" not in df.columns or "DEC" not in df.columns:
        msg = (
            f"NVSS table missing RA/DEC columns: {catalog_path}. "
            "Expected a VizieR-derived parquet/ASCII or a pre-normalized table."
        )
        raise ValueError(msg)
    return _finalize_nvss_frame(df)


def select_unique_nvss_matches(meta_flags: pd.DataFrame) -> pd.DataFrame:
    """Return meta rows with exactly one NVSS match (``n_nvss == 1``)."""
    if meta_flags.empty or "n_nvss" not in meta_flags.columns:
        return meta_flags.iloc[0:0].copy()
    return meta_flags.loc[meta_flags["n_nvss"] == 1].copy()


def _catalog_match_frame(
    catalog: pd.DataFrame,
    spec: CrossmatchRadiusSpec,
) -> pd.DataFrame:
    return catalog_match_frame(catalog, spec)


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
    lwa_match: pd.DataFrame | None = None,
) -> NvssMatchResult:
    """Cross-match an LWA catalog against NVSS and compute association metrics.

    Default target is the full input metacatalog (``config.target ==
    "metacatalog"``). Matching uses primary ``RA``/``DEC`` and configured
    :class:`~lwa_catalog.analyze.crossmatch_radius.CrossmatchRadiusSpec`
    radii via :func:`~lwa_catalog.create.merge.associate_catalogs`. NVSS
    sources outside the survey (Dec ``< config.dec_min_deg``, default −40°)
    are excluded from the footprint.

    Parameters
    ----------
    lwa_catalog
        Metacatalog table.
    nvss
        Pre-loaded NVSS catalog. Loaded from ``config.catalog_path`` when omitted.
    config
        Match configuration.
    lwa_match
        Optional prebuilt match frame (``RA``/``DEC``/``BMAJ``), iloc-aligned
        with the selected target. Used for cascaded VLSSR→NVSS bootstrap.

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
    ref_match = apply_match_radius(nvss_footprint, cfg.reference_radius)
    if lwa_match is None:
        resolved_lwa_match = _catalog_match_frame(target, cfg.lwa_radius)
    else:
        if len(lwa_match) != len(target):
            msg = (
                f"lwa_match length {len(lwa_match)} does not match target "
                f"length {len(target)}"
            )
            raise ValueError(msg)
        resolved_lwa_match = lwa_match[["RA", "DEC", "BMAJ"]].copy()
        resolved_lwa_match.index = target.index
    n_nvss_footprint = len(nvss_footprint)

    meta_hits: dict[int, list[int]] = {}
    nvss_hits: dict[int, list[int]] = {}
    if not resolved_lwa_match.empty and not ref_match.empty:
        meta_hits, _ = associate_catalogs(resolved_lwa_match, ref_match)
        nvss_hits, _ = associate_catalogs(ref_match, resolved_lwa_match)

    index_to_match_pos = {
        idx: pos for pos, idx in enumerate(resolved_lwa_match.index.tolist())
    }
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
    flux_kind: str = "total",
) -> float:
    """Extrapolate metacatalog flux to *frequency_hz* using Taylor ``spec_*`` coeffs.

    Prefers ``spec_total_*`` by default; falls back to ``spec_peak_*`` when the
    requested kind is unavailable. Returns NaN when no usable fit exists.
    """
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        return float("nan")

    kinds = ("total", "peak") if flux_kind == "total" else ("peak", "total")
    for kind in kinds:
        prefix = "spec_peak" if kind == "peak" else "spec_total"
        n_terms_col = f"{prefix}_n_terms"
        nu0_col = f"{prefix}_nu0_mhz"
        if n_terms_col not in row.index or nu0_col not in row.index:
            continue

        n_terms = int(pd.to_numeric(row.get(n_terms_col), errors="coerce") or 0)
        if n_terms < 2:
            continue

        nu0_mhz = pd.to_numeric(row.get(nu0_col), errors="coerce")
        if not np.isfinite(nu0_mhz) or float(nu0_mhz) <= 0.0:
            continue

        coeffs: list[float] = []
        ok = True
        for idx in range(n_terms):
            col = f"{prefix}_a{idx}"
            if col not in row.index:
                ok = False
                break
            coeff = pd.to_numeric(row.get(col), errors="coerce")
            if not np.isfinite(coeff):
                ok = False
                break
            coeffs.append(float(coeff))
        if not ok:
            continue

        x = np.log(frequency_hz / (float(nu0_mhz) * 1e6))
        ln_s = sum(c * x**j for j, c in enumerate(coeffs))
        return float(np.exp(ln_s))

    return float("nan")


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
