"""Radio Fundamental Catalogue (RFC) cross-match against LWA metacatalogs.

RFC provides VLBI absolute-astrometry positions of compact radio sources
(Petrov & Kovalev 2025). The main catalogue supplies positions / fluxes; the
companion cross-match table supplies NED ``type`` (and redshift / jet PA) used
for ``class_native`` on unique associations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from lwa_catalog.analyze.crossmatch_radius import (
    LWA_CROSSMATCH_RADIUS_BEAM,
    RFC_REFERENCE_RADIUS_LOCALIZATION,
    CrossmatchRadiusSpec,
    apply_match_radius,
    catalog_match_frame,
)
from lwa_catalog.constants import (
    RFC_CROSS_MATCH_DEFAULT_PATH,
    RFC_DEFAULT_PATH,
    RFC_POSITION_ERROR_DEFAULT_ARCSEC,
)
from lwa_catalog.create.merge import associate_catalogs

RfcTarget = Literal["metacatalog"]

# Main catalogue fixed-width colspecs (0-indexed half-open; ReadMe 1-indexed).
_RFC_CAT_COLSPECS: list[tuple[int, int]] = [
    (0, 14),  # Name
    (16, 24),  # Comnam
    (26, 28),  # RAh
    (29, 31),  # RAm
    (32, 41),  # RAs
    (42, 43),  # DE-
    (43, 45),  # DEd
    (46, 48),  # DEm
    (49, 57),  # DEs
    (59, 65),  # eRA (mas; no cos δ)
    (66, 72),  # eDE (mas)
    (104, 110),  # FsS
    (148, 154),  # FsX (short-baseline X-band; useful Peak proxy)
]
_RFC_CAT_NAMES: tuple[str, ...] = (
    "Name",
    "Comnam",
    "RAh",
    "RAm",
    "RAs",
    "DE_sign",
    "DEd",
    "DEm",
    "DEs",
    "eRA_mas",
    "eDE_mas",
    "FsS",
    "FsX",
)

# Cross-match table: Name + type / z / PA (ReadMe bytes).
_RFC_XM_COLSPECS: list[tuple[int, int]] = [
    (0, 14),  # Name
    (624, 630),  # PA (deg)
    (632, 638),  # z
    (661, 668),  # type (NED)
]
_RFC_XM_NAMES: tuple[str, ...] = ("Name", "PA_jet", "z", "type")

RFC_LOAD_COLUMNS: tuple[str, ...] = (
    "Name",
    "Comnam",
    "RA",
    "DEC",
    "E_RA",
    "E_DEC",
    "FsS",
    "FsX",
    "type",
    "z",
    "PA_jet",
)

# Null sentinel in RFC flux columns.
_RFC_FLUX_NULL = -9.99


@dataclass(frozen=True)
class RfcMatchConfig:
    """Configuration for :func:`match_catalog_to_rfc`."""

    catalog_path: Path = RFC_DEFAULT_PATH
    cross_match_path: Path = RFC_CROSS_MATCH_DEFAULT_PATH
    target: RfcTarget = "metacatalog"
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM
    reference_radius: CrossmatchRadiusSpec = RFC_REFERENCE_RADIUS_LOCALIZATION
    # When True, join NED ``type`` / ``z`` / ``PA_jet`` from the cross-match table.
    attach_cross_match: bool = True


@dataclass
class RfcMatchResult:
    """RFC cross-match metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    rfc_flags: pd.DataFrame
    rfc_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _strip_series(series: pd.Series) -> pd.Series:
    def _one(val: object) -> str:
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return ""
        if isinstance(val, (bytes, bytearray)):
            return bytes(val).decode("utf-8", errors="replace").strip()
        return str(val).strip()

    return series.map(_one)


def _sexagesimal_to_degrees(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert RFC sexagesimal RA/DEC columns to degrees."""
    rah = pd.to_numeric(df["RAh"], errors="coerce").to_numpy(dtype=float)
    ram = pd.to_numeric(df["RAm"], errors="coerce").to_numpy(dtype=float)
    ras = pd.to_numeric(df["RAs"], errors="coerce").to_numpy(dtype=float)
    ded = pd.to_numeric(df["DEd"], errors="coerce").to_numpy(dtype=float)
    dem = pd.to_numeric(df["DEm"], errors="coerce").to_numpy(dtype=float)
    des = pd.to_numeric(df["DEs"], errors="coerce").to_numpy(dtype=float)
    sign_raw = _strip_series(df["DE_sign"]).to_numpy()
    sign = np.where(np.asarray(sign_raw) == "-", -1.0, 1.0)

    ra_hours = rah + ram / 60.0 + ras / 3600.0
    dec_abs = ded + dem / 60.0 + des / 3600.0
    ra_deg = ra_hours * 15.0
    dec_deg = sign * dec_abs
    ok = (
        np.isfinite(ra_deg)
        & np.isfinite(dec_deg)
        & np.isfinite(rah)
        & np.isfinite(ram)
        & np.isfinite(ras)
        & np.isfinite(ded)
        & np.isfinite(dem)
        & np.isfinite(des)
    )
    ra_deg = np.where(ok, ra_deg, np.nan)
    dec_deg = np.where(ok, dec_deg, np.nan)
    return ra_deg, dec_deg


def _mas_to_deg(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    # mas → arcsec → degrees
    return vals / 1000.0 / 3600.0


def _null_flux(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals.mask(np.isclose(vals.to_numpy(dtype=float), _RFC_FLUX_NULL, atol=1e-6))


def _read_fwf_skipping_comments(
    path: Path,
    *,
    colspecs: list[tuple[int, int]],
    names: tuple[str, ...],
) -> pd.DataFrame:
    """Read a Petrov fixed-width table, skipping ``#`` comment / header lines."""
    return pd.read_fwf(
        path,
        colspecs=colspecs,
        names=list(names),
        comment="#",
        dtype=str,
        encoding="utf-8",
        encoding_errors="replace",
    )


def load_rfc_cross_match(path: Path | str | None = None) -> pd.DataFrame:
    """Load the RFC↔survey cross-match table (Name, type, z, PA_jet).

    Accepts the official Petrov fixed-width ``rfc_*_cross_match.txt``, or
    parquet/CSV with at least a ``Name`` column (and optional ``type`` / ``z`` /
    ``PA_jet``).

    Parameters
    ----------
    path
        Cross-match file. Defaults to
        :data:`~lwa_catalog.constants.RFC_CROSS_MATCH_DEFAULT_PATH`.

    Returns
    -------
    pandas.DataFrame
        One row per RFC ``Name`` with stripped ``type`` / numeric ``z`` /
        ``PA_jet``.
    """
    catalog_path = Path(RFC_CROSS_MATCH_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = f"RFC cross-match catalog not found: {catalog_path}"
        raise FileNotFoundError(msg)

    suffixes = "".join(catalog_path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        raw = pd.read_parquet(catalog_path)
    elif suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        raw = pd.read_csv(catalog_path)
    else:
        raw = _read_fwf_skipping_comments(
            catalog_path, colspecs=_RFC_XM_COLSPECS, names=_RFC_XM_NAMES
        )

    if "Name" not in raw.columns:
        msg = "RFC cross-match table missing Name column"
        raise ValueError(msg)

    out = pd.DataFrame(
        {
            "Name": _strip_series(raw["Name"]),
            "type": (
                _strip_series(raw["type"])
                if "type" in raw.columns
                else pd.Series([""] * len(raw))
            ),
            "z": (
                pd.to_numeric(raw["z"], errors="coerce")
                if "z" in raw.columns
                else pd.Series(np.nan, index=raw.index)
            ),
            "PA_jet": (
                pd.to_numeric(raw["PA_jet"], errors="coerce")
                if "PA_jet" in raw.columns
                else pd.Series(np.nan, index=raw.index)
            ),
        }
    )
    out = out.loc[out["Name"].str.len() > 0].drop_duplicates(subset=["Name"], keep="first")
    return out.reset_index(drop=True)


def _finalize_rfc_frame(
    df: pd.DataFrame,
    *,
    cross_match: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build RA/DEC/E_* columns and optionally join cross-match type/z/PA."""
    if "RA" in df.columns and "DEC" in df.columns:
        out = df.copy()
        out["RA"] = pd.to_numeric(out["RA"], errors="coerce")
        out["DEC"] = pd.to_numeric(out["DEC"], errors="coerce")
        if "Name" in out.columns:
            out["Name"] = _strip_series(out["Name"])
        if "Comnam" in out.columns:
            out["Comnam"] = _strip_series(out["Comnam"])
    else:
        required = {
            "Name",
            "RAh",
            "RAm",
            "RAs",
            "DE_sign",
            "DEd",
            "DEm",
            "DEs",
        }
        missing = sorted(required - set(df.columns))
        if missing:
            msg = f"RFC catalog missing required columns: {missing}"
            raise ValueError(msg)
        out = df.copy()
        out["Name"] = _strip_series(out["Name"])
        if "Comnam" in out.columns:
            out["Comnam"] = _strip_series(out["Comnam"])
        ra, dec = _sexagesimal_to_degrees(out)
        out["RA"] = ra
        out["DEC"] = dec

    if "eRA_mas" in out.columns and "eDE_mas" in out.columns:
        e_ra = _mas_to_deg(out["eRA_mas"])
        e_dec = _mas_to_deg(out["eDE_mas"])
    elif "E_RA" in out.columns and "E_DEC" in out.columns:
        e_ra = pd.to_numeric(out["E_RA"], errors="coerce").to_numpy(dtype=float)
        e_dec = pd.to_numeric(out["E_DEC"], errors="coerce").to_numpy(dtype=float)
    else:
        e_ra = np.full(len(out), np.nan)
        e_dec = np.full(len(out), np.nan)

    fallback_deg = RFC_POSITION_ERROR_DEFAULT_ARCSEC / 3600.0
    e_ra = np.where(np.isfinite(e_ra) & (e_ra > 0.0), e_ra, fallback_deg)
    e_dec = np.where(np.isfinite(e_dec) & (e_dec > 0.0), e_dec, fallback_deg)
    out["E_RA"] = e_ra
    out["E_DEC"] = e_dec

    for flux_col in ("FsS", "FsX"):
        if flux_col in out.columns:
            out[flux_col] = _null_flux(out[flux_col])

    if cross_match is not None and not cross_match.empty and "Name" in out.columns:
        xm = cross_match.drop_duplicates(subset=["Name"], keep="first")
        out = out.merge(xm, on="Name", how="left", suffixes=("", "_xm"))
        for col in ("type", "z", "PA_jet"):
            alt = f"{col}_xm"
            if alt in out.columns:
                if col not in out.columns:
                    out[col] = out[alt]
                else:
                    primary = out[col]
                    if col == "type":
                        blank = _strip_series(primary) == ""
                        out.loc[blank, col] = out.loc[blank, alt]
                    else:
                        miss = ~np.isfinite(pd.to_numeric(primary, errors="coerce"))
                        out.loc[miss, col] = out.loc[miss, alt]
                out = out.drop(columns=[alt])

    if "type" not in out.columns:
        out["type"] = ""
    else:
        out["type"] = _strip_series(out["type"])
    if "z" not in out.columns:
        out["z"] = np.nan
    if "PA_jet" not in out.columns:
        out["PA_jet"] = np.nan

    keep = [c for c in RFC_LOAD_COLUMNS if c in out.columns]
    out = out.loc[:, keep].copy()
    ra = out["RA"].to_numpy(dtype=float)
    dec = out["DEC"].to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return out.loc[ok].reset_index(drop=True)


def load_rfc_catalog(
    path: Path | str | None = None,
    *,
    cross_match_path: Path | str | None = None,
    attach_cross_match: bool = True,
) -> pd.DataFrame:
    """Load the RFC position catalogue for cross-matching.

    Accepts the official Petrov fixed-width ``rfc_*_cat.txt``, or parquet/CSV
    with ``RA``/``DEC`` (degrees) and ``Name``. When *attach_cross_match* is
    True, joins NED ``type`` / ``z`` / ``PA_jet`` from the companion cross-match
    table on ``Name``. Position errors ``eRA``/``eDE`` (mas) become ``E_RA`` /
    ``E_DEC`` (degrees) for localization matching.

    Parameters
    ----------
    path
        Main catalogue path. Defaults to
        :data:`~lwa_catalog.constants.RFC_DEFAULT_PATH`.
    cross_match_path
        Cross-match table path. Defaults to
        :data:`~lwa_catalog.constants.RFC_CROSS_MATCH_DEFAULT_PATH`.
    attach_cross_match
        Join classification columns from the cross-match table.
    """
    catalog_path = Path(RFC_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = f"RFC catalog not found: {catalog_path}"
        raise FileNotFoundError(msg)

    suffixes = "".join(catalog_path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        df = pd.read_parquet(catalog_path)
    elif suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        df = pd.read_csv(catalog_path)
    else:
        df = _read_fwf_skipping_comments(
            catalog_path, colspecs=_RFC_CAT_COLSPECS, names=_RFC_CAT_NAMES
        )

    cross_match: pd.DataFrame | None = None
    if attach_cross_match:
        xm_path = Path(
            RFC_CROSS_MATCH_DEFAULT_PATH if cross_match_path is None else cross_match_path
        )
        if xm_path.is_file():
            cross_match = load_rfc_cross_match(xm_path)
        elif cross_match_path is not None:
            msg = f"RFC cross-match catalog not found: {xm_path}"
            raise FileNotFoundError(msg)

    return _finalize_rfc_frame(df, cross_match=cross_match)


def select_unique_rfc_matches(meta_flags: pd.DataFrame) -> pd.DataFrame:
    """Return meta rows with exactly one RFC match (``n_rfc == 1``)."""
    if meta_flags.empty or "n_rfc" not in meta_flags.columns:
        return meta_flags.iloc[0:0].copy()
    return meta_flags.loc[meta_flags["n_rfc"] == 1].copy()


def rfc_type_counts(footprint: pd.DataFrame) -> pd.Series:
    """Value counts of NED ``type`` on an RFC footprint (blank → ``(none)``)."""
    if footprint is None or footprint.empty or "type" not in footprint.columns:
        return pd.Series(dtype="int64")
    labels = _strip_series(footprint["type"]).replace("", "(none)")
    return labels.value_counts(dropna=False)


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_rfc_footprint": 0,
        "n_meta_matched": 0,
        "match_completeness": float("nan"),
        "n_meta_unique": 0,
        "unique_match_fraction": float("nan"),
        "n_rfc_matched": 0,
        "rfc_recovery": float("nan"),
        "n_rfc_oversplit": 0,
        "n_meta_multi_rfc": 0,
        "meta_rfc_hits_max": 0,
        "n_unique_with_type": 0,
    }


def _footprint_filter_rfc(rfc: pd.DataFrame, lwa: pd.DataFrame) -> pd.DataFrame:
    """Keep RFC rows inside the LWA Dec range (with a small pad)."""
    if rfc.empty:
        return rfc.copy()
    if lwa.empty or "DEC" not in lwa.columns:
        return rfc.iloc[0:0].copy()

    lwa_dec = pd.to_numeric(lwa["DEC"], errors="coerce")
    finite_lwa = lwa_dec[np.isfinite(lwa_dec.to_numpy(dtype=float))]
    if finite_lwa.empty:
        return rfc.iloc[0:0].copy()

    pad = 1.0  # degrees
    dec_min = float(finite_lwa.min()) - pad
    dec_max = float(finite_lwa.max()) + pad
    rfc_dec = pd.to_numeric(rfc["DEC"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(rfc_dec) & (rfc_dec >= dec_min) & (rfc_dec <= dec_max)
    return rfc.loc[keep].copy()


def match_catalog_to_rfc(
    lwa_catalog: pd.DataFrame,
    rfc: pd.DataFrame | None = None,
    *,
    config: RfcMatchConfig | None = None,
    lwa_match: pd.DataFrame | None = None,
) -> RfcMatchResult:
    """Cross-match an LWA catalog against RFC and compute association metrics.

    Matching uses primary ``RA``/``DEC`` and configured
    :class:`~lwa_catalog.analyze.crossmatch_radius.CrossmatchRadiusSpec`
    radii via :func:`~lwa_catalog.create.merge.associate_catalogs`. RFC
    position errors are mas-level, so the gate is dominated by the LWA radius.

    Parameters
    ----------
    lwa_catalog
        Metacatalog table.
    rfc
        Pre-loaded RFC catalog (with optional ``type``). Loaded from
        ``config.catalog_path`` when omitted.
    config
        Match configuration.
    lwa_match
        Optional prebuilt match frame (``RA``/``DEC``/``BMAJ``), iloc-aligned
        with the selected target (cascaded bootstrap).
    """
    cfg = config or RfcMatchConfig()
    warnings: list[str] = []

    if rfc is None:
        rfc = load_rfc_catalog(
            cfg.catalog_path,
            cross_match_path=cfg.cross_match_path,
            attach_cross_match=cfg.attach_cross_match,
        )

    target = lwa_catalog
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty")
        return RfcMatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=["meta_id", "RA", "DEC", "n_rfc", "rfc_positions", "matched"]
            ),
            rfc_flags=pd.DataFrame(
                columns=[
                    "rfc_pos",
                    "Name",
                    "RA",
                    "DEC",
                    "type",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            rfc_footprint=pd.DataFrame(),
            warnings=warnings,
        )

    rfc_footprint = _footprint_filter_rfc(rfc, target)
    ref_match = apply_match_radius(rfc_footprint, cfg.reference_radius)
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
    n_rfc_footprint = len(rfc_footprint)

    meta_hits: dict[int, list[int]] = {}
    rfc_hits: dict[int, list[int]] = {}
    if not resolved_lwa_match.empty and not ref_match.empty:
        meta_hits, _ = associate_catalogs(resolved_lwa_match, ref_match)
        rfc_hits, _ = associate_catalogs(ref_match, resolved_lwa_match)

    index_to_match_pos = {
        idx: pos for pos, idx in enumerate(resolved_lwa_match.index.tolist())
    }
    match_pos_to_index = {pos: idx for idx, pos in index_to_match_pos.items()}
    has_meta_id = "meta_id" in target.columns

    meta_records: list[dict] = []
    for idx, row in target.iterrows():
        match_pos = index_to_match_pos.get(idx)
        hit_rfc = meta_hits.get(match_pos, []) if match_pos is not None else []
        n_rfc = len(hit_rfc)
        record: dict = {
            "RA": row.get("RA", np.nan),
            "DEC": row.get("DEC", np.nan),
            "n_rfc": n_rfc,
            "rfc_positions": list(hit_rfc),
            "matched": n_rfc >= 1,
        }
        if "meta_id" in target.columns:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    if "meta_id" in meta_flags.columns:
        cols = ["meta_id", "RA", "DEC", "n_rfc", "rfc_positions", "matched"]
        meta_flags = meta_flags[cols]

    rfc_records: list[dict] = []
    for pos in range(len(rfc_footprint)):
        row = rfc_footprint.iloc[pos]
        hit_meta = rfc_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                idx = match_pos_to_index.get(match_pos)
                if idx is not None:
                    meta_ids.append(target.loc[idx, "meta_id"])
        rfc_records.append(
            {
                "rfc_pos": pos,
                "Name": row.get("Name", ""),
                "RA": row["RA"],
                "DEC": row["DEC"],
                "type": row.get("type", ""),
                "FsX": row.get("FsX", np.nan),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    rfc_cols = [
        "rfc_pos",
        "Name",
        "RA",
        "DEC",
        "type",
        "FsX",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        rfc_cols = [c for c in rfc_cols if c != "meta_ids"]
    if rfc_records:
        rfc_flags = pd.DataFrame(rfc_records)[rfc_cols]
    else:
        rfc_flags = pd.DataFrame(columns=rfc_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_meta_unique = int((meta_flags["n_rfc"] == 1).sum()) if not meta_flags.empty else 0
    n_rfc_matched = int((rfc_flags["n_meta"] > 0).sum()) if not rfc_flags.empty else 0
    n_rfc_oversplit = int(rfc_flags["oversplit"].sum()) if not rfc_flags.empty else 0
    n_meta_multi_rfc = int((meta_flags["n_rfc"] > 1).sum()) if not meta_flags.empty else 0
    meta_rfc_hits_max = int(meta_flags["n_rfc"].max()) if not meta_flags.empty else 0

    n_unique_with_type = 0
    if n_meta_unique and not rfc_footprint.empty and "type" in rfc_footprint.columns:
        unique = select_unique_rfc_matches(meta_flags)
        for _, flag in unique.iterrows():
            positions = flag.get("rfc_positions", [])
            if isinstance(positions, (list, tuple)) and positions:
                pos = int(positions[0])
                if 0 <= pos < len(rfc_footprint):
                    t = str(rfc_footprint.iloc[pos].get("type", "") or "").strip()
                    if t:
                        n_unique_with_type += 1

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_rfc_footprint": n_rfc_footprint,
        "n_meta_matched": n_meta_matched,
        "match_completeness": n_meta_matched / n_lwa_target,
        "n_meta_unique": n_meta_unique,
        "unique_match_fraction": n_meta_unique / n_lwa_target,
        "n_rfc_matched": n_rfc_matched,
        "rfc_recovery": (
            n_rfc_matched / n_rfc_footprint if n_rfc_footprint else float("nan")
        ),
        "n_rfc_oversplit": n_rfc_oversplit,
        "n_meta_multi_rfc": n_meta_multi_rfc,
        "meta_rfc_hits_max": meta_rfc_hits_max,
        "n_unique_with_type": n_unique_with_type,
    }

    return RfcMatchResult(
        summary=summary,
        meta_flags=meta_flags,
        rfc_flags=rfc_flags,
        rfc_footprint=rfc_footprint,
        warnings=warnings,
    )


def summarize_rfc_match(result: RfcMatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:               {int(s['n_lwa_target']):6d}",
        f"RFC footprint (Dec box):       {int(s['n_rfc_footprint']):6d}",
        f"Meta matched (>=1 RFC):        {int(s['n_meta_matched']):6d}",
        f"Match completeness:            {s['match_completeness']:.3f}",
        f"Meta unique match (n=1):       {int(s['n_meta_unique']):6d}",
        f"Unique match fraction:         {s['unique_match_fraction']:.3f}",
        f"RFC matched (>=1 meta):        {int(s['n_rfc_matched']):6d}",
        f"RFC recovery:                  {s['rfc_recovery']:.3f}",
        f"RFC over-split (n_meta>1):     {int(s['n_rfc_oversplit']):6d}",
        f"Meta multi-RFC (n>1):          {int(s['n_meta_multi_rfc']):6d}",
        f"Max RFC hits per meta:         {int(s['meta_rfc_hits_max']):6d}",
        f"Unique matches with NED type:  {int(s['n_unique_with_type']):6d}",
        "",
        "RFC: VLBI compact-source positions (mas); classification from NED type.",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)


def verify_rfc_skycoord_roundtrip(df: pd.DataFrame, *, n: int = 5) -> float:
    """Return max |Δ| arcsec between Sexagesimal parse and Astropy SkyCoord.

    Utility for loader QA; not used on the hot match path.
    """
    if df.empty or not {"RAh", "RAm", "RAs", "DE_sign", "DEd", "DEm", "DEs"}.issubset(
        df.columns
    ):
        return float("nan")
    sample = df.head(n)
    ra_deg, dec_deg = _sexagesimal_to_degrees(sample)
    coords = []
    for _, row in sample.iterrows():
        sign = "-" if str(row["DE_sign"]).strip() == "-" else "+"
        coords.append(
            SkyCoord(
                f"{int(row['RAh'])}h{int(row['RAm'])}m{float(row['RAs'])}s",
                f"{sign}{int(row['DEd'])}d{int(row['DEm'])}m{float(row['DEs'])}s",
                frame="icrs",
            )
        )
    ref = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    other = SkyCoord(coords)
    return float(ref.separation(other).arcsec.max())
