"""MilliQUAS optical quasar/AGN cross-match against LWA metacatalogs."""

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
    MILLIQUAS_REFERENCE_RADIUS_FIXED,
    CrossmatchRadiusSpec,
    apply_match_radius,
    catalog_match_frame,
)
from lwa_catalog.constants import (
    MILLIQUAS_DEFAULT_PATH,
    MILLIQUAS_POSITION_ERROR_DEFAULT_ARCSEC,
)
from lwa_catalog.create.merge import associate_catalogs

MilliquasTarget = Literal["metacatalog"]

# Flesch MilliQUAS Note (3) primary optical classes (first letter in Type).
MILLIQUAS_CLASS_LABELS: dict[str, str] = {
    "Q": "QSO (type-I broad-line, core-dominated)",
    "A": "AGN (type-I Seyfert / host-dominated)",
    "B": "BL Lac",
    "K": "NLQSO (type-II narrow-line, core-dominated)",
    "N": "NLAGN (type-II Seyfert / host-dominated)",
    "S": "star / quasar-candidate photometry",
}

MILLIQUAS_LOAD_COLUMNS: tuple[str, ...] = (
    "RA",
    "DEC",
    "Name",
    "Type",
    "class_primary",
    "class_label",
    "has_radio",
    "has_xray",
    "has_double_lobes",
    "Rmag",
    "Bmag",
    "Comment",
    "R",
    "B",
    "z",
    "rName",
    "rz",
    "XName",
    "RName",
    "Lobe1",
    "Lobe2",
)

MILLIQUAS_ATTACH_COLUMNS: tuple[str, ...] = (
    "Name",
    "Type",
    "class_primary",
    "class_label",
    "has_radio",
    "has_xray",
    "has_double_lobes",
    "z",
    "Rmag",
    "Bmag",
    "Comment",
    "R",
    "B",
    "rName",
    "rz",
    "XName",
    "RName",
    "Lobe1",
    "Lobe2",
)


@dataclass(frozen=True)
class MilliquasMatchConfig:
    """Configuration for :func:`match_catalog_to_milliquas`."""

    catalog_path: Path = MILLIQUAS_DEFAULT_PATH
    target: MilliquasTarget = "metacatalog"
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM
    reference_radius: CrossmatchRadiusSpec = MILLIQUAS_REFERENCE_RADIUS_FIXED
    # Keep only rows whose primary class is in this set (``None`` keeps all).
    classes: tuple[str, ...] | None = None


@dataclass
class MilliquasMatchResult:
    """MilliQUAS cross-match metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    milliquas_flags: pd.DataFrame
    milliquas_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _decode_bytes(series: pd.Series) -> pd.Series:
    """Decode bytes/bytearray cells to stripped strings."""
    def _one(val: object) -> str:
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return ""
        if isinstance(val, (bytes, bytearray)):
            return bytes(val).decode("utf-8", errors="replace").strip()
        return str(val).strip()

    return series.map(_one)


def parse_milliquas_type(type_code: str) -> dict[str, object]:
    """Decode a MilliQUAS ``Type`` string into class / association fields.

    Parameters
    ----------
    type_code
        Raw ``Type`` field (e.g. ``\"QR2X\"``, ``\"A\"``, ``\"BX\"``).

    Returns
    -------
    dict
        ``class_primary``, ``class_label``, ``has_radio``, ``has_xray``,
        ``has_double_lobes``.
    """
    code = str(type_code or "").strip().upper()
    primary = ""
    for ch in code:
        if ch in MILLIQUAS_CLASS_LABELS:
            primary = ch
            break
    return {
        "class_primary": primary,
        "class_label": MILLIQUAS_CLASS_LABELS.get(primary, ""),
        "has_radio": "R" in code,
        "has_xray": "X" in code,
        "has_double_lobes": "2" in code,
    }


def _finalize_milliquas_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize MilliQUAS columns, decode Type classes, drop bad coords."""
    rename = {}
    if "RAdeg" in df.columns and "RA" not in df.columns:
        rename["RAdeg"] = "RA"
    if "DEdeg" in df.columns and "DEC" not in df.columns:
        rename["DEdeg"] = "DEC"
    out = df.rename(columns=rename)

    missing = [c for c in ("RA", "DEC") if c not in out.columns]
    if missing:
        msg = f"MilliQUAS table missing required columns: {missing}"
        raise ValueError(msg)

    for col in ("Name", "Type", "Comment", "R", "B", "rName", "rz", "XName", "RName", "Lobe1", "Lobe2"):
        if col in out.columns:
            out[col] = _decode_bytes(out[col])

    for col in ("RA", "DEC", "Rmag", "Bmag", "z"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    type_series = out["Type"] if "Type" in out.columns else pd.Series([""] * len(out))
    parsed = type_series.map(parse_milliquas_type)
    out["class_primary"] = parsed.map(lambda d: d["class_primary"])
    out["class_label"] = parsed.map(lambda d: d["class_label"])
    out["has_radio"] = parsed.map(lambda d: d["has_radio"]).astype(bool)
    out["has_xray"] = parsed.map(lambda d: d["has_xray"]).astype(bool)
    out["has_double_lobes"] = parsed.map(lambda d: d["has_double_lobes"]).astype(bool)

    # Fixed optical localization for radius-spec matching / overlays.
    sigma_deg = MILLIQUAS_POSITION_ERROR_DEFAULT_ARCSEC / 3600.0
    axis_sigma = sigma_deg / np.sqrt(2.0)
    out["E_RA"] = axis_sigma
    out["E_DEC"] = axis_sigma
    out["POS_ERR"] = float(MILLIQUAS_POSITION_ERROR_DEFAULT_ARCSEC)

    keep = [c for c in MILLIQUAS_LOAD_COLUMNS if c in out.columns]
    # Retain localization helpers used by match frames / overlays.
    for extra in ("E_RA", "E_DEC", "POS_ERR"):
        if extra in out.columns and extra not in keep:
            keep.append(extra)
    out = out.loc[:, keep].copy()

    ra = out["RA"].to_numpy(dtype=float)
    dec = out["DEC"].to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return out.loc[ok].reset_index(drop=True)


def load_milliquas_catalog(path: Path | str | None = None) -> pd.DataFrame:
    """Load the MilliQUAS FITS table and decode optical class columns.

    The operator file ``milliquas.dat`` is a FITS binary table despite the
    ``.dat`` suffix. ``RAdeg``/``DEdeg`` become ``RA``/``DEC``. ``Type`` is
    parsed into ``class_primary`` / ``class_label`` plus radio/X-ray/lobe
    association flags.

    Parameters
    ----------
    path
        Catalog path. Defaults to
        :data:`~lwa_catalog.constants.MILLIQUAS_DEFAULT_PATH`.
    """
    catalog_path = Path(MILLIQUAS_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = f"MilliQUAS catalog not found: {catalog_path}"
        raise FileNotFoundError(msg)

    from astropy.table import Table

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = Table.read(catalog_path)
    return _finalize_milliquas_frame(table.to_pandas())


def filter_milliquas_by_class(
    catalog: pd.DataFrame,
    classes: tuple[str, ...] | None,
) -> pd.DataFrame:
    """Keep rows whose ``class_primary`` is in *classes* (``None`` keeps all)."""
    if catalog.empty or not classes:
        return catalog.copy()
    wanted = {str(c).strip().upper() for c in classes}
    if "class_primary" not in catalog.columns:
        return catalog.copy()
    cls = catalog["class_primary"].astype(str).str.strip().str.upper()
    return catalog.loc[cls.isin(wanted)].reset_index(drop=True)


def select_unique_milliquas_matches(meta_flags: pd.DataFrame) -> pd.DataFrame:
    """Return meta rows with exactly one MilliQUAS match."""
    if meta_flags.empty or "n_milliquas" not in meta_flags.columns:
        return meta_flags.iloc[0:0].copy()
    return meta_flags.loc[meta_flags["n_milliquas"] == 1].copy()


def milliquas_class_counts(meta_attached: pd.DataFrame) -> pd.Series:
    """Value counts of ``MilliQUAS_class_primary`` (or unprefixed) labels."""
    for col in ("MilliQUAS_class_primary", "class_primary"):
        if col in meta_attached.columns:
            series = meta_attached[col].astype(str).str.strip()
            series = series.loc[series.ne("") & series.ne("nan")]
            return series.value_counts().sort_index()
    return pd.Series(dtype=int)


def _catalog_match_frame(
    catalog: pd.DataFrame,
    spec: CrossmatchRadiusSpec,
) -> pd.DataFrame:
    return catalog_match_frame(catalog, spec)


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_milliquas_footprint": 0,
        "n_meta_matched": 0,
        "match_completeness": float("nan"),
        "n_meta_unique": 0,
        "unique_match_fraction": float("nan"),
        "n_milliquas_matched": 0,
        "milliquas_recovery": float("nan"),
        "n_milliquas_oversplit": 0,
        "n_meta_multi_milliquas": 0,
        "meta_milliquas_hits_max": 0,
    }


def match_catalog_to_milliquas(
    lwa_catalog: pd.DataFrame,
    milliquas: pd.DataFrame | None = None,
    *,
    config: MilliquasMatchConfig | None = None,
    lwa_match: pd.DataFrame | None = None,
) -> MilliquasMatchResult:
    """Cross-match an LWA catalog against MilliQUAS optical AGN/quasars.

    Matching uses ``associate_catalogs`` with configured
    :class:`~lwa_catalog.analyze.crossmatch_radius.CrossmatchRadiusSpec`
    radii. The reference side defaults to a fixed 1″ optical radius; the LWA
    side defaults to beam mode, so the LWA ``BMAJ`` typically sets the gate.
    """
    cfg = config or MilliquasMatchConfig()
    warnings: list[str] = []

    if milliquas is None:
        milliquas = load_milliquas_catalog(cfg.catalog_path)

    n_loaded = len(milliquas)
    footprint = filter_milliquas_by_class(milliquas, cfg.classes)
    if len(footprint) < n_loaded:
        warnings.append(
            f"Class filter kept {len(footprint):,} / {n_loaded:,} MilliQUAS rows "
            f"(classes={cfg.classes})"
        )

    target = lwa_catalog
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty")
        return MilliquasMatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=[
                    "meta_id",
                    "RA",
                    "DEC",
                    "n_milliquas",
                    "milliquas_positions",
                    "matched",
                ]
            ),
            milliquas_flags=pd.DataFrame(
                columns=[
                    "milliquas_pos",
                    "RA",
                    "DEC",
                    "Name",
                    "Type",
                    "class_primary",
                    "z",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            milliquas_footprint=pd.DataFrame(),
            warnings=warnings,
        )

    ref_match = apply_match_radius(footprint, cfg.reference_radius)
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

    n_milliquas_footprint = len(footprint)

    meta_hits: dict[int, list[int]] = {}
    mq_hits: dict[int, list[int]] = {}
    if not resolved_lwa_match.empty and not ref_match.empty:
        meta_hits, _ = associate_catalogs(resolved_lwa_match, ref_match)
        mq_hits, _ = associate_catalogs(ref_match, resolved_lwa_match)

    index_to_match_pos = {
        idx: pos for pos, idx in enumerate(resolved_lwa_match.index.tolist())
    }
    match_pos_to_index = {pos: idx for idx, pos in index_to_match_pos.items()}
    has_meta_id = "meta_id" in target.columns

    meta_records: list[dict] = []
    for idx, row in target.iterrows():
        match_pos = index_to_match_pos.get(idx)
        hit_ref = meta_hits.get(match_pos, []) if match_pos is not None else []
        n_hit = len(hit_ref)
        record: dict = {
            "RA": row.get("RA", np.nan),
            "DEC": row.get("DEC", np.nan),
            "n_milliquas": n_hit,
            "milliquas_positions": list(hit_ref),
            "matched": n_hit >= 1,
        }
        if has_meta_id:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    cols = ["meta_id", "RA", "DEC", "n_milliquas", "milliquas_positions", "matched"]
    if "meta_id" not in meta_flags.columns:
        cols = [c for c in cols if c != "meta_id"]
    if not meta_flags.empty:
        meta_flags = meta_flags[cols]

    mq_records: list[dict] = []
    for pos in range(len(footprint)):
        row = footprint.iloc[pos]
        hit_meta = mq_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                midx = match_pos_to_index.get(match_pos)
                if midx is not None:
                    meta_ids.append(target.loc[midx, "meta_id"])
        mq_records.append(
            {
                "milliquas_pos": pos,
                "RA": row["RA"],
                "DEC": row["DEC"],
                "Name": row.get("Name", ""),
                "Type": row.get("Type", ""),
                "class_primary": row.get("class_primary", ""),
                "z": row.get("z", np.nan),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    mq_cols = [
        "milliquas_pos",
        "RA",
        "DEC",
        "Name",
        "Type",
        "class_primary",
        "z",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        mq_cols = [c for c in mq_cols if c != "meta_ids"]
    if mq_records:
        milliquas_flags = pd.DataFrame(mq_records)[mq_cols]
    else:
        milliquas_flags = pd.DataFrame(columns=mq_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_meta_unique = (
        int((meta_flags["n_milliquas"] == 1).sum()) if not meta_flags.empty else 0
    )
    n_mq_matched = (
        int((milliquas_flags["n_meta"] > 0).sum()) if not milliquas_flags.empty else 0
    )
    n_mq_oversplit = (
        int(milliquas_flags["oversplit"].sum()) if not milliquas_flags.empty else 0
    )
    n_meta_multi = (
        int((meta_flags["n_milliquas"] > 1).sum()) if not meta_flags.empty else 0
    )
    meta_hits_max = (
        int(meta_flags["n_milliquas"].max()) if not meta_flags.empty else 0
    )

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_milliquas_footprint": n_milliquas_footprint,
        "n_meta_matched": n_meta_matched,
        "match_completeness": n_meta_matched / n_lwa_target,
        "n_meta_unique": n_meta_unique,
        "unique_match_fraction": n_meta_unique / n_lwa_target,
        "n_milliquas_matched": n_mq_matched,
        "milliquas_recovery": (
            n_mq_matched / n_milliquas_footprint
            if n_milliquas_footprint
            else float("nan")
        ),
        "n_milliquas_oversplit": n_mq_oversplit,
        "n_meta_multi_milliquas": n_meta_multi,
        "meta_milliquas_hits_max": meta_hits_max,
    }

    return MilliquasMatchResult(
        summary=summary,
        meta_flags=meta_flags,
        milliquas_flags=milliquas_flags,
        milliquas_footprint=footprint,
        warnings=warnings,
    )


def summarize_milliquas_match(result: MilliquasMatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:                  {int(s['n_lwa_target']):6d}",
        f"MilliQUAS footprint:              {int(s['n_milliquas_footprint']):6d}",
        f"Meta matched (>=1 MilliQUAS):     {int(s['n_meta_matched']):6d}",
        f"Match completeness:               {s['match_completeness']:.3f}",
        f"Meta unique match (n=1):          {int(s['n_meta_unique']):6d}",
        f"Unique match fraction:            {s['unique_match_fraction']:.3f}",
        f"MilliQUAS matched (>=1 meta):     {int(s['n_milliquas_matched']):6d}",
        f"MilliQUAS recovery:               {s['milliquas_recovery']:.3f}",
        f"MilliQUAS over-split (n>1):       {int(s['n_milliquas_oversplit']):6d}",
        f"Meta multi-MilliQUAS (n>1):       {int(s['n_meta_multi_milliquas']):6d}",
        f"Max MilliQUAS hits per meta:      {int(s['meta_milliquas_hits_max']):6d}",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)


def _pick_milliquas_hit(
    milliquas_footprint: pd.DataFrame,
    hit_positions: list[int],
    *,
    meta_ra: float,
    meta_dec: float,
) -> int | None:
    """Choose one MilliQUAS hit: prefer radio-associated, then closest on sky."""
    if not hit_positions:
        return None
    if len(hit_positions) == 1:
        return int(hit_positions[0])

    rows = milliquas_footprint.iloc[hit_positions]
    radio = rows["has_radio"].to_numpy(dtype=bool) if "has_radio" in rows.columns else None
    candidates = list(hit_positions)
    if radio is not None and np.any(radio):
        candidates = [hit_positions[i] for i, flag in enumerate(radio) if flag]

    meta = SkyCoord(ra=meta_ra * u.deg, dec=meta_dec * u.deg)
    best_pos = candidates[0]
    best_sep = np.inf
    for pos in candidates:
        row = milliquas_footprint.iloc[pos]
        sep = meta.separation(
            SkyCoord(ra=float(row["RA"]) * u.deg, dec=float(row["DEC"]) * u.deg)
        ).deg
        if sep < best_sep:
            best_sep = sep
            best_pos = pos
    return int(best_pos)


def attach_milliquas_to_metacatalog(
    meta_df: pd.DataFrame,
    milliquas: pd.DataFrame,
    meta_flags: pd.DataFrame,
    *,
    column_prefix: str = "MilliQUAS_",
) -> pd.DataFrame:
    """Attach MilliQUAS class / photometry columns for matched metacatalog rows.

    Row count is unchanged. Matched rows (``n_milliquas >= 1``) receive the
    chosen counterpart's :data:`MILLIQUAS_ATTACH_COLUMNS` under *column_prefix*,
    plus ``n_milliquas`` and ``sep_arcsec_MilliQUAS``. Multi-match rows prefer a
    radio-associated hit, then closest on sky. Unmatched rows get NaN / empty
    defaults.
    """
    if len(meta_flags) != len(meta_df):
        msg = (
            f"meta_flags length {len(meta_flags)} does not match meta_df "
            f"length {len(meta_df)}"
        )
        raise ValueError(msg)

    out = meta_df.copy()
    n = len(out)
    out["n_milliquas"] = (
        pd.to_numeric(meta_flags["n_milliquas"], errors="coerce").to_numpy(dtype=float)
        if "n_milliquas" in meta_flags.columns
        else np.zeros(n, dtype=float)
    )
    sep_col = f"sep_arcsec_{column_prefix.rstrip('_')}"
    out[sep_col] = np.full(n, np.nan, dtype=float)

    for col in MILLIQUAS_ATTACH_COLUMNS:
        dest = f"{column_prefix}{col}"
        if col in ("has_radio", "has_xray", "has_double_lobes"):
            out[dest] = np.full(n, False, dtype=bool)
        elif col in ("z", "Rmag", "Bmag"):
            out[dest] = np.full(n, np.nan, dtype=float)
        else:
            out[dest] = np.full(n, "", dtype=object)

    positions_col = meta_flags["milliquas_positions"] if "milliquas_positions" in meta_flags.columns else None
    if positions_col is None or milliquas.empty:
        return out

    for i in range(n):
        hits = positions_col.iloc[i]
        if not isinstance(hits, (list, tuple)) or not hits:
            continue
        meta_ra = float(pd.to_numeric(out.iloc[i].get("RA"), errors="coerce"))
        meta_dec = float(pd.to_numeric(out.iloc[i].get("DEC"), errors="coerce"))
        if not (np.isfinite(meta_ra) and np.isfinite(meta_dec)):
            continue
        pos = _pick_milliquas_hit(
            milliquas,
            [int(p) for p in hits],
            meta_ra=meta_ra,
            meta_dec=meta_dec,
        )
        if pos is None:
            continue
        row = milliquas.iloc[int(pos)]
        for col in MILLIQUAS_ATTACH_COLUMNS:
            if col not in row.index:
                continue
            dest = f"{column_prefix}{col}"
            val = row[col]
            if col in ("has_radio", "has_xray", "has_double_lobes"):
                out.iat[i, out.columns.get_loc(dest)] = bool(val)
            elif col in ("z", "Rmag", "Bmag"):
                out.iat[i, out.columns.get_loc(dest)] = (
                    float(val) if pd.notna(val) else np.nan
                )
            else:
                out.iat[i, out.columns.get_loc(dest)] = "" if pd.isna(val) else str(val)

        mq_coord = SkyCoord(
            ra=float(row["RA"]) * u.deg, dec=float(row["DEC"]) * u.deg
        )
        meta_coord = SkyCoord(ra=meta_ra * u.deg, dec=meta_dec * u.deg)
        out.iat[i, out.columns.get_loc(sep_col)] = float(
            meta_coord.separation(mq_coord).arcsec
        )

    return out
