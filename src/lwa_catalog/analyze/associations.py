"""Long-form metacatalog↔survey associations for classification aggregation.

Crossmatch notebooks write unique matches (``n_* == 1`` / ``n_assoc_* == 1``)
into ``metacatalog_associations.parquet`` via replace-by-``catalog``. Harmonized
labels and conflict flags are computed at aggregation time, not at write.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from lwa_catalog.io import read_table, write_table
from lwa_catalog.paths import CatalogLayout

ASSOCIATION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "meta_id",
    "catalog",
    "ref_id",
    "ref_ra",
    "ref_dec",
    "class_native",
    "sep_arcsec",
    "n_ref",
)

# Documented examples of native → harmonized mapping (actual logic is
# catalog-aware in :func:`harmonize_class_native`).
HARMONIZED_CLASS_MAP: dict[str, str] = {
    "Freund2022": "star",
    "S": "star",  # MilliQuas primary (Xu Class S is handled via catalog)
    "Q": "agn",
    "A": "agn",
    "B": "agn",
    "K": "agn",
    "N": "agn",
    "blazar": "agn",
    "NEDLVS": "galaxy",
    "G": "galaxy",
    "Xu2022": "cluster",
    "2RXS": "xray",
    "eRASS3": "xray",
    "VLSSR": "radio",
    "NVSS": "radio",
    "VLASS": "radio",
    "LoDeSS": "radio",
}

RADIO_ASSOCIATION_CATALOGS: tuple[str, ...] = ("VLSSR", "NVSS", "VLASS", "LoDeSS")


@dataclass(frozen=True, slots=True)
class CatalogAssociationSpec:
    """How to extract unique association rows from a match-result footprint."""

    catalog: str
    id_column: str
    positions_col: str
    n_col: str
    class_column: str | None = None
    class_fallback_columns: tuple[str, ...] = ()
    class_detail_column: str | None = None
    id_fallback_columns: tuple[str, ...] = ()


# Specs used by crossmatch notebooks (stable ``catalog`` ids).
ASSOCIATION_SPEC_MILLIQUAS = CatalogAssociationSpec(
    catalog="MilliQuas",
    id_column="Name",
    positions_col="milliquas_positions",
    n_col="n_milliquas",
    class_column="class_primary",
    class_fallback_columns=("Type",),
    class_detail_column="class_label",
)
ASSOCIATION_SPEC_NEDLVS = CatalogAssociationSpec(
    catalog="NEDLVS",
    id_column="objname",
    positions_col="nedlvs_positions",
    n_col="n_nedlvs",
    class_column="objtype",
)
ASSOCIATION_SPEC_ERASS3 = CatalogAssociationSpec(
    catalog="eRASS3",
    id_column="LS10_OBJID",
    positions_col="erass3_positions",
    n_col="n_erass3",
    class_column="class_gal_exgal",
    id_fallback_columns=("main_id_simbad",),
)
ASSOCIATION_SPEC_FREUND2022 = CatalogAssociationSpec(
    catalog="Freund2022",
    id_column="2RXS",
    positions_col="freund2022_positions",
    n_col="n_freund2022",
    class_column=None,
)
ASSOCIATION_SPEC_XU2022 = CatalogAssociationSpec(
    catalog="Xu2022",
    id_column="RXGCC",
    positions_col="xu2022_positions",
    n_col="n_xu2022",
    class_column="Class",
)
ASSOCIATION_SPEC_TWORXS = CatalogAssociationSpec(
    catalog="2RXS",
    id_column="2RXS",
    positions_col="tworxs_positions",
    n_col="n_2rxs",
    class_column=None,
)


def ra_dec_key(ra: float, dec: float) -> str:
    """Stable string key from degrees for catalogs without a name column."""
    return f"{float(ra):.6f},{float(dec):.6f}"


def empty_associations_frame() -> pd.DataFrame:
    """Return an empty associations table with the required schema."""
    return pd.DataFrame(
        {
            "meta_id": pd.Series(dtype="int64"),
            "catalog": pd.Series(dtype="string"),
            "ref_id": pd.Series(dtype="string"),
            "ref_ra": pd.Series(dtype="float64"),
            "ref_dec": pd.Series(dtype="float64"),
            "class_native": pd.Series(dtype="string"),
            "sep_arcsec": pd.Series(dtype="float64"),
            "n_ref": pd.Series(dtype="int64"),
        }
    )


def validate_associations(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if required association columns are missing."""
    missing = [c for c in ASSOCIATION_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        msg = f"associations table missing required columns: {missing}"
        raise ValueError(msg)


def _as_str(value: object) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace").strip()
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>", "nat"}:
        return ""
    return text


def _sep_arcsec(meta_ra: float, meta_dec: float, ref_ra: float, ref_dec: float) -> float:
    if not all(np.isfinite(v) for v in (meta_ra, meta_dec, ref_ra, ref_dec)):
        return float("nan")
    meta = SkyCoord(ra=meta_ra * u.deg, dec=meta_dec * u.deg)
    ref = SkyCoord(ra=ref_ra * u.deg, dec=ref_dec * u.deg)
    return float(meta.separation(ref).arcsec)


def _ref_id_from_hit(hit: pd.Series, spec: CatalogAssociationSpec, ra: float, dec: float) -> str:
    for col in (spec.id_column, *spec.id_fallback_columns):
        if col in hit.index:
            text = _as_str(hit.get(col))
            if text:
                return text
    if np.isfinite(ra) and np.isfinite(dec):
        return ra_dec_key(ra, dec)
    return spec.catalog


def _class_native_from_hit(hit: pd.Series, spec: CatalogAssociationSpec) -> str:
    """Resolve native class; eRASS3 encodes blazar / gal-exgal specially."""
    if spec.catalog == "eRASS3":
        blazar = hit.get("is_blazar_in_simbad")
        if blazar is True or (isinstance(blazar, str) and blazar.strip().lower() in {"true", "1"}):
            return "blazar"
        code = pd.to_numeric(hit.get("class_gal_exgal"), errors="coerce")
        if np.isfinite(code):
            return str(int(code))
        return spec.catalog

    columns: list[str] = []
    if spec.class_column:
        columns.append(spec.class_column)
    columns.extend(spec.class_fallback_columns)
    for col in columns:
        if col in hit.index:
            text = _as_str(hit.get(col))
            if text:
                return text
    return spec.catalog


def association_rows_from_match(
    meta_flags: pd.DataFrame,
    footprint: pd.DataFrame,
    spec: CatalogAssociationSpec,
) -> pd.DataFrame:
    """Build unique-only association rows from match ``meta_flags`` + footprint.

    Keeps rows with ``meta_flags[n_col] == 1``, takes the first position index
    from ``positions_col``, and fills the associations schema.
    """
    if meta_flags is None or meta_flags.empty:
        return empty_associations_frame()
    if "meta_id" not in meta_flags.columns:
        msg = "meta_flags missing meta_id column"
        raise ValueError(msg)
    if spec.n_col not in meta_flags.columns:
        msg = f"meta_flags missing n-count column {spec.n_col!r}"
        raise ValueError(msg)
    if spec.positions_col not in meta_flags.columns:
        msg = f"meta_flags missing positions column {spec.positions_col!r}"
        raise ValueError(msg)

    unique = meta_flags.loc[
        pd.to_numeric(meta_flags[spec.n_col], errors="coerce").fillna(0).astype(int) == 1
    ]
    if unique.empty or footprint is None or footprint.empty:
        return empty_associations_frame()

    fp = footprint.reset_index(drop=True)
    records: list[dict[str, Any]] = []
    for _, flag in unique.iterrows():
        positions = flag.get(spec.positions_col, [])
        if not isinstance(positions, (list, tuple)) or not positions:
            continue
        pos = int(positions[0])
        if pos < 0 or pos >= len(fp):
            continue
        hit = fp.iloc[pos]
        meta_ra = float(pd.to_numeric(flag.get("RA"), errors="coerce"))
        meta_dec = float(pd.to_numeric(flag.get("DEC"), errors="coerce"))
        ref_ra = float(pd.to_numeric(hit.get("RA"), errors="coerce"))
        ref_dec = float(pd.to_numeric(hit.get("DEC"), errors="coerce"))
        class_native = _class_native_from_hit(hit, spec)
        row: dict[str, Any] = {
            "meta_id": int(flag["meta_id"]),
            "catalog": spec.catalog,
            "ref_id": _ref_id_from_hit(hit, spec, ref_ra, ref_dec),
            "ref_ra": ref_ra,
            "ref_dec": ref_dec,
            "class_native": class_native,
            "sep_arcsec": _sep_arcsec(meta_ra, meta_dec, ref_ra, ref_dec),
            "n_ref": 1,
        }
        if spec.class_detail_column and spec.class_detail_column in hit.index:
            detail = _as_str(hit.get(spec.class_detail_column))
            if detail:
                row["class_native_detail"] = detail
        records.append(row)

    if not records:
        return empty_associations_frame()
    out = pd.DataFrame.from_records(records)
    validate_associations(out)
    return out.reset_index(drop=True)


def association_rows_from_survey_attach(
    meta_df: pd.DataFrame,
    band: str,
) -> pd.DataFrame:
    """Build unique radio-survey association rows from an attached metacatalog.

    Keeps ``n_assoc_{band} == 1``. Uses ``RA_{band}``/``DEC_{band}`` when present,
    else ``match_RA``/``match_DEC`` when ``match_source`` equals *band*, else LWA
    ``RA``/``DEC``. ``class_native`` is the band name. ``ref_id`` prefers
    ``Component_name`` / ``Source_id`` columns when present, else a RA/DEC key.
    """
    if meta_df is None or meta_df.empty:
        return empty_associations_frame()
    if "meta_id" not in meta_df.columns:
        msg = "meta_df missing meta_id column"
        raise ValueError(msg)

    n_col = f"n_assoc_{band}"
    if n_col not in meta_df.columns:
        return empty_associations_frame()

    work = meta_df.loc[
        pd.to_numeric(meta_df[n_col], errors="coerce").fillna(0).astype(int) == 1
    ].copy()
    if work.empty:
        return empty_associations_frame()

    ra_band = f"RA_{band}"
    dec_band = f"DEC_{band}"
    records: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        meta_ra = float(pd.to_numeric(row.get("RA"), errors="coerce"))
        meta_dec = float(pd.to_numeric(row.get("DEC"), errors="coerce"))
        ref_ra = float("nan")
        ref_dec = float("nan")
        if ra_band in work.columns and dec_band in work.columns:
            ref_ra = float(pd.to_numeric(row.get(ra_band), errors="coerce"))
            ref_dec = float(pd.to_numeric(row.get(dec_band), errors="coerce"))
        if (not np.isfinite(ref_ra) or not np.isfinite(ref_dec)) and "match_source" in work.columns:
            if _as_str(row.get("match_source")) == band:
                ref_ra = float(pd.to_numeric(row.get("match_RA"), errors="coerce"))
                ref_dec = float(pd.to_numeric(row.get("match_DEC"), errors="coerce"))
        if not np.isfinite(ref_ra) or not np.isfinite(ref_dec):
            ref_ra, ref_dec = meta_ra, meta_dec

        ref_id = ""
        for col in (f"Component_name_{band}", "Component_name", f"Source_id_{band}", "Source_id"):
            if col in work.columns:
                ref_id = _as_str(row.get(col))
                if ref_id:
                    break
        if not ref_id:
            ref_id = ra_dec_key(ref_ra, ref_dec) if np.isfinite(ref_ra) else band

        records.append(
            {
                "meta_id": int(row["meta_id"]),
                "catalog": band,
                "ref_id": ref_id,
                "ref_ra": ref_ra,
                "ref_dec": ref_dec,
                "class_native": band,
                "sep_arcsec": _sep_arcsec(meta_ra, meta_dec, ref_ra, ref_dec),
                "n_ref": 1,
            }
        )

    out = pd.DataFrame.from_records(records)
    validate_associations(out)
    return out.reset_index(drop=True)


def _resolve_associations_path(path_or_layout: Path | str | CatalogLayout) -> Path:
    if isinstance(path_or_layout, CatalogLayout):
        return path_or_layout.metacatalog_associations()
    return Path(path_or_layout)


def read_associations(path_or_layout: Path | str | CatalogLayout) -> pd.DataFrame:
    """Load associations Parquet; return empty schema frame if missing."""
    path = _resolve_associations_path(path_or_layout)
    if not path.is_file():
        return empty_associations_frame()
    df = read_table(path, as_pandas=True)
    validate_associations(df)
    return df.reset_index(drop=True)


def write_associations(
    path_or_layout: Path | str | CatalogLayout,
    df: pd.DataFrame,
) -> Path:
    """Rewrite the full associations table."""
    path = _resolve_associations_path(path_or_layout)
    if df is None or df.empty:
        out = empty_associations_frame()
    else:
        validate_associations(df)
        out = df.reset_index(drop=True)
    return write_table(out, path, include_extras=True)


def replace_catalog_associations(
    path_or_layout: Path | str | CatalogLayout,
    catalog: str,
    rows: pd.DataFrame,
) -> Path:
    """Replace all rows for *catalog* with *rows* (idempotent re-run).

    Empty *rows* drops prior rows for that catalog so stale associations do not
    linger after a zero-unique re-run.
    """
    catalog_id = str(catalog).strip()
    if not catalog_id:
        msg = "catalog must be a non-empty string"
        raise ValueError(msg)

    existing = read_associations(path_or_layout)
    if not existing.empty:
        kept = existing.loc[existing["catalog"].astype(str) != catalog_id].copy()
    else:
        kept = empty_associations_frame()

    if rows is None or rows.empty:
        incoming = empty_associations_frame()
    else:
        validate_associations(rows)
        incoming = rows.copy()
        incoming["catalog"] = catalog_id
        if not (incoming["n_ref"] == 1).all():
            msg = "association rows must have n_ref == 1"
            raise ValueError(msg)

    if kept.empty and incoming.empty:
        combined = empty_associations_frame()
    elif kept.empty:
        combined = incoming.reset_index(drop=True)
    elif incoming.empty:
        combined = kept.reset_index(drop=True)
    else:
        combined = pd.concat([kept, incoming], ignore_index=True)

    return write_associations(path_or_layout, combined)


def harmonize_class_native(catalog: str, class_native: str) -> str:
    """Map ``(catalog, class_native)`` to a coarse harmonized label."""
    cat = _as_str(catalog)
    native = _as_str(class_native)

    if cat in RADIO_ASSOCIATION_CATALOGS:
        return "radio"
    if cat == "Freund2022":
        return "star"
    if cat == "Xu2022":
        return "cluster"
    if cat == "2RXS":
        return "xray"
    if cat == "NEDLVS":
        return "galaxy"
    if cat == "MilliQuas":
        primary = native[:1].upper() if native else ""
        if primary == "S":
            return "star"
        if primary in {"Q", "A", "B", "K", "N"}:
            return "agn"
        return HARMONIZED_CLASS_MAP.get(native, "unknown")
    if cat == "eRASS3":
        if native.lower() == "blazar":
            return "agn"
        try:
            code = int(native)
        except (TypeError, ValueError):
            return "xray" if native in {"", "eRASS3"} else "unknown"
        if code < 0:
            return "star"
        if code > 0:
            return "galaxy"
        return "xray"

    if native in HARMONIZED_CLASS_MAP:
        return HARMONIZED_CLASS_MAP[native]
    if cat in HARMONIZED_CLASS_MAP:
        return HARMONIZED_CLASS_MAP[cat]
    return "unknown"


def harmonize_associations(assoc_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``class_harmonized`` column filled."""
    if assoc_df is None or assoc_df.empty:
        out = empty_associations_frame()
        out["class_harmonized"] = pd.Series(dtype="string")
        return out
    validate_associations(assoc_df)
    out = assoc_df.copy()
    out["class_harmonized"] = [
        harmonize_class_native(str(c), str(n))
        for c, n in zip(out["catalog"], out["class_native"], strict=True)
    ]
    return out


def flag_association_conflicts(assoc_df: pd.DataFrame) -> pd.DataFrame:
    """Per-``meta_id`` conflict summary from harmonized labels.

    Returns columns ``meta_id``, ``n_associations``, ``n_harmonized``,
    ``harmonized_labels`` (sorted unique joined by ``,``), and ``conflict``
    (True when ≥2 distinct harmonized labels).
    """
    work = harmonize_associations(assoc_df)
    if work.empty:
        return pd.DataFrame(
            {
                "meta_id": pd.Series(dtype="int64"),
                "n_associations": pd.Series(dtype="int64"),
                "n_harmonized": pd.Series(dtype="int64"),
                "harmonized_labels": pd.Series(dtype="string"),
                "conflict": pd.Series(dtype="boolean"),
            }
        )

    rows: list[dict[str, Any]] = []
    for meta_id, group in work.groupby("meta_id", sort=True):
        labels = sorted({str(x) for x in group["class_harmonized"].tolist() if str(x)})
        rows.append(
            {
                "meta_id": int(meta_id),
                "n_associations": int(len(group)),
                "n_harmonized": int(len(labels)),
                "harmonized_labels": ",".join(labels),
                "conflict": len(labels) >= 2,
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_classifications(assoc_df: pd.DataFrame) -> dict[str, Any]:
    """Return counts suitable for logging / notebook display."""
    work = harmonize_associations(assoc_df)
    conflicts = flag_association_conflicts(work)
    summary: dict[str, Any] = {
        "n_associations": int(len(work)),
        "n_meta_ids": int(work["meta_id"].nunique()) if not work.empty else 0,
        "n_catalogs": int(work["catalog"].nunique()) if not work.empty else 0,
        "n_conflict_meta_ids": int(conflicts["conflict"].sum()) if not conflicts.empty else 0,
        "native_counts": {},
        "harmonized_counts": {},
        "catalog_counts": {},
    }
    if work.empty:
        return summary
    summary["catalog_counts"] = (
        work["catalog"].astype(str).value_counts(dropna=False).to_dict()
    )
    summary["native_counts"] = (
        work.groupby(["catalog", "class_native"], dropna=False)
        .size()
        .to_dict()
    )
    # Flatten tuple keys for JSON-ish display.
    summary["native_counts"] = {
        f"{cat}|{native}": int(n) for (cat, native), n in summary["native_counts"].items()
    }
    summary["harmonized_counts"] = (
        work["class_harmonized"].astype(str).value_counts(dropna=False).to_dict()
    )
    return summary


def metacatalog_with_erass3_association_columns(
    meta_df: pd.DataFrame,
    assoc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join unique ``eRASS3`` association rows onto *meta_df* for Visual QA.

    Synthesizes ``n_erass3``, ``sep_arcsec_eRASS3``, ``eRASS3_LS10_OBJID`` /
    ``eRASS3_main_id_simbad``-style id, ``eRASS3_class_gal_exgal`` (when
    ``class_native`` parses as int), and ``eRASS3_RA`` / ``eRASS3_DEC`` from
    association ``ref_*``. Does not restore full attach columns (DET_LIKE, etc.).
    """
    if meta_df is None or meta_df.empty:
        return meta_df.copy() if meta_df is not None else pd.DataFrame()
    if "meta_id" not in meta_df.columns:
        msg = "meta_df missing meta_id column"
        raise ValueError(msg)

    out = meta_df.reset_index(drop=True).copy()
    out["n_erass3"] = 0
    out["in_erass_footprint"] = False
    out["sep_arcsec_eRASS3"] = np.nan
    out["eRASS3_LS10_OBJID"] = pd.NA
    out["eRASS3_class_gal_exgal"] = np.nan
    out["eRASS3_RA"] = np.nan
    out["eRASS3_DEC"] = np.nan
    out["eRASS3_POS_ERR"] = np.nan

    if assoc_df is None or assoc_df.empty or "catalog" not in assoc_df.columns:
        return out

    erass = assoc_df.loc[assoc_df["catalog"].astype(str) == "eRASS3"].copy()
    if erass.empty:
        return out
    validate_associations(erass)
    # One row per meta_id (unique-only writers).
    erass = erass.drop_duplicates(subset=["meta_id"], keep="first")
    by_id = erass.set_index("meta_id", drop=False)

    for i, meta_id in enumerate(out["meta_id"].tolist()):
        if meta_id not in by_id.index:
            continue
        hit = by_id.loc[meta_id]
        if isinstance(hit, pd.DataFrame):
            hit = hit.iloc[0]
        out.at[i, "n_erass3"] = 1
        out.at[i, "sep_arcsec_eRASS3"] = float(
            pd.to_numeric(hit.get("sep_arcsec"), errors="coerce")
        )
        out.at[i, "eRASS3_RA"] = float(pd.to_numeric(hit.get("ref_ra"), errors="coerce"))
        out.at[i, "eRASS3_DEC"] = float(pd.to_numeric(hit.get("ref_dec"), errors="coerce"))
        ref_id = _as_str(hit.get("ref_id"))
        out.at[i, "eRASS3_LS10_OBJID"] = ref_id if ref_id else pd.NA
        native = _as_str(hit.get("class_native"))
        if native == "blazar":
            out.at[i, "eRASS3_is_blazar_in_simbad"] = True
        else:
            try:
                out.at[i, "eRASS3_class_gal_exgal"] = int(native)
            except (TypeError, ValueError):
                pass
    if "eRASS3_is_blazar_in_simbad" not in out.columns:
        out["eRASS3_is_blazar_in_simbad"] = pd.NA
    return out


def summarize_classifications_text(assoc_df: pd.DataFrame) -> str:
    """Human-readable classification summary."""
    s = summarize_classifications(assoc_df)
    lines = [
        f"Associations:                 {int(s['n_associations']):6d}",
        f"Unique meta_id:               {int(s['n_meta_ids']):6d}",
        f"Catalogs:                     {int(s['n_catalogs']):6d}",
        f"Conflict meta_ids:            {int(s['n_conflict_meta_ids']):6d}",
    ]
    if s["harmonized_counts"]:
        lines.append("Harmonized:")
        for label, n in sorted(s["harmonized_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  {label:12s} {int(n):6d}")
    return "\n".join(lines)
