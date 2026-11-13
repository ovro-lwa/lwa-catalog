"""GLEAM Extragalactic Catalog cross-match against LWA metacatalogs.

Hurley-Walker et al. 2017 (CDS VIII/100). Match once on position (Dec ≤ +30°),
then attach all 20 narrowband Peak/Total fluxes (+ errors) onto matched rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lwa_catalog.analyze.crossmatch_radius import (
    GLEAM_REFERENCE_RADIUS_LOCALIZATION,
    LWA_CROSSMATCH_RADIUS_BEAM,
    CrossmatchRadiusSpec,
    apply_match_radius,
    catalog_match_frame,
    match_radius_deg,
)
from lwa_catalog.analyze.survey_attach import (
    _bands_present_order,
    normalize_survey_band_catalog,
)
from lwa_catalog.constants import (
    GLEAM_BANDS,
    GLEAM_BMAJ_DEG,
    GLEAM_DEC_MAX_DEG,
    GLEAM_DEFAULT_PATH,
    GLEAM_NARROW_MHZ,
    SUBBAND_METACATALOG_FLUX_FIELDS,
)
from lwa_catalog.create.merge import (
    _pick_peak_flux_row,
    associate_band_into_metacatalog,
    associate_catalogs,
)

# Sentinel band key for ``n_assoc_GLEAM`` / associations (not a SED channel).
GLEAM_ASSOC_BAND: str = "GLEAM"

_GLEAM_ATTACH_CORE_FIELDS: tuple[str, ...] = (
    "Peak_flux",
    "Total_flux",
    "E_Peak_flux",
    "E_Total_flux",
    "RA",
    "DEC",
    "Component_name",
)

# CDS / VizieR / HEASARC column aliases → canonical names.
_COORD_ALIASES: dict[str, tuple[str, ...]] = {
    "RA": ("RA", "RAJ2000", "ra", "RA_J2000"),
    "DEC": ("DEC", "DEJ2000", "dec", "Dec", "DEC_J2000"),
    "E_RA": ("E_RA", "err_RAJ2000", "e_RAJ2000", "RA_Error", "e_RA"),
    "E_DEC": ("E_DEC", "err_DEJ2000", "e_DEJ2000", "Dec_Error", "DEC_Error", "e_DE"),
    "Name": ("Name", "GLEAM", "name", "ID"),
}

_WIDE_PEAK_ALIASES: tuple[str, ...] = (
    "peak_flux_wide",
    "Fpwide",
    "Flux_Wide",
    "Peak_flux",
)
_WIDE_PEAK_ERR_ALIASES: tuple[str, ...] = (
    "err_peak_flux_wide",
    "e_Fpwide",
    "Flux_Wide_Error",
    "E_Peak_flux",
)
_WIDE_INT_ALIASES: tuple[str, ...] = (
    "int_flux_wide",
    "Fintwide",
    "Int_Flux_Wide",
    "Total_flux",
)
_WIDE_INT_ERR_ALIASES: tuple[str, ...] = (
    "err_int_flux_wide",
    "e_Fintwide",
    "Int_Flux_Wide_Error",
    "E_Total_flux",
)
_PSF_A_ALIASES: tuple[str, ...] = ("psf_a_wide", "A_PSF_Wide", "psf_a")
_PSF_B_ALIASES: tuple[str, ...] = ("psf_b_wide", "B_PSF_Wide", "psf_b")


def _narrow_peak_aliases(mhz: int) -> tuple[str, ...]:
    code = f"{mhz:03d}"
    return (
        f"peak_flux_{code}",
        f"Fp{code}",
        f"Flux_{mhz}_MHz",
        f"Peak_flux_GLEAM{mhz}",
    )


def _narrow_peak_err_aliases(mhz: int) -> tuple[str, ...]:
    code = f"{mhz:03d}"
    return (
        f"err_peak_flux_{code}",
        f"e_Fp{code}",
        f"Flux_{mhz}_MHz_Error",
        f"E_Peak_flux_GLEAM{mhz}",
    )


def _narrow_int_aliases(mhz: int) -> tuple[str, ...]:
    code = f"{mhz:03d}"
    return (
        f"int_flux_{code}",
        f"Fint{code}",
        f"Int_Flux_{mhz}_MHz",
        f"Total_flux_GLEAM{mhz}",
    )


def _narrow_int_err_aliases(mhz: int) -> tuple[str, ...]:
    code = f"{mhz:03d}"
    return (
        f"err_int_flux_{code}",
        f"e_Fint{code}",
        f"Int_Flux_{mhz}_MHz_Error",
        f"E_Total_flux_GLEAM{mhz}",
    )


@dataclass(frozen=True)
class GleamMatchConfig:
    """Configuration for :func:`match_catalog_to_gleam`."""

    catalog_path: Path = GLEAM_DEFAULT_PATH
    dec_max_deg: float = GLEAM_DEC_MAX_DEG
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM
    reference_radius: CrossmatchRadiusSpec = GLEAM_REFERENCE_RADIUS_LOCALIZATION


@dataclass
class GleamMatchResult:
    """GLEAM cross-match metrics and per-row flags."""

    summary: dict[str, float | int]
    meta_flags: pd.DataFrame
    gleam_flags: pd.DataFrame
    gleam_footprint: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _first_present(columns: Sequence[str], aliases: Sequence[str]) -> str | None:
    colset = set(columns)
    for name in aliases:
        if name in colset:
            return name
    return None


def _series_from_aliases(df: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    name = _first_present(df.columns, aliases)
    if name is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def _position_errors_to_deg(
    e_ra: pd.Series,
    e_dec: pd.Series,
    *,
    src_col: str | None,
) -> tuple[pd.Series, pd.Series]:
    """Return E_RA/E_DEC in degrees.

    CDS ``err_RAJ2000`` is already degrees. HEASARC ``RA_Error`` is arcseconds.
    """
    heasarc_like = src_col is not None and "Error" in src_col and "err_" not in src_col
    if heasarc_like:
        return e_ra / 3600.0, e_dec / 3600.0
    # Heuristic: values ≳ 0.01 are almost certainly arcsec, not deg.
    sample = pd.concat([e_ra, e_dec], ignore_index=True)
    finite = sample[np.isfinite(sample.to_numpy(dtype=float))]
    if not finite.empty and float(finite.median()) > 0.01:
        return e_ra / 3600.0, e_dec / 3600.0
    return e_ra, e_dec


def _flux_to_jy(values: pd.Series, *, src_col: str | None) -> pd.Series:
    """CDS FITS is Jy; HEASARC tables are mJy — convert when HEASARC-like."""
    if src_col is not None and (
        src_col.startswith("Flux_") or src_col.startswith("Int_Flux_")
    ):
        return values / 1000.0
    return values


def _psf_a_to_deg(values: pd.Series) -> pd.Series:
    """PSF semi-major from CDS is arcsec; convert to degrees for BMAJ."""
    out = pd.to_numeric(values, errors="coerce")
    # Values ≳ 0.1 are arcsec-scale (~120″), not already degrees.
    finite = out[np.isfinite(out.to_numpy(dtype=float))]
    if not finite.empty and float(finite.median()) > 0.1:
        return out / 3600.0
    return out


def _finalize_gleam_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize coordinates, id, beam, and narrowband fluxes (Jy)."""
    if df.empty:
        cols = [
            "Name",
            "RA",
            "DEC",
            "E_RA",
            "E_DEC",
            "BMAJ",
            "BMIN",
            "Peak_flux",
            "Total_flux",
            "E_Peak_flux",
            "E_Total_flux",
            "Component_name",
        ]
        for band in GLEAM_BANDS:
            for field in SUBBAND_METACATALOG_FLUX_FIELDS:
                cols.append(f"{field}_{band}")
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(index=df.index)
    for dest, aliases in _COORD_ALIASES.items():
        src = _first_present(df.columns, aliases)
        if src is None:
            out[dest] = np.nan if dest != "Name" else ""
        elif dest == "Name":
            out[dest] = df[src].astype(str)
        else:
            out[dest] = pd.to_numeric(df[src], errors="coerce")

    e_ra_src = _first_present(df.columns, _COORD_ALIASES["E_RA"])
    e_dec_src = _first_present(df.columns, _COORD_ALIASES["E_DEC"])
    out["E_RA"], out["E_DEC"] = _position_errors_to_deg(
        out["E_RA"], out["E_DEC"], src_col=e_ra_src or e_dec_src
    )

    if "RA" not in out.columns or "DEC" not in out.columns:
        msg = "GLEAM table missing RA/DEC columns"
        raise ValueError(msg)

    peak_src = _first_present(df.columns, _WIDE_PEAK_ALIASES)
    peak_err_src = _first_present(df.columns, _WIDE_PEAK_ERR_ALIASES)
    int_src = _first_present(df.columns, _WIDE_INT_ALIASES)
    int_err_src = _first_present(df.columns, _WIDE_INT_ERR_ALIASES)
    out["Peak_flux"] = _flux_to_jy(_series_from_aliases(df, _WIDE_PEAK_ALIASES), src_col=peak_src)
    out["E_Peak_flux"] = _flux_to_jy(
        _series_from_aliases(df, _WIDE_PEAK_ERR_ALIASES), src_col=peak_err_src
    )
    out["Total_flux"] = _flux_to_jy(_series_from_aliases(df, _WIDE_INT_ALIASES), src_col=int_src)
    out["E_Total_flux"] = _flux_to_jy(
        _series_from_aliases(df, _WIDE_INT_ERR_ALIASES), src_col=int_err_src
    )
    # Prefer total for representative pick when wide peak is missing.
    missing_peak = ~np.isfinite(out["Peak_flux"].to_numpy(dtype=float))
    out.loc[missing_peak, "Peak_flux"] = out.loc[missing_peak, "Total_flux"]

    for mhz in GLEAM_NARROW_MHZ:
        band = f"GLEAM{mhz}"
        p_src = _first_present(df.columns, _narrow_peak_aliases(mhz))
        pe_src = _first_present(df.columns, _narrow_peak_err_aliases(mhz))
        t_src = _first_present(df.columns, _narrow_int_aliases(mhz))
        te_src = _first_present(df.columns, _narrow_int_err_aliases(mhz))
        out[f"Peak_flux_{band}"] = _flux_to_jy(
            _series_from_aliases(df, _narrow_peak_aliases(mhz)), src_col=p_src
        )
        out[f"E_Peak_flux_{band}"] = _flux_to_jy(
            _series_from_aliases(df, _narrow_peak_err_aliases(mhz)), src_col=pe_src
        )
        out[f"Total_flux_{band}"] = _flux_to_jy(
            _series_from_aliases(df, _narrow_int_aliases(mhz)), src_col=t_src
        )
        out[f"E_Total_flux_{band}"] = _flux_to_jy(
            _series_from_aliases(df, _narrow_int_err_aliases(mhz)), src_col=te_src
        )

    psf_a = _psf_a_to_deg(_series_from_aliases(df, _PSF_A_ALIASES))
    psf_b = _psf_a_to_deg(_series_from_aliases(df, _PSF_B_ALIASES))
    out["BMAJ"] = psf_a
    out["BMIN"] = psf_b
    out.loc[~np.isfinite(out["BMAJ"].to_numpy(dtype=float)), "BMAJ"] = GLEAM_BMAJ_DEG
    out.loc[~np.isfinite(out["BMIN"].to_numpy(dtype=float)), "BMIN"] = out.loc[
        ~np.isfinite(out["BMIN"].to_numpy(dtype=float)), "BMAJ"
    ]

    names = out["Name"].fillna("").astype(str)
    blank = names.str.strip().eq("") | names.str.lower().eq("nan")
    out.loc[blank, "Name"] = [
        f"GLEAM_{i}" for i in out.index[blank].to_numpy()
    ]
    out["Component_name"] = out["Name"]

    ra = out["RA"].to_numpy(dtype=float)
    dec = out["DEC"].to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return out.loc[ok].reset_index(drop=True)


def load_gleam_catalog(path: Path | str | None = None) -> pd.DataFrame:
    """Load the GLEAM EG catalog for cross-matching.

    Accepts CDS ``GLEAM_EGC_v2.fits`` (preferred), parquet, or CSV. Returns
    positions, wide Peak/Total (for match representative), circular ``BMAJ``/
    ``BMIN``, and ``Peak_flux_GLEAMnn`` / ``Total_flux_GLEAMnn`` (+ errors) for
    all 20 narrowbands in Jy. Rows with non-finite ``RA``/``DEC`` are dropped.

    Parameters
    ----------
    path
        Catalog file path. Defaults to
        :data:`~lwa_catalog.constants.GLEAM_DEFAULT_PATH`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing.
    """
    catalog_path = Path(GLEAM_DEFAULT_PATH if path is None else path)
    if not catalog_path.is_file():
        msg = (
            f"GLEAM catalog not found: {catalog_path}. "
            "Download CDS VIII/100 once, e.g.\n"
            "  curl -L -o /fast/claw/GLEAM_EGC_v2.fits.gz \\\n"
            "    http://vizier.china-vo.org/ftp/cats/viii/100/GLEAM_EGC_v2.fits.gz\n"
            "  gunzip -f /fast/claw/GLEAM_EGC_v2.fits.gz"
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

    return _finalize_gleam_frame(df)


def select_unique_gleam_matches(meta_flags: pd.DataFrame) -> pd.DataFrame:
    """Return meta rows with exactly one GLEAM match (``n_gleam == 1``)."""
    if meta_flags.empty or "n_gleam" not in meta_flags.columns:
        return meta_flags.iloc[0:0].copy()
    return meta_flags.loc[meta_flags["n_gleam"] == 1].copy()


def _footprint_filter_gleam(
    gleam: pd.DataFrame,
    lwa: pd.DataFrame,
    *,
    dec_max_deg: float,
) -> pd.DataFrame:
    """Keep GLEAM rows in the LWA Dec box and at/below the survey Dec limit."""
    if gleam.empty:
        return gleam.copy()
    if lwa.empty or "DEC" not in lwa.columns:
        return gleam.iloc[0:0].copy()

    lwa_dec = pd.to_numeric(lwa["DEC"], errors="coerce")
    finite_lwa = lwa_dec[np.isfinite(lwa_dec.to_numpy(dtype=float))]
    if finite_lwa.empty:
        return gleam.iloc[0:0].copy()

    dec_min = float(finite_lwa.min())
    dec_max = min(float(finite_lwa.max()), float(dec_max_deg))
    gleam_dec = pd.to_numeric(gleam["DEC"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(gleam_dec) & (gleam_dec >= dec_min) & (gleam_dec <= dec_max)
    return gleam.loc[keep].copy()


def _empty_summary() -> dict[str, float | int]:
    return {
        "n_lwa_target": 0,
        "n_gleam_footprint": 0,
        "n_meta_matched": 0,
        "match_completeness": float("nan"),
        "n_meta_unique": 0,
        "unique_match_fraction": float("nan"),
        "n_gleam_matched": 0,
        "gleam_recovery": float("nan"),
        "n_gleam_oversplit": 0,
        "n_meta_multi_gleam": 0,
        "meta_gleam_hits_max": 0,
    }


def match_catalog_to_gleam(
    lwa_catalog: pd.DataFrame,
    gleam: pd.DataFrame | None = None,
    *,
    config: GleamMatchConfig | None = None,
    lwa_match: pd.DataFrame | None = None,
) -> GleamMatchResult:
    """Cross-match an LWA catalog against GLEAM and compute association metrics.

    GLEAM sources above ``config.dec_max_deg`` (default +30°) are excluded from
    the footprint. Matching uses ``RA``/``DEC`` and configured
    :class:`~lwa_catalog.analyze.crossmatch_radius.CrossmatchRadiusSpec` radii.

    Parameters
    ----------
    lwa_catalog
        Metacatalog table.
    gleam
        Pre-loaded GLEAM catalog. Loaded from ``config.catalog_path`` when omitted.
    config
        Match configuration.
    lwa_match
        Optional prebuilt match frame (``RA``/``DEC``/``BMAJ``), iloc-aligned
        with *lwa_catalog* (cascaded bootstrap).
    """
    cfg = config or GleamMatchConfig()
    warnings: list[str] = []

    if gleam is None:
        gleam = load_gleam_catalog(cfg.catalog_path)

    target = lwa_catalog
    n_lwa_target = len(target)
    if n_lwa_target == 0:
        warnings.append("LWA target catalog is empty")
        return GleamMatchResult(
            summary=_empty_summary(),
            meta_flags=pd.DataFrame(
                columns=["meta_id", "RA", "DEC", "n_gleam", "gleam_positions", "matched"]
            ),
            gleam_flags=pd.DataFrame(
                columns=[
                    "gleam_pos",
                    "Name",
                    "RA",
                    "DEC",
                    "Peak_flux",
                    "Total_flux",
                    "n_meta",
                    "meta_ids",
                    "oversplit",
                ]
            ),
            gleam_footprint=pd.DataFrame(),
            warnings=warnings,
        )

    gleam_footprint = _footprint_filter_gleam(
        gleam, target, dec_max_deg=cfg.dec_max_deg
    )
    ref_match = apply_match_radius(gleam_footprint, cfg.reference_radius)
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
    n_gleam_footprint = len(gleam_footprint)

    meta_hits: dict[int, list[int]] = {}
    gleam_hits: dict[int, list[int]] = {}
    if not resolved_lwa_match.empty and not ref_match.empty:
        meta_hits, _ = associate_catalogs(resolved_lwa_match, ref_match)
        gleam_hits, _ = associate_catalogs(ref_match, resolved_lwa_match)

    index_to_match_pos = {
        idx: pos for pos, idx in enumerate(resolved_lwa_match.index.tolist())
    }
    match_pos_to_index = {pos: idx for idx, pos in index_to_match_pos.items()}
    has_meta_id = "meta_id" in target.columns

    meta_records: list[dict] = []
    for idx, row in target.iterrows():
        match_pos = index_to_match_pos.get(idx)
        hit_gleam = meta_hits.get(match_pos, []) if match_pos is not None else []
        n_gleam = len(hit_gleam)
        record: dict = {
            "RA": row.get("RA", np.nan),
            "DEC": row.get("DEC", np.nan),
            "n_gleam": n_gleam,
            "gleam_positions": list(hit_gleam),
            "matched": n_gleam >= 1,
        }
        if has_meta_id:
            record["meta_id"] = row.get("meta_id", np.nan)
        meta_records.append(record)

    meta_flags = pd.DataFrame(meta_records)
    if "meta_id" in meta_flags.columns:
        meta_flags = meta_flags[
            ["meta_id", "RA", "DEC", "n_gleam", "gleam_positions", "matched"]
        ]

    gleam_records: list[dict] = []
    for pos in range(len(gleam_footprint)):
        row = gleam_footprint.iloc[pos]
        hit_meta = gleam_hits.get(pos, [])
        n_meta = len(hit_meta)
        meta_ids: list[object] = []
        if has_meta_id:
            for match_pos in hit_meta:
                idx = match_pos_to_index.get(match_pos)
                if idx is not None:
                    meta_ids.append(target.loc[idx, "meta_id"])
        gleam_records.append(
            {
                "gleam_pos": pos,
                "Name": row.get("Name", ""),
                "RA": row["RA"],
                "DEC": row["DEC"],
                "Peak_flux": row.get("Peak_flux", np.nan),
                "Total_flux": row.get("Total_flux", np.nan),
                "n_meta": n_meta,
                "meta_ids": meta_ids,
                "oversplit": n_meta > 1,
            }
        )

    gleam_cols = [
        "gleam_pos",
        "Name",
        "RA",
        "DEC",
        "Peak_flux",
        "Total_flux",
        "n_meta",
        "meta_ids",
        "oversplit",
    ]
    if not has_meta_id:
        gleam_cols = [c for c in gleam_cols if c != "meta_ids"]
    if gleam_records:
        gleam_flags = pd.DataFrame(gleam_records)[gleam_cols]
    else:
        gleam_flags = pd.DataFrame(columns=gleam_cols)

    n_meta_matched = int(meta_flags["matched"].sum()) if not meta_flags.empty else 0
    n_meta_unique = int((meta_flags["n_gleam"] == 1).sum()) if not meta_flags.empty else 0
    n_gleam_matched = (
        int((gleam_flags["n_meta"] > 0).sum()) if not gleam_flags.empty else 0
    )
    n_gleam_oversplit = (
        int(gleam_flags["oversplit"].sum()) if not gleam_flags.empty else 0
    )
    n_meta_multi_gleam = (
        int((meta_flags["n_gleam"] > 1).sum()) if not meta_flags.empty else 0
    )
    meta_gleam_hits_max = (
        int(meta_flags["n_gleam"].max()) if not meta_flags.empty else 0
    )

    summary: dict[str, float | int] = {
        "n_lwa_target": n_lwa_target,
        "n_gleam_footprint": n_gleam_footprint,
        "n_meta_matched": n_meta_matched,
        "match_completeness": n_meta_matched / n_lwa_target,
        "n_meta_unique": n_meta_unique,
        "unique_match_fraction": n_meta_unique / n_lwa_target,
        "n_gleam_matched": n_gleam_matched,
        "gleam_recovery": (
            n_gleam_matched / n_gleam_footprint if n_gleam_footprint else float("nan")
        ),
        "n_gleam_oversplit": n_gleam_oversplit,
        "n_meta_multi_gleam": n_meta_multi_gleam,
        "meta_gleam_hits_max": meta_gleam_hits_max,
    }

    return GleamMatchResult(
        summary=summary,
        meta_flags=meta_flags,
        gleam_flags=gleam_flags,
        gleam_footprint=gleam_footprint,
        warnings=warnings,
    )


def summarize_gleam_match(result: GleamMatchResult) -> str:
    """Return a multi-line text summary suitable for notebook printout."""
    s = result.summary
    lines = [
        f"LWA target rows:               {int(s['n_lwa_target']):6d}",
        f"GLEAM footprint (Dec≤+30):     {int(s['n_gleam_footprint']):6d}",
        f"Meta matched (>=1 GLEAM):      {int(s['n_meta_matched']):6d}",
        f"Match completeness:            {s['match_completeness']:.3f}",
        f"Meta unique match (n=1):       {int(s['n_meta_unique']):6d}",
        f"Unique match fraction:         {s['unique_match_fraction']:.3f}",
        f"GLEAM matched (>=1 meta):      {int(s['n_gleam_matched']):6d}",
        f"GLEAM recovery:                {s['gleam_recovery']:.3f}",
        f"GLEAM over-split (n_meta>1):   {int(s['n_gleam_oversplit']):6d}",
        f"Meta multi-GLEAM (n>1):        {int(s['n_meta_multi_gleam']):6d}",
        f"Max GLEAM hits per meta:       {int(s['meta_gleam_hits_max']):6d}",
        "",
        f"GLEAM narrowband channels:     {len(GLEAM_BANDS)} "
        f"({GLEAM_NARROW_MHZ[0]}–{GLEAM_NARROW_MHZ[-1]} MHz)",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)


def attach_gleam_to_metacatalog(
    meta_df: pd.DataFrame,
    gleam_df: pd.DataFrame,
    *,
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM,
    reference_radius: CrossmatchRadiusSpec | None = None,
    dec_max_deg: float = GLEAM_DEC_MAX_DEG,
    footprint_filter: bool = True,
    match_base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach GLEAM as photometric bands after a single position match.

    Writes ``n_assoc_GLEAM`` / ``n_confused_GLEAM`` / ``RA_GLEAM`` / ``DEC_GLEAM``
    / wide ``Peak_flux_GLEAM`` plus ``Peak_flux_GLEAMnn`` / ``Total_flux_GLEAMnn``
    (+ errors) for all 20 narrowbands from the same matched GLEAM row.
    Unmatched survey rows are discarded. LWA astrometry is unchanged.
    """
    if reference_radius is None:
        reference_radius = GLEAM_REFERENCE_RADIUS_LOCALIZATION

    work = meta_df.reset_index(drop=True)
    survey = normalize_survey_band_catalog(gleam_df)
    if "Component_name" not in survey.columns and "Name" in survey.columns:
        survey["Component_name"] = survey["Name"].astype(str)
    if footprint_filter:
        survey = _footprint_filter_gleam(survey, work, dec_max_deg=dec_max_deg)
    survey = apply_match_radius(survey, reference_radius)

    color_bands = _bands_present_order(work, (GLEAM_ASSOC_BAND,))

    if match_base is None:
        base_ra = None
        base_dec = None
        base_bmaj = np.asarray(
            [match_radius_deg(row, lwa_radius) for _, row in work.iterrows()],
            dtype=float,
        )
    else:
        if len(match_base) != len(work):
            msg = (
                f"match_base length {len(match_base)} does not match "
                f"metacatalog rows {len(work)}"
            )
            raise ValueError(msg)
        base_ra = match_base["RA"].to_numpy(dtype=float)
        base_dec = match_base["DEC"].to_numpy(dtype=float)
        base_bmaj = match_base["BMAJ"].to_numpy(dtype=float)

    out = associate_band_into_metacatalog(
        work,
        survey,
        GLEAM_ASSOC_BAND,
        assoc_bands=(GLEAM_ASSOC_BAND,),
        band_fields=_GLEAM_ATTACH_CORE_FIELDS,
        color_bands=color_bands,
        primary_flux=False,
        astrometry_from_highest_frequency=False,
        append_unmatched=False,
        update_bmaj_match=False,
        representative="peak_flux",
        base_bmaj=base_bmaj,
        base_ra=base_ra,
        base_dec=base_dec,
    )

    # Fan out narrowband fluxes from the same representative GLEAM row.
    for band in GLEAM_BANDS:
        for fld in SUBBAND_METACATALOG_FLUX_FIELDS:
            col = f"{fld}_{band}"
            if col not in out.columns:
                out[col] = np.nan

    if out.empty or survey.empty:
        return out

    if base_ra is None:
        match_frame = pd.DataFrame(
            {
                "RA": work["RA"].to_numpy(dtype=float),
                "DEC": work["DEC"].to_numpy(dtype=float),
                "BMAJ": base_bmaj,
            }
        )
    else:
        match_frame = pd.DataFrame(
            {"RA": base_ra, "DEC": base_dec, "BMAJ": base_bmaj}
        )
    hits, _ = associate_catalogs(match_frame, survey)
    for i, band_hits in hits.items():
        if not band_hits:
            continue
        best = _pick_peak_flux_row(survey.iloc[band_hits])
        for band in GLEAM_BANDS:
            for fld in SUBBAND_METACATALOG_FLUX_FIELDS:
                col = f"{fld}_{band}"
                out.at[i, col] = best.get(col, np.nan)
    return out
