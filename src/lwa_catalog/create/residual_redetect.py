"""Stage-3 residual re-detect on HEALPix coadds.

After tile detect (stage 1) and ``merge_tile_metacatalog`` (stage 2), paint the
stage-2 catalog onto the observed HEALPix MAP, form a positive-clipped
residual, run single-tier tile PyBDSF, then:

- **append** residual islands that do **not** match any stage-2 source
  (``detect_stage=3``);
- write **matched** leftover islands to a QA sidecar (not the science catalog);
- mark stage-2 rows that had leftover flux with ``resid_leftover=True``.

Fusion runs once on the enriched LST-merged catalogs. Use
:func:`mark_resid_leftover_on_metacatalog` after fusion so reliability can set
``SourceQualityFlag.RESID_LEFTOVER``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lwa_catalog.analyze.healpix_cutout import (
    healpix_background_path,
    healpix_coadd_path,
    healpix_residual_path,
)
from lwa_catalog.analyze.healpix_map import (
    catalog_with_peak_from_total_flux,
    metacatalog_to_healpix,
)
from lwa_catalog.create.merge import associate_catalogs

__all__ = [
    "DETECT_STAGE_COL",
    "RESID_LEFTOVER_COL",
    "ResidualStage3Config",
    "ResidualStage3Result",
    "build_residual_healpix_map",
    "healpix_background_path",
    "healpix_residual_path",
    "mark_resid_leftover_on_metacatalog",
    "merge_residual_into_stage2",
    "summarize_residual_stage3",
    "write_residual_healpix_fits",
]

DETECT_STAGE_COL = "detect_stage"
RESID_LEFTOVER_COL = "resid_leftover"


def _import_lwa_healpix():
    try:
        import lwa_healpix
    except ImportError as exc:  # pragma: no cover
        msg = (
            "lwa_healpix is required for residual HEALPix maps; "
            "pip install 'lwa-catalog[analyze]'"
        )
        raise ImportError(msg) from exc
    return lwa_healpix


@dataclass(frozen=True)
class ResidualStage3Config:
    """Configuration for residual map build / stage-2 merge."""

    clip_negative: bool = True
    nested: bool = True
    primary_detect_stage: int = 1


@dataclass
class ResidualStage3Result:
    """Enriched stage-2 catalog plus matched-leftover sidecar."""

    summary: dict[str, Any]
    catalog: pd.DataFrame
    matched: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def summarize_residual_stage3(result: ResidualStage3Result) -> str:
    """One-line human summary of a stage-3 merge result."""
    s = result.summary
    return (
        f"residual stage3: n_stage2={s.get('n_stage2', 0)}, "
        f"n_residual={s.get('n_residual', 0)}, "
        f"n_append={s.get('n_append', 0)}, "
        f"n_matched={s.get('n_matched', 0)}, "
        f"n_out={s.get('n_out', len(result.catalog))}"
    )


def build_residual_healpix_map(
    obs_map: np.ndarray,
    catalog: pd.DataFrame | None,
    *,
    nside: int,
    nest: bool = True,
    clip_negative: bool = True,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``obs − background − Gaussian paint``; negatives clipped by default.

    Unobserved / non-finite ``obs`` pixels stay NaN. Model paint uses
    :func:`~lwa_catalog.analyze.healpix_map.catalog_with_peak_from_total_flux`
    when ``Total_flux`` is present so Peak is Jy/beam consistent with residual
    HiPS. *background* is the PyBDSF mean map on the same HEALPix grid
    (``healpix_background_{band}``); PyBDSF fits Gaussians above it, so
    omitting it leaves the coadd's (negative) zero level in the residual.
    Non-finite background pixels count as 0.
    """
    obs = np.asarray(obs_map, dtype=float)
    npix = int(obs.size)
    expected = 12 * int(nside) ** 2
    if npix != expected:
        msg = f"obs_map length {npix} != 12*nside^2 ({expected}) for nside={nside}"
        raise ValueError(msg)
    if background is not None:
        bkg = np.asarray(background, dtype=float)
        if bkg.shape != obs.shape:
            msg = f"background shape {bkg.shape} != obs_map shape {obs.shape}"
            raise ValueError(msg)
        obs = obs - np.where(np.isfinite(bkg), bkg, 0.0)

    if catalog is None or catalog.empty:
        model = np.zeros(npix, dtype=float)
    else:
        paint_cat = catalog
        if "Total_flux" in catalog.columns:
            try:
                paint_cat = catalog_with_peak_from_total_flux(catalog)
            except (KeyError, ValueError):
                paint_cat = catalog
        model = metacatalog_to_healpix(
            paint_cat,
            nside=int(nside),
            nest=bool(nest),
            profile="gaussian",
            weight_col="Peak_flux",
        )

    resid = obs - model
    finite_obs = np.isfinite(obs)
    resid = np.where(finite_obs, resid, np.nan)
    if clip_negative:
        neg = finite_obs & np.isfinite(resid) & (resid < 0.0)
        resid = resid.copy()
        resid[neg] = 0.0
    return resid.astype(float, copy=False)


def write_residual_healpix_fits(
    root: Path | str,
    band: str,
    catalog: pd.DataFrame | None,
    *,
    nside: int,
    nested: bool = True,
    clip_negative: bool = True,
    obs_path: Path | str | None = None,
    background_path: Path | str | None = None,
    overwrite: bool = False,
) -> Path:
    """Build and write residual HEALPix MAP+WEIGHT FITS for *band*.

    Weight is copied from the observed coadd. *background_path* (typically
    :func:`~lwa_catalog.analyze.healpix_cutout.healpix_background_path`) is
    subtracted along with the model when given. Returns the output path.
    """
    lwa_healpix = _import_lwa_healpix()
    root_path = Path(root)
    obs_fits = (
        Path(obs_path)
        if obs_path is not None
        else healpix_coadd_path(root_path, band, nside=nside)
    )
    if not obs_fits.is_file():
        raise FileNotFoundError(obs_fits)
    obs_map, weight, meta = lwa_healpix.read_healpix_fits(obs_fits)
    nest = bool(meta.get("nested", nested))
    file_nside = int(meta.get("nside", nside))
    background = None
    if background_path is not None:
        background, _bkg_weight, bkg_meta = lwa_healpix.read_healpix_fits(
            Path(background_path)
        )
        if int(bkg_meta.get("nside", file_nside)) != file_nside or bool(
            bkg_meta.get("nested", nest)
        ) != nest:
            msg = (
                f"background {background_path} (nside={bkg_meta.get('nside')}, "
                f"nested={bkg_meta.get('nested')}) does not match coadd "
                f"(nside={file_nside}, nested={nest})"
            )
            raise ValueError(msg)
    resid = build_residual_healpix_map(
        obs_map,
        catalog,
        nside=file_nside,
        nest=nest,
        clip_negative=clip_negative,
        background=background,
    )
    out = healpix_residual_path(root_path, band, nside=file_nside)
    return lwa_healpix.write_healpix_fits(
        out,
        resid.astype(np.float32, copy=False),
        np.asarray(weight, dtype=np.float32),
        nside=file_nside,
        nested=nest,
        coord_frame=str(meta.get("coord_frame", "icrs")),
        overwrite=overwrite,
    )


def _ensure_bmaj(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "BMAJ" in out.columns:
        out["BMAJ"] = pd.to_numeric(out["BMAJ"], errors="coerce").fillna(0.0)
    else:
        out["BMAJ"] = 0.0
    return out


def _stage2_match_key_cols(stage2: pd.DataFrame) -> list[str]:
    keys = []
    for col in ("Source_id", "Isl_id", "tile_ipix", "meta_id", "Gaus_id"):
        if col in stage2.columns:
            keys.append(col)
    return keys


def merge_residual_into_stage2(
    stage2: pd.DataFrame,
    residual: pd.DataFrame,
    *,
    band: str,
    config: ResidualStage3Config | None = None,
) -> ResidualStage3Result:
    """Append unmatched residual rows; return matched leftover sidecar.

    Matched residual islands are **not** added to the science catalog. Stage-2
    rows that associate to residual leftovers get ``resid_leftover=True``.
    Appended rows get ``detect_stage=3``.
    """
    cfg = config or ResidualStage3Config()
    warn: list[str] = []
    band_label = str(band)

    if stage2 is None:
        stage2 = pd.DataFrame()
    if residual is None:
        residual = pd.DataFrame()

    primary = _ensure_bmaj(stage2.reset_index(drop=True))
    resid = _ensure_bmaj(residual.reset_index(drop=True))

    if DETECT_STAGE_COL not in primary.columns:
        primary[DETECT_STAGE_COL] = int(cfg.primary_detect_stage)
    else:
        primary[DETECT_STAGE_COL] = (
            pd.to_numeric(primary[DETECT_STAGE_COL], errors="coerce")
            .fillna(int(cfg.primary_detect_stage))
            .astype(int)
        )
    primary[RESID_LEFTOVER_COL] = False

    summary: dict[str, Any] = {
        "band": band_label,
        "n_stage2": int(len(primary)),
        "n_residual": int(len(resid)),
        "n_append": 0,
        "n_matched": 0,
        "n_out": int(len(primary)),
    }

    if resid.empty:
        warn.append(f"{band_label}: empty residual catalog; no stage-3 append")
        return ResidualStage3Result(
            summary=summary,
            catalog=primary,
            matched=pd.DataFrame(),
            warnings=warn,
        )

    if primary.empty:
        append = resid.copy()
        append[DETECT_STAGE_COL] = 3
        append[RESID_LEFTOVER_COL] = False
        if "band" not in append.columns:
            append["band"] = band_label
        summary["n_append"] = int(len(append))
        summary["n_out"] = int(len(append))
        warn.append(f"{band_label}: empty stage-2; appending all residual rows")
        return ResidualStage3Result(
            summary=summary,
            catalog=append.reset_index(drop=True),
            matched=pd.DataFrame(),
            warnings=warn,
        )

    mapping, matched_resid = associate_catalogs(primary, resid)
    matched_base = sorted(mapping.keys())
    primary.loc[matched_base, RESID_LEFTOVER_COL] = True

    key_cols = _stage2_match_key_cols(primary)
    matched_rows: list[dict[str, Any]] = []
    for base_i, resid_idxs in mapping.items():
        base_row = primary.iloc[int(base_i)]
        for resid_i in resid_idxs:
            r = resid.iloc[int(resid_i)]
            entry: dict[str, Any] = {
                "band": band_label,
                "stage2_iloc": int(base_i),
                "residual_iloc": int(resid_i),
                "RA_stage2": float(base_row["RA"]) if "RA" in base_row.index else float("nan"),
                "DEC_stage2": float(base_row["DEC"]) if "DEC" in base_row.index else float("nan"),
                "RA_residual": float(r["RA"]) if "RA" in r.index else float("nan"),
                "DEC_residual": float(r["DEC"]) if "DEC" in r.index else float("nan"),
                "Peak_flux_residual": (
                    float(r["Peak_flux"]) if "Peak_flux" in r.index else float("nan")
                ),
                "Total_flux_residual": (
                    float(r["Total_flux"]) if "Total_flux" in r.index else float("nan")
                ),
                "BMAJ_stage2": (
                    float(base_row["BMAJ"]) if "BMAJ" in base_row.index else float("nan")
                ),
                "BMAJ_residual": (
                    float(r["BMAJ"]) if "BMAJ" in r.index else float("nan")
                ),
            }
            for col in key_cols:
                entry[f"{col}_stage2"] = base_row[col]
            matched_rows.append(entry)

    matched_df = pd.DataFrame(matched_rows)
    unmatched_mask = np.ones(len(resid), dtype=bool)
    if matched_resid:
        unmatched_mask[list(matched_resid)] = False
    append = resid.loc[unmatched_mask].copy()
    if not append.empty:
        append[DETECT_STAGE_COL] = 3
        append[RESID_LEFTOVER_COL] = False
        if "band" not in append.columns:
            append["band"] = band_label

    out = pd.concat([primary, append], ignore_index=True, sort=False)
    summary["n_append"] = int(len(append))
    summary["n_matched"] = int(len(matched_resid))
    summary["n_stage2_leftover"] = int(primary[RESID_LEFTOVER_COL].sum())
    summary["n_out"] = int(len(out))

    return ResidualStage3Result(
        summary=summary,
        catalog=out,
        matched=matched_df,
        warnings=warn,
    )


def mark_resid_leftover_on_metacatalog(
    metacatalog: pd.DataFrame,
    matched: pd.DataFrame | Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Set ``resid_leftover`` on fused rows that match residual-leftover stage-2 positions.

    *matched* is one sidecar table, a sequence of tables, or a band→table map
    from :func:`merge_residual_into_stage2`. Association uses stage-2 RA/DEC
    (preferred) or residual RA/DEC with ``BMAJ_stage2`` / ``BMAJ_residual``.
    """
    meta = metacatalog.copy() if metacatalog is not None else pd.DataFrame()
    if meta.empty:
        if RESID_LEFTOVER_COL not in meta.columns:
            meta[RESID_LEFTOVER_COL] = pd.Series(dtype=bool)
        return meta

    meta = _ensure_bmaj(meta.reset_index(drop=True))
    meta[RESID_LEFTOVER_COL] = False

    frames: list[pd.DataFrame] = []
    if isinstance(matched, Mapping):
        frames = [df for df in matched.values() if df is not None and not df.empty]
    elif isinstance(matched, pd.DataFrame):
        if not matched.empty:
            frames = [matched]
    else:
        frames = [df for df in matched if df is not None and not getattr(df, "empty", True)]

    if not frames:
        return meta

    side = pd.concat(frames, ignore_index=True, sort=False)
    ra_col = "RA_stage2" if "RA_stage2" in side.columns else "RA_residual"
    dec_col = "DEC_stage2" if "DEC_stage2" in side.columns else "DEC_residual"
    bmaj_col = "BMAJ_stage2" if "BMAJ_stage2" in side.columns else "BMAJ_residual"
    if ra_col not in side.columns or dec_col not in side.columns:
        return meta

    seeds = pd.DataFrame(
        {
            "RA": pd.to_numeric(side[ra_col], errors="coerce"),
            "DEC": pd.to_numeric(side[dec_col], errors="coerce"),
            "BMAJ": (
                pd.to_numeric(side[bmaj_col], errors="coerce").fillna(0.0)
                if bmaj_col in side.columns
                else 0.0
            ),
        }
    )
    seeds = seeds[np.isfinite(seeds["RA"]) & np.isfinite(seeds["DEC"])].reset_index(drop=True)
    if seeds.empty:
        return meta

    mapping, _ = associate_catalogs(meta, seeds)
    if mapping:
        meta.loc[sorted(mapping.keys()), RESID_LEFTOVER_COL] = True
    return meta
