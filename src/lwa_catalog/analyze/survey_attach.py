"""Attach external radio surveys onto an LWA metacatalog as photometric bands.

Unmatched survey sources are not seeded as new rows. Top-level LWA astrometry
and ``BMAJ_match`` are left unchanged. Stored survey flux is the brightest
associated hit (preferring ``Total_flux`` when available, else peak);
``n_assoc_{band}`` counts every hit inside the match radius.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from lwa_catalog.analyze.crossmatch_radius import (
    LWA_CROSSMATCH_RADIUS_BEAM,
    NVSS_REFERENCE_RADIUS_LOCALIZATION,
    VLASS_REFERENCE_RADIUS_LOCALIZATION,
    VLSSR_REFERENCE_RADIUS_FIXED,
    CrossmatchRadiusSpec,
    apply_match_radius,
    match_radius_deg,
)
from lwa_catalog.analyze.nvss import _footprint_filter_nvss
from lwa_catalog.analyze.vlass import _footprint_filter_vlass
from lwa_catalog.analyze.vlssr import _footprint_filter_vlssr
from lwa_catalog.constants import (
    NVSS_DEC_MIN_DEG,
    SUBBAND_METACATALOG_FLUX_FIELDS,
    VLASS_DEC_MIN_DEG,
)
from lwa_catalog.create.merge import associate_band_into_metacatalog

RADIO_SURVEY_BANDS: tuple[str, ...] = ("VLASS", "NVSS", "VLSSR")

_DEFAULT_REFERENCE_RADIUS: dict[str, CrossmatchRadiusSpec] = {
    "VLASS": VLASS_REFERENCE_RADIUS_LOCALIZATION,
    "NVSS": NVSS_REFERENCE_RADIUS_LOCALIZATION,
    "VLSSR": VLSSR_REFERENCE_RADIUS_FIXED,
}


def normalize_survey_band_catalog(survey_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with merge-ready ``Peak_flux`` / ``Total_flux`` / ``source_file``."""
    out = survey_df.copy()
    if "Peak_flux" not in out.columns and "Peak_intensity" in out.columns:
        out["Peak_flux"] = pd.to_numeric(out["Peak_intensity"], errors="coerce")
    if "Peak_flux" not in out.columns:
        out["Peak_flux"] = np.nan
    if "Total_flux" not in out.columns:
        out["Total_flux"] = np.nan
    if "source_file" not in out.columns:
        if "Component_name" in out.columns:
            out["source_file"] = out["Component_name"].astype(str)
        elif "Field" in out.columns:
            out["source_file"] = out["Field"].astype(str)
        else:
            out["source_file"] = ""
    return out


def _bands_present_order(meta_df: pd.DataFrame, extra_bands: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    if "bands_present" in meta_df.columns:
        for val in meta_df["bands_present"].tolist():
            for part in str(val).split(","):
                text = part.strip()
                if text and text not in seen:
                    seen.append(text)
    for band in extra_bands:
        if band not in seen:
            seen.append(band)
    return tuple(seen)


def _footprint_filter_survey(
    band: str,
    survey_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    if band == "NVSS":
        return _footprint_filter_nvss(survey_df, meta_df, dec_min_deg=NVSS_DEC_MIN_DEG)
    if band == "VLASS":
        return _footprint_filter_vlass(survey_df, meta_df, dec_min_deg=VLASS_DEC_MIN_DEG)
    if band == "VLSSR":
        return _footprint_filter_vlssr(survey_df, meta_df)
    return survey_df


def attach_survey_to_metacatalog(
    meta_df: pd.DataFrame,
    survey_df: pd.DataFrame,
    band: str,
    *,
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM,
    reference_radius: CrossmatchRadiusSpec | None = None,
    band_fields: Sequence[str] = SUBBAND_METACATALOG_FLUX_FIELDS,
    color_bands: Sequence[str] | None = None,
    footprint_filter: bool = True,
) -> pd.DataFrame:
    """Attach one external catalog onto *meta_df* as photometric band *band*.

    Unmatched survey rows are discarded. Top-level ``RA``/``DEC``/shape,
    ``BMAJ_match``, ``origin_band``, and ``astrometry_band`` are not updated.
    """
    if reference_radius is None:
        reference_radius = _DEFAULT_REFERENCE_RADIUS.get(band, LWA_CROSSMATCH_RADIUS_BEAM)

    work = meta_df.reset_index(drop=True)
    if color_bands is None:
        color_bands = _bands_present_order(work, (band,))
    elif band not in color_bands:
        color_bands = tuple(color_bands) + (band,)

    survey = normalize_survey_band_catalog(survey_df)
    if footprint_filter:
        survey = _footprint_filter_survey(band, survey, work)
    survey = apply_match_radius(survey, reference_radius)

    if work.empty:
        return associate_band_into_metacatalog(
            work,
            survey,
            band,
            assoc_bands=(band,),
            band_fields=band_fields,
            color_bands=color_bands,
            primary_flux=False,
            astrometry_from_highest_frequency=False,
            append_unmatched=False,
            update_bmaj_match=False,
            representative="peak_flux",
        )

    base_bmaj = np.asarray(
        [match_radius_deg(row, lwa_radius) for _, row in work.iterrows()],
        dtype=float,
    )
    return associate_band_into_metacatalog(
        work,
        survey,
        band,
        assoc_bands=(band,),
        band_fields=band_fields,
        color_bands=color_bands,
        primary_flux=False,
        astrometry_from_highest_frequency=False,
        append_unmatched=False,
        update_bmaj_match=False,
        representative="peak_flux",
        base_bmaj=base_bmaj,
    )


def attach_radio_surveys_to_metacatalog(
    meta_df: pd.DataFrame,
    surveys: Mapping[str, pd.DataFrame],
    *,
    lwa_radius: CrossmatchRadiusSpec = LWA_CROSSMATCH_RADIUS_BEAM,
    reference_radii: Mapping[str, CrossmatchRadiusSpec] | None = None,
    band_order: Sequence[str] = RADIO_SURVEY_BANDS,
    band_fields: Sequence[str] = SUBBAND_METACATALOG_FLUX_FIELDS,
    footprint_filter: bool = True,
) -> pd.DataFrame:
    """Attach VLASS, NVSS, and/or VLSSR onto *meta_df* as photometric bands.

    Keys in *surveys* should be ``VLASS``, ``NVSS``, and/or ``VLSSR``. Missing
    keys are skipped. Row count is unchanged.
    """
    radii = dict(_DEFAULT_REFERENCE_RADIUS)
    if reference_radii is not None:
        radii.update(reference_radii)

    present = [band for band in band_order if band in surveys]
    color_bands = _bands_present_order(meta_df, present)
    out = meta_df
    for band in present:
        out = attach_survey_to_metacatalog(
            out,
            surveys[band],
            band,
            lwa_radius=lwa_radius,
            reference_radius=radii[band],
            band_fields=band_fields,
            color_bands=color_bands,
            footprint_filter=footprint_filter,
        )
    return out
