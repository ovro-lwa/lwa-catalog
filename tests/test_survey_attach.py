"""Tests for photometric survey attach onto LWA metacatalogs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lwa_catalog.analyze.crossmatch_radius import CrossmatchRadiusSpec
from lwa_catalog.analyze.survey_attach import (
    attach_radio_surveys_to_metacatalog,
    attach_survey_to_metacatalog,
    normalize_survey_band_catalog,
)
from lwa_catalog.constants import NVSS_BMAJ_DEG, VLASS_BMAJ_DEG, VLSSR_BMAJ_DEG


def _meta_row(
    *,
    ra: float = 10.0,
    dec: float = 20.0,
    bmaj: float = 0.5,
    bands_present: str = "82MHz,18MHz",
    origin_band: str = "82MHz",
) -> dict:
    return {
        "meta_id": 0,
        "RA": ra,
        "DEC": dec,
        "Maj": 0.2,
        "Min": 0.1,
        "PA": 0.0,
        "BMAJ_match": bmaj,
        "origin_band": origin_band,
        "astrometry_band": origin_band,
        "bands_present": bands_present,
        "Peak_flux_82MHz": 2.0,
        "Peak_flux_18MHz": 1.0,
    }


def test_normalize_survey_maps_nvss_peak_intensity() -> None:
    raw = pd.DataFrame(
        {
            "RA": [10.0],
            "DEC": [20.0],
            "Peak_intensity": [0.05],
            "Field": ["C0100P20"],
        }
    )
    out = normalize_survey_band_catalog(raw)
    assert float(out.iloc[0]["Peak_flux"]) == 0.05
    assert np.isnan(float(out.iloc[0]["Total_flux"]))
    assert out.iloc[0]["source_file"] == "C0100P20"


def test_attach_survey_does_not_seed_unmatched_or_rewrite_astrometry() -> None:
    meta = pd.DataFrame([_meta_row()])
    nvss = pd.DataFrame(
        [
            {
                "RA": 10.01,
                "DEC": 20.0,
                "Peak_intensity": 0.04,
                "BMAJ": NVSS_BMAJ_DEG,
                "Field": "hit",
            },
            {
                "RA": 80.0,
                "DEC": 0.0,
                "Peak_intensity": 9.0,
                "BMAJ": NVSS_BMAJ_DEG,
                "Field": "miss",
            },
        ]
    )
    out = attach_survey_to_metacatalog(
        meta,
        nvss,
        "NVSS",
        lwa_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=1800.0),
        reference_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=45.0),
        footprint_filter=False,
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert float(row["RA"]) == 10.0
    assert row["origin_band"] == "82MHz"
    assert row["astrometry_band"] == "82MHz"
    assert float(row["BMAJ_match"]) == 0.5
    assert int(row["n_assoc_NVSS"]) == 1
    assert float(row["Peak_flux_NVSS"]) == 0.04
    present = str(row["bands_present"]).split(",")
    assert "82MHz" in present
    assert "18MHz" in present
    assert "NVSS" in present


def test_attach_survey_picks_brightest_total_when_available() -> None:
    meta = pd.DataFrame([_meta_row(bmaj=1.0)])
    vlass = pd.DataFrame(
        [
            {
                "RA": 10.02,
                "DEC": 20.0,
                "Peak_flux": 0.9,
                "Total_flux": 0.2,
                "BMAJ": VLASS_BMAJ_DEG,
            },
            {
                "RA": 10.03,
                "DEC": 20.0,
                "Peak_flux": 0.1,
                "Total_flux": 0.8,
                "BMAJ": VLASS_BMAJ_DEG,
            },
        ]
    )
    out = attach_survey_to_metacatalog(
        meta,
        vlass,
        "VLASS",
        lwa_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=3600.0),
        reference_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=2.5),
        footprint_filter=False,
    )
    row = out.iloc[0]
    assert int(row["n_assoc_VLASS"]) == 2
    assert float(row["Total_flux_VLASS"]) == 0.8
    assert float(row["Peak_flux_VLASS"]) == 0.1


def test_attach_survey_picks_brightest_of_multiple_hits() -> None:
    meta = pd.DataFrame([_meta_row(bmaj=1.0)])
    nvss = pd.DataFrame(
        [
            {"RA": 10.02, "DEC": 20.0, "Peak_intensity": 0.01, "BMAJ": NVSS_BMAJ_DEG},
            {"RA": 10.03, "DEC": 20.0, "Peak_intensity": 0.08, "BMAJ": NVSS_BMAJ_DEG},
        ]
    )
    out = attach_survey_to_metacatalog(
        meta,
        nvss,
        "NVSS",
        lwa_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=3600.0),
        reference_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=45.0),
        footprint_filter=False,
    )
    row = out.iloc[0]
    assert int(row["n_assoc_NVSS"]) == 2
    assert float(row["Peak_flux_NVSS"]) == 0.08


def test_attach_survey_respects_small_match_radius() -> None:
    meta = pd.DataFrame([_meta_row()])
    nvss = pd.DataFrame([{"RA": 10.2, "DEC": 20.0, "Peak_intensity": 0.05, "BMAJ": NVSS_BMAJ_DEG}])
    out = attach_survey_to_metacatalog(
        meta,
        nvss,
        "NVSS",
        lwa_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=1.0),
        reference_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=1.0),
        footprint_filter=False,
    )
    row = out.iloc[0]
    assert int(row["n_assoc_NVSS"]) == 0
    assert np.isnan(float(row["Peak_flux_NVSS"]))
    assert "NVSS" not in str(row["bands_present"])


def test_attach_radio_surveys_preserves_row_count_and_lwa_columns() -> None:
    meta = pd.DataFrame(
        [
            _meta_row(),
            _meta_row(ra=50.0, dec=0.0, bands_present="18MHz", origin_band="18MHz"),
        ]
    )
    surveys = {
        "VLASS": pd.DataFrame(
            [
                {
                    "RA": 10.0,
                    "DEC": 20.0,
                    "Peak_flux": 0.02,
                    "Total_flux": 0.03,
                    "BMAJ": VLASS_BMAJ_DEG,
                    "Component_name": "J004000+200000",
                }
            ]
        ),
        "NVSS": pd.DataFrame(
            [
                {
                    "RA": 10.0,
                    "DEC": 20.0,
                    "Peak_intensity": 0.05,
                    "BMAJ": NVSS_BMAJ_DEG,
                }
            ]
        ),
        "VLSSR": pd.DataFrame(
            [
                {
                    "RA": 50.0,
                    "DEC": 0.0,
                    "Peak_flux": 1.2,
                    "BMAJ": VLSSR_BMAJ_DEG,
                }
            ]
        ),
    }
    out = attach_radio_surveys_to_metacatalog(
        meta,
        surveys,
        lwa_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=1800.0),
        footprint_filter=False,
    )
    assert len(out) == 2
    first = out.iloc[0]
    second = out.iloc[1]
    assert float(first["RA"]) == 10.0
    assert float(second["RA"]) == 50.0
    assert first["astrometry_band"] == "82MHz"
    assert second["astrometry_band"] == "18MHz"
    assert int(first["n_assoc_VLASS"]) == 1
    assert int(first["n_assoc_NVSS"]) == 1
    assert int(first["n_assoc_VLSSR"]) == 0
    assert int(second["n_assoc_VLSSR"]) == 1
    assert "VLASS" in str(first["bands_present"])
    assert "18MHz" in str(second["bands_present"])
    assert "VLSSR" in str(second["bands_present"])


def test_attach_empty_metacatalog_stays_empty() -> None:
    survey = pd.DataFrame([{"RA": 10.0, "DEC": 20.0, "Peak_flux": 1.0, "BMAJ": VLSSR_BMAJ_DEG}])
    out = attach_survey_to_metacatalog(
        pd.DataFrame(),
        survey,
        "VLSSR",
        footprint_filter=False,
    )
    assert len(out) == 0
