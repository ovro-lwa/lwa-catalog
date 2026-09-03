"""Tests for cascaded survey position bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lwa_catalog.analyze.bootstrap import (
    advance_bootstrap_frame,
    advance_from_match,
    bijective_map_from_hits,
    bootstrap_source_counts,
    native_lwa_frame,
    select_bijective_pairs,
)
from lwa_catalog.analyze.crossmatch_radius import (
    CrossmatchRadiusSpec,
    NVSS_REFERENCE_RADIUS_LOCALIZATION,
    VLSSR_REFERENCE_RADIUS_FIXED,
)
from lwa_catalog.analyze.nvss import match_catalog_to_nvss
from lwa_catalog.analyze.survey_attach import attach_radio_surveys_to_metacatalog
from lwa_catalog.analyze.vlass import match_catalog_to_vlass
from lwa_catalog.analyze.vlssr import VlssrMatchResult, match_catalog_to_vlssr
from lwa_catalog.constants import (
    NVSS_BMAJ_DEG,
    VLASS_BMAJ_DEG,
    VLSSR_BMAJ_DEG,
    VLSSR_POSITION_ERROR_ARCSEC,
)


def _meta_row(meta_id: int, ra: float, dec: float, *, bmaj: float = 0.01) -> dict:
    return {
        "meta_id": meta_id,
        "RA": ra,
        "DEC": dec,
        "BMAJ_match": bmaj,
        "bands_present": "Full,Blue",
        "Peak_flux": 1.0,
        "origin_band": "Full",
    }


def test_select_bijective_pairs_excludes_multi() -> None:
    meta_flags = pd.DataFrame(
        {
            "n_vlssr": [1, 2, 1, 0],
            "vlssr_positions": [[0], [0, 1], [2], []],
        }
    )
    ref_flags = pd.DataFrame(
        {
            "vlssr_pos": [0, 1, 2],
            "n_meta": [1, 1, 2],  # pos 2 oversplit
        }
    )
    pairs = select_bijective_pairs(
        meta_flags,
        ref_flags,
        n_ref_col="n_vlssr",
        positions_col="vlssr_positions",
        ref_pos_col="vlssr_pos",
    )
    assert pairs == {0: 0}


def test_advance_adopts_vlssr_coords_and_fixed_radius() -> None:
    lwa_radius = CrossmatchRadiusSpec(mode="localization", sigma_scale=3.0, default_arcsec=5.0)
    meta = pd.DataFrame([_meta_row(0, 10.0, 20.0), _meta_row(1, 11.0, 21.0)])
    frame = native_lwa_frame(meta, lwa_radius)
    assert list(frame["bootstrap_source"]) == ["LWA", "LWA"]

    footprint = pd.DataFrame(
        {
            "RA": [10.001, 99.0],
            "DEC": [20.001, 99.0],
            "BMAJ": [VLSSR_BMAJ_DEG, VLSSR_BMAJ_DEG],
        }
    )
    advanced = advance_bootstrap_frame(
        frame,
        footprint,
        {0: 0},
        radius_spec=VLSSR_REFERENCE_RADIUS_FIXED,
        source="VLSSR",
    )
    assert advanced.iloc[0]["RA"] == pytest.approx(10.001)
    assert advanced.iloc[0]["DEC"] == pytest.approx(20.001)
    assert advanced.iloc[0]["BMAJ"] == pytest.approx(VLSSR_POSITION_ERROR_ARCSEC / 3600.0)
    assert advanced.iloc[0]["bootstrap_source"] == "VLSSR"
    # Non-bijective row unchanged.
    assert advanced.iloc[1]["RA"] == pytest.approx(11.0)
    assert advanced.iloc[1]["bootstrap_source"] == "LWA"


def test_advance_adopts_nvss_localization_radius() -> None:
    frame = pd.DataFrame(
        [
            {
                "RA": 10.0,
                "DEC": 20.0,
                "BMAJ": 3.0 / 3600.0,
                "bootstrap_source": "VLSSR",
                "bootstrap_ref_pos": 0,
            }
        ]
    )
    e_ra = 0.0001
    e_dec = 0.0002
    footprint = pd.DataFrame(
        {
            "RA": [10.05],
            "DEC": [20.05],
            "E_RA": [e_ra],
            "E_DEC": [e_dec],
            "BMAJ": [NVSS_BMAJ_DEG],
        }
    )
    advanced = advance_bootstrap_frame(
        frame,
        footprint,
        {0: 0},
        radius_spec=NVSS_REFERENCE_RADIUS_LOCALIZATION,
        source="NVSS",
    )
    assert advanced.iloc[0]["RA"] == pytest.approx(10.05)
    assert advanced.iloc[0]["BMAJ"] == pytest.approx(float(np.hypot(e_ra, e_dec)))
    assert advanced.iloc[0]["bootstrap_source"] == "NVSS"


def test_cascade_fallback_keeps_vlssr_when_no_nvss() -> None:
    frame = pd.DataFrame(
        [
            {
                "RA": 10.001,
                "DEC": 20.001,
                "BMAJ": 3.0 / 3600.0,
                "bootstrap_source": "VLSSR",
                "bootstrap_ref_pos": 0,
            }
        ]
    )
    # Empty bijective map → unchanged.
    advanced = advance_bootstrap_frame(
        frame,
        pd.DataFrame({"RA": [1.0], "DEC": [2.0], "E_RA": [0.001], "E_DEC": [0.001]}),
        {},
        radius_spec=NVSS_REFERENCE_RADIUS_LOCALIZATION,
        source="NVSS",
    )
    assert advanced.iloc[0]["bootstrap_source"] == "VLSSR"
    assert advanced.iloc[0]["RA"] == pytest.approx(10.001)


def test_nvss_match_with_bootstrapped_frame_recovers_offset() -> None:
    from lwa_catalog.analyze.nvss import NvssMatchConfig

    # LWA coords are offset; VLSSR-bootstrap frame sits on the true NVSS position.
    meta = pd.DataFrame([_meta_row(0, 10.05, 20.0, bmaj=0.001)])
    nvss = pd.DataFrame(
        [
            {
                "RA": 10.0,
                "DEC": 20.0,
                "E_RA": 0.0005,
                "E_DEC": 0.0005,
                "Peak_intensity": 0.05,
                "BMAJ": NVSS_BMAJ_DEG,
                "BMIN": NVSS_BMAJ_DEG,
            }
        ]
    )
    cfg = NvssMatchConfig(
        lwa_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=1.0),
        reference_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=1.0),
    )
    native = match_catalog_to_nvss(meta, nvss=nvss, config=cfg)
    assert int(native.summary["n_meta_matched"]) == 0

    boot = pd.DataFrame(
        {
            "RA": [10.0],
            "DEC": [20.0],
            "BMAJ": [3.0 / 3600.0],
            "bootstrap_source": ["VLSSR"],
            "bootstrap_ref_pos": [0],
        }
    )
    recovered = match_catalog_to_nvss(meta, nvss=nvss, config=cfg, lwa_match=boot)
    assert int(recovered.summary["n_meta_matched"]) == 1


def test_advance_from_match_vlssr_roundtrip() -> None:
    meta = pd.DataFrame([_meta_row(0, 10.0, 20.0, bmaj=0.5)])
    vlssr = pd.DataFrame(
        [
            {
                "RA": 10.001,
                "DEC": 20.0,
                "Peak_flux": 1.0,
                "BMAJ": VLSSR_BMAJ_DEG,
                "BMIN": VLSSR_BMAJ_DEG,
            }
        ]
    )
    from lwa_catalog.analyze.vlssr import VlssrMatchConfig

    result = match_catalog_to_vlssr(
        meta,
        vlssr=vlssr,
        config=VlssrMatchConfig(
            target="metacatalog",
            lwa_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=30.0),
            reference_radius=VLSSR_REFERENCE_RADIUS_FIXED,
        ),
    )
    assert isinstance(result, VlssrMatchResult)
    assert int(result.summary["n_meta_matched"]) == 1
    frame = native_lwa_frame(
        meta, CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=30.0)
    )
    advanced = advance_from_match(
        frame, result, VLSSR_REFERENCE_RADIUS_FIXED, source="VLSSR"
    )
    assert advanced.iloc[0]["bootstrap_source"] == "VLSSR"
    assert advanced.iloc[0]["RA"] == pytest.approx(10.001)
    counts = bootstrap_source_counts(advanced)
    assert counts.get("VLSSR") == 1


def test_vlass_match_with_nvss_bootstrapped_frame() -> None:
    meta = pd.DataFrame([_meta_row(0, 10.1, 20.0, bmaj=0.001)])
    vlass = pd.DataFrame(
        [
            {
                "RA": 10.0,
                "DEC": 20.0,
                "E_RA": 1e-5,
                "E_DEC": 1e-5,
                "Peak_flux": 0.05,
                "Total_flux": 0.08,
                "BMAJ": VLASS_BMAJ_DEG,
                "BMIN": VLASS_BMAJ_DEG,
                "Component_name": "c0",
                "S_Code": "S",
            }
        ]
    )
    from lwa_catalog.analyze.vlass import VlassMatchConfig

    boot = pd.DataFrame(
        {
            "RA": [10.0],
            "DEC": [20.0],
            "BMAJ": [0.0005],
            "bootstrap_source": ["NVSS"],
            "bootstrap_ref_pos": [0],
        }
    )
    result = match_catalog_to_vlass(
        meta,
        vlass=vlass,
        config=VlassMatchConfig(
            apply_quality_filter=False,
            lwa_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=1.0),
            reference_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=1.0),
        ),
        lwa_match=boot,
    )
    assert int(result.summary["n_meta_matched"]) == 1


def test_attach_cascade_bootstrap_does_not_mutate_ra() -> None:
    meta = pd.DataFrame([_meta_row(0, 10.0, 20.0, bmaj=0.5)])
    orig_ra = float(meta.iloc[0]["RA"])
    vlssr = pd.DataFrame(
        [
            {
                "RA": 10.001,
                "DEC": 20.0,
                "Peak_flux": 2.0,
                "BMAJ": VLSSR_BMAJ_DEG,
            }
        ]
    )
    nvss = pd.DataFrame(
        [
            {
                "RA": 10.002,
                "DEC": 20.0,
                "Peak_intensity": 0.05,
                "E_RA": 0.0002,
                "E_DEC": 0.0002,
                "BMAJ": NVSS_BMAJ_DEG,
            }
        ]
    )
    vlass = pd.DataFrame(
        [
            {
                "RA": 10.0025,
                "DEC": 20.0,
                "Peak_flux": 0.04,
                "Total_flux": 0.06,
                "E_RA": 1e-5,
                "E_DEC": 1e-5,
                "BMAJ": VLASS_BMAJ_DEG,
                "Component_name": "c0",
            }
        ]
    )
    out = attach_radio_surveys_to_metacatalog(
        meta,
        {"VLSSR": vlssr, "NVSS": nvss, "VLASS": vlass},
        lwa_radius=CrossmatchRadiusSpec(mode="fixed", fixed_arcsec=60.0),
        cascade_bootstrap=True,
    )
    assert float(out.iloc[0]["RA"]) == pytest.approx(orig_ra)
    assert len(out) == 1
    assert "Peak_flux_VLSSR" in out.columns or "Total_flux_VLSSR" in out.columns


def test_bijective_map_from_hits() -> None:
    assert bijective_map_from_hits({0: [1], 1: [0, 2]}, {1: [0], 0: [1], 2: [1]}) == {
        0: 1
    }
